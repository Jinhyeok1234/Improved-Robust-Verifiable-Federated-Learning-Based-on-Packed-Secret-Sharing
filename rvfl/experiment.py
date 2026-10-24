"""Setup shared by run.py and the end-to-end worker processes: seeding, VPSS configurations,
party construction and the in-process round drivers."""
import math
import random

import numpy as np
import torch
import vpss_module as vpss

from . import attacks
from .baselines import BreaClient, BreaServer
from .party import Client, Server


def seed_everything(s: int):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def seed_round(seed: int, iteration: int, party_id: int):
    """One seed per (round, party) before its local training: the data shuffling, the
    poisoning draw and the model-poisoning attacks that follow draw from it, so on the CPU a
    point run twice gives the same updates in every round, whatever else ran in between (on a
    GPU some kernels are not deterministic)."""
    seed_everything(seed + iteration * 4096 + (party_id + 1))


def resolve_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def vpss_config(t, k, n, secret_count, deterministic_rand=False):
    """Message points 1..secret_count, share points n + 1..2n (client i holds n + 1 + i)."""
    config = vpss.VPSSConfig()
    config.t = t
    config.k = k
    config.N = n
    config.padType = vpss.PaddingMode.Zero
    secrets = vpss.BigIntVector()
    for i in range(1, secret_count + 1):
        secrets.append(vpss.BigInt(i))
    shares = vpss.BigIntVector()
    for i in range(n + 1, 2 * n + 1):
        shares.append(vpss.BigInt(i))
    config.secretIndices = secrets
    config.shareIndices = shares
    if deterministic_rand:
        config.deterministicRand = True
    return config


def vpss_configs(n, t1, k1, t2, k2):
    """(config1, config2, config1_det) of a round with every client taking part."""
    secret_count = max(t1, t2)
    return (vpss_config(t1, k1, n, secret_count), vpss_config(t2, k2, n, secret_count),
            vpss_config(t1, k1, n, secret_count, deterministic_rand=True))


def thresholds(protocol, n, k1, t=None):
    """(t1, k1, t2, k2): t1 = t2 = n // 3 unless t is given, k2 = 1. BREA packs nothing and
    reconstructs the pairwise distances (products of two shares) directly: k1 = 1,
    t2 = 2 t1 - 1."""
    t1 = n // 3 if t is None else t
    if protocol == "brea":
        return t1, 1, 2 * t1 - 1, 1
    if not 1 <= k1 <= t1:
        raise ValueError(f"k1 = {k1} must be in [1, t1 = {t1}] (N = {n})")
    return t1, k1, t1, 1


def make_parties(protocol, n, addr_map, common, pp, configs):
    """The server and the n clients of an in-process run, with every party in addr_map."""
    server_cls, client_cls = (BreaServer, BreaClient) if protocol == "brea" else (Server, Client)
    server = server_cls(ip=addr_map[-1][0], port=addr_map[-1][1], **common)
    clients = [client_cls(id=i, ip=addr_map[i][0], port=addr_map[i][1], server_addr=addr_map[-1], **common)
               for i in range(n)]
    for party in [server] + clients:
        party.party_map = dict(addr_map)
        party.set_VPSS_params(pp, *configs)
    return server, clients


def set_up_attack(clients, attack, malicious_rate):
    """Malicious ids, with their data poisoning installed. attack None: no attack."""
    if attack is None:
        return []
    if attack not in attacks.ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; one of {attacks.ATTACKS}")
    ids = attacks.malicious_ids(len(clients), malicious_rate)
    cfg = attacks.poisoning_config(attack)
    for cid in ids:
        if cfg is not None:
            clients[cid].set_poisoning(cfg)
        clients[cid].submission_scale = attacks.submission_scale(attack, len(clients))
    return ids


def train_round(server, clients, iteration, seed, lr_local, attack, malicious, train_server=True):
    """Local training of every party (seeded per party), then the model-poisoning attack."""
    if train_server:
        seed_round(seed, iteration, server.id)
        server.train_phase(epoch=1, lr=lr_local)
    for c in clients:
        seed_round(seed, iteration, c.id)
        c.train_phase(epoch=1, lr=lr_local)
    if attack in attacks.MODEL_POISON_ATTACKS and malicious:
        attacks.apply_model_poisoning(clients, malicious, attack,
                                      g_server=server.g_server if train_server else None)


def run_rvfl_round(server, clients, lr_global, threads):
    """One protocol round with every party in this process (after train_round)."""
    sp = server.enrollment_phase(threads)
    cps = [c.enrollment_phase(threads) for c in clients]
    next(sp)                                  # the server publishes g_S
    for cp in cps:
        next(cp)                              # g_S received; cosine reported or shares dealt
    if server.cos_at_enrollment:
        next(sp)                              # cosines received (and the participants published)
        for cp in cps:
            next(cp)                          # first-level shares dealt
    for cp in cps:
        next(cp)                              # first-level shares received
    for _ in sp:
        pass

    cps = [c.verification_phase(threads) for c in clients]
    for cp in cps:
        next(cp)                              # second-level shares and commitments sent
    for cp in cps:
        next(cp)                              # verdicts sent
    for cp in cps:
        next(cp)                              # s_tilde sent
    server.verification_phase(threads)        # trust scores published
    for cp in cps:
        next(cp)                              # trust scores received

    cps = [c.aggregation_phase() for c in clients]
    for cp in cps:
        next(cp)                              # aggregated shares sent
    server.aggregation_phase(lr_global, threads)
    for cp in cps:
        next(cp)                              # new global model received


def run_brea_round(server, clients, lr_global, threads, byzantine_rate):
    cps = [c.brea_phase(threads) for c in clients]
    sp = server.brea_phase(lr_global=lr_global, byzantine_rate=byzantine_rate, thread_count=threads)
    for cp in cps:
        next(cp)                              # shares and commitments dealt
    for cp in cps:
        next(cp)                              # shares verified, pairwise-distance shares sent
    next(sp)                                  # distances reconstructed, Multi-Krum set published
    for cp in cps:
        next(cp)                              # aggregated shares sent
    next(sp)                                  # aggregate reconstructed, new model published
    for cp in cps:
        next(cp)                              # new global model received


def val_max_of(q_int):
    """sqrt(q / 2): the bound on the norm of a quantized vector, so its square stays below q / 2."""
    return math.isqrt(q_int // 2)
