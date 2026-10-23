"""
Single-party worker for the netns-based end-to-end experiment.

Each party (server + N clients) runs as its own process, spawned by main.py. At
startup the worker enters its pre-created network namespace (so its sockets bind to
that namespace's tc-shaped veth), rebuilds its party from primitive parameters, wires
the cross-process BSP barriers / compute lock, and runs the protocol for a few
iterations. Bandwidth and propagation delay are enforced by `tc` on the veth links
(see net_setup.py); the transport itself (party.PlainTCPCommsManager) does no shaping.

Not launched directly — main.py spawns worker_main() via multiprocessing (spawn).
"""
import os
import json
import time
import warnings
from pathlib import Path

# Silence the noisy "Can't initialize NVML" UserWarning (sudo strips the env that lets torch
# read NVML; CUDA compute via cudart still works). Set before torch is imported in worker_main.
warnings.filterwarnings("ignore", message="Can't initialize NVML")


# --- namespace entry -------------------------------------------------------
def _enter_netns(ns_name: str):
    """Move this process into the named network namespace before any socket is created."""
    CLONE_NEWNET = 0x40000000
    fd = os.open(f"/run/netns/{ns_name}", os.O_RDONLY)
    try:
        if hasattr(os, "setns"):                       # Python 3.12+
            os.setns(fd, CLONE_NEWNET)
        else:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            if libc.setns(fd, CLONE_NEWNET) != 0:
                e = ctypes.get_errno()
                raise OSError(e, os.strerror(e))
    finally:
        os.close(fd)


# --- crypto/config rebuild (duplicated from main.py to avoid importing __main__) ---
def _build_pp(vpss, crypto):
    # [B2] crypto is the spec dict crypto_params.build_pp returned in the parent, so the
    # child gets the same generators and the same backend (Z_p* draws g and h at random,
    # so they cannot be re-derived here).
    import crypto_params
    return crypto_params.pp_from_spec(vpss, crypto)


def _build_config(vpss, t, k, N, secret_ints, share_ints, deterministic_rand=False):
    cfg = vpss.VPSSConfig()
    cfg.t = t
    cfg.k = k
    cfg.N = N
    cfg.padType = vpss.PaddingMode.Zero
    si = vpss.BigIntVector()
    for v in secret_ints:
        si.append(vpss.BigInt(int(v)))
    sh = vpss.BigIntVector()
    for v in share_ints:
        sh.append(vpss.BigInt(int(v)))
    cfg.secretIndices = si
    cfg.shareIndices = sh
    if deterministic_rand:
        cfg.deterministicRand = True
    return cfg


def _exhaust(x):
    """Client phases are generators (drive to completion); server phases return None."""
    if x is not None and hasattr(x, "__next__"):
        for _ in x:
            pass


def worker_main(spec, all_barrier, client_barrier, compute_lock, start_gate, end_gate):
    # 1. enter our namespace FIRST, so every socket binds to the shaped veth
    _enter_netns(spec["netns"])

    # 2. heavy imports (safe after setns — none open network sockets at import time)
    import torch
    import vpss_module as vpss
    import party as P
    from models import get_model
    from trainer import LocalTrainer

    is_server = spec["is_server"]
    dev = torch.device(spec["device"])
    tc = spec["thread_count"]
    use_perp = spec["use_perp"]

    # 3. process-wide party module state (BSP + backend)
    P.NETWORK_CONFIG["backend"] = "netns_tcp"
    P._SERIALIZE_COMPUTE = True
    P._COMPUTE_LOCK = compute_lock
    P.BROADCAST_PARALLEL = spec["broadcast_parallel"]
    P.SERVER_COMPUTES_SHARES = spec["server_computes_shares"]

    # 4. rebuild crypto material + configs
    pp = _build_pp(vpss, spec["crypto"])
    config1 = _build_config(vpss, spec["t1"], spec["k1"], spec["N"],
                            spec["secret_idx"], spec["share_idx"])
    config2 = _build_config(vpss, spec["t2"], spec["k2"], spec["N"],
                            spec["secret_idx"], spec["share_idx"])
    config1_det = None
    if not spec["server_computes_shares"]:
        config1_det = _build_config(vpss, spec["t1"], spec["k1"], spec["N"],
                                    spec["secret_idx"], spec["share_idx"],
                                    deterministic_rand=True)

    # 5. model + trainer + party
    model = get_model(spec["model_name"]).to(dev)
    trainer = LocalTrainer(model, device=dev)
    ip, port = spec["ip"], spec["port"]
    common = dict(model_name=spec["model_name"], dataset_name=spec["dataset_name"],
                  model=model, trainer=trainer, partition_path=spec["partition_path"],
                  device=dev, dataset_path=spec["dataset_path"],
                  batch_size=spec["batch_size"], val_max=spec["val_max"])
    addr_map = {int(k): tuple(v) for k, v in spec["addr_map"].items()}  # JSON sent str keys
    if is_server:
        party = P.Server(ip=ip, port=port, **common)
    else:
        party = P.Client(id=spec["party_id"], ip=ip, port=port,
                         server_addr=addr_map[-1], **common)

    party.party_map = addr_map
    # Diagnostic net-fault log (created lazily only if a reset/drop actually happens).
    if hasattr(party.comms, "set_reset_log"):
        party.comms.set_reset_log(str(Path(spec["log_dir"]) / f"netreset_{spec['party_id']}.log"))
    party.set_VPSS_params(pp, config1, config2, config1_det)
    # Per-round participant filter (RVFL_E2E_ALPHA). Off, the party keeps the fixed configs
    # above and the protocol runs exactly as without this block.
    party.participant_filter = spec.get("participant_filter", False)
    if party.participant_filter:
        party.set_vpss_shape(spec["N"], spec["k1"], spec["k2"],   # configs rebuilt per round
                             corrupt_bound=spec.get("corrupt_bound", 0),
                             vark=spec.get("vark", ""))
        if is_server:
            party.min_participants = spec.get("min_participants", 0)
            party.addback_strategy = spec.get("filter_addback", "random")
            party.cost_fit = spec.get("cost_fit")        # variable k plan 2 only
            party.cost_points = spec.get("cost_points")  # variable k plan 3 only
    party.weights = torch.load(spec["weights_path"], map_location="cpu")
    # Resume (RVFL_E2E_RESUME): the shared vector holds the parameters only, so the global
    # model's batch-norm running statistics are put back from the server's checkpoint of the
    # resumed round. They matter for the server's model only (test accuracy): training runs
    # in train mode, where batch norm normalises with each batch's own statistics, so the
    # running statistics a client holds never enter its update.
    if spec.get("bn_state_path"):
        _bufs = dict(model.named_buffers())
        for _name, _val in torch.load(spec["bn_state_path"], map_location="cpu").items():
            if _name in _bufs and _bufs[_name].dtype.is_floating_point:
                _bufs[_name].data.copy_(_val.to(_bufs[_name].device))
    party._all_barrier = all_barrier
    party._client_barrier = None if is_server else client_barrier

    # 6. optional attack (opt-in; only clients, only when configured malicious)
    if (not is_server) and spec.get("poison") is not None and spec["party_id"] in spec["malicious_ids"]:
        from dataloader import PoisoningConfig
        party.set_poisoning(PoisoningConfig(**spec["poison"]))

    log_path = Path(spec["log_dir"]) / f"worker_{spec['party_id']}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # a resumed combo keeps the rounds already logged (main.py has cut any partial round)
    fout = log_path.open("a" if spec.get("resume") else "w", encoding="utf-8")

    # 7. iterations
    seed_base = spec.get("seed")
    for it in range(spec["iter_start"], spec["num_iterations"] + 1):
        # --- train: serialized (compute lock) but NOT part of the timed protocol ---
        P._acquire_compute()
        try:
            if seed_base is not None:
                # RVFL_E2E_SEED: one seed per (round, party). The DataLoader shuffle and the
                # augmentation draw from these generators, so two sweeps with the same seed and
                # the same start weights give every client the same update, hence the same
                # cosine, in every round; that is what lets runs at different alpha be
                # compared round by round. Unset, nothing is seeded, as before.
                _s = seed_base + it * 4096 + (spec["party_id"] + 1)
                import random as _random
                _random.seed(_s)
                try:
                    import numpy as _np
                    _np.random.seed(_s % (2 ** 32))
                except ImportError:
                    pass
                torch.manual_seed(_s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(_s)
            party.train_phase(epoch=1, lr=spec["lr_local"], logging=True)
        finally:
            P._release_compute()

        # --- protocol: all parties released together, then run under the compute lock ---
        if is_server and party.participant_filter:
            # per-round seed for the random add-back, so a repeated run picks the same clients
            party.addback_seed = None if seed_base is None else seed_base + it
        start_gate.wait()                         # sync t0 with the parent
        P._acquire_compute()
        party._lock_dropped = False               # held now; comm methods drop it, stamps re-take it
        party._lock_wait_s = 0.0                  # per-round total, reported below
        try:
            if is_server:
                _exhaust(party.enrollment_phase(tc, logging=True))
                _exhaust(party.verification_phase(use_perp, tc, logging=True))
                _exhaust(party.aggregation_phase(spec["lr_global"], use_error_correction=True,
                                                 use_perp=use_perp, thread_count=tc, logging=True))
            else:
                _exhaust(party.enrollment_phase(use_perp, tc, logging=True))
                _exhaust(party.verification_phase(use_perp, tc, logging=True))
                _exhaust(party.aggregation_phase(logging=True))
        finally:
            if not getattr(party, "_lock_dropped", False):
                P._release_compute()
            party._lock_dropped = False
        end_gate.wait()                           # sync t1 with the parent

        # --- record this iteration's per-phase timing and byte counts -----------------
        # [B6] comm_logs_bytes used to be dropped here, so the netns sweep could report time
        # but not volume and the volume had to come from a separate in-process run. Both
        # analyzers select by key (comp_log_analyzer_revised reads time_logs_ms,
        # comm_log_analyzer reads comm_logs_bytes), so one record carrying both is readable
        # by each of them unchanged. values_log is still dropped: it is not used here.
        rec = {"party_id": spec["party_id"], "iteration": it,
               "time_logs_ms": party._time_logs,
               "comm_logs_bytes": getattr(party, "_comm_logs", {}),
               # time this party spent waiting for the compute lock (other parties' serialized
               # computation); already excluded from every timer above, kept for inspection
               "lock_wait_ms": round(getattr(party, "_lock_wait_s", 0.0) * 1000, 3)}
        if is_server and party.participant_filter:
            # Who took part this round and on what cosines; tabulate_filter.py reads these.
            # Written only with the filter on, so a filter-off record is unchanged.
            vals = getattr(party, "_value_logs", {}) or {}
            participants = list(party.participants or [])
            rec["values"] = {
                "num_participants": vals.get("num_participants", len(participants)),
                "participants": participants,
                "round_t": vals.get("round_t"),
                "round_k1": vals.get("round_k1"),
                "num_positive": vals.get("num_positive"),
                # variable k plans 2 and 3: the chosen size (and k, t) with its predicted time
                "round_size_choice": vals.get("round_size_choice"),
                "cos_reported": vals.get("cos_reported"),
            }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        party._time_logs = {}
        party._comm_logs = {'unicast': {}, 'broadcast': {}}
        if is_server and hasattr(party, "_value_logs"):
            party._value_logs = {}          # discard values_log (unused in the netns comm test)

        # --- server: checkpoint the updated global model ---
        if is_server and spec.get("ckpt_dir"):
            torch.nn.utils.vector_to_parameters(party.weights.to(dev), model.parameters())
            ck = Path(spec["ckpt_dir"]) / f"{spec['dataset_name']}_{spec['model_name']}_N{spec['N']}_Iter{it}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ck)

    fout.close()
    try:
        party.comms.close()
    except Exception:
        pass