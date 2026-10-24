#!/usr/bin/env python3
"""Run the experiments listed in TOML files.

    python run.py experiments/accuracy.toml                 # every point of every [[run]]
    python run.py experiments/cost.toml --only table6       # the [[run]] entries named table6
    python run.py experiments/cost.toml --list              # print the points and stop
    sudo -E "$(which python)" run.py experiments/e2e.toml   # end-to-end points need root

Each [[run]] entry of a file is one experiment. A setting given as a list is varied: the
entry expands into one point per combination, in the order the lists are written (the first
list varies slowest). One folder per point is written under --out
(<out>/<name>/<point>/, see rvfl/logs.py for what it holds). A point whose folder already
holds a finished run (status "ok") is skipped; any other is run from the start. --overwrite
runs every point again.

Kinds and their settings (default in brackets):

  every kind
    name            folder name of the experiment (required)
    kind            "train" | "cost" | "e2e" | "curve_bench" (required)
    n               number of clients N (required)
    seed            [0] fixes the data partition, the initial model, every local training
                    (one seed per round and party) and the filter's add-back draw
    dataset         ["CIFAR10"]         model ["ResNet20"]
    data_dir        ["./dataset"]       partition_dir ["./partitions"]
    device          ["auto"] (CUDA when available)
    batch_size [64]  lr_local [0.1]  lr_global [1.0]  server_samples [100] (root dataset size)
    iid             data split: [false] for train and cost, [true] for e2e
    threads         [60] threads of the VPSS primitives
    backend         ["ec"] commitment group ("zp": legacy, slower, unused)
    level           ["256"] security level: "128" P-256, "192" P-384, "256" P-521
    attack          ["none"] | "label_flip_fltrust" | "trim_fltrust" | "adaptive_fltrust"
                    | "krum_fltrust" | "scaling_fltrust" (malicious: the first clients)
    malicious_rate  [0.0] fraction of malicious clients

  train   accuracy under attack, aggregation computed in the clear (rvfl.baselines)
    protocol        "rvfl" | "fltrust" (required)
    rounds          [500]
    eval_every      [10] evaluate the global model every this many rounds (0: never); test
                    accuracy, training accuracy (eval_train) and, under the Scaling attack,
                    the attack success rate
    eval_train      [true]   train_subset [0] (training samples evaluated; 0: all)
    eval_batch      [256]
    checkpoint_every [0] keep the global model every this many rounds (0: never)

  cost    computation time and bytes of the protocol, every party in one process
    protocol        "rvfl" | "brea" (required)
    k1              [1] first-level packing; "t" = t1 = N // 3; "doubling" = 1, 2, 4, ...
                    below N // 3, then N // 3
    t               [N // 3] threshold t1 = t2 (rvfl), t1 (brea)
    rounds          [6]
    cos_reveal      ["verify"] when each client reports its cosine: "verify" (with its
                    verification shares) or "enroll" (at enrollment); the filter needs "enroll"
    filter ...      as for e2e
    byzantine_rate  [0.3] BREA's assumed fraction of malicious clients (Multi-Krum)
    checkpoints     [false] keep the global model after every round

  e2e     wall-clock rounds, one process per party in its own network namespace (root)
    every cost setting except protocol (rvfl only), and
    link            ["10000:0.2"] "bandwidth Mbit/s : RTT ms" of every party's link
    filter          ["none"] | "max" | "max_variable_k" | "optimal_nk" (rvfl.filtering)
    alpha           [0.0] assumed malicious fraction the filter's thresholds cover
    threshold       ["n_over_3"] | "lower_bound" (optimal_nk only)
    addback         ["random"] | "highest_cos"
    cost_model      optimal_nk only: the file python -m rvfl.filtering writes
    cost_link       [the point's link] the link label to read from cost_model
    start_checkpoint  a checkpoint to start from instead of the initial model (its data split
                    must be the point's: same seed, n and iid)
    checkpoints     [false] keep the global model after every round
    broadcast_parallel [false] the server sends a broadcast over one link per receiver at once
    round_timeout_s [3600] a round taking longer ends the point as "timeout"

  curve_bench   per-curve cost of the VPSS primitives (rvfl.curve_bench)
    level           ["256"]   k1 [1] (as cost)   model_size [272474]   threads [60]
    ops             [["pss_share", "pedersen_share", "pedersen_verify"]] (and "reconstruct")
    repeats [3]  warmup [1]  mult_ts [1e8]  gs_norm [8.0]

--jobs J runs J points at the same time, each in its own process (default 1, one after
another). Only train points should run side by side: they share the machine with each other,
which is harmless for accuracy but would distort the timings of cost and curve_bench points,
and e2e points share the network namespaces, so those always run one at a time. With several
GPUs, --gpus 0 1 hands the parallel points the GPUs in turn (CUDA_VISIBLE_DEVICES).

The timings of the paper were taken with jemalloc preloaded (see README.md).
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import tomllib
import warnings
from pathlib import Path

from rvfl.logs import link_label

# torchvision's CIFAR-10 loader trips a NumPy 2.4 deprecation warning on every load, and torch
# warns that it cannot initialize NVML when sudo has dropped the environment (CUDA still works)
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")
warnings.filterwarnings("ignore", message="Can't initialize NVML")

KINDS = ("train", "cost", "e2e", "curve_bench")
FILTERS = ("none", "max", "max_variable_k", "optimal_nk")

COMMON = dict(name=None, kind=None, n=None, seed=0, dataset="CIFAR10", model="ResNet20",
              data_dir="./dataset", partition_dir="./partitions", device="auto", batch_size=64,
              lr_local=0.1, lr_global=1.0, server_samples=100, iid=None, threads=60,
              backend="ec", level="256", attack="none", malicious_rate=0.0)
FILTER = dict(filter="none", alpha=0.0, threshold="n_over_3", addback="random", cost_model=None,
              cost_link=None)
PER_KIND = {
    "train": dict(protocol=None, rounds=500, eval_every=10, eval_train=True, train_subset=0,
                  eval_batch=256, checkpoint_every=0),
    "cost": dict(protocol=None, k1=1, t=None, rounds=6, cos_reveal="verify", byzantine_rate=0.3,
                 checkpoints=False, **FILTER),
    "e2e": dict(k1=1, t=None, rounds=6, cos_reveal="verify", link="10000:0.2",
                start_checkpoint=None, checkpoints=False, broadcast_parallel=False,
                round_timeout_s=3600, **FILTER),
    "curve_bench": dict(k1=1, model_size=272474,
                        ops=["pss_share", "pedersen_share", "pedersen_verify"], repeats=3,
                        warmup=1, mult_ts=1e8, gs_norm=8.0),
}
# list-valued settings that are one value, not a grid axis
LIST_VALUED = {"ops"}


# --- the experiment list -------------------------------------------------------------------

def k1_doubling(n):
    t = n // 3
    vals, k = [], 1
    while k < t:
        vals.append(k)
        k *= 2
    return vals + [t]


def expand(entry):
    """The points of one [[run]] entry, in the order its lists are written."""
    kind = entry.get("kind")
    if kind not in KINDS:
        raise SystemExit(f"[[run]] {entry.get('name')!r}: kind must be one of {KINDS}")
    allowed = {**COMMON, **PER_KIND[kind]}
    unknown = set(entry) - set(allowed)
    if unknown:
        raise SystemExit(f"[[run]] {entry.get('name')!r} ({kind}): unknown setting(s) {sorted(unknown)}")
    for req in ("name", "n") + (("protocol",) if kind in ("train", "cost") else ()):
        if entry.get(req) is None:
            raise SystemExit(f"[[run]] {entry.get('name')!r}: '{req}' is required")
    axes = [(k, v) for k, v in entry.items() if isinstance(v, list) and k not in LIST_VALUED]
    points = []
    for combo in itertools.product(*[v for _, v in axes]):
        p = {**allowed, **entry, **dict(zip([k for k, _ in axes], combo))}
        # k1 given by rule: resolved per n, then varied too
        k1s = [p["k1"]] if "k1" in p else [None]
        if p.get("k1") == "t":
            k1s = [p["n"] // 3 if p.get("t") is None else p["t"]]
        elif p.get("k1") == "doubling":
            k1s = k1_doubling(p["n"])
        for k1 in k1s:
            q = dict(p)
            if k1 is not None:
                q["k1"] = k1
            points.append(finalize(q))
    return points


def finalize(p):
    kind = p["kind"]
    if p["iid"] is None:
        p["iid"] = kind == "e2e"
    if p["attack"] in ("none", None):
        p["attack"] = None
    if kind == "cost" and p["protocol"] not in ("rvfl", "brea"):
        raise SystemExit(f"{p['name']}: cost protocol must be 'rvfl' or 'brea'")
    if kind == "train" and p["protocol"] not in ("rvfl", "fltrust"):
        raise SystemExit(f"{p['name']}: train protocol must be 'rvfl' or 'fltrust'")
    if kind == "cost" and p["protocol"] == "brea" and p["k1"] != 1:
        raise SystemExit(f"{p['name']}: BREA packs nothing; leave k1 out (it is 1)")
    if kind in ("cost", "e2e"):
        if p["filter"] not in FILTERS:
            raise SystemExit(f"{p['name']}: filter must be one of {FILTERS}")
        if p["filter"] != "none":
            if p.get("protocol", "rvfl") != "rvfl":
                raise SystemExit(f"{p['name']}: the participant filter is for the rvfl protocol")
            if p["t"] is not None:
                raise SystemExit(f"{p['name']}: t is set per round by the filter; leave it out")
            p["cos_reveal"] = "enroll"
        if p["cos_reveal"] not in ("verify", "enroll"):
            raise SystemExit(f"{p['name']}: cos_reveal must be 'verify' or 'enroll'")
    if kind == "e2e":
        from rvfl.attacks import MODEL_POISON_ATTACKS
        if p["attack"] in MODEL_POISON_ATTACKS:
            raise SystemExit(f"{p['name']}: model-poisoning attacks need every update in one "
                             f"process; use the cost kind")
        if p["t"] is not None and not 2 * p["t"] - 1 <= p["n"]:
            raise SystemExit(f"{p['name']}: t = {p['t']} at N = {p['n']} needs 2t - 1 <= N")
    return p


def point_dir(out, p):
    kind = p["kind"]
    tag = f"N{p['n']}"
    if kind == "train":
        tag += f"_{p['protocol']}_{p['attack'] or 'noattack'}"
        if p["attack"]:
            tag += f"_m{int(round(p['malicious_rate'] * 100))}"
    elif kind in ("cost", "e2e"):
        if kind == "cost":
            tag += f"_{p['protocol']}"
        tag += f"_k{p['k1']}" + (f"_t{p['t']}" if p["t"] is not None else "")
        if p["filter"] != "none":
            tag += f"_{p['filter']}_a{p['alpha']:g}"
            if p["filter"] == "optimal_nk":
                tag += f"_{p['threshold']}"
            if p["addback"] != "random":
                tag += f"_{p['addback']}"
        elif p["cos_reveal"] == "enroll":
            tag += "_cosenroll"
        if p["attack"]:
            tag += f"_{p['attack']}_m{int(round(p['malicious_rate'] * 100))}"
        if p["backend"] != "ec" or p["level"] != "256":
            tag += f"_{p['backend']}{p['level']}"
    elif kind == "curve_bench":
        from rvfl.crypto_params import CURVES
        tag = f"{CURVES[p['level']]}_N{p['n']}_k{p['k1']}_M{p['model_size']}_th{p['threads']}"
    if p["seed"] != 0:
        tag += f"_s{p['seed']}"
    d = Path(out) / p["name"] / tag
    return d / link_label(p["link"]) if kind == "e2e" else d


# --- shared setup --------------------------------------------------------------------------

def setup(p):
    """Crypto parameters, data partition and initial model of a point."""
    import torch
    import vpss_module as vpss
    from rvfl import crypto_params
    from rvfl.data import partition_and_save
    from rvfl.experiment import resolve_device, val_max_of
    from rvfl.models import get_model
    from rvfl.trainer import LocalTrainer

    pp, spec = crypto_params.build_pp(vpss, p["backend"], p["level"])
    part_path, _, _ = partition_and_save(p["dataset"], p["n"], p["partition_dir"], is_iid=p["iid"], q=0.5,
                                         dataset_path=p["data_dir"], server_samples=p["server_samples"],
                                         seed=p["seed"])
    device = resolve_device(p["device"])
    torch.manual_seed(p["seed"])
    model = get_model(p["model"])
    trainer = LocalTrainer(model, device=device)
    # the global model is kept on the CPU, as every update is (LocalTrainer moved the model to device)
    weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().clone()
    common = dict(model_name=p["model"], dataset_name=p["dataset"], model=model, trainer=trainer,
                  partition_path=str(part_path), device=device, dataset_path=p["data_dir"],
                  batch_size=p["batch_size"], val_max=val_max_of(spec["q"]))
    return pp, spec, model, weights, common, device


def addr_map_local(n):
    return {-1: ("127.0.0.1", 5000), **{i: ("127.0.0.1", 5001 + i) for i in range(n)}}


def save_checkpoint(model, weights, path):
    import torch
    device = next(model.parameters()).device
    torch.nn.utils.vector_to_parameters(weights.to(device, copy=True), model.parameters())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# --- kinds ---------------------------------------------------------------------------------

def run_train(p, out):
    import torch
    from torch.utils.data import DataLoader, Subset
    from rvfl import comms
    from rvfl.attacks import SCALING_TARGET
    from rvfl.baselines import fltrust_aggregate, rvfl_plaintext_aggregate
    from rvfl.data import load_dataset, stamp_trigger
    from rvfl.experiment import make_parties, set_up_attack, train_round
    from rvfl.trainer import accuracy, attack_success_rate

    comms.BACKEND = "none"
    pp, spec, model, weights, common, device = setup(p)
    n = p["n"]
    server, clients = make_parties("rvfl", n, addr_map_local(n), common, pp, (None, None, None))
    for party in [server] + clients:
        party.weights = weights
    malicious = set_up_attack(clients, p["attack"], p["malicious_rate"])

    test_loader = train_loader = None
    if p["eval_every"]:
        test_loader = DataLoader(load_dataset(p["dataset"], False, p["model"], p["data_dir"]),
                                 batch_size=p["eval_batch"], shuffle=False)
        if p["eval_train"]:
            train_set = load_dataset(p["dataset"], True, p["model"], p["data_dir"])
            if p["train_subset"] and p["train_subset"] < len(train_set):
                g = torch.Generator().manual_seed(0)
                train_set = Subset(train_set, torch.randperm(len(train_set), generator=g)[:p["train_subset"]].tolist())
            train_loader = DataLoader(train_set, batch_size=p["eval_batch"], shuffle=False)

    def evaluate(it):
        torch.nn.utils.vector_to_parameters(server.weights.to(device, copy=True), model.parameters())
        rec = {"iteration": it, "test_acc": accuracy(model, test_loader, device)}
        if train_loader is not None:
            rec["train_acc"] = accuracy(model, train_loader, device)
        if p["attack"] == "scaling_fltrust":
            rec["asr"] = attack_success_rate(model, test_loader, device, SCALING_TARGET, stamp_trigger)
        append_jsonl(out / "eval.jsonl", rec)
        print(f"[{p['name']}] round {it}: " + ", ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "iteration"),
              flush=True)

    if p["checkpoint_every"]:
        save_checkpoint(model, server.weights, out / "checkpoints" / "iter0.pt")
    if p["eval_every"]:
        evaluate(0)
    q_int = spec["q"]
    for it in range(1, p["rounds"] + 1):
        train_round(server, clients, it, p["seed"], p["lr_local"], p["attack"], malicious)
        updates = {c.id: c.g_i for c in clients}
        if p["protocol"] == "fltrust":
            final_g, values = fltrust_aggregate(server.g_server, updates)
        else:
            final_g, values = rvfl_plaintext_aggregate(server.g_server, updates, server.val_max, q_int,
                                                       p["attack"], malicious)
        server.weights += final_g * p["lr_global"]
        append_jsonl(out / "values.jsonl", {"iteration": it, "values": values})
        if p["checkpoint_every"] and it % p["checkpoint_every"] == 0:
            save_checkpoint(model, server.weights, out / "checkpoints" / f"iter{it}.pt")
        if p["eval_every"] and it % p["eval_every"] == 0:
            evaluate(it)


def filter_policy(p, link=None):
    from rvfl.filtering import FilterPolicy, corrupt_bound, load_cost_model
    if p["filter"] == "none":
        return None
    cost_model = None
    if p["filter"] == "optimal_nk":
        cost_model = load_cost_model(p["cost_model"], p["cost_link"] or link, p["threshold"],
                                     corrupt_bound(p["alpha"], p["n"]))
    return FilterPolicy(p["filter"], p["n"], p["k1"], 1, p["alpha"], threshold=p["threshold"],
                        addback=p["addback"], cost_model=cost_model)


def run_cost(p, out):
    from rvfl import comms
    from rvfl.experiment import (make_parties, run_brea_round, run_rvfl_round, set_up_attack,
                                 thresholds, train_round, vpss_configs)

    comms.BACKEND = "inmemory"
    pp, spec, model, weights, common, device = setup(p)
    n = p["n"]
    t1, k1, t2, k2 = thresholds(p["protocol"], n, p["k1"], p["t"])
    server, clients = make_parties(p["protocol"], n, addr_map_local(n), common, pp,
                                   vpss_configs(n, t1, k1, t2, k2))
    for party in [server] + clients:
        party.weights = weights
        party.cos_at_enrollment = p["cos_reveal"] == "enroll"
    policy = filter_policy(p)
    if policy is not None:
        server.enable_filter(n, k1, k2, policy.c, policy)
        for c in clients:
            c.enable_filter(n, k1, k2, policy.c)
    malicious = set_up_attack(clients, p["attack"], p["malicious_rate"])
    brea = p["protocol"] == "brea"
    if p["checkpoints"]:
        save_checkpoint(model, server.weights, out / "checkpoints" / "iter0.pt")
    for it in range(1, p["rounds"] + 1):
        t0 = time.perf_counter()
        train_round(server, clients, it, p["seed"], p["lr_local"], p["attack"], malicious,
                    train_server=not brea)
        if brea:
            run_brea_round(server, clients, p["lr_global"], p["threads"], p["byzantine_rate"])
        else:
            server.addback_seed = p["seed"] + it
            run_rvfl_round(server, clients, p["lr_global"], p["threads"])
        values = None
        for party in [server] + clients:
            rec = party.extract_and_clear_logs(it)
            values = rec.pop("values", values)
            append_jsonl(out / "parties.jsonl", rec)
        append_jsonl(out / "values.jsonl", {"iteration": it, "values": values})
        if p["checkpoints"]:
            save_checkpoint(model, server.weights, out / "checkpoints" / f"iter{it}.pt")
        print(f"[{p['name']}] round {it}/{p['rounds']}: {time.perf_counter() - t0:.1f} s", flush=True)


def run_e2e(p, out):
    import multiprocessing
    from threading import BrokenBarrierError
    import torch
    from rvfl import testbed
    from rvfl.experiment import thresholds
    from rvfl.models import get_model
    from rvfl.worker import worker_main

    pp, spec, model, weights, common, device = setup(p)
    n = p["n"]
    t1, k1, t2, k2 = thresholds("rvfl", n, p["k1"], p["t"])
    weights_path = out / "start_weights.pt"
    bn_state_path = None
    if p["start_checkpoint"]:
        m = get_model(p["model"])
        m.load_state_dict(torch.load(p["start_checkpoint"], map_location="cpu"))
        weights = torch.nn.utils.parameters_to_vector(m.parameters()).detach().cpu().clone()
        bn_state_path = str(p["start_checkpoint"])
    torch.save(weights, weights_path)
    policy = filter_policy(p, link_label(p["link"]))
    filter_settings = None
    if policy is not None:
        filter_settings = dict(alpha=p["alpha"], threshold=p["threshold"], addback=p["addback"],
                               c=policy.c, cost_model=policy.cost_model)
    from rvfl.attacks import malicious_ids
    slot_of = {-1: 0, **{i: i + 1 for i in range(n)}}
    addr_map = {pid: (f"10.0.0.{s + 2}", 5000) for pid, s in slot_of.items()}
    base = dict(n=n, t1=t1, k1=k1, t2=t2, k2=k2, crypto=spec, device=p["device"], threads=p["threads"],
                broadcast_parallel=p["broadcast_parallel"], model=p["model"], dataset=p["dataset"],
                partition_path=common["partition_path"], data_dir=p["data_dir"],
                batch_size=p["batch_size"], val_max=common["val_max"],
                addr_map={str(k): v for k, v in addr_map.items()}, cos_reveal=p["cos_reveal"],
                filter=p["filter"], filter_settings=filter_settings, weights_path=str(weights_path),
                bn_state_path=bn_state_path, seed=p["seed"], rounds=p["rounds"],
                lr_local=p["lr_local"], lr_global=p["lr_global"], out_dir=str(out),
                checkpoints=p["checkpoints"], poison=p["attack"],
                malicious_ids=malicious_ids(n, p["malicious_rate"]) if p["attack"] else [])

    bw, rtt = (float(x) for x in p["link"].split(":"))
    parties = n + 1
    ctx = multiprocessing.get_context("spawn")
    procs, status = [], "ok"
    try:
        testbed.setup(parties, bw if bw != int(bw) else int(bw), rtt)
        all_barrier, client_barrier = ctx.Barrier(n + 1), ctx.Barrier(n)
        start_gate, end_gate = ctx.Barrier(n + 2), ctx.Barrier(n + 2)
        compute_lock = ctx.RLock()
        for pid, s in slot_of.items():
            wspec = dict(base, party_id=pid, netns=f"rvfl{s}")
            procs.append(ctx.Process(target=worker_main, name=f"party{pid}",
                                     args=(wspec, all_barrier, client_barrier, compute_lock,
                                           start_gate, end_gate)))
        for proc in procs:
            proc.start()
        try:
            for it in range(1, p["rounds"] + 1):
                start_gate.wait(timeout=p["round_timeout_s"])
                t0 = time.perf_counter()
                end_gate.wait(timeout=p["round_timeout_s"])
                wall_ms = (time.perf_counter() - t0) * 1000
                append_jsonl(out / "rounds.jsonl", {"iteration": it, "wall_ms": round(wall_ms, 2)})
                print(f"[{p['name']}] {out.parent.name}/{out.name} round {it}: {wall_ms / 1000:.2f} s", flush=True)
            for proc in procs:
                proc.join(timeout=60)
            codes = [proc.exitcode for proc in procs]
            if any(c not in (0, None) for c in codes):
                status = f"error: worker exit codes {codes}"
        except BrokenBarrierError:
            status = f"timeout: a round took longer than {p['round_timeout_s']} s"
    except Exception as e:
        status = f"error: {e!r}"
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        try:
            testbed.teardown(parties, announce=False)
        except Exception:
            pass
        # every worker's records into one file
        with open(out / "parties.jsonl", "w", encoding="utf-8") as f:
            for wf in sorted(out.glob("worker_*.jsonl")):
                f.write(wf.read_text(encoding="utf-8"))
                wf.unlink()
    return status


def run_curve_bench(p, out):
    import random
    from rvfl.curve_bench import COLUMNS, measure
    if not 1 <= p["k1"] <= p["n"] // 3:
        raise SystemExit(f"{p['name']}: k1 = {p['k1']} must be in [1, N // 3 = {p['n'] // 3}]")
    row = measure(p["level"], p["n"], p["k1"], p["model_size"], p["threads"], p["ops"], p["repeats"],
                  p["warmup"], p["mult_ts"], p["gs_norm"], random.Random(p["seed"]))
    (out / "result.json").write_text(json.dumps({c: row.get(c) for c in COLUMNS}, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"[{p['name']}] {row['curve']} N={p['n']} k1={p['k1']}: "
          + ", ".join(f"{k[:-2]} {v:.3f} s" for k, v in row.items() if k.endswith("_s")), flush=True)


RUNNERS = {"train": run_train, "cost": run_cost, "e2e": run_e2e, "curve_bench": run_curve_bench}


def run_point(p, out):
    """Run one point into its folder: config.json first, status.json last."""
    import shutil
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
    t0 = time.time()
    try:
        status = RUNNERS[p["kind"]](p, out) or "ok"
    except Exception as e:
        import traceback
        traceback.print_exc()
        status = f"error: {e!r}"
    (out / "status.json").write_text(json.dumps({"status": status, "seconds": round(time.time() - t0, 1)},
                                                indent=2) + "\n", encoding="utf-8")
    return status


def is_done(out):
    st = out / "status.json"
    return st.exists() and json.loads(st.read_text(encoding="utf-8")).get("status") == "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="TOML experiment lists")
    ap.add_argument("--out", default="runs", help="output folder (default: runs)")
    ap.add_argument("--only", nargs="+", help="run only the [[run]] entries with these names")
    ap.add_argument("--list", action="store_true", help="print the points and stop")
    ap.add_argument("--overwrite", action="store_true", help="run finished points again")
    ap.add_argument("--jobs", type=int, default=1,
                    help="points run at the same time, each in its own process (default 1); "
                         "for train points only, see above")
    ap.add_argument("--gpus", nargs="+", help="GPU ids handed to parallel points in turn")
    ap.add_argument("--point", help=argparse.SUPPRESS)      # internal: one point, as JSON
    args = ap.parse_args()

    if args.point:                                          # a child of --jobs
        p = json.loads(args.point)
        sys.exit(0 if run_point(p, point_dir(args.out, p)) == "ok" else 1)

    points = []
    for f in args.files:
        doc = tomllib.loads(Path(f).read_text(encoding="utf-8"))
        for entry in doc.get("run", []):
            if args.only and entry.get("name") not in args.only:
                continue
            points += expand(entry)
    if not points:
        raise SystemExit("no points (check the file names and --only)")
    dirs = [point_dir(args.out, p) for p in points]
    if len(set(dirs)) != len(dirs):
        dup = sorted({str(d) for d in dirs if dirs.count(d) > 1})
        raise SystemExit(f"several points map to the same folder: {dup}; they differ only in settings "
                         "the folder name does not show, so put them in entries with different names")
    if args.list:
        for p, d in zip(points, dirs):
            print(f"{'done ' if is_done(d) else '     '}{d}")
        return
    todo = [(p, d) for p, d in zip(points, dirs) if args.overwrite or not is_done(d)]
    print(f"[run] {len(points)} points, {len(points) - len(todo)} already done, {len(todo)} to run", flush=True)

    serial = [(p, d) for p, d in todo if args.jobs <= 1 or p["kind"] != "train"]
    parallel = [(p, d) for p, d in todo if args.jobs > 1 and p["kind"] == "train"]
    failed = []
    if parallel:
        running = []
        queue = list(parallel)
        gpu_turn = itertools.cycle(args.gpus or [None])
        while queue or running:
            while queue and len(running) < args.jobs:
                p, d = queue.pop(0)
                d.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, PYTHONUNBUFFERED="1")
                gpu = next(gpu_turn)
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                log = open(d.parent / f"{d.name}.log", "w", encoding="utf-8")
                proc = subprocess.Popen([sys.executable, __file__, "--out", args.out, "--point", json.dumps(p)],
                                        stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append((proc, d, log))
                print(f"[run] started {d} (output: {d.parent / (d.name + '.log')})", flush=True)
            time.sleep(1.0)
            for proc, d, log in list(running):
                if proc.poll() is not None:
                    log.close()
                    running.remove((proc, d, log))
                    print(f"[run] {'finished' if proc.returncode == 0 else 'FAILED'} {d}", flush=True)
                    if proc.returncode != 0:
                        failed.append(d)
    for p, d in serial:
        print(f"\n[run] {d}", flush=True)
        status = run_point(p, d)
        print(f"[run] {d}: {status}", flush=True)
        if status != "ok":
            failed.append(d)
    if failed:
        print("\n[run] not finished:\n  " + "\n  ".join(str(d) for d in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
