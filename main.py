import multiprocessing
from multiprocessing.synchronize import Lock, Barrier
import os
import re
import time
import random
import math
import torch
import numpy as np
from typing import Dict, Tuple
import logging
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from pathlib import Path
import json

# os.add_dll_directory(r"C:\dev\vcpkg\installed\x64-windows\bin")
import vpss_module as vpss
import dataset_partitioner
import crypto_params as cryptoparams          # [B2]
from party import Server, Client
from models import get_model
from trainer import LocalTrainer

import warnings
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")
warnings.filterwarnings("ignore", message="Can't initialize NVML")


# the participant floor of a filtered round (plan3_cost.filter_floor, with the other threshold rules)
from plan3_cost import filter_floor  # noqa: E402


def load_cost_fit(path, label=None):
    """[[n, A, B], ...] of make_cost_model.py's fit for one link label (round time = A / k + B
    at n clients), for variable k plan 2. label None takes the file's only label."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"variable k plan 2 needs the cost model {path} (make_cost_model.py)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    labels = [k for k in doc if not k.startswith("_")]
    if label is None:
        if len(labels) != 1:
            raise SystemExit(f"{path} has the labels {labels}; set RVFL_E2E_COST_LABEL")
        label = labels[0]
    if label not in doc or not doc[label].get("fit"):
        raise SystemExit(f"{path} has no fit for {label} (it has {labels})")
    return [[e["n"], e["A"], e["B"]] for e in doc[label]["fit"]]


def load_cost_points(path, label, c):
    """[[n, k, seconds], ...] measured on the plan-3 grid for one link label and one c
    (make_cost_model.py --plan3), for variable k plan 3. label None takes the file's only label."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"variable k plan 3 needs the cost model {path} (make_cost_model.py --plan3)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    labels = [k for k in doc if not k.startswith("_")]
    if label is None:
        if len(labels) != 1:
            raise SystemExit(f"{path} has the labels {labels}; set RVFL_E2E_COST_LABEL")
        label = labels[0]
    entry = doc.get(label, {}).get("plan3", {}).get(str(c))
    if not entry or not entry.get("points"):
        raise SystemExit(f"{path} has no plan-3 grid for {label}, c={c}")
    return entry["points"]


def _read_records(out_dir):
    """party_id -> JSON lines (sorted by round) of one combo folder: the live per-worker files
    and the combined processing_time_log.jsonl an ended run (normally, by timeout or by error)
    folded them into, merged; for a (party, round) found in both, the per-worker line wins."""
    seen = {}
    for src in sorted(out_dir.glob("worker_*.jsonl")) + [out_dir / "processing_time_log.jsonl"]:
        if not src.exists():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                seen.setdefault((rec["party_id"], rec["iteration"]), line)
    by_pid = {}
    for (pid, it) in sorted(seen):
        by_pid.setdefault(pid, []).append(seen[(pid, it)])
    return by_pid


def _write_worker_logs(out_dir, by_pid):
    for f in out_dir.glob("worker_*.jsonl"):
        f.unlink()
    for pid, lines in by_pid.items():
        (out_dir / f"worker_{pid}.jsonl").write_text("".join(l + "\n" for l in lines), encoding="utf-8")


def _retire_combined_logs(out_dir):
    """The combined logs of an earlier end are stale once a resumed run is going (their records
    are in the per-worker files by now) and would hide the live files from tabulate_filter.py.
    They are rewritten when this run ends."""
    for name in ("processing_time_log.jsonl", "communication_cost_log.jsonl", "terminal_log.txt"):
        if (out_dir / name).exists():
            (out_dir / name).unlink()


def _restore_worker_logs(out_dir):
    """Before a plain resume: bring every round already logged back into the per-worker files
    (an ended run folded them into processing_time_log.jsonl and removed them), so they are
    kept when the resumed run folds its logs again at the end."""
    by_pid = _read_records(out_dir)
    if by_pid:
        _write_worker_logs(out_dir, by_pid)
        _retire_combined_logs(out_dir)


def _set_aside_rounds(out_dir, ck_dir, wall_log, keep_upto):
    """Move everything of the rounds after keep_upto out of the way, into
    out_dir/rewound_after_{keep_upto}_<time>/, so the combo can be rerun from round keep_upto+1.
    Kept rounds stay in the per-worker files; nothing is deleted."""
    by_pid = _read_records(out_dir)
    dest = out_dir / f"rewound_after_{keep_upto}_{time.strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=False)

    kept, moved = {}, []
    for pid, lines in by_pid.items():
        for line in lines:
            (kept.setdefault(pid, []) if json.loads(line)["iteration"] <= keep_upto else moved).append(line)
    if moved:
        (dest / "processing_time_log.jsonl").write_text("".join(l + "\n" for l in moved), encoding="utf-8")

    if wall_log.exists():
        stay, go = [], []
        for line in wall_log.read_text(encoding="utf-8").splitlines():
            m = re.match(r"iteration (\d+):", line)
            (stay if (m and int(m.group(1)) <= keep_upto) else go).append(line)
        if go:
            (dest / "wall_log.txt").write_text("".join(l + "\n" for l in go), encoding="utf-8")
        wall_log.write_text("".join(l + "\n" for l in stay), encoding="utf-8")

    n_ck = 0
    if ck_dir.is_dir():
        for f in ck_dir.glob("*_Iter*.pt"):
            m = re.search(r"_Iter(\d+)\.pt$", f.name)
            if m and int(m.group(1)) > keep_upto:
                (dest / "checkpoints").mkdir(exist_ok=True)
                f.rename(dest / "checkpoints" / f.name)
                n_ck += 1
    if (out_dir / "terminal_log.txt").exists():
        (dest / "terminal_log.txt").write_bytes((out_dir / "terminal_log.txt").read_bytes())
    # only now, with the set-aside copies written, touch the originals
    _write_worker_logs(out_dir, kept)
    _retire_combined_logs(out_dir)
    print(f"[sweep] rounds after {keep_upto} set aside in {dest}: {len(moved)} log records, "
          f"{n_ck} checkpoints", flush=True)

def is_prime(n, k=5):
    """Miller-Rabin primality test"""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_safe_prime(bits):
    """Generate a safe prime p such that p = 2q + 1, where q is also prime."""
    print(f"Generating {bits}-bit safe primes... (this might take a moment)")
    while True:
        q = random.getrandbits(bits - 1)
        q |= (1 << (bits - 2)) | 1 
        
        if is_prime(q):
            p = 2 * q + 1
            if is_prime(p):
                return p, q

def setup_crypto_params(bits=128, p_int: int|None = None, q_int: int|None = None):
    if p_int is None:
        p_int, q_int = generate_safe_prime(bits)
        print(f"Selected p : {p_int}")
        print(f"Selected q : {q_int}")
    
    p_bi = vpss.BigInt(str(p_int))
    q_bi = vpss.BigInt(str(q_int))
    
    g_bi = vpss.find_generator(p_bi, q_bi)
    
    while True:
        h_bi = vpss.find_generator(p_bi, q_bi)
        if h_bi.to_int() != g_bi.to_int():
            break
            
    return {
        "p": str(p_bi),
        "q": str(q_bi),
        "g": str(g_bi),
        "h": str(h_bi)
    }

def init_public_parameter(p, q, g, h):
    pp = vpss.PublicParameters()
    pp.p = vpss.BigInt(p)
    pp.q = vpss.BigInt(q)
    pp.g = vpss.BigInt(g)
    pp.h = vpss.BigInt(h)
    return pp

def init_config_parameter(t, k, N, secret_indices: vpss.BigIntVector, share_indices: vpss.BigIntVector, padType: str = "Zero", deterministic_rand: bool = False):
    config = vpss.VPSSConfig()
    config.t = t
    config.k = k
    config.N = N
    if padType == "Zero":
        config.padType = vpss.PaddingMode.Zero
    elif padType == "Random":
        config.padType = vpss.PaddingMode.Random
    else:
        # dummy
        config.padType = vpss.PaddingMode.Zero
    config.secretIndices = secret_indices
    config.shareIndices = share_indices
    # Only set when clients recompute the public g_server shares (zeroes the [k,t) randomness so
    # every honest client derives the identical share). Left at the C++ default (False) otherwise.
    if deterministic_rand:
        config.deterministicRand = True
    return config

def init_party(
    node_type: str,
    my_id: int,
    party_addr_map: Dict[int, Tuple[str, int]],
    dataset_name: str,
    dataset_path: str,
    partition_path: str,
    device: int,
    model_name: str,
    model: torch.nn.Module,
    trainer: LocalTrainer,
    batch_size: int,
    val_max: int,
    initial_weights: torch.Tensor,
    pp: vpss.PublicParameters,
    config1: vpss.VPSSConfig,
    config2: vpss.VPSSConfig,
    config1_det: vpss.VPSSConfig = None,
):
    my_ip, my_port = party_addr_map[my_id]
    if node_type == 'Server':
        # Server ID = -1
        party = Server(
            ip=my_ip,
            port=my_port,
            party_map={},
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            partition_path=partition_path,
            model=model,
            trainer=trainer,
            device=device,
            batch_size=batch_size,
            val_max=val_max
        )
    else:
        server_addr = party_addr_map[-1] # ID -1 is Server
        party = Client(
            id=my_id,
            ip=my_ip,
            port=my_port,
            server_addr=server_addr,
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            partition_path=partition_path,
            model=model,
            trainer=trainer,
            device=device,
            batch_size=batch_size,
            val_max=val_max
        )
    party.party_map = party_addr_map
    party.set_VPSS_params(pp, config1, config2, config1_det)
    party.weights = initial_weights
    return party


if __name__ == "__main__":
    # Linux: fork, Windows: spawn
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # ==========================
    # Configuration
    # ==========================
    # DATASET_NAME = "MNIST"
    DATASET_NAME = "CIFAR10"
    # MODEL_NAME = "MNIST_FLTrust"
    # MODEL_NAME = "CIFAR_CNN"
    MODEL_NAME = "ResNet20"
    # [B6] Every knob a cost sweep varies is overridable from the environment, so a sweep
    # never has to edit this file. The values above stay the defaults for a manual run.
    DATASET_NAME = os.environ.get("RVFL_DATASET", DATASET_NAME)
    MODEL_NAME   = os.environ.get("RVFL_MODEL", MODEL_NAME)
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SERVER_DATASET_SIZE = 100
    DATASET_PATH = "./dataset"
    PARTITION_ROOT = "./partitions"
    # CKPT_ROOT / LOG_DIR / INIT_WEIGHT_PATH are set PER-N right after N is known (see RUN_ROOT below),
    # so runs at different N never share a checkpoint dir, eval curve, or log file.
    INIT_WEIGHT_PATH = None   # set per-run right after RUN_TAG below (avoids shared-file race)

    # MODE = "FedAvg"
    # MODE = "BREA"
    # MODE = "FLTrust"
    # MODE = "FLTrust_quantized"
    # MODE = "FLTrust_quantized_perp"
    MODE = "FLTrust_quantized_perp"   # RVFL's exact aggregation in plaintext (quantized + perp):
                                      # reproduces RVFL's accuracy without the per-round crypto,
                                      # which is what makes a multi-hundred-round accuracy run feasible.
                                      # (set MODE="RVFL" only for a short cross-check of a few rounds.)
    use_perp = True # only for RVFL

    # === Attack experiment config (opt-in; ATTACK=None reproduces published no-attack runs) ===
    # ATTACK value == dataloader poison_type (1:1), to avoid name confusion.
    ATTACK = "label_flip_fltrust"   # None | "label_flip_fltrust"  (FLTrust-style LF: l -> M-1-l)
    MALICIOUS_RATE = 0.2   # fraction of clients that are malicious (FLTrust default: 20%)
    # Sweep driver hook: run_attack_sweep.py sets these per combo (a single manual run ignores them).
    MODE           = os.environ.get("RVFL_MODE", MODE)
    if os.environ.get("RVFL_ATTACK") is not None:
        _a = os.environ["RVFL_ATTACK"]
        ATTACK = None if _a in ("", "none", "None") else _a
    if os.environ.get("RVFL_MRATE") is not None:
        MALICIOUS_RATE = float(os.environ["RVFL_MRATE"])

    # Broadcast model for the netns E2E run:
    #   False -> serial single-uplink broadcast (honest N x payload/bandwidth; default)
    #   True  -> per-link parallel broadcast (well-provisioned server)
    E2E_BROADCAST_PARALLEL = False

    # === Who deals the g_server shares during enrollment ===
    #   True  -> server computes and broadcasts them (DEFAULT; no C++ rebuild needed).
    #   False -> server sends only g_server(+exp); each client recomputes the shares itself,
    #            deterministically (zeroed [k,t) randomness), so all honest clients agree.
    #            Requires the C++ VPSS rebuilt with deterministicRand support.
    SERVER_COMPUTES_SHARES = False

    # === netns end-to-end config (opt-in; runs each party as a process in its own network
    #     namespace over tc-shaped links -- the faithful WAN emulation). Requires root. ===
    E2E_NETNS = False           # True -> netns WAN emulation; False -> the FL training loop (attack/accuracy runs)
    # [B6] RVFL_E2E=1 turns the netns sweep on without editing this file.
    E2E_NETNS = os.environ.get("RVFL_E2E", "1" if E2E_NETNS else "0") not in ("", "0", "false", "False")
    # (BW Mbit/s, RTT ms) combos to sweep, in priority order. Each runs NUM_ITERATIONS times and
    # writes its own folder N{N}_k{k1}/{label}/ with processing_time_log.jsonl + terminal_log.txt.
    COMM_SWEEP = [
        (10, 80),                    # WAN, narrow
        (1000, 0.2),                 # LAN
        (50, 80),                    # WAN
        (100, 80),                   # WAN, wide
        (10000, 0.2),                # LAN, 10 Gbit
    ]
    E2E_GATE_TIMEOUT_S = 3600   # per-iteration cap; a combo exceeding it is killed and skipped
    E2E_ITERATIONS = 1          # iters PER COMBO for the sweep. Single iteration: at ~10^3 s protocol
                                #   times the TCP setup/slow-start warm-up (~5-17 s) is <1%, so the
                                #   iter1 outlier we used to discard is now negligible.
                                #   separate from NUM_ITERATIONS below (that's the FL-training path).

    # (N, k1) combos to sweep, in priority order (t1=t2=N//3, k2=1 derived per N).
    # [B6] k1 runs 1, 2, 4, 8, ... while it stays below floor(N/3), and floor(N/3) itself
    # closes the row. t1 = t2 = floor(N/3), so the last k1 of each row leaves a privacy
    # threshold (t - k) of zero; it is kept because it is the packing limit the tables report.
    def _k1_values(n_clients):
        t = n_clients // 3
        vals, k = [], 1
        while k < t:
            vals.append(k)
            k *= 2
        vals.append(t)
        return vals

    # Peak memory grows as N^2 * ceil(M/k1), so only (50, 1) (about 210 GB for ResNet20) is
    # dropped. The two next heaviest run last, lightest first: (50, 2) at ~105 GB, then
    # (40, 1) at ~134 GB. cost_sweep.py applies the same rule to the in-process measurement.
    # The end-to-end sweep covers N = 10 to 50 only. Which (N, k1) actually run, and in what
    # order, comes from cost_sweep.schedule, so the netns sweep and the in-memory sweep are
    # filtered by the same memory rule and move together when the security level changes.
    # Every party is its own process here, so the per-process peak is 1/N of the estimate,
    # but the machine still has to hold the whole of it.
    import cost_sweep as _sched
    _nk_head, _nk_tail, _nk_cut = _sched.schedule(
        [10, 20, 30, 40, 50], None,
        os.environ.get("RVFL_BACKEND", cryptoparams.DEFAULT_BACKEND),
        os.environ.get("RVFL_SECLEVEL", cryptoparams.DEFAULT_LEVEL),
        _sched.DEFAULT_MODEL_SIZE, _sched.DEFER_GB, _sched.EXCLUDE_GB)
    NK_SWEEP = [(n, k) for n, k, _ in _nk_head + _nk_tail]

    # === netns sweep overrides from the environment (all optional; unset keeps the values
    #     above and the sweep runs exactly as before) ===
    #   RVFL_E2E_ROUNDS  rounds per combo                                  e.g. 5
    #   RVFL_E2E_LINKS   bw_mbit:rtt_ms pairs, comma-separated             e.g. 10000:0.2,10:80
    #   RVFL_E2E_K1      k1 values at N = RVFL_N, comma-separated (set below, after N)
    #   RVFL_E2E_CKPT    state_dict checkpoint to start every combo from    (default: fresh init)
    #   RVFL_E2E_SEED    base seed; every party's local training is seeded per (round, party)
    #   RVFL_E2E_ALPHA   turn the per-round participant filter on, at this alpha (set below)
    E2E_ITERATIONS = int(os.environ.get("RVFL_E2E_ROUNDS", E2E_ITERATIONS))
    if os.environ.get("RVFL_E2E_LINKS"):
        COMM_SWEEP = []
        for _item in os.environ["RVFL_E2E_LINKS"].split(","):
            _bw, _rtt = _item.split(":")
            COMM_SWEEP.append((int(_bw), float(_rtt) if "." in _rtt else int(_rtt)))
    E2E_CKPT = os.environ.get("RVFL_E2E_CKPT") or None
    E2E_SEED = int(os.environ["RVFL_E2E_SEED"]) if os.environ.get("RVFL_E2E_SEED") else None
    #   RVFL_E2E_SAVE_CKPT=1  the server saves the global model after every round under
    #                         <combo folder>/checkpoints/, so a long sweep (a whole training run
    #                         on the protocol, e.g. RVFL_E2E_ROUNDS=160) can be evaluated for
    #                         accuracy afterwards. Unset, nothing is saved, as before.
    E2E_SAVE_CKPT = os.environ.get("RVFL_E2E_SAVE_CKPT", "0") == "1"
    #   RVFL_E2E_RESUME=1  continue every combo from the last round checkpointed in its own
    #                      <combo folder>/checkpoints/ (so the earlier run needs
    #                      RVFL_E2E_SAVE_CKPT=1), keeping the rounds already logged. A combo
    #                      with no checkpoint starts from round 1; one already at
    #                      RVFL_E2E_ROUNDS is skipped. Same seed -> same rounds as an
    #                      uninterrupted run (the per-round seeds depend on the round only).
    E2E_RESUME = os.environ.get("RVFL_E2E_RESUME", "0") == "1"
    #   RVFL_E2E_RESUME_AT=r  rerun every combo from round r: start from the checkpoint of round
    #                         r-1 instead of the latest one. Nothing is deleted: the checkpoints
    #                         of rounds >= r, their log records and wall times are moved into
    #                         <combo folder>/rewound_after_{r-1}_<time>/ and the kept rounds stay
    #                         where they are. Implies RVFL_E2E_RESUME=1 for the rounds after that.
    E2E_RESUME_AT = int(os.environ["RVFL_E2E_RESUME_AT"]) if os.environ.get("RVFL_E2E_RESUME_AT") else None
    if E2E_RESUME_AT is not None and E2E_RESUME_AT < 1:
        raise SystemExit("RVFL_E2E_RESUME_AT must be a round number >= 1")
    #   RVFL_PARTITION_IID=0|1  how the training set is split over the clients. Unset keeps what
    #                         each path had: the training path non-IID q=0.5 (FLTrust), the netns
    #                         sweep IID. Set, both paths use the same split, which a sweep started
    #                         from a checkpoint (RVFL_E2E_CKPT) needs anyway.
    _iid_env = os.environ.get("RVFL_PARTITION_IID")
    TRAIN_PARTITION_IID = (_iid_env == "1") if _iid_env is not None else False
    E2E_PARTITION_IID = (_iid_env == "1") if _iid_env is not None else True

    ITER_START = 0
    NUM_ITERATIONS = 600

    CHECKPOINT_INTERVAL       = 1

    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 5000
    BASE_CLIENT_PORT = 5001

    N = int(os.environ.get("RVFL_N", 100))   # single-sourced by run_attack_sweep via RVFL_N; 100 = manual-run default
    if os.environ.get("RVFL_E2E_K1"):        # (N, k1) grid for the netns sweep, at this N only
        NK_SWEEP = [(N, int(_k)) for _k in os.environ["RVFL_E2E_K1"].split(",")]
    # RVFL_E2E_KT  k1:t pairs at this N, comma-separated (filter off only): the sweep at the
    #              given threshold t instead of N // 3, for the plan-3 cost grid. Results go to
    #              N{N}_k{k1}_t{t}/.  e.g. RVFL_E2E_KT=1:11,2:12
    if os.environ.get("RVFL_E2E_KT"):
        NK_SWEEP = [(N, int(_p.split(":")[0]), int(_p.split(":")[1]))
                    for _p in os.environ["RVFL_E2E_KT"].split(",")]
    # Per-N storage root: every output (checkpoints, logs, init weights) lives under ./runs/N{N}/ so
    # different-N runs never mix. run_attack_sweep / extend_sweep resolve to the same ./runs/N{N} paths.
    RUN_ROOT  = Path(f"./runs/N{N}")
    CKPT_ROOT = RUN_ROOT / "checkpoints"
    t1 = N // 3     # 1 to N/2; for robustness, 1 to N/3
    t2 = N // 3
    # k1 = t1       # 1 to t
    k1 = 1
    k2 = 1

    # [B6] VPSS shape from the environment, so an (N, k1) sweep needs no source edit.
    #   RVFL_K1 / RVFL_K2  packing widths;  RVFL_T1 / RVFL_T2  polynomial degrees.
    # t1/t2 default to N//3 and follow RVFL_N unless given explicitly.
    k1 = int(os.environ.get("RVFL_K1", k1))
    k2 = int(os.environ.get("RVFL_K2", k2))
    t1 = int(os.environ.get("RVFL_T1", t1))
    t2 = int(os.environ.get("RVFL_T2", t2))

    # [B6] k > t makes createPolynomialFromSecrets read past the end of the interpolation
    # point list, which the C++ side does not check: the process dies with a segmentation
    # fault and no message. The sweep grid keeps k1 <= floor(N/3) = t1, so this only fires on
    # a typo, but a typo should say what is wrong rather than crash.
    if k1 > t1 or k2 > t2:
        raise SystemExit(f"[config] k must not exceed t: k1={k1} t1={t1}, k2={k2} t2={t2}. "
                         f"With N={N} the packing width k1 can go up to {t1}.")

    if MODE == "BREA":
        k1 = 1          # no packing
        t2 = 2*t1-1     # no degree reduction

    # === Per-round participant filter (RVFL only; off unless RVFL_E2E_ALPHA is set) ===
    # A client whose cosine with the root update is non-positive gets trust score 0 and so
    # contributes exactly zero to the aggregate. Letting it sit the round out leaves the
    # result unchanged and shrinks the all-to-all share exchange. With the filter off every
    # client takes part every round and nothing else in the protocol changes.
    #   With c = ceil(a * N) assumed malicious clients, cp = max(c, 1) share holders that may
    #   pool their shares (a client always sees its own share; with t = k a share carries no
    #   randomness), and k = max(k1, k2), a round with n participants needs a threshold t with
    #       privacy     t - k >= cp         (t >= k + cp)
    #       robustness  (n - t) / 2 >= c    (t <= n - 2c)
    #       degree      2t - 1 <= n - c     (pairwise verification's degree reduction)
    #   On top of that k may not exceed the round's filter-off threshold n // 3 (no packing
    #   beyond what the baseline grid uses, whatever alpha is). filter_floor is the least n for
    #   which all of this holds; the rejected clients are added back when the cosine filter
    #   would go below it. The round's t is n // 3 clipped into [k + cp, n - 2c]
    #   (Party.set_round_participants).
    #   FILTER_ALPHA = 0.0 -> c = 0, cp = 1: k = 1 floor 3, t = max(n // 3, k + 1)
    PARTICIPANT_FILTER = False
    FILTER_ALPHA = 0.0
    if os.environ.get("RVFL_E2E_ALPHA"):
        PARTICIPANT_FILTER = True
        FILTER_ALPHA = float(os.environ["RVFL_E2E_ALPHA"])
    FILTER_C = math.ceil(FILTER_ALPHA * N)
    MIN_PARTICIPANTS = filter_floor(max(k1, k2), FILTER_C)
    # Which non-positive-cosine clients are added back to reach the floor: "random" (default,
    # seeded per round from RVFL_E2E_SEED) or "highest_cos". See Server._addback.
    FILTER_ADDBACK = os.environ.get("RVFL_E2E_ADDBACK", "random")
    # Variable k (filter on only). RVFL_E2E_VARK=1 is plan 1: the participants are chosen
    # first with the fixed k1's floor, then the round's k1 takes the privacy margin left over,
    # k1 = max(k1, n // 3 - cp), cp = max(c, 1), with t kept at n // 3 (Party.round_k1_for).
    # Results go to N{N}_k{k1}_a{alpha}_vk1/. RVFL_E2E_VARK=2 is plan 2: the same k1 rule, but the round
    # size is chosen too, by adding non-positive-cosine clients back (they contribute zero)
    # wherever the predicted round time is lower (Server.choose_round_size). The prediction
    # is the fit make_cost_model.py writes, read from RVFL_E2E_COST_MODEL (default
    # cost_model.json) under the run's link label. Results go to N{N}_k{k1}_a{alpha}_vk2/.
    # Unset, k1 stays fixed.
    # RVFL_E2E_VARK=3 is plan 3: the round size n and k are chosen together and t = k + cp moves
    # with k (no k <= n // 3 cap; 2t - 1 <= n - c), from the measured plan-3 grid of this c in
    # the same cost model file (Server.choose_round_shape). Results go to N{N}_k{k1}_a{alpha}_vk3/.
    FILTER_VARK = {"": "", "0": "", "1": "margin", "2": "opt", "3": "full"}.get(
        os.environ.get("RVFL_E2E_VARK", ""), None)
    if FILTER_VARK is None:
        raise SystemExit("RVFL_E2E_VARK must be 1 (variable k, plan 1), 2 (plan 2), 3 (plan 3) or unset")
    if FILTER_VARK and not PARTICIPANT_FILTER:
        raise SystemExit("RVFL_E2E_VARK needs the participant filter (RVFL_E2E_ALPHA)")
    VARK_TAG = {"margin": "_vk1", "opt": "_vk2", "full": "_vk3"}.get(FILTER_VARK, "")
    COST_MODEL_PATH = Path(os.environ.get("RVFL_E2E_COST_MODEL", "cost_model.json"))

    # === Auto run tag: keep checkpoints/logs separated per (MODE, attack, malicious rate) ===
    attack_tag = ATTACK if ATTACK is not None else "noattack"
    RUN_TAG = f"{MODE}_{attack_tag}"
    if ATTACK is not None:
        RUN_TAG += f"_m{int(round(MALICIOUS_RATE * 100))}"
    # Optional per-instance suffix (default empty = no change). Used only by the parallelism
    # benchmark to give otherwise-identical concurrent runs distinct ckpt/log folders.
    # [B6] A cost sweep varies k1 and the commitment group while N, MODE and the attack stay
    # fixed, and the log file name is keyed on RUN_TAG and N only. Without this the runs of a
    # sweep all append to one file and cannot be told apart afterwards, so each varied knob
    # joins the tag as soon as it is set explicitly. A run that sets none keeps the old name.
    if os.environ.get("RVFL_K1") is not None:
        RUN_TAG += f"_k{k1}"
    if MODE == "RVFL":
        # The commitment group only exists on the RVFL path, and it changes both the timings
        # and the byte counts, so it belongs in the name of every RVFL log. Other modes keep
        # the name they had.
        RUN_TAG += "_" + os.environ.get("RVFL_BACKEND", cryptoparams.DEFAULT_BACKEND) \
                 + os.environ.get("RVFL_SECLEVEL", cryptoparams.DEFAULT_LEVEL)
    RUN_TAG += os.environ.get("RVFL_RUN_SUFFIX", "")
    CKPT_DIR = CKPT_ROOT / RUN_TAG
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-run-unique init-weight file. Every run writes its OWN random init here and re-reads
    # it, so a single shared path ("./init_model.pt") makes concurrent runs read a half-written
    # pickle -> EOFError (the crash seen at P>=4). Keying it on RUN_TAG (which already includes
    # RVFL_RUN_SUFFIX and the MODE/attack/rate) gives every concurrent run its own file.
    INIT_WEIGHT_PATH = str(RUN_ROOT / f"init_model_{RUN_TAG}.pt")   # per-N (under RUN_ROOT), per-RUN_TAG

    # NUM_ITERATIONS += int((N-10)*15)   # (N-scaling; disabled for the attack run)
    NUM_ITERATIONS = int(os.environ.get("RVFL_ITERS", 1000))   # global rounds (R_g); driver may override

    # === Resume: continue from the latest checkpoint of THIS RUN_TAG (set RVFL_RESUME=0 to force
    # a fresh start). Scans CKPT_DIR for the max Iter{k}; ITER_START=k+1 then loads Iter{k} below
    # (via the ITER_START != 0 branch) and the loop runs range(ITER_START, NUM_ITERATIONS+1). With
    # CHECKPOINT_INTERVAL=1 every round is saved, so an interrupted run resumes with no lost work. ===
    if os.environ.get("RVFL_RESUME", "1") != "0" and CKPT_DIR.is_dir():
        _ck_re = re.compile(rf"^{re.escape(DATASET_NAME)}_{re.escape(MODEL_NAME)}_N{N}_Iter(\d+)\.pt$")
        _done = [int(m.group(1)) for f in CKPT_DIR.glob("*.pt") if (m := _ck_re.match(f.name))]
        _last = max(_done) if _done else 0
        if 0 < _last < NUM_ITERATIONS:
            ITER_START = _last + 1
            print(f"[Resume] {RUN_TAG}: found Iter{_last} -> continuing from Iter{ITER_START}/{NUM_ITERATIONS}")
        elif _last >= NUM_ITERATIONS:
            # Already complete: make the loop empty (and skip the ITER_START==0 re-init that would
            # otherwise overwrite every checkpoint by retraining from scratch).
            ITER_START = NUM_ITERATIONS + 1
            print(f"[Resume] {RUN_TAG}: already trained to Iter{_last} (>= {NUM_ITERATIONS}); nothing to do.")

    lr_local = 0.1
    lr_global = 1.0

    LOG_DIR = RUN_ROOT / "logs"          # per-N (matches run_attack_sweep's ./runs/N{N}/logs)
    (LOG_DIR / MODE).mkdir(parents=True, exist_ok=True)
    TIME_LOG_PATH = LOG_DIR / MODE / f"{RUN_TAG}_N{N}_processing_time_log.jsonl"
    COMM_LOG_PATH = LOG_DIR / MODE / f"{RUN_TAG}_N{N}_communication_cost_log.jsonl"
    VALUE_LOG_PATH = LOG_DIR / MODE / f"{RUN_TAG}_N{N}_values_log.jsonl"
    # if TIME_LOG_PATH.exists(): TIME_LOG_PATH.unlink()
    # if COMM_LOG_PATH.exists(): COMM_LOG_PATH.unlink()
    # if VALUE_LOG_PATH.exists(): VALUE_LOG_PATH.unlink()
    ENABLE_LOGGING = True

    print("=== 1. Setup Crypto Parameters ===")
    # [B2] Commitment group: "ec" (prime-order NIST curve, default) or "zp" (Schnorr group
    # in Z_p*), at security level 128 / 192 / 256. Shamir shares are elements of Z_q in both
    # backends, so the backend changes only the commitment size and the exponentiation cost.
    #   RVFL_BACKEND=ec|zp   RVFL_SECLEVEL=128|192|256
    # To reproduce the submitted measurements exactly, use the old 512-bit safe prime:
    #   RVFL_BACKEND=zp RVFL_SECLEVEL=legacy512
    CRYPTO_BACKEND = os.environ.get("RVFL_BACKEND", cryptoparams.DEFAULT_BACKEND)
    SECURITY_LEVEL = os.environ.get("RVFL_SECLEVEL", cryptoparams.DEFAULT_LEVEL)
    pp, crypto_spec = cryptoparams.build_pp(vpss, CRYPTO_BACKEND, SECURITY_LEVEL)
    print(f"[Crypto] backend={CRYPTO_BACKEND} level={SECURITY_LEVEL} "
          f"|q|={crypto_spec['q'].bit_length()} bits"
          + (f" curve={crypto_spec['curve']}" if crypto_spec['curve'] else
             f" |p|={crypto_spec['p'].bit_length()} bits"))

    vpss_params = {
        "p": str(crypto_spec["p"]), "q": str(crypto_spec["q"]),
        "g": str(crypto_spec["g"]), "h": str(crypto_spec["h"]),
        "t1": t1,
        "k1": k1,
        "t2": t2,
        "k2": k2,
        "N": N
    }
    print(f"Generated Params: N={N}, t1={t1}, k1={k1}, t2={t2}, k2={k2}")

    secret_indices = vpss.BigIntVector()
    # secret_pts: 1 to t_max
    t_max = max(vpss_params["t1"], vpss_params["t2"])
    for i in range(1, t_max + 1):
        secret_indices.append(vpss.BigInt(i))
        
    share_indices = vpss.BigIntVector()
    # share_pts: N+1 to 2N
    for i in range(N + 1, 2 * N + 1):
        share_indices.append(vpss.BigInt(i))
    print(f"e = {secret_indices}\nalpha = {share_indices}")

    config1 = init_config_parameter(vpss_params["t1"], vpss_params["k1"], vpss_params["N"], secret_indices, share_indices)
    config2 = init_config_parameter(vpss_params["t2"], vpss_params["k2"], vpss_params["N"], secret_indices, share_indices)
    # Deterministic-randomness variant of config1, only built when clients recompute g_server shares.
    config1_det = None
    if not SERVER_COMPUTES_SHARES:
        config1_det = init_config_parameter(vpss_params["t1"], vpss_params["k1"], vpss_params["N"],
                                            secret_indices, share_indices, deterministic_rand=True)

    print("\n=== 2. Partitioning Dataset ===")
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        
    PARTITION_PATH, _, ratio = dataset_partitioner.partition_and_save(
        dataset_name=DATASET_NAME,
        num_clients=N,
        save_path=PARTITION_ROOT,
        is_iid=TRAIN_PARTITION_IID,   # non-IID q=0.5 (FLTrust) unless RVFL_PARTITION_IID says otherwise
        q=0.5,
        dataset_path=DATASET_PATH,
        server_samples=SERVER_DATASET_SIZE
    )

    print("\n=== 3. Network Configuration ===")
    # ID -1: Server, ID 0~N-1: Clients
    party_addr_map = {}
    party_addr_map[-1] = (SERVER_IP, SERVER_PORT)
    for i in range(N):
        party_addr_map[i] = (SERVER_IP, BASE_CLIENT_PORT + i)


    print("\n=== 4. Initial Weight Generation ===")

    model = get_model(MODEL_NAME)
    trainer = LocalTrainer(model, device=DEVICE)

    if ITER_START == 0:
        weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        torch.save(weights, INIT_WEIGHT_PATH)
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR, exist_ok=True)
        filename = f"{DATASET_NAME}_{MODEL_NAME}_N{N}_Iter0.pt"
        save_path = CKPT_DIR / filename
        torch.save(model.state_dict(), save_path)
        print("Initial global model saved.")
    M = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_int = int(vpss_params["q"])
    val_max = math.isqrt(q_int//2)
    print(f"val_max : {val_max} \t Model Params M : {M}")

    weights = None
    if ITER_START == 0:
        if os.path.exists(INIT_WEIGHT_PATH):
            weights = torch.load(INIT_WEIGHT_PATH, map_location='cpu')
            ITER_START += 1
        else:
            print(f"Warning: Initial weights file not found.")
    elif ITER_START != 0 and os.path.exists(CKPT_DIR):
        filename = f"{DATASET_NAME}_{MODEL_NAME}_N{N}_Iter{ITER_START-1}.pt"
        if os.path.exists(CKPT_DIR / filename):
            model.load_state_dict(torch.load(CKPT_DIR / filename))
            weights = torch.nn.utils.parameters_to_vector(model.parameters()).cpu().detach().clone()
        else:
            print(f"Warning: Weights file for Iter{ITER_START-1} not found.")


    # ==========================================================================
    # netns end-to-end: run each party as a process in its own tc-shaped namespace.
    # ==========================================================================
    if E2E_NETNS:
        import net_setup
        import dataset_partitioner
        from worker import worker_main
        from threading import BrokenBarrierError

        def _ip(slot):   return f"10.0.0.{slot + 2}"
        def _ns(slot):   return f"rvfl{slot}"
        def _bw_label(bw):
            return f"{bw // 1000}Gbps" if (bw >= 1000 and bw % 1000 == 0) else f"{bw}Mbps"

        _crypto = dict(crypto_spec)   # [B2] carries the backend and both generators
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== 5. netns E2E sweep: {len(NK_SWEEP)} (N,k1) x {len(COMM_SWEEP)} (BW,RTT), "
              f"{E2E_ITERATIONS} iters/combo ===")
        # [B6] The commitment group belongs in the path. Without it a second sweep at another
        # backend or security level writes into the folders of the first and overwrites them.
        # RVFL_E2E_RESULTS: another base folder (default e2e_results), so a separate set of
        # runs never mixes with the results already there
        E2E_ROOT = Path(os.environ.get("RVFL_E2E_RESULTS", "e2e_results")) / f"{CRYPTO_BACKEND}{SECURITY_LEVEL}"
        print(f"[sweep] results root: {E2E_ROOT.resolve()}", flush=True)

        # Starting point. By default every combo starts from the fresh init weights. With
        # RVFL_E2E_CKPT it starts from a trained state_dict (a checkpoint of the training
        # path, e.g. runs/N48/checkpoints/FLTrust_quantized_perp_noattack/..._Iter200.pt), so
        # the cosines the participant filter sees are the ones of a model trained that far.
        # The workers take a flat parameter vector, so the state_dict is flattened once here.
        start_weights_path = INIT_WEIGHT_PATH
        start_iter = 0
        if E2E_CKPT:
            _ck = Path(E2E_CKPT)
            if not _ck.is_file():
                sys.exit(f"RVFL_E2E_CKPT not found: {_ck}")
            model.load_state_dict(torch.load(_ck, map_location="cpu"))
            _m = re.search(r"Iter(\d+)", _ck.name)
            start_iter = int(_m.group(1)) if _m else 0
            E2E_ROOT.mkdir(parents=True, exist_ok=True)
            start_weights_path = str(E2E_ROOT / f"start_{_ck.stem}_flat.pt")
            torch.save(torch.nn.utils.parameters_to_vector(model.parameters()).cpu().detach().clone(),
                       start_weights_path)
            print(f"[sweep] start weights: {_ck} (iteration {start_iter}) -> {start_weights_path}")
        # Folder suffix so a filter run never overwrites the filter-off sweep of the same (N, k1).
        alpha_tag = (f"_a{FILTER_ALPHA:g}{VARK_TAG}") if PARTICIPANT_FILTER else ""

        # plan 2 reads its cost model per link; a missing one stops the run before anything starts
        cost_fits = {}
        if FILTER_VARK in ("opt", "full"):
            for (bw, rtt) in COMM_SWEEP:
                _lab = f"{_bw_label(bw)}_{rtt}ms"
                cost_fits[_lab] = (load_cost_fit(COST_MODEL_PATH, _lab) if FILTER_VARK == "opt"
                                   else load_cost_points(COST_MODEL_PATH, _lab, FILTER_C))

        def _write_combo_end(out_dir, wall_log, status, Ns, k1s, k2s, bw, rtt, note=None, t=None):
            """How a combo ends: the per-worker logs folded into processing_time_log.jsonl and
            communication_cost_log.jsonl (the per-worker files removed), and terminal_log.txt
            with the settings, every round's wall time and the status."""
            _blob = "".join(_f.read_text(encoding="utf-8")
                            for _f in sorted(out_dir.glob("worker_*.jsonl")))
            for _name in ("processing_time_log.jsonl", "communication_cost_log.jsonl"):
                (out_dir / _name).write_text(_blob, encoding="utf-8")
            for _f in out_dir.glob("worker_*.jsonl"):
                _f.unlink()
            with (out_dir / "terminal_log.txt").open("w", encoding="utf-8") as _tl:
                _tl.write(f"N={Ns} k1={k1s}  BW={bw} Mbit/s  RTT={rtt} ms  iters={E2E_ITERATIONS}"
                          + (f"  t={t}" if t is not None else "") + "\n")
                if PARTICIPANT_FILTER or E2E_CKPT or E2E_SEED is not None or E2E_SAVE_CKPT:
                    _tl.write(f"filter={PARTICIPANT_FILTER} alpha={FILTER_ALPHA} vark={FILTER_VARK or 'off'} "
                              f"min_participants={filter_floor(max(k1s, k2s), math.ceil(FILTER_ALPHA * Ns))} "
                              f"start_iter={start_iter} ckpt={E2E_CKPT} seed={E2E_SEED} "
                              f"partition_iid={E2E_PARTITION_IID} save_ckpt={E2E_SAVE_CKPT}\n")
                if note:
                    _tl.write(f"note: {note}\n")
                # every round's wall time, including the ones before a resume
                if wall_log.exists():
                    _tl.write(wall_log.read_text(encoding="utf-8"))
                _tl.write(f"STATUS: {status}\n")

        for _entry in NK_SWEEP:
            Ns, k1s = _entry[0], _entry[1]
            t_over = _entry[2] if len(_entry) > 2 else None
            t1s, t2s, k2s = Ns // 3, Ns // 3, 1        # per-N derived VPSS params
            combo_name = f"N{Ns}_k{k1s}"
            if t_over is not None:
                # a given threshold (RVFL_E2E_KT): every party is in the degree reduction here,
                # so 2t - 1 <= N is what the pairwise verification needs
                if PARTICIPANT_FILTER:
                    sys.exit("RVFL_E2E_KT is for the filter-off sweep; unset RVFL_E2E_ALPHA")
                if not (1 <= k1s <= t_over and 2 * t_over - 1 <= Ns):
                    sys.exit(f"RVFL_E2E_KT: k1={k1s}, t={t_over} at N={Ns} needs k1 <= t and 2t - 1 <= N")
                t1s = t2s = t_over
                combo_name += f"_t{t_over}"
            t_max = max(t1s, t2s)
            if E2E_CKPT:
                # The checkpoint was trained on the training path's partition (non-IID, N
                # clients), so each client keeps the data it trained on: reuse that partition.
                if Ns != N:
                    sys.exit(f"RVFL_E2E_CKPT was trained at N={N}; the sweep asks for N={Ns}")
                part_path = PARTITION_PATH
            else:
                part_path, _, _ = dataset_partitioner.partition_and_save(   # (re)partition for this N
                    dataset_name=DATASET_NAME, num_clients=Ns, save_path=PARTITION_ROOT,
                    is_iid=E2E_PARTITION_IID, q=0.5, dataset_path=DATASET_PATH, server_samples=SERVER_DATASET_SIZE)
            P = Ns + 1
            slot_of = {-1: 0, **{i: i + 1 for i in range(Ns)}}
            addr_map = {pid: (_ip(s), SERVER_PORT) for pid, s in slot_of.items()}
            common = dict(
                model_name=MODEL_NAME, dataset_name=DATASET_NAME, dataset_path=DATASET_PATH,
                partition_path=part_path, batch_size=BATCH_SIZE, val_max=val_max,
                t1=t1s, k1=k1s, t2=t2s, k2=k2s, N=Ns,
                secret_idx=list(range(1, t_max + 1)), share_idx=list(range(Ns + 1, 2 * Ns + 1)),
                crypto=_crypto, server_computes_shares=SERVER_COMPUTES_SHARES,
                participant_filter=PARTICIPANT_FILTER,
                min_participants=filter_floor(max(k1s, k2s), math.ceil(FILTER_ALPHA * Ns)),
                corrupt_bound=math.ceil(FILTER_ALPHA * Ns),
                vark=FILTER_VARK,
                filter_addback=FILTER_ADDBACK,
                broadcast_parallel=E2E_BROADCAST_PARALLEL, use_perp=use_perp, thread_count=60,
                lr_local=lr_local, lr_global=lr_global, iter_start=1, num_iterations=E2E_ITERATIONS,
                device=_device, weights_path=start_weights_path,
                addr_map={str(k): v for k, v in addr_map.items()},
                poison=None, malicious_ids=[],
                seed=E2E_SEED,
            )
            nk_root = E2E_ROOT / f"{combo_name}{alpha_tag}"
            print(f"\n########## (N={Ns}, k1={k1s}), {P} parties ##########", flush=True)

            for (bw, rtt) in COMM_SWEEP:
                label = f"{_bw_label(bw)}_{rtt}ms"
                out_dir = nk_root / label
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n--- [sweep] {combo_name} / {label}: {bw} Mbit/s, RTT {rtt} ms ---", flush=True)

                # --- where this combo starts: round 1, or after the last checkpointed round ---
                wall_log = out_dir / "wall_log.txt"      # one line per finished round, written live
                ck_dir = out_dir / "checkpoints"
                it_first, resume_over = 1, {}
                if E2E_RESUME or E2E_RESUME_AT is not None:
                    _done = sorted(int(m.group(1)) for f in ck_dir.glob("*_Iter*.pt")
                                   if (m := re.search(r"_Iter(\d+)\.pt$", f.name)))
                    if E2E_RESUME_AT is not None:
                        _keep_upto = E2E_RESUME_AT - 1
                        if _keep_upto > 0 and _keep_upto not in _done:
                            raise SystemExit(
                                f"[sweep] {combo_name}/{label}: no checkpoint of round {_keep_upto} in "
                                f"{ck_dir} (have {_done[0] if _done else '-'}..{_done[-1] if _done else '-'})")
                        _set_aside_rounds(out_dir, ck_dir, wall_log, _keep_upto)
                        _done = [d for d in _done if d <= _keep_upto]
                    if _done and _done[-1] >= E2E_ITERATIONS:
                        # Finished earlier: nothing to run, and its logs stay as they are. A
                        # finished combo that an earlier resume took apart (per-worker files back,
                        # terminal_log.txt removed, then skipped) gets its end written again.
                        if not (out_dir / "terminal_log.txt").exists():
                            _restore_worker_logs(out_dir)
                            _write_combo_end(out_dir, wall_log, "ok", Ns, k1s, k2s, bw, rtt, t=t_over,
                                             note=f"end rewritten on a later resume; checkpoints up to "
                                                  f"round {_done[-1]} present")
                            print(f"[sweep] {combo_name}/{label}: end rewritten (rounds up to {_done[-1]})",
                                  flush=True)
                        print(f"[sweep] {combo_name}/{label}: already at round {_done[-1]}, skipping", flush=True)
                        continue
                    if E2E_RESUME_AT is None and _done:
                        _restore_worker_logs(out_dir)
                    if _done:
                        last = _done[-1]
                        _gm = get_model(MODEL_NAME)
                        _gck = next(ck_dir.glob(f"*_Iter{last}.pt"))
                        _gm.load_state_dict(torch.load(_gck, map_location="cpu"))
                        _flat = out_dir / f"resume_Iter{last}_flat.pt"
                        # the shared vector is the parameters only (vector_to_parameters);
                        # the batch-norm statistics come back from the checkpoint itself
                        torch.save(torch.nn.utils.parameters_to_vector(_gm.parameters()).cpu().detach().clone(),
                                   _flat)
                        it_first = last + 1
                        resume_over = dict(resume=True, iter_start=it_first, weights_path=str(_flat),
                                           bn_state_path=str(_gck))
                        # drop any record of the round that was cut off, in the logs and wall times
                        for _f in out_dir.glob("worker_*.jsonl"):
                            _keep = [l for l in _f.read_text(encoding="utf-8").splitlines()
                                     if l.strip() and json.loads(l)["iteration"] <= last]
                            _f.write_text("".join(l + "\n" for l in _keep), encoding="utf-8")
                        if wall_log.exists():
                            _keep = [l for l in wall_log.read_text(encoding="utf-8").splitlines()
                                     if (m := re.match(r"iteration (\d+):", l)) and int(m.group(1)) <= last]
                            wall_log.write_text("".join(l + "\n" for l in _keep), encoding="utf-8")
                        print(f"[sweep] {combo_name}/{label}: resuming after round {last} ({_gck.name})", flush=True)
                    else:
                        print(f"[sweep] {combo_name}/{label}: no checkpoint in {ck_dir}, starting at round 1", flush=True)

                if label in cost_fits:
                    # the fit this run chooses its round sizes with, kept next to its results
                    (out_dir / "cost_model_used.json").write_text(json.dumps(
                        {"source": str(COST_MODEL_PATH.resolve()), "label": label,
                         ("fit_n_A_B" if FILTER_VARK == "opt" else f"plan3_points_n_k_s_c{FILTER_C}"):
                             cost_fits[label]}, indent=2) + "\n", encoding="utf-8")
                e2e_lines, status, procs = [], "ok", []
                try:
                    net_setup.setup(P, bw, rtt)
                    if it_first == 1:
                        for _f in out_dir.glob("worker_*.jsonl"):
                            _f.unlink()
                        if wall_log.exists():
                            wall_log.unlink()
                    ctx = multiprocessing.get_context("spawn")
                    all_barrier = ctx.Barrier(Ns + 1)     # server + N clients
                    client_barrier = ctx.Barrier(Ns)      # clients only
                    start_gate = ctx.Barrier(Ns + 2)      # workers + this parent
                    end_gate = ctx.Barrier(Ns + 2)
                    compute_lock = ctx.RLock()

                    for pid, s in slot_of.items():
                        wspec = dict(common, party_id=pid, is_server=(pid == -1),
                                     netns=_ns(s), ip=_ip(s), port=SERVER_PORT,
                                     log_dir=str(out_dir),
                                     # no ckpt_dir -> nothing is checkpointed
                                     ckpt_dir=str(ck_dir) if E2E_SAVE_CKPT else None)
                        if resume_over:
                            wspec.update(resume_over)
                        if pid == -1 and label in cost_fits:
                            wspec["cost_fit" if FILTER_VARK == "opt" else "cost_points"] = cost_fits[label]
                        procs.append(ctx.Process(
                            target=worker_main, name=f"party{pid}",
                            args=(wspec, all_barrier, client_barrier, compute_lock,
                                  start_gate, end_gate)))
                    for p in procs:
                        p.start()
                    try:
                        for it in range(it_first, E2E_ITERATIONS + 1):
                            start_gate.wait(timeout=E2E_GATE_TIMEOUT_S)
                            t0 = time.perf_counter()
                            end_gate.wait(timeout=E2E_GATE_TIMEOUT_S)
                            e2e_ms = (time.perf_counter() - t0) * 1000
                            line = f"iteration {it}: {e2e_ms:.2f} ms ({e2e_ms / 1000:.2f} s)"
                            print(f"[sweep] {combo_name}/{label} {line}", flush=True)
                            with wall_log.open("a", encoding="utf-8") as _wl:
                                _wl.write(line + "\n")
                        for p in procs:
                            p.join(timeout=60)
                        codes = [p.exitcode for p in procs]
                        if any(c not in (0, None) for c in codes):
                            status = f"error: worker exitcodes {codes}"
                    except BrokenBarrierError:
                        status = "timeout (deadlock or slower than E2E_GATE_TIMEOUT_S)"
                        print(f"[sweep] {label}: TIMED OUT -> terminating workers", flush=True)
                except Exception as e:
                    status = f"error: {e!r}"
                    print(f"[sweep] {label}: {status}", flush=True)
                finally:
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    try:
                        net_setup.teardown(P)
                    except Exception:
                        pass

                # per-combo logs: processing_time_log.jsonl + terminal_log.txt (e2e times + status)
                # [B6] Each worker record now carries both time_logs_ms and comm_logs_bytes,
                # so the same combined content is written under both names: one netns sweep
                # yields the phase-wise timing AND the communication volume, and the analyzer
                # scripts take either file by the name they already expect.
                _write_combo_end(out_dir, wall_log, status, Ns, k1s, k2s, bw, rtt, t=t_over)
                print(f"[sweep] {combo_name}/{label}: {status}  -> {out_dir}", flush=True)

        print("\n[sweep] done.")
        sys.exit(0)

    print("\n=== 5. Spawning Parties ===")

    import party as _party
    _party.SERVER_COMPUTES_SHARES = SERVER_COMPUTES_SHARES

    common_args = (
        party_addr_map,
        DATASET_NAME, DATASET_PATH, PARTITION_PATH,
        DEVICE, MODEL_NAME, model, trainer,
        BATCH_SIZE, val_max, weights,
        pp, config1, config2, config1_det,
    )

    server = init_party('Server', -1, *common_args)
    clients = []
    for id in range(N):
        clients.append(init_party('Client', id, *common_args))

    if PARTICIPANT_FILTER and MODE == "RVFL":
        # The VPSS configs are rebuilt every round from the round's participant list, so
        # every party needs the shape rather than the fixed configs passed above.
        for _party_obj in [server] + clients:
            _party_obj.participant_filter = True
            _party_obj.set_vpss_shape(N, k1, k2, corrupt_bound=FILTER_C, vark=FILTER_VARK)
        server.min_participants = MIN_PARTICIPANTS
        server.addback_strategy = FILTER_ADDBACK
        if FILTER_VARK == "opt":
            server.cost_fit = load_cost_fit(COST_MODEL_PATH, os.environ.get("RVFL_E2E_COST_LABEL"))
        elif FILTER_VARK == "full":
            server.cost_points = load_cost_points(COST_MODEL_PATH, os.environ.get("RVFL_E2E_COST_LABEL"),
                                                  FILTER_C)
        print(f"[Participation] filter on: alpha={FILTER_ALPHA} "
              f"min_participants={MIN_PARTICIPANTS} (of N={N})")

    # === Apply attack (opt-in). No effect when ATTACK is None. ===
    #   Data-poisoning (label-flip): set up here (rebuilds malicious dataloaders once).
    #   Model-poisoning (trim): nothing to set up here -- the malicious g_i is crafted every
    #   round after train_phase by attacks.apply_model_poisoning (see the training loop).
    from attacks import MODEL_POISON_ATTACKS
    malicious_ids = []
    if ATTACK is not None:
        num_malicious = int(math.floor(N * MALICIOUS_RATE))
        malicious_ids = list(range(num_malicious))   # first m clients are malicious
        if ATTACK == "label_flip_fltrust":
            from dataloader import PoisoningConfig
            poison_cfg = PoisoningConfig(poison_type="label_flip_fltrust",
                                         poison_rate=1.0, num_classes=10)  # CIFAR10/MNIST
            for cid in malicious_ids:
                clients[cid].set_poisoning(poison_cfg)
        elif ATTACK == "scaling_fltrust":
            # Scaling/backdoor (targeted): malicious clients train on trigger-stamped,
            # target-relabeled data; the g_i is then scaled by lambda=n in the per-round hook.
            # CIFAR-10 target = "bird" (class 2); SCALING_P = fraction of data triggered.
            from dataloader import PoisoningConfig
            SCALING_TARGET, SCALING_P = 2, 0.5      # bird; augment fraction (tune per ref [5])
            poison_cfg = PoisoningConfig(poison_type="backdoor", target_label=SCALING_TARGET,
                                         poison_rate=SCALING_P, num_classes=10)
            for cid in malicious_ids:
                clients[cid].set_poisoning(poison_cfg)
        elif ATTACK in MODEL_POISON_ATTACKS:
            pass   # trim/adaptive: crafted per round in the training loop (pure model poisoning)
        else:
            raise ValueError(f"Unknown ATTACK: {ATTACK}")
        print(f"[Attack] type={ATTACK}, malicious clients ({len(malicious_ids)}/{N}) = {malicious_ids}")

    THREAD_COUNT = 60
    for iteration in range(ITER_START, NUM_ITERATIONS + 1):
        t_start = time.perf_counter()

        print(f"=== Iteration {iteration} / {NUM_ITERATIONS} Start ===")

        # In the case of BREA, server training is not required.
        if MODE == "BREA":
            print(f"\t[Train Phase]")
            total = len(clients)
            for i, client in enumerate(clients):
                print(f"\r\t\tClients' Train Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="")
                client.train_phase(epoch=1, lr=lr_local, logging=ENABLE_LOGGING)
            print()

            if ATTACK in MODEL_POISON_ATTACKS and malicious_ids:
                from attacks import apply_model_poisoning
                apply_model_poisoning(clients, malicious_ids, ATTACK,
                                      g_server=getattr(server, "g_server", None))

            client_processes = [client.BREA(THREAD_COUNT, logging=ENABLE_LOGGING) for client in clients]
            server_process = server.BREA(lr_global=lr_global, byzantine_rate=0.3, thread_count=THREAD_COUNT, logging=ENABLE_LOGGING)
            try:
                # deal share of local update
                print(f"\t[Client - Deal Shares]")
                for cp in client_processes:
                    next(cp)
                # calculate share of pairwise distance
                print(f"\t[Client - Verify Shares and Calculate Pairwise Distance]")
                for cp in client_processes:
                    next(cp)
                # select clients by Multi-Krum
                print(f"\t[Server - Reconstruct and Multi-Krum]")
                next(server_process)
                # make share of aggregated update
                print(f"\t[Client - Calculate Aggregated Shares]")
                for cp in client_processes:
                    next(cp)
                # update weights
                print(f"\t[Server - Reconstruct and Update Model]")
                next(server_process)
                # receive new weights
                print(f"\t[Client - Receive Updated Model]")
                for cp in client_processes:
                    next(cp)
            except StopIteration:
                pass
        else:
            # 0. Train Phase
            print(f"\t[Train Phase]")
            server.train_phase(epoch=1, lr=lr_local, logging=ENABLE_LOGGING)
            total = len(clients)
            for i, client in enumerate(clients):
                print(f"\r\t\tClients' Train Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="")
                client.train_phase(epoch=1, lr=lr_local, logging=ENABLE_LOGGING)
            print()

            # Model-poisoning: overwrite malicious clients' g_i (crafted from benign updates).
            # No-op for None / label-flip (which already acts via poisoned data). Adaptive also
            # needs the server root update g_0 (server.g_server) to score/optimize against.
            if ATTACK in MODEL_POISON_ATTACKS and malicious_ids:
                from attacks import apply_model_poisoning
                apply_model_poisoning(clients, malicious_ids, ATTACK,
                                      g_server=getattr(server, "g_server", None))

            if MODE == "RVFL":
                # 1. Enrollment Phase
                print(f"\t[Enrollment Phase]")
                processes = [client.enrollment_phase(use_perp, THREAD_COUNT, logging=ENABLE_LOGGING) for client in clients]
                if PARTICIPANT_FILTER:
                    # With the filter on the server's enrollment is a generator with two
                    # exchanges in between, so the two sides are interleaved explicitly.
                    server_process = server.enrollment_phase(THREAD_COUNT, logging=ENABLE_LOGGING)
                    next(server_process)      # server broadcasts g_server
                    for cp in processes:
                        next(cp)              # receive g_server, compute cos, send cos
                    next(server_process)      # collect cos, choose participants, broadcast the list
                    for cp in processes:
                        next(cp)              # receive the list, deal and send first-level shares
                    for cp in processes:
                        next(cp)              # receive first-level shares
                else:
                    server.enrollment_phase(THREAD_COUNT, logging=ENABLE_LOGGING)
                    while True:
                        try:
                            for process in processes:
                                next(process)
                        except StopIteration:
                            break

                # 2. Verification Phase
                print(f"\t[Verification Phase]")
                processes = [client.verification_phase(use_perp, THREAD_COUNT, logging=ENABLE_LOGGING) for client in clients]
                for cp in processes:
                    next(cp)              # send second-level share with Pedersen VSS
                for cp in processes:
                    next(cp)              # send verify list
                for cp in processes:
                    next(cp)              # send s_tilde to server
                server.verification_phase(use_perp, THREAD_COUNT, logging=ENABLE_LOGGING)      # send TS
                for cp in processes:
                    next(cp)              # receive TS

                # 3. Aggregation & Final Update Phase
                print(f"\t[Aggregation Phase]")
                processes = [client.aggregation_phase(logging=ENABLE_LOGGING) for client in clients]
                for cp in processes:
                    next(cp)
                server.aggregation_phase(lr_global, use_error_correction=True, use_perp=use_perp, thread_count=THREAD_COUNT, logging=ENABLE_LOGGING)
                for cp in processes:
                    next(cp)
            else:
                print(f"\t[Aggregation Phase]")
                processes = [client.plaintext_agg(logging=ENABLE_LOGGING) for client in clients]
                for cp in processes:
                    next(cp)
                if MODE == "FLTrust":
                    # FLTrust
                    server.FLTrust_agg(lr_global = lr_global, logging=ENABLE_LOGGING)
                elif MODE == "FLTrust_quantized":
                    # FLTrust with quantized vector (simulate RVFL without perp)
                    server.FLTrust_quantized_agg(lr_global = lr_global, logging=ENABLE_LOGGING)
                elif MODE == "FLTrust_quantized_perp":
                    # FLTrust with quantized perpendicular vector (simulate RVFL with perp).
                    # Pass attack + malicious ids so the server can model the Scaling submission
                    # (lambda-scaled perp) and reject it via the dot-product verification.
                    server.FLTrust_quantized_perp_agg(lr_global=lr_global, attack=ATTACK,
                                                      malicious_ids=malicious_ids, logging=ENABLE_LOGGING)
                elif MODE == "FedAvg":
                    # FedAvg
                    server.FedAvg_agg(lr_global = lr_global, logging=ENABLE_LOGGING)
                for cp in processes:
                    next(cp)
            
        t_end = time.perf_counter()
        print(f"Processing time for 1 iteration: {(t_end - t_start) * 1000:.2f} ms")
            


        if (iteration-1)%CHECKPOINT_INTERVAL == 0:
            # save weights
            # format: {dataset}_{model}_{N}_{iteration}.pt
            torch.nn.utils.vector_to_parameters(server.weights, model.parameters())
            filename = f"{DATASET_NAME}_{MODEL_NAME}_N{N}_Iter{iteration}.pt"
            save_path = CKPT_DIR / filename
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

        if ENABLE_LOGGING:
            with TIME_LOG_PATH.open("a", encoding="utf-8") as f_time, \
                 COMM_LOG_PATH.open("a", encoding="utf-8") as f_comm, \
                 VALUE_LOG_PATH.open("a", encoding="utf-8") as f_value :
                 
                for client in [server] + clients:
                    logs = client.extract_and_clear_logs(iteration)
                    
                    time_entry = {
                        "party_id": logs["party_id"],
                        "iteration": logs["iteration"],
                        "time_logs_ms": logs["processing_time_ms"]
                    }
                    f_time.write(json.dumps(time_entry) + "\n")
                    
                    comm_entry = {
                        "party_id": logs["party_id"],
                        "iteration": logs["iteration"],
                        "comm_logs_bytes": logs["communication_cost_bytes"]
                    }
                    f_comm.write(json.dumps(comm_entry) + "\n")

                    if logs["party_id"] == -1:
                        value_entry = {
                            "iteration": logs["iteration"],
                            "values": logs["values"]
                        }
                        f_value.write(json.dumps(value_entry) + "\n")
        
        print(f"=== Iteration {iteration} Finished ===")

    print(f"=== All Iterations Completed. ===")
    print("\n=== Simulation Finished ===")