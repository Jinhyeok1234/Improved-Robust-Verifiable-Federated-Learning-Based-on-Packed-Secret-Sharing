#!/usr/bin/env python3
"""Crypto-cost sweep driver: runs the RVFL path over an (N, k1) grid and collects the
phase-wise computation time and the communication volume into one CSV.

Grid. For N = 10 to 50 the k1 values follow the doubling rule: 1, 2, 4, 8, ... while they stay
below floor(N/3), then floor(N/3) itself closes the row. For N = 60 to 100 the combinations are
the specific ones the manuscript reports. t1 = t2 = floor(N/3) and k2 = 1, exactly as main.py
derives them.

    N= 10 -> k1 in 1, 2, 3            N= 60 -> k1 = 20
    N= 20 -> k1 in 1, 2, 4, 6         N= 70 -> k1 = 23
    N= 30 -> k1 in 1, 2, 4, 8, 10     N= 80 -> k1 = 26
    N= 40 -> k1 in 1, 2, 4, 8, 13     N= 90 -> k1 = 30
    N= 50 -> k1 in 1, 2, 4, 8, 16     N=100 -> k1 in 1, 2, 4, 8, 16, 32, 33

Which of those actually run is decided by the estimated peak memory, not by a fixed list: at or
below --defer-gb they keep grid order, between --defer-gb and --exclude-gb they run afterwards
lightest first, and above --exclude-gb they are not scheduled. Changing the backend or the
security level changes |q| and therefore the schedule. --dry-run prints the split with the
estimate of every combination.

Each combination is one `python main.py` run with the settings passed in the environment, so
no source file is edited. Logs land in ./runs/N{N}/logs/RVFL/ under a name that carries k1,
the backend and the security level, so no two combinations share a file.

    python cost_sweep.py --dry-run              # print the plan and stop
    python cost_sweep.py                        # the full grid at the default backend/level
    python cost_sweep.py --backend zp --level 128
    python cost_sweep.py --n 10 20 --iters 5
    python cost_sweep.py --collect-only         # rebuild the CSV from logs already on disk
    python cost_sweep.py --preview              # summarise what has finished, from another
                                                # terminal, while the sweep is still running

A combination is skipped only when its logs already hold the full `--iters` rounds for the
server and every client, so an interrupted sweep resumes by re-running the same command and a
run killed part-way is redone rather than mistaken for a finished one.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import comm_log_analyzer
import comp_log_analyzer_revised


# ---------------------------------------------------------------------------------------
# Memory model.
#
# During enrollment every one of N dealers holds a share for each of N parties, and a share
# is ceil(M/k1) blocks of two Z_q elements (the value s and the Pedersen blind r). So the
# number of Z_q elements alive at the peak is
#
#     N * N * 2 * ceil(M / k1)
#
# and the bytes per element depend on |q|, which the security level fixes. MEM_PER_ELEMENT
# below is measured: the RSS delta across one PedersenVPSS.distribute with 20,000 blocks gave
# 84.7 / 107.5 / 128.0 bytes per element at |q| = 256 / 384 / 521, rescaled by 1.195 so that
# the 256 entry reproduces the 153 B/element a full main.py run actually showed (the isolated
# call misses the serialisation buffers and the receiving copies).
#
# The table is keyed on the security level, not on the backend, because a share is an element
# of Z_q in both backends and |q| is what the level fixes. What the backend changes is the
# commitments, and those are O(N * t) rather than O(N^2 * M / k1) -- a client broadcasts about
# 8 KB of them per round at N = 10 -- so they do not move this estimate. The Z_p backend does
# carry a larger fixed working set, because GMP keeps big temporaries for 3072- to 15360-bit
# modular exponentiation: the same isolated measurement came out at 141.8 and 256.8 bytes per
# element for zp at levels 128 and 192, which at N = 6 and 20,000 blocks is a few hundred MB of
# scratch mislabelled as a per-element cost. Allow roughly that much headroom on top of the
# estimate when running the Z_p backend.
#
# The estimate is a fit anchored on one end-to-end measurement, not a derivation. Check it on
# the real machine with a light combination before trusting the heavy end of the table.
# ---------------------------------------------------------------------------------------
MEM_PER_ELEMENT = {          # bytes of RSS per Z_q element held
    "128": 101.2,            # |q| = 256
    "192": 128.5,            # |q| = 384
    "256": 153.0,            # |q| = 521 (ec) / 512 (zp) -- the anchor, measured end to end
    "legacy512": 153.0,      # |q| = 511
}
DEFAULT_MODEL_SIZE = 274_442     # ResNet20 as RoFL counts it, the heaviest task in the plan

# A combination above DEFER_GB runs after everything lighter; one above EXCLUDE_GB is not
# scheduled at all. Both are overridable from the command line.
DEFER_GB = 100.0
EXCLUDE_GB = 130.0


def est_memory_gb(n_clients, k1, backend, level, model_size=DEFAULT_MODEL_SIZE):
    per = MEM_PER_ELEMENT.get(level, MEM_PER_ELEMENT["256"])
    blocks = -(-model_size // k1)          # ceil
    return n_clients * n_clients * 2 * blocks * per / 2**30


# The (N, k1) candidates. N = 10..50 follow the doubling rule; the larger N are the specific
# combinations the manuscript reports, not a full row each.
BASE_N = (10, 20, 30, 40, 50)
EXTRA_COMBOS = [(60, 20), (70, 23), (80, 26), (90, 30),
                (100, 1), (100, 2), (100, 4), (100, 8), (100, 16), (100, 32), (100, 33)]

def k1_values(n_clients):
    """1, 2, 4, 8, ... below floor(n/3), then floor(n/3) itself."""
    t = n_clients // 3
    vals, k = [], 1
    while k < t:
        vals.append(k)
        k *= 2
    vals.append(t)
    return vals


def candidates(n_list, k1_list):
    if k1_list:
        return [(n, k) for n in n_list for k in k1_list]
    out = [(n, k) for n in n_list if n in BASE_N for k in k1_values(n)]
    out += [c for c in EXTRA_COMBOS if c[0] in n_list]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def schedule(n_list, k1_list, backend, level, model_size, defer_gb, exclude_gb):
    """Split the candidates into (run now, run last, not scheduled) by estimated memory.

    Everything at or below defer_gb keeps grid order, because those are expected to finish.
    Everything between defer_gb and exclude_gb runs afterwards, lightest first, so that a
    combination that does fit is never blocked behind one that does not. Above exclude_gb
    nothing is scheduled. The split therefore moves on its own when the backend or the
    security level changes |q|, rather than being a fixed list of pairs.
    """
    rows = [(n, k, est_memory_gb(n, k, backend, level, model_size))
            for n, k in candidates(n_list, k1_list)]
    head = [(n, k, g) for n, k, g in rows if g <= defer_gb]
    tail = sorted((r for r in rows if defer_gb < r[2] <= exclude_gb), key=lambda r: r[2])
    cut = sorted((r for r in rows if r[2] > exclude_gb), key=lambda r: r[2])
    return head, tail, cut


def read_records(path):
    """Every JSON record in a log, in file order. Missing or malformed -> empty."""
    out = []
    if not path.exists():
        return out
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except (OSError, ValueError):
        return []
    return out


def dedup(records):
    """Keep the last record for each (party_id, iteration).

    main.py opens its logs in append mode, so a combination that was run more than once has
    several records per round in one file. Every run restarts at iteration 1 (the driver sets
    RVFL_RESUME=0), so the later record for a given (party, iteration) is the one from the most
    recent run, and the earlier ones are superseded -- including the half-written round a
    killed run leaves behind. Averaging over all of them mixes runs together.
    """
    latest = {}
    for rec in records:
        latest[(rec.get("party_id"), rec.get("iteration"))] = rec
    return list(latest.values())


def is_complete(path, n_clients, iters):
    """True when the log holds `iters` distinct rounds for the server and every client."""
    rounds = {}
    for rec in dedup(read_records(path)):
        rounds.setdefault(rec.get("party_id"), set()).add(rec.get("iteration"))
    expected = set(range(n_clients)) | {-1}
    if not expected.issubset(rounds):
        return False
    return min(len(rounds[p]) for p in expected) >= iters


def repair(path):
    """Rewrite a log with the superseded duplicates removed. Returns lines dropped."""
    records = read_records(path)
    kept = dedup(records)
    if len(kept) == len(records):
        return 0
    kept.sort(key=lambda r: (r.get("iteration", 0), r.get("party_id", 0)))
    with open(path, "w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec) + "\n")
    return len(records) - len(kept)


def log_paths(n_clients, k1, backend, level):
    tag = f"RVFL_noattack_k{k1}_{backend}{level}_N{n_clients}"
    root = Path(f"./runs/N{n_clients}/logs/RVFL")
    return root / f"{tag}_processing_time_log.jsonl", root / f"{tag}_communication_cost_log.jsonl"


def value_log(n_clients, k1, args):
    tag = f"RVFL_noattack_k{k1}_{args.backend}{args.level}_N{n_clients}"
    return Path(f"./runs/N{n_clients}/logs/RVFL") / f"{tag}_values_log.jsonl"


def run_one(n_clients, k1, args):
    env = dict(
        os.environ,
        RVFL_MODE="RVFL",
        RVFL_ATTACK="none",
        RVFL_RESUME="0",
        RVFL_E2E="0",
        RVFL_N=str(n_clients),
        RVFL_K1=str(k1),
        RVFL_ITERS=str(args.iters),
        RVFL_BACKEND=args.backend,
        RVFL_SECLEVEL=args.level,
        PYTHONUNBUFFERED="1",
    )
    if args.dataset:
        env["RVFL_DATASET"] = args.dataset
    if args.model:
        env["RVFL_MODEL"] = args.model

    out_dir = Path("./logs/cost_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"N{n_clients}_k{k1}_{args.backend}{args.level}.out"
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run([sys.executable, "main.py"], env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
    print(f"    {status}  ({elapsed:.1f} s wall)  terminal output -> {out_path}", flush=True)
    return proc.returncode == 0


def collect(n_clients, k1, args, read_only=False):
    """Read the two logs of one combination and flatten them into a CSV row.

    read_only leaves the log files untouched. The repair pass rewrites them, and a sweep that
    is still running has main.py appending to the very file being rewritten, so a preview taken
    while the sweep is in flight must not repair.
    """
    time_log, comm_log = log_paths(n_clients, k1, args.backend, args.level)
    if not time_log.exists() or not comm_log.exists():
        return None

    if read_only:
        # Only finished combinations: the one being written right now holds a partial round
        # whose average would be meaningless.
        if not (is_complete(time_log, n_clients, args.iters)
                and is_complete(comm_log, n_clients, args.iters)):
            return None
    else:
        # A log written by more than one run holds superseded records; drop them before averaging.
        dropped = repair(time_log) + repair(comm_log)
        if dropped:
            print(f"    N={n_clients} k1={k1}: dropped {dropped} superseded log record(s)")

    comp = comp_log_analyzer_revised.summarize(str(time_log))
    comm = comm_log_analyzer.calculate_communication_averages(str(comm_log))

    row = {
        "N": n_clients, "k1": k1, "t1": n_clients // 3, "k2": 1,
        "backend": args.backend, "level": args.level, "iters": args.iters,
    }
    for party in ("server", "client"):
        for kind in ("comp", "comm"):
            block = comp.get(party, {}).get(kind, {})
            row[f"{party}_{kind}_total_s"] = block.get("total_s", 0.0)
            for phase, ms in block.get("per_phase_ms", {}).items():
                row[f"{party}_{kind}_{phase}_ms"] = ms
    for key, val in comm.items():
        row[f"{key}_bytes"] = val
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=None,
                    help="client counts to sweep (default: 10 through 100). Naming --n or --k1 "
                         "makes this a hand-picked run, and then nothing is dropped for memory")
    ap.add_argument("--k1", type=int, nargs="+", default=None,
                    help="explicit k1 list, applied to every N (default: the doubling rule)")
    ap.add_argument("--backend", default=None, help="ec or zp (default: crypto_params default)")
    ap.add_argument("--level", default=None, help="128, 192, 256 or legacy512")
    ap.add_argument("--iters", type=int, default=5, help="rounds per combination (default: 5)")
    ap.add_argument("--dataset", default=None, help="overrides RVFL_DATASET")
    ap.add_argument("--model", default=None, help="overrides RVFL_MODEL")
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default: cost_sweep_<backend><level>.csv, so that a "
                         "second configuration does not overwrite the first)")
    ap.add_argument("--model-size", type=int, default=DEFAULT_MODEL_SIZE,
                    help=f"parameter count M used for the memory estimate "
                         f"(default {DEFAULT_MODEL_SIZE}, ResNet20)")
    ap.add_argument("--defer-gb", type=float, default=DEFER_GB,
                    help=f"above this estimate a combination runs last (default {DEFER_GB})")
    ap.add_argument("--exclude-gb", type=float, default=None,
                    help=f"above this estimate a combination is not scheduled. Default {EXCLUDE_GB} "
                         f"for the full grid, and no ceiling at all when --n or --k1 names the "
                         f"combinations by hand")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--collect-only", action="store_true",
                    help="skip the runs, rebuild the CSV from logs already on disk")
    ap.add_argument("--preview", action="store_true",
                    help="summarise the combinations finished so far WITHOUT touching any file "
                         "the running sweep is writing; safe to use from a second terminal "
                         "while cost_sweep.py is still going. Writes its own CSV.")
    ap.add_argument("--repair-only", action="store_true",
                    help="only clean superseded records out of the logs, run and collect nothing")
    ap.add_argument("--rerun", action="store_true",
                    help="run every combination even if its log is already complete "
                         "(its old log is deleted first, not appended to)")
    args = ap.parse_args()

    import crypto_params
    if args.backend is None:
        args.backend = crypto_params.DEFAULT_BACKEND
    if args.level is None:
        args.level = crypto_params.DEFAULT_LEVEL
    if args.csv is None:
        suffix = "_preview" if args.preview else ""
        args.csv = f"cost_sweep_{args.backend}{args.level}{suffix}.csv"

    # A grid named by hand is a grid that is meant to run. The memory estimate then only orders
    # the combinations -- heavy ones last, lightest first -- instead of removing any, because a
    # combination typed on the command line was chosen deliberately and silently dropping it
    # leaves a hole in the table that is only noticed much later. --exclude-gb still imposes a
    # ceiling when one is asked for, and the default grid keeps the ceiling it always had.
    hand_picked = args.n is not None or args.k1 is not None
    if args.n is None:
        args.n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    if args.exclude_gb is not None:
        ceiling = args.exclude_gb
    elif hand_picked:
        ceiling = float("inf")
    else:
        ceiling = EXCLUDE_GB

    head, tail, cut = schedule(args.n, args.k1, args.backend, args.level,
                               args.model_size, args.defer_gb, ceiling)
    combos = [(n, k) for n, k, _ in head + tail]
    gb = {(n, k): g for n, k, g in head + tail + cut}

    print(f"backend={args.backend} level={args.level} iters={args.iters}  "
          f"model M={args.model_size:,}  {len(combos)} combinations")
    for n_clients in sorted({n for n, _ in combos} | {n for n, _, _ in cut}):
        t = n_clients // 3
        run = [k for n, k in combos if n == n_clients]
        late = [k for n, k, _ in tail if n == n_clients]
        gone = [k for n, k, _ in cut if n == n_clients]
        line = f"  N={n_clients:3d}  t1=t2={t:2d}  k1 = {sorted(run)}"
        if late:
            line += f"   (deferred: {sorted(late)})"
        if gone:
            line += f"   (over {ceiling:.0f} GB, not scheduled: {sorted(gone)})"
        print(line)
    print("\n  order, with the estimated peak memory of each:")
    deferred = {(n, k) for n, k, _ in tail}
    for n_clients, k1 in combos:
        estimate = gb[(n_clients, k1)]
        mark = " <- deferred" if (n_clients, k1) in deferred else ""
        # With no ceiling in force, a combination past the usual one still runs, so name it:
        # that is the one most likely to be killed by the machine rather than by this script.
        if estimate > EXCLUDE_GB:
            mark += f" (over the usual {EXCLUDE_GB:.0f} GB ceiling)"
        print(f"    N={n_clients:3d} k1={k1:2d}   {estimate:7.1f} GB{mark}")
    if cut:
        print("\n  not scheduled:")
        for n_clients, k1, g in cut:
            print(f"    N={n_clients:3d} k1={k1:2d}   {g:7.1f} GB")
    print()
    if args.dry_run:
        return

    if args.repair_only:
        total = 0
        for n_clients, k1 in combos:
            tl, cl = log_paths(n_clients, k1, args.backend, args.level)
            dropped = repair(tl) + repair(cl)
            if dropped:
                print(f"  N={n_clients} k1={k1}: dropped {dropped} record(s)")
            total += dropped
        print(f"\n{total} superseded record(s) removed")
        return

    if not args.collect_only and not args.preview:
        for idx, (n_clients, k1) in enumerate(combos, 1):
            time_log, comm_log = log_paths(n_clients, k1, args.backend, args.level)
            if not args.rerun and is_complete(time_log, n_clients, args.iters) \
                    and is_complete(comm_log, n_clients, args.iters):
                print(f"[{idx}/{len(combos)}] N={n_clients} k1={k1}: {args.iters} rounds already "
                      f"logged, skipping", flush=True)
                continue
            # main.py appends to its logs, so anything left from an earlier attempt has to go
            # before the run starts -- including when --rerun forced us here past a complete
            # log. Leaving it turns one round into two records and silently averages the two
            # runs together.
            stale = [q for q in (time_log, comm_log, value_log(n_clients, k1, args))
                     if q.exists()]
            if stale:
                print(f"[{idx}/{len(combos)}] N={n_clients} k1={k1}: removing "
                      f"{len(stale)} incomplete or superseded log(s)", flush=True)
                for q in stale:
                    q.unlink()
            print(f"[{idx}/{len(combos)}] N={n_clients} k1={k1} ...", flush=True)
            run_one(n_clients, k1, args)

    rows = [r for r in (collect(n, k, args, read_only=args.preview)
                        for n, k in sorted(combos)) if r is not None]
    if not rows:
        print("no finished combination yet" if args.preview else "no logs to collect")
        return

    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    if args.preview:
        done = {(int(r["N"]), int(r["k1"])) for r in rows}
        pending = [c for c in combos if c not in done]
        print(f"\n{len(rows)} of {len(combos)} combinations finished -> {args.csv}")
        if pending:
            print("   still to come: " + ", ".join(f"N{n}k{k}" for n, k in pending))
    else:
        print(f"\n{len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()