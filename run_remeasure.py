#!/usr/bin/env python3
"""Run the re-measurement of the paper's tables unattended: the cost sweep of tables 4, 5 and
6 (with BREA for table 6) and the netns end-to-end sweep of the discussion, every timing with
jemalloc preloaded.

    sudo -E "$(which python)" run_remeasure.py                       # start now
    sudo -E "$(which python)" run_remeasure.py --after-pid <pid>     # start once <pid> has ended
    python run_remeasure.py --status                                 # what is done, what is left
    python run_remeasure.py --dry-run                                # the plan and the commands

Run it from this folder (main.py resolves every path relative to the working directory). Root
is needed for the network namespaces.

Every combination: t1 = t2 = N // 3, k2 = 1, backend ec, level 256, model ResNet20 (272,474
parameters), ROUNDS = 6 rounds (BREA: BREA_ROUNDS, 6 at N = 10, 1 at N = 20..40); the tables
leave round 1 out (not where only one round is run)
(remeasure_tables.py --drop-first 1).

What is measured, from the tables it serves:

    table 4   cost sweep at t = k: (N, N // 3) for N = 10, 20, ..., 100
    table 5   cost sweep at N = 50, k1 = 2 4 8 16, and N = 100, k1 = 8 16 32;
              N = 50, k1 = 1 (COST_LAST) is tried after the netns runs and may fail: its
              memory estimate is far above the others
    table 6   cost sweep at N = 10, 20, 30, 40, k1 = 1, ours and BREA
    e2e       the cost rows with N <= E2E_MAX_N (50), the five links of LINKS

    COST_GRID  N= 10 -> 1 3     N= 50 -> 2 4 8 16    N= 90 -> 30
               N= 20 -> 1 6     N= 60 -> 20          N=100 -> 8 16 32 33
               N= 30 -> 1 10    N= 70 -> 23
               N= 40 -> 1 13    N= 80 -> 26
    E2E_GRID   N = 10..40 -> 1 and N // 3,  N = 50 -> 2 4 8 16
    BREA_N     10 20 30 40 (k1 = 1)
    E2E_LAST   N = 50, k1 = 1: the netns run of COST_LAST, after BREA; and E2E_EXTRA, the netns
               runs left out of E2E_GRID to save time and added afterwards (no cost sweep):
               N = 10 -> 2,  N = 20 -> 2 4,  N = 30 and 40 -> 2 4 8

Steps, in order:

    cost_N<N>     cost_sweep.py --n N --k1 <COST_GRID[N]> --iters 6 (in-memory transport:
                  computation time and communication volume); one step per N because
                  cost_sweep.py applies its --k1 list to every N it is given
    e2e_N<N>      netns sweep at N (RVFL_E2E=1, RVFL_N=N, RVFL_E2E_K1=<E2E_GRID[N]>), the five
                  links, 6 rounds, filter off; a checkpoint per round (RVFL_E2E_SAVE_CKPT=1) so
                  that a cut-off combination resumes (RVFL_E2E_RESUME=1) and a finished one is
                  skipped
    cost_N50_k1   COST_LAST, optional: tried once; if it fails it is not tried again on the next
                  invocation (--redo cost_N50_k1 tries it again), and it never holds up the tables
    tables        remeasure_tables.py: cost_sweep_ec256_drop1.csv and remeasure_ec256.md, every
                  figure over rounds 2..6 (written before BREA so ours does not wait for it)
    brea_N<N>     main.py with RVFL_MODE=BREA at N, BREA_ROUNDS[N] rounds (cost_sweep.py runs the
                  RVFL mode only). main.py leaves the backend out of a BREA log name, so
                  RVFL_RUN_SUFFIX adds it: runs/N<N>/logs/BREA/BREA_noattack_k1_ec256_N<N>_*.
                  BREA's Pedersen commitments cost M * (2tN + (t+2)N^2) group exponentiations a
                  round (0.6 / 3.7 / about 12 / about 27 hours at N = 10 / 20 / 30 / 40, ec256),
                  so these run last, N = 20..40 one round each, and resume: main.py continues from its last checkpoint (RVFL_RESUME=1), and
                  before that the checkpoints and log records after the last fully logged round
                  are moved into _set_aside/<time>/brea_N<N>_resume/
    tables_final  the tables again, now with BREA in table 6 (brea_ec256_drop1.csv)
    e2e_N<N>_k<k1..>  E2E_LAST, one step per N: e2e_N10_k2, e2e_N20_k2_4, e2e_N30_k2_4_8,
                  e2e_N40_k2_4_8, e2e_N50_k1; the netns sweep of those k1, as e2e_N<N> (the five
                  links, 6 rounds, resumable); a failure is tried again on the next invocation,
                  and holds nothing up
    tables_last   the tables again, now with the E2E_LAST runs in (c1)

A step that finishes is recorded in logs/remeasure/state.json and skipped the next time, so
after an interruption the same command continues where it stopped: cost_sweep.py skips the
combinations already complete, main.py resumes or skips each netns combination and each BREA
run. A step that fails is recorded as failed, the queue moves on, and the failed one is tried
again on the next invocation (except the optional one).

Nothing is deleted. The first time a step starts, the earlier results it would replace are
moved into _set_aside/<time>/<step>/ with their relative paths: for cost_N<N> every ec256
cost-sweep log, terminal output and checkpoint folder of that N (whatever its k1, except the
k1 of the optional step, which moves its own), and the step of the smallest N also moves
cost_sweep_ec256.csv; for brea_N<N> the ec256 BREA logs and checkpoint folder of that N; for
e2e_N<N> every filter-off folder e2e_results/ec256/N<N>_k<k1>/ except the k1 of E2E_LAST, which
their own step moves itself (the filter runs N<N>_k<k1>_a.../ stay where they are). --redo STEP forgets a step's record so it runs again, and
sets its current results aside the same way first.

Each step's output goes to logs/remeasure/<step>.log; this script prints one line per step.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- what the paper's tables need --------------------------------------------------------
TABLE4_N = list(range(10, 101, 10))           # t = k = N // 3
TABLE5 = {50: [1, 2, 4, 8, 16], 100: [8, 16, 32]}
TABLE6_N = [10, 20, 30, 40]                   # k1 = 1, ours and BREA
COST_LAST = [(50, 1)]                          # tried after everything else, may fail
E2E_MAX_N = 50
# netns runs left out of E2E_GRID to save time and added afterwards (2026-10-21): the k1 between
# 1 and N // 3 at N = 10..40. Netns only, no cost sweep; they run after BREA, one step per N.
E2E_EXTRA = [(10, 2), (20, 2), (20, 4), (30, 2), (30, 4), (30, 8), (40, 2), (40, 4), (40, 8)]
E2E_LAST = [(50, 1)] + E2E_EXTRA               # netns runs after BREA ((50, 1): the cost run of the
                                               # same pair needed swap, so it may fail too)


def _rows(pairs):
    out = {}
    for n, k in sorted(set(pairs)):
        out.setdefault(n, []).append(k)
    return out


COST_PAIRS = ({(n, n // 3) for n in TABLE4_N}
              | {(n, k) for n, ks in TABLE5.items() for k in ks}
              | {(n, 1) for n in TABLE6_N}) - set(COST_LAST)
COST_GRID = _rows(COST_PAIRS)
E2E_GRID = _rows((n, k) for n, k in COST_PAIRS if n <= E2E_MAX_N)
BREA_N = list(TABLE6_N)


def e2e_step_name(n, k):
    """The step that runs the netns combination (n, k): e2e_N<n>, or for E2E_LAST its own."""
    last = _rows(E2E_LAST)
    if k in last.get(n, ()):
        return f"e2e_N{n}_k{'_'.join(map(str, last[n]))}"
    return f"e2e_N{n}"

BACKEND, LEVEL = "ec", "256"
MODEL_SIZE = 272_474                  # ResNet20 (models.py), the model main.py trains by default
ROUNDS = 6
# BREA's rounds, by N. Its Pedersen commitments cost M * (2tN + (t+2)N^2) group exponentiations
# a round: 2,165 s at N = 10 and 13,155 s at N = 20 measured (ec256), so about 12 h at N = 30 and
# 27 h at N = 40. N = 10 runs ROUNDS rounds like everything else; N = 20..40 run one round, and
# the tables use that round (remeasure_tables.py leaves no round out where only one is run).
BREA_ROUNDS = {10: ROUNDS, 20: 1, 30: 1, 40: 1}
DROP_FIRST = 1                        # rounds the tables leave out


def brea_rounds(n):
    """The rounds BREA runs at N = n."""
    return BREA_ROUNDS.get(n, ROUNDS)
# (Mbit/s, RTT ms), the links of main.py's COMM_SWEEP, passed explicitly so the check below
# knows the folder names
LINKS = [(10, "80"), (1000, "0.2"), (50, "80"), (100, "80"), (10000, "0.2")]

STATE_DIR = Path("logs/remeasure")
STATE_FILE = STATE_DIR / "state.json"
CONFIG = f"{BACKEND}{LEVEL}"
BREA_TAG = f"BREA_noattack_k1_{CONFIG}"      # main.py: MODE_attack, _k1 (RVFL_K1), RVFL_RUN_SUFFIX


def link_label(bw, rtt):
    """The folder name main.py gives a link."""
    rate = f"{bw // 1000}Gbps" if (bw >= 1000 and bw % 1000 == 0) else f"{bw}Mbps"
    return f"{rate}_{rtt}ms"


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_elapsed(sec):
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m {sec % 60:02d}s"


# ------------------------------------------------------------------------------------------
# state
# ------------------------------------------------------------------------------------------
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_FILE)


def set_aside(step, paths, stamp):
    """Move the given files/folders into _set_aside/<stamp>/<step>/, keeping relative paths."""
    moved = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        dest = Path("_set_aside") / stamp / step / p
        n = 2
        while dest.exists():                       # a second set-aside within the same second
            dest = Path("_set_aside") / f"{stamp}_{n}" / step / p
            n += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        moved.append(f"{p} -> {dest}")
    return moved


# ------------------------------------------------------------------------------------------
# environment of the measurement processes
# ------------------------------------------------------------------------------------------
def base_env(args):
    """The caller's environment without any RVFL_* setting or preload it may carry, so that
    each step sees exactly the settings written here."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RVFL_") and k != "LD_PRELOAD"}
    env["PYTHONUNBUFFERED"] = "1"
    env["LD_PRELOAD"] = args.jemalloc
    return env


def rvfl_env(args, **extra):
    env = base_env(args)
    env.update(RVFL_BACKEND=BACKEND, RVFL_SECLEVEL=LEVEL)
    env.update({k: str(v) for k, v in extra.items()})
    return env


# ------------------------------------------------------------------------------------------
# what a step leaves behind
# ------------------------------------------------------------------------------------------
def status_ok(combo_dir):
    path = Path(combo_dir) / "terminal_log.txt"
    if not path.exists():
        return False
    return any(line.strip() == "STATUS: ok" for line in path.read_text(encoding="utf-8").splitlines())


def last_checkpoint(combo_dir):
    ck = Path(combo_dir) / "checkpoints"
    its = [int(m.group(1)) for f in ck.glob("*_Iter*.pt") if (m := re.search(r"_Iter(\d+)\.pt$", f.name))]
    return max(its) if its else 0


def e2e_root():
    return Path("e2e_results") / CONFIG


def old_cost_outputs(n, only=None, besides=()):
    """The ec256 cost-sweep outputs of N = n on disk: every k1 (only None) or the k1 in only,
    leaving out the k1 in besides."""
    def k_of(p):
        m = re.search(r"(?:RVFL_noattack_k|N\d+_k)(\d+)_" + re.escape(CONFIG), p.name)
        return int(m.group(1)) if m else None
    base = Path(f"runs/N{n}")
    out = sorted(base.glob(f"logs/RVFL/RVFL_noattack_k*_{CONFIG}_N{n}_*_log.jsonl"))
    out += sorted(p for p in base.glob(f"checkpoints/RVFL_noattack_k*_{CONFIG}") if p.is_dir())
    out += sorted(Path("logs/cost_sweep").glob(f"N{n}_k*_{CONFIG}.out"))
    return [p for p in out if (only is None or k_of(p) in only) and k_of(p) not in besides]


def brea_log_paths(n):
    base = Path(f"runs/N{n}/logs/BREA")
    return [base / f"{BREA_TAG}_N{n}_{kind}_log.jsonl"
            for kind in ("processing_time", "communication_cost", "values")]


def brea_outputs(n):
    return brea_log_paths(n) + [Path(f"runs/N{n}/checkpoints/{BREA_TAG}")]


def brea_logged_rounds(n):
    """L such that rounds 1..L of the BREA run at N = n are logged in full (every party, in both
    the time and the communication log)."""
    import cost_sweep
    parties = set(range(n)) | {-1}

    def full(path):
        by = {}
        for rec in cost_sweep.read_records(path):
            by.setdefault(rec.get("iteration"), set()).add(rec.get("party_id"))
        return {it for it, ps in by.items() if parties <= ps}
    tl, cl, _ = brea_log_paths(n)
    done = full(tl) & full(cl)
    last = 0
    while last + 1 in done:
        last += 1
    return last


def brea_resume_point(n, step, stamp):
    """Before a BREA run: main.py resumes from its last checkpoint (RVFL_RESUME=1) and appends to
    its logs, so the checkpoints after the last fully logged round L and the log records after L
    (a round cut off part-way) are moved into _set_aside/<stamp>/<step>_resume/; the kept records
    are written back. Returns L."""
    import cost_sweep
    last = brea_logged_rounds(n)
    ck_dir = Path(f"runs/N{n}/checkpoints/{BREA_TAG}")
    later = [f for f in ck_dir.glob("*_Iter*.pt")
             if (m := re.search(r"_Iter(\d+)\.pt$", f.name)) and int(m.group(1)) > last]
    keep = {}
    for path in brea_log_paths(n):
        recs = cost_sweep.read_records(path)
        if not recs:
            continue
        kept = [r for r in recs if int(r.get("iteration", 0)) <= last]   # file order kept
        if len(kept) != len(recs):
            keep[path] = kept
    set_aside(f"{step}_resume", later + list(keep), stamp)
    for path, kept in keep.items():
        if kept:
            path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    return last


def old_e2e_outputs(n, only=None, besides=()):
    """The filter-off folders e2e_results/ec256/N<n>_k<k1>/ (not the filter runs): every k1
    (only None) or the k1 in only, leaving out the k1 in besides."""
    pat = re.compile(rf"^N{n}_k(\d+)$")
    out = []
    for p in sorted(e2e_root().glob(f"N{n}_k*")):
        m = pat.match(p.name)
        if p.is_dir() and m and (only is None or int(m.group(1)) in only) \
                and int(m.group(1)) not in besides:
            out.append(p)
    return out


class Step:
    def __init__(self, name, describe, commands, outputs, check, needs=(), prepare=None,
                 optional=False):
        self.name = name
        self.describe = describe
        self.commands = commands      # callable -> list of (argv, env, cwd)
        self.outputs = outputs        # callable -> paths set aside before the first run
        self.check = check            # callable -> (ok, message)
        self.needs = needs            # steps that must be done before this one runs
        self.prepare = prepare        # callable(stamp) -> False when nothing has to run
        self.optional = optional      # a failure is not retried on the next invocation


def build_steps(args):
    import cost_sweep
    py = sys.executable
    steps = []

    def cost_step(name, n, ks, extra_outputs=(), optional=False):
        def cs_cmd():
            argv = [py, "cost_sweep.py", "--n", str(n), "--k1", *map(str, ks),
                    "--backend", BACKEND, "--level", LEVEL, "--iters", str(ROUNDS),
                    "--model-size", str(MODEL_SIZE),
                    # its own CSV holds every round; the tables step writes the one to use
                    "--csv", str(STATE_DIR / f"{name}_all_rounds.csv")]
            return [(argv, rvfl_env(args), None)]

        def cs_out():
            # the optional step moves only its own k1; the others every k1 of their N except the
            # optional one's, which that step moves itself when it first starts
            if optional:
                return old_cost_outputs(n, only=ks)
            return old_cost_outputs(n, besides=[k for m, k in COST_LAST if m == n]) + list(extra_outputs)

        def cs_check():
            missing = []
            for k in ks:
                tl, cl = cost_sweep.log_paths(n, k, BACKEND, LEVEL)
                if not (cost_sweep.is_complete(tl, n, ROUNDS) and cost_sweep.is_complete(cl, n, ROUNDS)):
                    missing.append(k)
            return (not missing, f"k1 {ks} all at {ROUNDS} rounds" if not missing
                    else f"k1 {missing} incomplete")

        gb = max(cost_sweep.est_memory_gb(n, k, BACKEND, LEVEL, MODEL_SIZE) for k in ks)
        return Step(name, f"cost sweep N={n} k1={ks}, {ROUNDS} rounds (peak estimate {gb:.0f} GB)"
                    + (", optional" if optional else ""), cs_cmd, cs_out, cs_check, optional=optional)

    first_n = min(COST_GRID)
    for n, ks in sorted(COST_GRID.items()):
        steps.append(cost_step(f"cost_N{n}", n, ks,
                               extra_outputs=[Path(f"cost_sweep_{CONFIG}.csv")] if n == first_n else ()))

    links_env = ",".join(f"{bw}:{rtt}" for bw, rtt in LINKS)

    def e2e_step(name, n, ks, outputs):
        def e_cmd():
            env = rvfl_env(args, RVFL_E2E=1, RVFL_N=n, RVFL_E2E_K1=",".join(map(str, ks)),
                           RVFL_E2E_LINKS=links_env, RVFL_E2E_ROUNDS=ROUNDS,
                           RVFL_E2E_SAVE_CKPT=1, RVFL_E2E_RESUME=1)
            return [([py, "main.py"], env, None)]

        def e_check():
            bad = [f"k1={k} {link_label(bw, rtt)}" for k in ks for bw, rtt in LINKS
                   if not (status_ok(e2e_root() / f"N{n}_k{k}" / link_label(bw, rtt))
                           and last_checkpoint(e2e_root() / f"N{n}_k{k}" / link_label(bw, rtt)) >= ROUNDS)]
            total = len(ks) * len(LINKS)
            return (not bad, f"all {total} combinations at {ROUNDS} rounds" if not bad else
                    f"{len(bad)} of {total} not done: " + ", ".join(bad[:4]) + (" ..." if len(bad) > 4 else ""))

        return Step(name, f"netns e2e N={n} k1={ks}, {len(LINKS)} links, {ROUNDS} rounds",
                    e_cmd, outputs, e_check)

    e2e_last = _rows(E2E_LAST)
    for n, ks in sorted(E2E_GRID.items()):
        # the k1 of E2E_LAST are left to their own step, which moves them itself
        steps.append(e2e_step(f"e2e_N{n}", n, ks,
                              lambda n=n: old_e2e_outputs(n, besides=e2e_last.get(n, ()))))

    required = tuple(s.name for s in steps)
    for n, ks in sorted(_rows(COST_LAST).items()):
        steps.append(cost_step(f"cost_N{n}_k{'_'.join(map(str, ks))}", n, ks, optional=True))

    csv_path = Path(f"cost_sweep_{CONFIG}_drop{DROP_FIRST}.csv")
    brea_csv = Path(f"brea_{CONFIG}_drop{DROP_FIRST}.csv")
    md_path = Path(f"remeasure_{CONFIG}.md")

    def t_cmd():
        return [([py, "remeasure_tables.py", "--backend", BACKEND, "--level", LEVEL,
                  "--drop-first", str(DROP_FIRST), "--csv", str(csv_path), "--brea-csv", str(brea_csv),
                  "--out", str(md_path)], base_env(args), None)]

    def t_check():
        ok = csv_path.exists() and md_path.exists()
        return (ok, f"{csv_path} and {md_path} written" if ok else f"{csv_path} or {md_path} missing")

    # the tables of ours first, so they do not wait for BREA; written again once BREA is done
    steps.append(Step("tables", f"{csv_path} and {md_path}, rounds after the first {DROP_FIRST}",
                      t_cmd, lambda: [csv_path, brea_csv, md_path], t_check, needs=required))

    brea_steps = []
    for n in BREA_N:
        name = f"brea_N{n}"

        def b_cmd(n=n):
            env = rvfl_env(args, RVFL_MODE="BREA", RVFL_ATTACK="none", RVFL_RESUME=1, RVFL_E2E=0,
                           RVFL_N=n, RVFL_K1=1, RVFL_ITERS=brea_rounds(n), RVFL_RUN_SUFFIX=f"_{CONFIG}")
            return [([py, "main.py"], env, None)]

        def b_prepare(stamp, n=n, name=name):
            return brea_resume_point(n, name, stamp) < brea_rounds(n)

        def b_check(n=n):
            last = brea_logged_rounds(n)
            return (last >= brea_rounds(n), f"rounds 1..{last} of {brea_rounds(n)} logged")

        steps.append(Step(name, f"BREA cost N={n} k1=1, {brea_rounds(n)} round(s) -> runs/N{n}/logs/BREA/{BREA_TAG}_N{n}_*",
                          b_cmd, lambda n=n: brea_outputs(n), b_check, prepare=b_prepare))
        brea_steps.append(name)

    steps.append(Step("tables_final", f"the tables again, with BREA (table 6)", t_cmd,
                      lambda: [], t_check, needs=required + tuple(brea_steps)))

    # E2E_LAST after BREA. Not optional: like the other netns steps, one that fails resumes on
    # the next invocation (a checkpoint a round). It holds nothing up: only tables_last follows,
    # and that runs whether it ended well or not (the tables list a run that did not).
    for n, ks in sorted(e2e_last.items()):
        steps.append(e2e_step(e2e_step_name(n, ks[0]), n, ks, lambda n=n, ks=ks: old_e2e_outputs(n, only=ks)))
    if e2e_last:
        steps.append(Step("tables_last", "the tables again, with the netns runs of E2E_LAST", t_cmd,
                          lambda: [], t_check, needs=required + tuple(brea_steps)))
    return steps


# ------------------------------------------------------------------------------------------
# running
# ------------------------------------------------------------------------------------------
def other_runs():
    """Anything else measuring on this machine: a netns sweep holding processes in its
    namespaces, or a cost sweep. Namespaces left behind by a run that died hold no process and
    do not count: the next setup clears them. A runner that is only waiting (--after-pid) does
    not count either, so two runners can be chained on each other's pid."""
    busy = []
    try:
        listing = subprocess.run(["ip", "netns", "list"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        listing = ""
    for line in listing.splitlines():
        ns = line.split()[0] if line.strip() else ""
        if not ns.startswith("rvfl"):
            continue
        pids = subprocess.run(["ip", "netns", "pids", ns], capture_output=True, text=True).stdout.split()
        if pids:
            busy.append(f"namespace {ns} holds processes {' '.join(pids)}")
    out = subprocess.run(["pgrep", "-af", r"cost_sweep\.py"], capture_output=True, text=True).stdout
    busy += [f"process {line}" for line in out.splitlines() if line.strip()]
    return busy


def record_environment(args):
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    def read(path):
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return "?"
    lines = [f"[{now()}] remeasure started",
             f"  numa_balancing = {read('/proc/sys/kernel/numa_balancing')}",
             f"  LD_PRELOAD     = {args.jemalloc}",
             f"  python         = {sys.executable}",
             f"  cpus           = {os.cpu_count()}"]
    with (STATE_DIR / "environment.txt").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def check_jemalloc(path):
    if not Path(path).exists():
        sys.exit(f"jemalloc not found at {path}; install it (apt install libjemalloc2) or pass --jemalloc")
    probe = subprocess.run([sys.executable, "-c", "pass"], env=dict(os.environ, LD_PRELOAD=path),
                           capture_output=True, text=True)
    if "cannot be preloaded" in probe.stderr:
        sys.exit(f"LD_PRELOAD={path} is rejected by the loader:\n{probe.stderr}")


def wait_for_pids(pids):
    """Block until none of pids is alive any more, checking once a minute."""
    def alive(pid):
        return Path(f"/proc/{pid}").exists()
    waiting = [p for p in pids if alive(p)]
    if waiting:
        print(f"[remeasure] {now()} waiting for pid {' '.join(map(str, waiting))} to end", flush=True)
    while any(alive(p) for p in pids):
        time.sleep(60)
    if waiting:
        print(f"[remeasure] {now()} pid {' '.join(map(str, waiting))} ended; starting", flush=True)


def run_step(step, state):
    commands = step.commands()
    entry = state.get(step.name, {})
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not entry.get("set_aside_done"):
        moved = set_aside(step.name, step.outputs(), stamp)
        entry.update(set_aside_done=True, set_aside=moved)
    entry.update(status="running", started=now())
    state[step.name] = entry
    save_state(state)

    log_path = STATE_DIR / f"{step.name}.log"
    t0 = time.time()
    rc = 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {now()} {step.name}: {step.describe}\n")
        if step.prepare is not None and not step.prepare(stamp):
            log.write("already complete, nothing to run\n")
            commands = []
        for argv, env, cwd in commands:
            log.write(f"$ {' '.join(shlex.quote(a) for a in argv)}\n")
            log.flush()
            rc = subprocess.run(argv, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode
            log.write(f"(exit {rc})\n")
            log.flush()
            if rc != 0:
                break
    ok, message = step.check()
    entry.update(status="done" if (ok and rc == 0) else "failed", finished=now(),
                 elapsed=fmt_elapsed(time.time() - t0), message=message, exit_code=rc)
    state[step.name] = entry
    save_state(state)
    return entry["status"] == "done", message, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jemalloc", default="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2")
    ap.add_argument("--after-pid", type=int, nargs="+", default=[],
                    help="wait until these processes have ended before starting, checking once a minute")
    ap.add_argument("--steps", nargs="+", default=None, help="run only these steps (in queue order)")
    ap.add_argument("--redo", nargs="+", default=[], help="forget these steps' records and run them again")
    ap.add_argument("--status", action="store_true", help="print the state of every step and stop")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the commands and stop")
    args = ap.parse_args()

    steps = build_steps(args)
    names = [s.name for s in steps]
    for name in (args.steps or []) + args.redo:
        if name not in names:
            sys.exit(f"unknown step {name}; steps are: {' '.join(names)}")
    state = load_state()

    if args.status or args.dry_run:
        for s in steps:
            e = state.get(s.name, {})
            line = f"  {s.name:15s} {e.get('status', 'pending'):8s} {s.describe}"
            if e.get("elapsed"):
                line += f"  [{e['elapsed']}; {e.get('message', '')}]"
            print(line)
        if args.dry_run:
            print("\ncommands:")
            for s in steps:
                if args.steps and s.name not in args.steps:
                    continue
                for argv, env, cwd in s.commands():
                    extra = {k: env[k] for k in sorted(env) if k.startswith("RVFL_") or k == "LD_PRELOAD"}
                    print(f"  [{s.name}] {' '.join(f'{k}={v}' for k, v in extra.items())} "
                          f"{' '.join(shlex.quote(a) for a in argv)}")
            print("\nset aside before the first run of each step (what is on disk now):")
            for s in steps:
                if args.steps and s.name not in args.steps:
                    continue
                if state.get(s.name, {}).get("set_aside_done"):
                    continue
                outs = [str(p) for p in s.outputs() if Path(p).exists()]
                print(f"  [{s.name}] " + (", ".join(outs) if outs else "(nothing)"))
        return

    if os.geteuid() != 0:
        sys.exit("run under sudo: the netns steps create network namespaces")
    if args.after_pid:
        wait_for_pids(args.after_pid)
    check_jemalloc(args.jemalloc)
    busy = other_runs()
    if busy:
        sys.exit("another run is measuring on this machine; wait for it or stop it first:\n  " + "\n  ".join(busy))

    for name in args.redo:
        state.pop(name, None)
    save_state(state)
    record_environment(args)

    failed, optional_failed = [], []
    for s in steps:
        if args.steps and s.name not in args.steps:
            continue
        prev = state.get(s.name, {}).get("status")
        if prev == "done":
            print(f"[remeasure] {s.name}: done earlier, skipping")
            continue
        if s.optional and prev == "failed":
            print(f"[remeasure] {s.name}: optional, failed earlier; not tried again (--redo {s.name} to try)")
            continue
        not_done = [n for n in s.needs if state.get(n, {}).get("status") != "done"]
        if not_done:
            print(f"[remeasure] {s.name}: waiting for {', '.join(not_done)}; skipped this time")
            continue
        print(f"[remeasure] {now()} {s.name}: {s.describe}  (log: {STATE_DIR / (s.name + '.log')})", flush=True)
        ok, message, elapsed = run_step(s, state)
        print(f"[remeasure] {now()} {s.name}: {'done' if ok else 'FAILED'} in {fmt_elapsed(elapsed)} -- {message}"
              + (" (optional)" if (s.optional and not ok) else ""), flush=True)
        if not ok:
            (optional_failed if s.optional else failed).append(s.name)

    tail = f"; failed: {' '.join(failed)} (run again to retry)" if failed else "; every required step done"
    if optional_failed:
        tail += f"; optional step failed: {' '.join(optional_failed)}"
    print(f"\n[remeasure] finished at {now()}{tail}")


if __name__ == "__main__":
    main()