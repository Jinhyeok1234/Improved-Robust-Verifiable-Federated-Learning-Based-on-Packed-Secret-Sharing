#!/usr/bin/env python3
"""
Attack-experiment sweep driver (parallel, train/eval decoupled).

Each (MODE, ATTACK, malicious-rate) combo in SWEEP is one INDEPENDENT full training
(a fresh `python main.py` process). Because a single run underutilizes a GPU (tiny
per-client shards), the driver runs several combos CONCURRENTLY, spread across GPUS.

    python run_attack_sweep.py            # TRAIN every combo (one at a time; no eval)
    python run_attack_sweep.py --eval     # EVALUATE present checkpoints -> logs/eval/<TAG>.jsonl
                                          #   (incremental + content-based: evaluates whatever
                                          #   checkpoints exist now and MERGES into the one sorted
                                          #   file; safe to re-run anytime, fills in new checkpoints)

Training and evaluation are separate passes: train with no flag, evaluate with --eval. --eval is
incremental -- it evaluates whatever checkpoints exist and merges into the one canonical curve, so
it is safe to run mid-training (it just picks up more on the next run). Both passes skip work
already done (content-based / resumable); each combo writes its own folders (checkpoints/<TAG>/,
logs/...), so nothing collides.

main.py reads MODE / ATTACK / MALICIOUS_RATE / rounds from the env vars this driver sets
(RVFL_MODE / RVFL_ATTACK / RVFL_MRATE / RVFL_ITERS). Set NUM_ITERS=10 here for a quick
smoke test (no need to touch main.py). Edit SWEEP to change the matrix.

MONITORING: each child's stdout/stderr is redirected to its own file under
logs/sweep/<TAG>.out (so the terminal is NOT flooded with interleaved output). The
driver reads the current round from each log and prints ONE self-updating line per job:
    <TAG>   GPU <g>   <state>   <cur>/<total>
Open logs/sweep/<TAG>.out to see a single job's full output.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import eval_io

# ---- experiment matrix: (MODE, ATTACK, malicious_rate). EDIT ME. --------------------
LF    = "label_flip_fltrust"    # data-poisoning (dataloader)
TRIM  = "trim_fltrust"          # model-poisoning (Fang et al. Trim attack; attacks.py)
ADAPT = "adaptive_fltrust"      # model-poisoning (FLTrust paper's adaptive attack, Alg. 3)
KRUM  = "krum_fltrust"          # model-poisoning (Fang et al. Krum attack; -lambda*s, Krum-selected)
SCALE = "scaling_fltrust"       # TARGETED backdoor (trigger+relabel) + lambda=n scaling
OURS  = "FLTrust_quantized_perp"     # RVFL's exact aggregation, plaintext over Z_q
FT    = "FLTrust"                    # plaintext FLTrust baseline
# === Camera-ready sweep (post-discussion): finish EVERY attack x fraction. ============================
# Shared server -> MAX_PARALLEL=1 (one combo at a time). EVERY combo is PAIRED per fraction:
# (Ours,f) then (FT,f), so each fraction's Ours-vs-FLTrust comparison lands together (fraction outer,
# protocol inner) -- the same pairing we used for adaptive in the discussion. FLTrust is the essential
# same-setting baseline (absolute accuracy is read relative to FLTrust, not the paper's numbers).
#
# SWEEP lists the FULL schedule (no-attack + every attack x every fraction) so the whole plan stays
# on record. The driver's resume SKIPS whatever already finished (and re-runs anything not actually
# complete), so keeping the done combos in the list costs nothing. Run order is priority-first: the
# discussion-period block below (no-attack; LF & Trim {20,80,90,95}; Adaptive {20,40}) is already
# trained and skipped instantly, so the first ACTUAL training is Adaptive 60%.
#   (1) finish Adaptive's breaking curve up to 95%   -> {60,80,90,95}
#   (2) Krum & Scaling at the 20% operating point    -> {20}
#   (3) Krum & Scaling around their collapse region  -> {80,90,95}
#       (Krum breaks ~90% [FLTrust Fig 6, MNIST-0.5]; Scaling ASR stays ~0, so {80,90,95} probes the top)
#   (4) fill EVERY remaining fraction of {10,20,40,60,80,90,95}, per attack, ordered
#       LF, Trim, Adaptive, Krum, Scaling.
# End state: {LF, Trim, Adaptive, Krum, Scaling} x {10,20,40,60,80,90,95}, both protocols.


def _pairs(attack, fracs, protos=(OURS, FT)):
    """One attack over `fracs`, PAIRED by fraction: (Ours,f) then (FT,f) for each f in order --
    so each fraction's Ours-vs-FLTrust comparison finishes together (fraction outer, protocol inner)."""
    return [(p, attack, f) for f in fracs for p in protos]


SWEEP = (
    [(OURS, None, 0.0), (FT, None, 0.0)]          # no-attack references (0%, shared baseline)
    # --- (0) discussion-period block: already trained; resume skips it (re-runs any that isn't) ---
    + _pairs(LF,    [0.20, 0.80, 0.90, 0.95])     # Exp-A 20% + LF breaking neighborhood
    + _pairs(TRIM,  [0.20, 0.80, 0.90, 0.95])     # Exp-A 20% + Trim breaking neighborhood
    + _pairs(ADAPT, [0.20, 0.40])                 # adaptive, discussion scope (<= N/3)
    # --- remaining work, priority-first ---
    # (1) finish adaptive's breaking curve up to 95%
    + _pairs(ADAPT, [0.60, 0.80, 0.90, 0.95])
    # (2) Krum & Scaling at the 20% operating point
    + _pairs(KRUM,  [0.20])
    + _pairs(SCALE, [0.20])
    # (3) Krum & Scaling around their collapse region
    + _pairs(KRUM,  [0.80, 0.90, 0.95])
    + _pairs(SCALE, [0.80, 0.90, 0.95])
    # (4) fill every remaining fraction of {10,20,40,60,80,90,95}, per attack
    + _pairs(LF,    [0.10, 0.40, 0.60])
    + _pairs(TRIM,  [0.10, 0.40, 0.60])
    + _pairs(ADAPT, [0.10])
    + _pairs(KRUM,  [0.10, 0.40, 0.60])
    + _pairs(SCALE, [0.10, 0.40, 0.60])
)

# ---- run params (must match main.py where noted) -----------------------------------
DATASET, MODEL, N = "CIFAR10", "ResNet20", 30   # <-- EDIT N HERE ONLY. Single source of N: it is
                                                 #     passed to main.py via RVFL_N and picks the
                                                 #     per-N storage root below, so every path follows.
NUM_ITERS    = 500          # global rounds; passed to main.py via RVFL_ITERS (set 10 to smoke-test)
EVAL_STRIDE  = 1            # evaluate every k-th checkpoint
GPUS         = [0, 1]        # GPU ids to spread runs over
MAX_PARALLEL = 8           # SHARED SERVER: one combo at a time (do NOT raise on a shared box)
# Per-N storage: every result path hangs off ./runs/N{N}/, so changing N above re-roots checkpoints,
# eval curves, and sweep logs automatically -- N=30 and N=100 never share a file. extend_sweep imports
# these three from here, so it follows too; main.py writes checkpoints to the SAME ./runs/N{N}/
# checkpoints because it now reads N from RVFL_N (which this driver sets).
RUN_ROOT      = Path(f"./runs/N{N}")
CKPT_ROOT     = RUN_ROOT / "checkpoints"
EVAL_DIR      = RUN_ROOT / "logs" / "eval"
SWEEP_LOG_DIR = RUN_ROOT / "logs" / "sweep"   # per-job captured stdout/stderr lives here
REFRESH_S = 1.0              # monitor redraw interval

# "=== Iteration 12 / 1000 Start ===" -> current round / total, from a job's captured log
_ITER_RE = re.compile(r"Iteration\s+(\d+)\s*/\s*(\d+)\s+Start")
_TAIL_BYTES = 32768          # how much of each log's tail to scan for the latest round


def run_tag(mode, attack, rate):
    tag = f"{mode}_{attack if attack else 'noattack'}"
    if attack:
        tag += f"_m{int(round(rate * 100))}"
    return tag


def _train_job(mode, attack, rate, gpu):
    env = dict(os.environ, RVFL_MODE=mode,
               RVFL_ATTACK=(attack if attack else "none"),
               RVFL_MRATE=str(rate), RVFL_ITERS=str(NUM_ITERS),
               RVFL_N=str(N),          # single-source N -> main.py (keeps train N == sweep/eval N)
               CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONUNBUFFERED="1")   # unbuffered so the log shows the current round live
    return [sys.executable, "main.py"], env


SCALING_TARGET = 2   # CIFAR-10 backdoor target = "bird" (must match main.py scaling setup)


def _eval_job(mode, attack, rate, gpu):
    # Untargeted attacks (None/LF/Trim/Adaptive/Krum) -> test error via eval_checkpoints.py.
    # Scaling is TARGETED -> Attack Success Rate via eval_backdoor.py (a DIFFERENT metric, so its
    # _asr file legitimately lives on its own). Each writes the ONE canonical file for its metric
    # and MERGES into it (incremental + sorted); there is no separate peek file.
    tag = run_tag(mode, attack, rate)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    if attack == SCALE:
        cmd = [sys.executable, "eval_backdoor.py",
               "--dataset", DATASET, "--model", MODEL, "--N", str(N),
               "--ckpt_dir", str(CKPT_ROOT / tag), "--out", str(EVAL_DIR / f"{tag}_asr.jsonl"),
               "--stride", str(EVAL_STRIDE), "--target_label", str(SCALING_TARGET)]
    else:
        cmd = [sys.executable, "eval_checkpoints.py",
               "--dataset", DATASET, "--model", MODEL, "--N", str(N),
               "--ckpt_dir", str(CKPT_ROOT / tag), "--out", str(EVAL_DIR / f"{tag}.jsonl"),
               "--stride", str(EVAL_STRIDE), "--eval_train"]
    return cmd, env


def train_done(tag):
    return (CKPT_ROOT / tag / f"{DATASET}_{MODEL}_N{N}_Iter{NUM_ITERS}.pt").exists()


def has_ckpts(tag):
    d = CKPT_ROOT / tag
    return d.is_dir() and any(d.glob(f"{DATASET}_{MODEL}_N{N}_Iter*.pt"))


def _ckpt_iters(tag):
    """Sorted checkpoint iterations present for a tag."""
    d = CKPT_ROOT / tag
    if not d.is_dir():
        return []
    pat = re.compile(rf"^{re.escape(DATASET)}_{re.escape(MODEL)}_N{N}_Iter(\d+)\.pt$")
    return sorted(int(m.group(1)) for f in d.iterdir() if (m := pat.match(f.name)))


def eval_done(tag, attack=None):
    # CONTENT-BASED: done iff every checkpoint --eval would target (present checkpoints at
    # EVAL_STRIDE) is already recorded in the ONE canonical file -- never "the file exists".
    # A leftover legacy peek file forces one more run so it gets folded in first.
    out = EVAL_DIR / (f"{tag}_asr.jsonl" if attack == SCALE else f"{tag}.jsonl")
    if eval_io.peek_path(out).exists():
        return False
    targets = _ckpt_iters(tag)[::EVAL_STRIDE]
    if not targets:
        return False
    have = eval_io.evaluated_iters(out)
    return all(i in have for i in targets)


def _read_progress(log_path):
    """Latest (cur, total) round from a job's captured log, or None if not started yet."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            data = f.read().decode("utf-8", "replace")
    except (FileNotFoundError, OSError):
        return None
    matches = _ITER_RE.findall(data)
    if not matches:
        return None
    cur, total = matches[-1]
    return int(cur), int(total)


# ---- display: show the SWEEP triple (protocol / attack / fraction), not the file tag ----
_ATTACK_LABEL = {
    None:                  "no-attack",
    LF:                    "LF",
    TRIM:                  "Trim",
    ADAPT:                 "Adaptive",
    KRUM:                  "Krum",
    SCALE:                 "Scaling",
}
_PROTO_W, _ATK_W, _FRAC_W = 7, 9, 4   # column widths ("FLTrust" / "no-attack" / "100%")


def _proto_label(mode):
    if mode == FT:
        return "FLTrust"
    if mode == OURS:
        return "Ours"
    return mode


def _frac_label(attack, rate):
    return "-" if attack is None else f"{int(round(rate * 100))}%"


def _triple(job):
    """'Ours     LF         20%' -- the (protocol, attack, fraction) columns."""
    return (f"{_proto_label(job.mode):<{_PROTO_W}}  "
            f"{_ATTACK_LABEL.get(job.attack, job.attack):<{_ATK_W}}  "
            f"{_frac_label(job.attack, job.rate):>{_FRAC_W}}")


class Job:
    __slots__ = ("tag", "mode", "attack", "rate", "cmd", "env",
                 "gpu", "log_path", "fh", "proc", "state", "rc")

    def __init__(self, tag, mode, attack, rate, cmd, env, gpu, log_path):
        self.tag = tag
        self.mode = mode
        self.attack = attack
        self.rate = rate
        self.cmd = cmd
        self.env = env
        self.gpu = gpu
        self.log_path = log_path
        self.fh = None
        self.proc = None
        self.state = "waiting"   # waiting -> running -> done/failed
        self.rc = None

    def start(self):
        self.fh = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd, env=self.env,
                                     stdout=self.fh, stderr=subprocess.STDOUT)
        self.state = "running"

    def finish(self, rc):
        self.rc = rc
        self.state = "done" if rc == 0 else "failed"
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def _running_row(job):
    prog = _read_progress(job.log_path)
    detail = f"{prog[0]}/{prog[1]}" if prog else "setting up..."
    gpu = "-" if job.gpu is None else str(job.gpu)
    return f"  {_triple(job)}  GPU {gpu:>2}  {detail}"


def _finish_line(job):
    prog = _read_progress(job.log_path)
    if job.rc == 0:
        detail = f"{prog[0]}/{prog[1]}" if prog else "OK"
        return f"[done]   {_triple(job)}  {detail}"
    detail = f"stopped at {prog[0]}/{prog[1]}" if prog else ""
    return f"[FAILED] {_triple(job)}  exit {job.rc}  {detail}"


def run_pool(jobs):
    """jobs: list[Job]. Keep at most MAX_PARALLEL running.

    Only RUNNING jobs occupy the live, self-updating block (<= MAX_PARALLEL+1 lines, so it
    always fits the terminal -- the earlier full-table redraw broke when 26 rows were taller
    than the window). Each finished job is flushed once as a permanent line ABOVE that block.
    """
    SWEEP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_tty = sys.stdout.isatty()
    total = len(jobs)
    col_head = (f"  {'PROTOCOL':<{_PROTO_W}}  {'ATTACK':<{_ATK_W}}  "
                f"{'FRAC':>{_FRAC_W}}  GPU  ROUND")
    pending, running = list(jobs), []
    done_ct = 0
    live_lines = 0   # lines the live block currently occupies (0 on non-tty / nothing drawn)

    def clear_live():
        nonlocal live_lines
        if is_tty and live_lines:
            sys.stdout.write(f"\033[{live_lines}F\033[J")   # up to block top, clear downward
            live_lines = 0

    def draw_live():
        nonlocal live_lines
        header = f"[monitor] {len(running)} running | {done_ct}/{total} finished"
        rows = [header, col_head] + [_running_row(j) for j in running]
        if is_tty:
            sys.stdout.write("\n".join(rows) + "\n")
            live_lines = len(rows)
        else:
            sys.stdout.write(header + "\n")
        sys.stdout.flush()

    while pending or running:
        clear_live()                       # erase live block before printing permanent lines
        while pending and len(running) < MAX_PARALLEL:
            job = pending.pop(0)
            job.start()
            running.append(job)
        for job in running[:]:
            rc = job.proc.poll()
            if rc is not None:
                job.finish(rc)
                running.remove(job)
                done_ct += 1
                sys.stdout.write(_finish_line(job) + "\n")   # permanent, scrolls up
        draw_live()
        if pending or running:
            time.sleep(REFRESH_S)


def main(argv):
    eval_mode = "--eval" in argv                  # evaluate present checkpoints (incremental, merges)
    jobs, gpu_i = [], 0
    for (mode, attack, rate) in SWEEP:
        tag = run_tag(mode, attack, rate)
        gpu = GPUS[gpu_i % len(GPUS)]
        gpu_i += 1
        if eval_mode:
            # Content-based, incremental: evaluate whatever checkpoints exist NOW (partial or final)
            # and merge into the one canonical file; re-running later fills in new checkpoints. This
            # is what the old --peek did, minus the throwaway file -- so there is no separate pass.
            if not has_ckpts(tag):
                print(f"[skip-eval] {tag} (no checkpoints yet)", flush=True)
                continue
            if eval_done(tag, attack):
                print(f"[skip-eval] {tag} (all present checkpoints evaluated)", flush=True)
                continue
            cmd, env = _eval_job(mode, attack, rate, gpu)
        else:
            if train_done(tag):
                print(f"[skip-train] {tag}", flush=True)
                continue
            cmd, env = _train_job(mode, attack, rate, gpu)
        jobs.append(Job(tag, mode, attack, rate, cmd, env, gpu, SWEEP_LOG_DIR / f"{tag}.out"))
    mode_name = "EVAL" if eval_mode else "TRAIN"
    print(f"[sweep] {mode_name}: {len(jobs)} job(s), "
          f"{MAX_PARALLEL} at a time over GPUs {GPUS}", flush=True)
    print(f"[sweep] per-job full output -> {SWEEP_LOG_DIR}/<TAG>.out\n", flush=True)
    if jobs:
        run_pool(jobs)
    print("\n[sweep] done.")


if __name__ == "__main__":
    main(sys.argv[1:])