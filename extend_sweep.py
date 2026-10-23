#!/usr/bin/env python3
"""
Auto-extend phase (runs AFTER run_attack_sweep.py has trained every combo to its base target).

GOAL
    A run at N=100 has ~500 examples/client, so gradients are noisy and learning is slow: many
    settings finish their base iterations still short of the plateau (train acc 80-85%, i.e. test
    acc ~70-80% -- parameter-dependent). This driver gives every such setting +500 iterations at a
    time until it converges.

    STOP CONDITION IS ONE-SIDED: a run stops ONLY when it SUCCEEDS (its last-window accuracy reaches
    the plateau band). We do NOT try to detect collapse -- broken runs come in incompatible shapes
    (an early plateau at low accuracy, accuracy that regresses, large-amplitude oscillation), and
    telling those apart from "still slowly learning" is unreliable. So every non-converged run just
    keeps extending to --iter-cap; you truncate its curve to whatever iteration you want at OUTPUT
    time. The eval jsonl keeps the full history, so training past the interesting point loses
    nothing. (see classify())

ARCHITECTURE  --  PHASES, not a serial loop  (this is the whole point)
    A pass does four DISTINCT phases over the FULL canonical grid; it never interleaves "train
    this one, then eval this one, then train the next". Concretely, each pass:

      PHASE 1  EVAL  (pooled + monitored)  -- stride-1 eval for every tag whose checkpoints aren't
               fully evaluated yet, using the SAME either/or as the base sweep: accuracy
               (eval_checkpoints) for the untargeted attacks, ASR (eval_backdoor) for Scaling --
               never both, so an ASR-only Scaling run is never re-flagged for a missing accuracy
               file. Runs through run_attack_sweep's run_pool (obeys MAX_PARALLEL, same monitor).
      PHASE 2  PRUNE (fast, in-process)    -- coarsen every eval-complete tag to every-Kth + final
               (reuses prune_checkpoints' OWN safety gate: never prunes an un-evaluated range,
               never drops the resume/final checkpoint).
      PHASE 3  SCAN  (read-only, instant)  -- classify every tag from its (now dense) curve and
               FILTER OUT ONLY THE EXTENSION TARGETS (not yet converged and under the iter cap).
               Prints the target list BEFORE any training starts.
      PHASE 4  TRAIN (pooled + monitored)  -- extend exactly those targets by +step, EACH via a
               fresh `python main.py` resume job handed to run_pool -- i.e. run_attack_sweep style,
               parallel-capable and monitorable. NOT a blocking inline subprocess.run loop.

    Then it passes again (round-robin from the top: no-attack, LF, Trim, Adaptive, Krum, Scaling,
    each fraction, paired (Ours,f)(FT,f)), because a tag extended this pass needs a fresh eval to
    know if it converged. It stops when a SCAN finds nothing to extend -- i.e. every setting has
    either converged (plateau) or hit --iter-cap. Non-converged runs therefore keep training to the
    cap; that cap is the total budget you set. The terminating pass's PHASE 1-2 leave every curve
    dense and every run coarsened.

MAX_PARALLEL / GPUS come from run_attack_sweep (shared server -> MAX_PARALLEL=1). DRY-RUN by
default: SCAN from whatever eval jsonls already exist and print the plan; nothing is evaluated,
pruned, or trained until --apply.

    python extend_sweep.py                 # dry run: SCAN existing curves, print the target list
    python extend_sweep.py --apply         # execute the 4 phases, round-robin to plateau
    python extend_sweep.py --apply --iter-cap 3000            # never extend any tag past 3000
    python extend_sweep.py --apply --never-extend FLTrust_label_flip_fltrust_m95   # force-skip
"""
import argparse
import json
import os
import sys
from pathlib import Path

import run_attack_sweep as R          # Job, run_pool, run_tag, constants, GPUS, SCALING_TARGET, ...
import eval_io
from prune_checkpoints import ckpt_iters, eval_iterations, eval_status, plan_tag, human

GRID = [0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95]   # full fraction sweep, ascending
ATTACKS_IN_ORDER = [R.LF, R.TRIM, R.ADAPT, R.KRUM, R.SCALE]


def canonical():
    """no-attack (Ours,FT), then each attack over every fraction, paired (Ours,f)(FT,f)."""
    combos = [(R.OURS, None, 0.0), (R.FT, None, 0.0)]
    for atk in ATTACKS_IN_ORDER:
        combos += R._pairs(atk, GRID)
    return combos


# ---- plateau (SUCCESS) decision -- the ONLY thing that stops a run early --------------------------
# We do NOT try to classify collapse. Collapsed runs come in incompatible shapes -- (i) an early
# plateau at low accuracy, (ii) accuracy that regresses, (iii) large-amplitude oscillation -- and
# telling "collapsed" from "still slowly learning" is unreliable. So the rule is one-sided: a run
# stops ONLY when it SUCCEEDS (reaches the plateau band); every non-converged run just keeps
# extending to --iter-cap, and the curve is truncated to whatever iteration you want at OUTPUT time
# (the eval jsonl keeps the full history, so nothing is lost by training past the interesting point).

def _window(curve, lo, hi, key):
    """mean, std, n of curve[key] over iterations in (lo, hi]."""
    vals = [rec[key] for it, rec in curve.items() if lo < it <= hi and key in rec]
    if not vals:
        return None, None, 0
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5, n


def classify(curve, reached, *, metric="train_acc", plateau_lo=0.80, window=100):
    """Return (status, detail). ONLY success stops a run:
        plateau   -- last-window mean accuracy >= plateau_lo   -> DONE (converged), stop extending
        climbing  -- anything below plateau                    -> EXTEND (to --iter-cap), no matter
                                                                  the collapse shape
        no_eval   -- the curve does not reach `reached` yet    -> needs a stride-1 eval first
    Scaling passes metric='clean_acc' (its clean test accuracy); untargeted attacks use train_acc.
    The test-side keys use a threshold 0.05 lower than the train-acc band."""
    key = metric if any(metric in r for r in curve.values()) else "test_acc"
    lo_thr = plateau_lo if key == "train_acc" else max(0.0, plateau_lo - 0.05)  # test band ~5% lower
    mean, std, n = _window(curve, reached - window, reached, key)
    if n == 0:
        return "no_eval", f"no {key} at Iter{reached-window+1}..{reached}"
    tag = f"{key}={mean:.3f}±{std:.3f}"
    if mean >= lo_thr:
        return "plateau", f"{tag} >= {lo_thr:.2f} (converged)"
    return "climbing", f"{tag} < {lo_thr:.2f} (not converged -- keep training)"


# ---- iteration bookkeeping ----------------------------------------------------------------------

def milestone(reached, base, step):
    """The 'step'-grid target this run is at or heading to: base, base+step, base+2step, ...
    reached exactly on the grid -> that milestone; between -> the next one up (interrupted run)."""
    if reached <= base:
        return base
    over = reached - base
    k = (over + step - 1) // step          # ceil(over/step)
    return base + k * step


def next_target(reached, base, step):
    """Iterations a CLIMBING run should train to: finish an interrupted milestone if `reached` is
    between grid points, else the next +step above a completed one."""
    m = milestone(reached, base, step)
    return m if reached < m else reached + step


def acc_file(tag, attack):
    """The eval file carrying this tag's ACCURACY signal for the plateau decision. Scaling is
    ASR-only -- its clean (test) accuracy lives in <tag>_asr.jsonl; every untargeted attack writes
    train/test accuracy to <tag>.jsonl. Same either/or as run_attack_sweep._eval_job (never both)."""
    return R.EVAL_DIR / (f"{tag}_asr.jsonl" if attack == R.SCALE else f"{tag}.jsonl")


def acc_metric(attack):
    """Plateau-decision key in that file: Scaling's ASR eval reports clean (test) accuracy as
    `clean_acc`; untargeted attacks report `train_acc` (classify falls back to test_acc)."""
    return "clean_acc" if attack == R.SCALE else "train_acc"


def load_curve(tag, attack):
    """{iter: record} from the tag's accuracy-signal file (acc_file). Empty if absent."""
    return eval_io.read_records(acc_file(tag, attack))


def present_iters(tag):
    """Sorted checkpoint iterations currently on disk for a tag ([] if none)."""
    d = R.CKPT_ROOT / tag
    return sorted(ckpt_iters(d, R.DATASET, R.MODEL, R.N)) if d.is_dir() else []


def needs_eval(tag, attack):
    """True iff some checkpoint on disk isn't yet in the tag's ONE eval file (content-based, stride
    independent): the ASR file <tag>_asr.jsonl for Scaling, the accuracy file <tag>.jsonl otherwise
    -- the same file that tag's eval actually writes. Checking the wrong file (e.g. the missing
    accuracy file for an ASR-only Scaling run) would flag an already-evaluated tag forever. After a
    prune the on-disk set is sparse and already-evaluated -> False (no re-eval)."""
    present = set(present_iters(tag))
    if not present:
        return False
    return not present <= eval_io.evaluated_iters(acc_file(tag, attack))


# ---- PHASE 1: stride-1 eval jobs (run through run_attack_sweep.run_pool, so pooled + monitored) --

def eval_jobs_for(tag, mode, attack, rate, gpu):
    """The stride-1 eval Job for this tag, run through run_attack_sweep.run_pool -- the SAME
    either/or as run_attack_sweep._eval_job, never both: Scaling (targeted) -> ASR via
    eval_backdoor.py -> <tag>_asr.jsonl; every untargeted attack -> accuracy via
    eval_checkpoints.py --eval_train -> <tag>.jsonl. Scaling gets no accuracy file; the untargeted
    attacks get no ASR file. Returns a one-element list (the pool takes a flat job list)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    if attack == R.SCALE:
        cmd = [sys.executable, "eval_backdoor.py",
               "--dataset", R.DATASET, "--model", R.MODEL, "--N", str(R.N),
               "--ckpt_dir", str(R.CKPT_ROOT / tag), "--out", str(R.EVAL_DIR / f"{tag}_asr.jsonl"),
               "--stride", "1", "--target_label", str(R.SCALING_TARGET)]
    else:
        cmd = [sys.executable, "eval_checkpoints.py",
               "--dataset", R.DATASET, "--model", R.MODEL, "--N", str(R.N),
               "--ckpt_dir", str(R.CKPT_ROOT / tag), "--out", str(R.EVAL_DIR / f"{tag}.jsonl"),
               "--stride", "1", "--eval_train"]
    return [R.Job(tag, mode, attack, rate, cmd, env, gpu, R.SWEEP_LOG_DIR / f"{tag}.eval.out")]


# ---- PHASE 2: prune one tag in-process, reusing prune_checkpoints' own safety gate --------------

def prune_one(tag, keep_every, min_coverage=0.9):
    """Coarsen a single tag to every-Kth + final, but ONLY if its stride-1 eval is complete & dense
    up to the latest checkpoint (prune_checkpoints.eval_status). Returns a one-line report."""
    d = R.CKPT_ROOT / tag
    imap = ckpt_iters(d, R.DATASET, R.MODEL, R.N) if d.is_dir() else {}
    if not imap:
        return None
    reached = max(imap)                                    # the resume/final marker for THIS tag
    ev_iters, _ = eval_iterations(R.EVAL_DIR, tag)
    ok, reason = eval_status(ev_iters, reached, min_coverage)
    if not ok:
        return f"  - {tag:<52} skip prune ({reason})"
    keep, del_paths, freed = plan_tag(imap, keep_every, reached)
    if not del_paths:
        return f"  = {tag:<52} already coarse ({len(keep)} kept)"
    for p in del_paths:
        try:
            p.unlink()
        except OSError as e:
            print(f"      ! could not delete {p.name}: {e}")
    return f"  * {tag:<52} kept {len(keep):>4}, deleted {len(del_paths):>4} = {human(freed)}"


# ---- PHASE 3: SCAN -- classify every tag, filter out the extension targets (read-only) ----------

def scan(combos, args, cls_kw, force, never):
    """Classify every trained tag. Returns (targets, rows):
        targets -- list of dicts {mode,attack,rate,tag,reached,target} for tags to EXTEND now
        rows    -- (tag, reached_or_None, mark, detail) for the full printed report, canonical order

    Only SUCCESS (plateau) stops a run; every non-converged tag keeps extending until --iter-cap.

    MILESTONE-SYNCHRONIZED: a pass advances toward ONE shared milestone
        M = next_target(min reached among active runs).
    Settings below M are extended TO M; settings already at/above M WAIT this pass. So no run races
    ahead to milestone M+step while others are still short of M -- which is exactly what happens if a
    previous pass was interrupted mid-grid (front of the grid at 1500, the rest still at 1000/1381):
    without this, the 1500 runs would jump to 2000 while the 1000/1381 runs are ignored. Here they
    all catch up to 1500 first; only once every active run reaches 1500 does M become 2000.
    """
    # 1) classify everything (read-only); remember reached + status per combo
    info = []
    for (mode, attack, rate) in combos:
        tag = R.run_tag(mode, attack, rate)
        present = present_iters(tag)
        if not present:
            info.append((mode, attack, rate, tag, None, "skip", "not trained yet"))
            continue
        reached = max(present)
        curve = load_curve(tag, attack)
        if tag in never:
            status, detail = "never", "forced skip"
        elif tag in force:
            status, detail = "climbing", "forced extend"
        else:
            metric = "clean_acc" if attack == R.SCALE else cls_kw["metric"]
            status, detail = classify(curve, reached, **{**cls_kw, "metric": metric})
        info.append((mode, attack, rate, tag, reached, status, detail))

    # 2) the shared milestone this pass drives toward = where the LEAST-advanced active run is headed
    active = [r for r in info if r[5] == "climbing" and r[4] < args.iter_cap]
    milestone_target = next_target(min(r[4] for r in active), args.base_iters, args.step) if active else None
    if active:
        n_ext = sum(1 for r in active if r[4] < milestone_target)
        print(f"  [milestone] pass target = Iter{milestone_target}: {n_ext} run(s) below it will extend, "
              f"{len(active) - n_ext} already there wait for the rest")

    # 3) only runs BELOW the shared milestone extend (to it); ones already there WAIT
    targets, rows = [], []
    for (mode, attack, rate, tag, reached, status, detail) in info:
        if reached is None:
            rows.append((tag, None, "skip", detail))
            continue
        if status == "climbing" and reached < args.iter_cap:
            if reached < milestone_target:
                rows.append((tag, reached, "EXTEND", detail + f" -> {milestone_target}"))
                targets.append(dict(mode=mode, attack=attack, rate=rate, tag=tag,
                                    reached=reached, target=milestone_target))
            else:
                rows.append((tag, reached, "WAIT", detail + f" (holds until all reach {milestone_target})"))
        elif status == "climbing":                       # climbing but already at/over the cap
            rows.append((tag, reached, "CAP", detail))
        else:
            rows.append((tag, reached, status.upper(), detail))
    return targets, rows


def _print_rows(rows):
    for tag, reached, mark, detail in rows:
        at = f"@{reached:<5}" if reached is not None else "      "
        print(f"  {tag:<52} {at} {mark:<9} {detail}")


# ---- PHASE 4: extension TRAIN jobs (resume main.py to the new target; pooled + monitored) --------

def train_job_for(t, gpu):
    """A run_attack_sweep Job that RESUMES this combo and trains to t['target'] iterations, exactly
    like run_attack_sweep's own training jobs (fresh `python main.py`, RVFL_* env) -- so it goes
    through run_pool with the live per-round monitor, NOT a blocking inline subprocess.run."""
    mode, attack, rate = t["mode"], t["attack"], t["rate"]
    env = dict(os.environ, RVFL_MODE=mode, RVFL_ATTACK=(attack if attack else "none"),
               RVFL_MRATE=str(rate), RVFL_ITERS=str(t["target"]), RVFL_RESUME="1",
               RVFL_N=str(R.N),        # same single-source N as the base sweep (resume to the right N-root)
               CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    return R.Job(t["tag"], mode, attack, rate, [sys.executable, "main.py"], env, gpu,
                 R.SWEEP_LOG_DIR / f"{t['tag']}.out")


def _round_robin_gpus(n):
    return [R.GPUS[i % len(R.GPUS)] for i in range(n)]


# ---- driver -------------------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description="Extend un-converged runs by +step to plateau or --iter-cap, in phases.")
    ap.add_argument("--apply", action="store_true", help="execute the 4 phases (default: dry run / SCAN only)")
    ap.add_argument("--base-iters", type=int, default=R.NUM_ITERS, help="base target of the main sweep (1000)")
    ap.add_argument("--step", type=int, default=500, help="iterations added per extension")
    ap.add_argument("--iter-cap", type=int, default=1000, help="never extend a run past this many iters")
    ap.add_argument("--keep-every", type=int, default=10, help="checkpoint grain kept after each eval")
    ap.add_argument("--metric", default="train_acc", choices=["train_acc", "test_acc"],
                    help="accuracy signal for the plateau (success) check; Scaling always uses its clean_acc")
    ap.add_argument("--plateau-lo", type=float, default=0.80,
                    help="SUCCESS threshold: last-window train-acc >= this = converged -> stop (test/clean uses -0.05). "
                         "The ONLY stop condition; everything below just keeps extending to --iter-cap.")
    ap.add_argument("--window", type=int, default=100, help="last-N-iteration window for the plateau check")
    ap.add_argument("--force-extend", default="", help="comma tags to always extend (bypass the plateau check)")
    ap.add_argument("--never-extend", default="", help="comma tags to never extend")
    ap.add_argument("--max-passes", type=int, default=100, help="safety bound on round-robin passes")
    args = ap.parse_args(argv)

    cls_kw = dict(metric=args.metric, plateau_lo=args.plateau_lo, window=args.window)
    force = {t for t in args.force_extend.split(",") if t}
    never = {t for t in args.never_extend.split(",") if t}
    combos = canonical()

    print(f"[extend] {'APPLY' if args.apply else 'DRY RUN'} | base {args.base_iters} +{args.step}/step "
          f"| plateau {args.metric}>= {args.plateau_lo} | cap {args.iter_cap} | keep 1/{args.keep_every} "
          f"| {len(combos)} settings | pool<= {R.MAX_PARALLEL} over GPUs {R.GPUS}\n")

    # DRY RUN: SCAN from existing curves only; classify + list what WOULD extend. No eval/prune/train.
    if not args.apply:
        targets, rows = scan(combos, args, cls_kw, force, never)
        _print_rows(rows)
        tags = ", ".join(t["tag"] for t in targets) or "(none)"
        print(f"\n[extend] dry run: {len(targets)} setting(s) WOULD extend -> {tags}")
        print("[extend] re-run with --apply to eval + prune + extend (needs stride-1 evals to decide).")
        return

    R.SWEEP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    R.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for pass_no in range(1, args.max_passes + 1):
        print(f"\n===== pass {pass_no} =====")

        # PHASE 1 -- EVAL (pooled + monitored): stride-1 eval every tag with un-evaluated checkpoints
        ejobs = []
        for (mode, attack, rate) in combos:
            tag = R.run_tag(mode, attack, rate)
            if needs_eval(tag, attack):
                ejobs.append((mode, attack, rate, tag))
        if ejobs:
            gpus = _round_robin_gpus(len(ejobs))
            jobs = []
            for (mode, attack, rate, tag), gpu in zip(ejobs, gpus):
                jobs += eval_jobs_for(tag, mode, attack, rate, gpu)
            print(f"[phase 1/4 EVAL] stride-1 eval for {len(ejobs)} tag(s) ({len(jobs)} job(s)) "
                  f"via run_pool -> logs under {R.SWEEP_LOG_DIR}/<tag>.eval.out")
            R.run_pool(jobs)
        else:
            print("[phase 1/4 EVAL] nothing new to evaluate (all checkpoints already scored)")

        # PHASE 2 -- PRUNE (fast, in-process): coarsen every eval-complete tag to every-Kth + final
        print(f"[phase 2/4 PRUNE] coarsen to every-{args.keep_every}th + final:")
        any_pruned = False
        for (mode, attack, rate) in combos:
            line = prune_one(R.run_tag(mode, attack, rate), args.keep_every)
            if line:
                print(line)
                any_pruned = any_pruned or line.lstrip().startswith("*")
        if not any_pruned:
            print("  (nothing to coarsen this pass)")

        # PHASE 3 -- SCAN (read-only): classify + FILTER OUT ONLY THE EXTENSION TARGETS
        print("[phase 3/4 SCAN] classify settings, select extension targets:")
        targets, rows = scan(combos, args, cls_kw, force, never)
        _print_rows(rows)
        if not targets:
            print(f"\n[extend] pass {pass_no}: no settings to extend -- every setting has converged "
                  f"(plateau) or hit the iter cap. Curves dense, checkpoints coarsened. Done.")
            break
        tags = ", ".join(t["tag"] for t in targets)
        print(f"  -> {len(targets)} target(s): {tags}")

        # PHASE 4 -- TRAIN (pooled + monitored, run_attack_sweep style): drive targets to the shared milestone
        gpus = _round_robin_gpus(len(targets))
        jobs = [train_job_for(t, gpu) for t, gpu in zip(targets, gpus)]
        milestone = targets[0]["target"]                 # all targets share this pass's milestone
        print(f"[phase 4/4 TRAIN] driving {len(jobs)} run(s) up to Iter{milestone} via run_pool "
              f"(<= {R.MAX_PARALLEL} at a time) -> logs under {R.SWEEP_LOG_DIR}/<tag>.out")
        R.run_pool(jobs)
        print(f"[extend] pass {pass_no}: trained {len(jobs)} extension(s); re-evaluating next pass.")
    else:
        print(f"\n[extend] hit --max-passes ({args.max_passes}); stopping.")


if __name__ == "__main__":
    main(sys.argv[1:])