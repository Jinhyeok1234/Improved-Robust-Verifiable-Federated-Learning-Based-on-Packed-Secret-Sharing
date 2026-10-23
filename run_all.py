#!/usr/bin/env python3
"""
One-command unattended pipeline: train the whole sweep to the base target, THEN auto-extend
every un-plateaued (non-collapsed) setting -- with NO manual "did training finish yet?" hand-off
in between. Launch it once and leave it.

    python run_all.py                        # stage 1 (train base sweep) -> stage 2 (auto-extend)
    python run_all.py -- --iter-cap 3000     # everything after `--` is forwarded to extend_sweep.py
    python run_all.py -- --require-improving --plateau-lo 0.82

Stage 1  (run_attack_sweep.py, train mode): trains every combo in SWEEP to NUM_ITERS. Resume-safe --
    already-finished combos are skipped, so this blocks only on the work that remains, then returns.
Stage 2  (extend_sweep.py --apply): walks the full canonical grid, does the stride-1 accuracy eval
    (+ASR for Scaling), compacts checkpoints to every-10th, and adds +500 iterations to any setting
    below the train-accuracy plateau that has NOT collapsed -- round-robin until nothing extends.

Because stage 2 finishes any partially-trained run before judging it (main.py resume), a base combo
that crashed mid-way is completed here too. The whole pipeline is resume-safe: re-run `python
run_all.py` after any interruption and it continues where it left off.

Stage 2 forces stride-1 evaluation itself, so you do NOT need a separate `run_attack_sweep.py --eval`
pass; the authoritative per-iteration curves come out of this pipeline.
"""
import sys

import run_attack_sweep
import extend_sweep


def main(argv):
    extend_args = argv[argv.index("--") + 1:] if "--" in argv else []

    bar = "=" * 80
    print(bar)
    print("[run_all] STAGE 1/2  --  base training sweep (train every combo to the base target)")
    print(bar, flush=True)
    run_attack_sweep.main([])                       # TRAIN mode (no --eval): resume-skips finished combos

    print("\n" + bar)
    print("[run_all] STAGE 2/2  --  auto-extend un-plateaued settings "
          "(stride-1 eval + every-10 compaction + +500 loop)")
    print(bar, flush=True)
    extend_sweep.main(["--apply"] + extend_args)    # round-robin until nothing extends

    print("\n[run_all] pipeline complete.")


if __name__ == "__main__":
    main(sys.argv[1:])