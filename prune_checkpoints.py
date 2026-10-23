#!/usr/bin/env python3
"""
Coarsen (thin out) per-iteration checkpoints to reclaim disk, AFTER the fine 1-iteration
evaluation for a run is safely on disk.

WHY THIS IS SAFE
    main.py saves a checkpoint EVERY iteration (CHECKPOINT_INTERVAL=1) so that
    eval_checkpoints.py / eval_backdoor.py can produce a 1-iteration-resolution curve.
    Once that curve is written to logs/eval/<TAG>[_asr].jsonl, the fine checkpoints are
    redundant: the per-iteration accuracy/ASR is already archived in the jsonl. What the
    checkpoints are still good for is (a) RESUMING / extending training and (b) RE-evaluating
    a NEW metric later -- both of which a coarse (every-K) subset serves fine. So we keep
    every K-th checkpoint plus the final one and delete the rest.

TWO HARD INVARIANTS (the reason this won't silently break a run)
    1. A tag is pruned ONLY if its 1-iteration eval is COMPLETE:
         - the training marker  Iter{NUM_ITERS}.pt  exists (train finished), AND
         - the eval jsonl exists, reaches Iter{NUM_ITERS}, and is DENSE (coverage >= --min-coverage,
           i.e. it really is the stride-1 curve, not a sparse stride-25 pass).
       In-progress or not-yet-evaluated runs are left completely untouched.
    2. The FINAL checkpoint Iter{NUM_ITERS}.pt is ALWAYS kept. The sweep driver uses it as the
       train-done marker AND as the resume point (main.py sees max Iter >= NUM_ITERS -> "already
       trained, nothing to do"). Deleting it would trigger a full retrain. It is never deleted.

DRY-RUN BY DEFAULT. Nothing is deleted unless you pass --apply.

    python prune_checkpoints.py                 # dry run: report what WOULD be freed
    python prune_checkpoints.py --apply         # actually delete (keep every 10th + final)
    python prune_checkpoints.py --keep-every 25 # keep every 25th (matches EVAL_STRIDE; frees more)
    python prune_checkpoints.py --tags FLTrust_quantized_perp_krum_fltrust_m20   # restrict to some tags

Idempotent: re-running finds nothing new to delete once a tag is coarsened.
"""
import argparse
import json
import re
import sys
from pathlib import Path


def human(n):
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024


def ckpt_iters(tag_dir, dataset, model, N):
    """{iteration: (path, size_bytes)} for every {dataset}_{model}_N{N}_Iter{i}.pt in tag_dir."""
    pat = re.compile(rf"^{re.escape(dataset)}_{re.escape(model)}_N{N}_Iter(\d+)\.pt$")
    out = {}
    for f in tag_dir.iterdir():
        m = pat.match(f.name)
        if m and f.is_file():
            out[int(m.group(1))] = (f, f.stat().st_size)
    return out


def eval_iterations(eval_dir, tag):
    """Set of iterations present in this tag's eval jsonl(s). Scaling writes <tag>_asr.jsonl;
    the untargeted attacks write <tag>.jsonl. Use whichever exist (union if both)."""
    iters, files = set(), []
    for name in (f"{tag}.jsonl", f"{tag}_asr.jsonl"):
        p = eval_dir / name
        if p.exists():
            files.append(p)
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    iters.add(int(json.loads(line)["iteration"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
    return iters, files


def eval_status(iters, num_iters, min_coverage):
    """(-> ok, reason). ok=True only if the fine 1-iter eval is complete AND dense."""
    if not iters:
        return False, "no eval jsonl"
    hi = max(iters)
    if hi < num_iters:
        return False, f"eval only to Iter{hi} (< {num_iters}); still running?"
    coverage = len(iters) / (num_iters + 1)          # stride-1 -> ~1.0 ; stride-25 -> ~0.04
    if coverage < min_coverage:
        return False, f"eval sparse (coverage {coverage:.2f} < {min_coverage:.2f}); not the 1-iter curve"
    return True, f"complete & dense (coverage {coverage:.2f})"


def plan_tag(iters_map, keep_every, num_iters):
    """(keep_set, delete_paths, freed_bytes). Keeps every-K, the final NUM_ITERS, and the max present."""
    present = set(iters_map)
    keep = {i for i in present if i % keep_every == 0}
    keep |= {num_iters, max(present)}                 # never drop the final / resume marker
    delete = present - keep
    paths = [iters_map[i][0] for i in sorted(delete)]
    freed = sum(iters_map[i][1] for i in delete)
    return keep, paths, freed


def main(argv):
    ap = argparse.ArgumentParser(description="Coarsen per-iteration checkpoints after the 1-iter eval is saved.")
    ap.add_argument("--ckpt-root", default="./checkpoints")
    ap.add_argument("--eval-dir",  default="./logs/eval")
    ap.add_argument("--keep-every", type=int, default=10, help="keep every K-th checkpoint (default 10)")
    ap.add_argument("--num-iters",  type=int, default=1000, help="final/target iteration = train-done marker (default 1000)")
    ap.add_argument("--min-coverage", type=float, default=0.9,
                    help="min eval density to treat the curve as the stride-1 curve (default 0.9)")
    ap.add_argument("--dataset", default="CIFAR10")
    ap.add_argument("--model",   default="ResNet20")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--tags", default=None, help="comma-separated tag names to restrict to (default: all)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args(argv)

    if args.keep_every < 1:
        ap.error("--keep-every must be >= 1")

    ckpt_root, eval_dir = Path(args.ckpt_root), Path(args.eval_dir)
    if not ckpt_root.is_dir():
        print(f"[prune] no checkpoint root at {ckpt_root} -- nothing to do.")
        return
    only = set(args.tags.split(",")) if args.tags else None

    tag_dirs = sorted(d for d in ckpt_root.iterdir() if d.is_dir())
    if only:
        tag_dirs = [d for d in tag_dirs if d.name in only]

    mode = "APPLY (deleting)" if args.apply else "DRY RUN (nothing deleted)"
    print(f"[prune] {mode} | keep every {args.keep_every}th + final Iter{args.num_iters} | "
          f"{len(tag_dirs)} tag(s) under {ckpt_root}\n")

    total_freed = total_del = 0
    pruned = skipped = 0
    for d in tag_dirs:
        imap = ckpt_iters(d, args.dataset, args.model, args.N)
        if not imap:
            print(f"  - {d.name:<52} skip (no checkpoints)")
            skipped += 1
            continue
        # invariant 1: only prune when the fine eval is complete & dense
        ev_iters, ev_files = eval_iterations(eval_dir, d.name)
        has_marker = args.num_iters in imap
        ok, reason = eval_status(ev_iters, args.num_iters, args.min_coverage)
        if not has_marker:
            print(f"  - {d.name:<52} skip (train not finished: no Iter{args.num_iters}.pt)")
            skipped += 1
            continue
        if not ok:
            print(f"  - {d.name:<52} skip ({reason})")
            skipped += 1
            continue

        keep, del_paths, freed = plan_tag(imap, args.keep_every, args.num_iters)
        if not del_paths:
            print(f"  = {d.name:<52} already coarse ({len(keep)} kept, 0 to delete)")
            skipped += 1
            continue

        if args.apply:
            for p in del_paths:
                try:
                    p.unlink()
                except OSError as e:
                    print(f"      ! could not delete {p.name}: {e}")
        verb = "deleted" if args.apply else "would free"
        print(f"  * {d.name:<52} keep {len(keep):>4}, {verb} {len(del_paths):>4} ckpt = {human(freed)}")
        total_freed += freed
        total_del += len(del_paths)
        pruned += 1

    print(f"\n[prune] {pruned} tag(s) pruned, {skipped} skipped/untouched.")
    print(f"[prune] {'freed' if args.apply else 'would free'} {human(total_freed)} "
          f"across {total_del} checkpoint file(s).")
    if not args.apply and total_del:
        print("[prune] re-run with --apply to delete. Final checkpoints and un-evaluated runs are never touched.")


if __name__ == "__main__":
    main(sys.argv[1:])