#!/usr/bin/env python3
"""Measure the per-curve costs of the VPSS primitives, for the curve comparison table.

    python bench_curves.py                                  # the default grid, see below
    python bench_curves.py --n 100 --k1 1 8 32 --repeats 5  # a narrower grid
    python bench_curves.py --ops pss_share reconstruct      # other primitives

Run it from the folder that holds crypto_params.py and the built vpss_module, as main.py is.
The other timings of the paper were taken with jemalloc preloaded; to measure under the same
conditions, start it as

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python bench_curves.py

For every commitment curve (--levels: 128 = P-256, 192 = P-384, 256 = P-521) and every
combination of client count N, packing k1, model size M and thread count in the grid, it
times the primitives the protocol runs (--ops) and records the wire size of what they
produce:

  pss_share        first-level sharing of an M-dimensional quantized update: NoneVPSS, no
                   commitments, t1 = N // 3 (the dealing of the enrollment phase)
  pedersen_share   second-level sharing of N values with Pedersen commitments: PedersenVPSS,
                   t2 = N // 3, k2 = 1 (one of the two dealings of the verification phase)
  pedersen_verify  checking one received second-level batch against its commitments
  reconstruct      error-correcting reconstruction of the M-dimensional vector from all N
                   first-level shares (what the server does in the aggregation phase). Off by
                   default: at N = 100 and k1 = 1 a single run takes many minutes.

Sizes are pickled bytes per element, measured the way the protocol sends them: a first-level
share element ({'share': share}, one per receiver) and a commitment group element
({'commitment': commitments}). They are recorded whenever pss_share or pedersen_share runs.

Why the grid varies what it varies:
  - The curve fixes |q|, hence the size of a share element and the cost of every Z_q
    operation, and the cost of a group operation in the commitments.
  - N and t = N // 3 set how many shares a block yields and the degree each is evaluated at.
  - k1 sets the number of blocks, ceil(M / k1), and with k1 = t no random coefficient is left,
    so the shares become deterministic functions of the update and their serialized size can
    drop below that of a random element.
  - M scales the first-level work linearly.
  - The thread count changes the absolute times and, through memory bandwidth, possibly the
    ratios between curves.
By default every k1 the cost tables use is run: powers of two below N // 3, then N // 3 itself.

The update that gets shared is a random direction scaled to the norm the quantization of the
protocol produces: the multiplier is the largest one allowed by
    N * mult_TS * (mult * ||g_S|| + sqrt(M) / 2) < sqrt(q / 2),
so the quantized vector has norm sqrt(q / 2) / (N * mult_TS) - sqrt(M) / 2 whatever ||g_S||
is. The same bound gives the analytic columns of the table, also written to the output:
log2 of the largest multiplier for ||g_S|| <= --gs-norm, log2 of the relative l2
quantization error sqrt(M) / (2 * mult * ||g_S||), and log2 of the smallest entry that is
recovered exactly (2^-149, every float32 value, once mult > 2^149; 1 / mult below that).

Every timing is the mean over --repeats runs after --warmup runs that are discarded, with the
standard deviation over the kept runs. One CSV row per (curve, N, k1, M, threads) goes to
--out, and a summary relative to the first curve is printed at the end.
"""
import argparse
import csv
import math
import pickle
import random
import statistics
import sys
import time

import crypto_params
import vpss_module as vpss

CURVE_NAMES = crypto_params.CURVES          # level -> curve name
OPS = ("pss_share", "pedersen_share", "pedersen_verify", "reconstruct")
DEFAULT_OPS = ("pss_share", "pedersen_share", "pedersen_verify")
FLOAT32_MIN_SUBNORMAL_LOG2 = -149


def default_k1_values(n):
    """The packings the cost tables use at n clients: 1, 2, 4, ... below n // 3, then n // 3."""
    t = n // 3
    values, k = [], 1
    while k < t:
        values.append(k)
        k *= 2
    values.append(t)
    return values


def make_config(t, k, n):
    """A VPSS configuration as main.py builds it: message points 1..t, share points n+1..2n."""
    config = vpss.VPSSConfig()
    config.t = t
    config.k = k
    config.N = n
    config.padType = vpss.PaddingMode.Zero
    secret_indices = vpss.BigIntVector()
    for x in range(1, t + 1):
        secret_indices.append(vpss.BigInt(x))
    share_indices = vpss.BigIntVector()
    for x in range(n + 1, 2 * n + 1):
        share_indices.append(vpss.BigInt(x))
    config.secretIndices = secret_indices
    config.shareIndices = share_indices
    return config


def quantized_norm(q, n, mult_ts, m):
    """Norm of the quantized update under the largest multiplier the protocol allows."""
    return math.isqrt(q // 2) / (n * mult_ts) - math.sqrt(m) / 2


def analytic_columns(q, n, mult_ts, m, gs_norm):
    """log2 of the largest multiplier, of the relative error and of the smallest exact entry."""
    scaled = quantized_norm(q, n, mult_ts, m)        # mult * ||g_S|| at the largest mult
    if scaled <= 0:
        return float("nan"), float("nan"), float("nan")
    log2_mult = math.log2(scaled) - math.log2(gs_norm)
    log2_rel_error = math.log2(math.sqrt(m)) - 1 - math.log2(scaled)
    log2_min_exact = max(FLOAT32_MIN_SUBNORMAL_LOG2, -log2_mult)
    return log2_mult, log2_rel_error, log2_min_exact


def random_update(rng, q, n, mult_ts, m):
    """A quantized update of the norm the protocol produces, as signed integers."""
    direction = [rng.gauss(0.0, 1.0) for _ in range(m)]
    length = math.sqrt(sum(x * x for x in direction))
    scale = quantized_norm(q, n, mult_ts, m) / length
    return [int(round(x * scale)) for x in direction]


def to_vector(values):
    vec = vpss.BigIntVector()
    for v in values:
        vec.append(vpss.BigInt(str(v)))
    return vec


def timed(fn, warmup, repeats):
    """Run fn warmup + repeats times; return (mean s, sd s, result of the last run)."""
    result = None
    times = []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    mean = statistics.fmean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, sd, result


def share_bytes_per_element(shares):
    total_bytes = sum(len(pickle.dumps({"share": s})) for s in shares)
    total_elems = sum(len(s.s_values) for s in shares)
    return total_bytes / total_elems


def commitment_bytes_per_element(commitments):
    count = sum(len(row) for row in commitments)
    return len(pickle.dumps({"commitment": commitments})) / count


def measure(level, pp, q, n, k1, m, threads, args, rng, update_cache):
    t = n // 3
    row = dict(curve=CURVE_NAMES[level], level=level, q_bits=q.bit_length(), N=n, t=t, k1=k1,
               M=m, threads=threads, repeats=args.repeats)
    row["log2_mult_max"], row["log2_rel_error"], row["log2_min_exact_entry"] = \
        analytic_columns(q, n, args.mult_ts, m, args.gs_norm)

    key = (level, n, m)
    if key not in update_cache:
        update_cache.clear()                            # one update in memory at a time
        update_cache[key] = random_update(rng, q, n, args.mult_ts, m)
    update = update_cache[key]

    first_shares = None
    if "pss_share" in args.ops or "reconstruct" in args.ops:
        config1 = make_config(t, k1, n)
        secrets = to_vector(update)
        mean, sd, (first_shares, _) = timed(
            lambda: vpss.NoneVPSS.distribute(pp, config1, secrets, thread_count=threads),
            args.warmup, args.repeats)
        if "pss_share" in args.ops:
            row["pss_share_s"], row["pss_share_sd"] = mean, sd
            row["share_bytes_per_elem"] = share_bytes_per_element(first_shares)

    if "reconstruct" in args.ops:
        mean, sd, recovered = timed(
            lambda: vpss.NoneVPSS.reconstruct(pp, config1, first_shares, True, thread_count=threads),
            args.warmup, args.repeats)
        for i in range(min(8, m)):                      # a reconstruction that is wrong is not a timing
            if recovered[i].to_int() % q != update[i] % q:
                raise RuntimeError(f"reconstruction mismatch at {i} ({CURVE_NAMES[level]}, N={n}, k1={k1})")
        row["reconstruct_s"], row["reconstruct_sd"] = mean, sd
    first_shares = None

    if "pedersen_share" in args.ops or "pedersen_verify" in args.ops:
        config2 = make_config(t, 1, n)
        values = to_vector(rng.randrange(q) for _ in range(n))   # stand-ins for first-level shares
        mean, sd, (second_shares, commitments) = timed(
            lambda: vpss.PedersenVPSS.distribute(pp, config2, values, thread_count=threads),
            args.warmup, args.repeats)
        if "pedersen_share" in args.ops:
            row["pedersen_share_s"], row["pedersen_share_sd"] = mean, sd
            row["commit_bytes_per_elem"] = commitment_bytes_per_element(commitments)
        if "pedersen_verify" in args.ops:
            mean, sd, ok = timed(
                lambda: vpss.PedersenVPSS.verify_batch_share(pp, second_shares[0], commitments,
                                                             thread_count=threads),
                args.warmup, args.repeats)
            if not ok:
                raise RuntimeError(f"Pedersen verification failed ({CURVE_NAMES[level]}, N={n})")
            row["pedersen_verify_s"], row["pedersen_verify_sd"] = mean, sd
    return row


COLUMNS = ["curve", "level", "q_bits", "N", "t", "k1", "M", "threads", "repeats",
           "share_bytes_per_elem", "commit_bytes_per_elem",
           "pss_share_s", "pss_share_sd", "pedersen_share_s", "pedersen_share_sd",
           "pedersen_verify_s", "pedersen_verify_sd", "reconstruct_s", "reconstruct_sd",
           "log2_mult_max", "log2_rel_error", "log2_min_exact_entry"]


def print_summary(rows, ops):
    """Per (N, k1, M, threads): sizes and times relative to the first curve measured."""
    metrics = [("share_bytes_per_elem", "share B/elem")]
    if "pedersen_share" in ops:
        metrics.append(("commit_bytes_per_elem", "commit B/elem"))
    metrics += [(f"{op}_s", op) for op in OPS if op in ops]
    groups = {}
    for r in rows:
        groups.setdefault((r["N"], r["k1"], r["M"], r["threads"]), []).append(r)
    for (n, k1, m, threads), group in groups.items():
        base = group[0]
        print(f"\nN={n} t={n // 3} k1={k1} M={m} threads={threads} (ratio to {base['curve']})")
        print(f"  {'curve':<7} {'log2 mult':>9} {'log2 err':>9} {'log2 x_min':>10}"
              + "".join(f" {label:>22}" for _, label in metrics))
        for r in group:
            cells = []
            for key, _ in metrics:
                v, b = r.get(key), base.get(key)
                cells.append(f" {v:>12.4g} ({v / b:>5.2f}x)" if v is not None and b else f" {'-':>22}")
            print(f"  {r['curve']:<7} {r['log2_mult_max']:>9.1f} {r['log2_rel_error']:>9.1f} "
                  f"{r['log2_min_exact_entry']:>10.1f}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=["128", "192", "256"], choices=sorted(CURVE_NAMES),
                    help="security levels, i.e. curves (default: all three)")
    ap.add_argument("--n", nargs="+", type=int, default=[10, 50, 100], help="client counts")
    ap.add_argument("--k1", nargs="+", type=int, default=None,
                    help="first-level packings (default: per N, powers of two below N // 3, then N // 3); "
                         "a value above N // 3 is skipped for that N")
    ap.add_argument("--model-size", nargs="+", type=int, default=[272_474],
                    help="update dimensions M (default: ResNet-20, 272474)")
    ap.add_argument("--threads", nargs="+", type=int, default=[60], help="VPSS thread counts")
    ap.add_argument("--ops", nargs="+", default=list(DEFAULT_OPS), choices=OPS,
                    help="primitives to time (default: %(default)s)")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per measurement")
    ap.add_argument("--warmup", type=int, default=1, help="untimed runs before them")
    ap.add_argument("--mult-ts", type=float, default=1e8, help="trust-score multiplier mult_TS")
    ap.add_argument("--gs-norm", type=float, default=8.0,
                    help="bound on ||g_S|| for the analytic columns (the table uses 8)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random updates")
    ap.add_argument("--out", default="curve_bench.csv", help="CSV output")
    args = ap.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        sys.exit("--repeats must be >= 1 and --warmup >= 0")

    plan = []
    for n in args.n:
        if n // 3 < 1:
            sys.exit(f"N={n} leaves t = N // 3 = 0")
        k1s = [k for k in (args.k1 or default_k1_values(n)) if 1 <= k <= n // 3]
        for m in args.model_size:
            for k1 in k1s:
                for threads in args.threads:
                    plan.append((n, k1, m, threads))
    print(f"[bench] {len(args.levels)} curve(s) x {len(plan)} (N, k1, M, threads) combination(s), "
          f"ops {' '.join(args.ops)}, {args.warmup} warm-up + {args.repeats} timed run(s) each", flush=True)

    rows = []
    rng = random.Random(args.seed)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        # curve innermost so the rows of one (N, k1, M, threads) sit together for the summary
        for n, k1, m, threads in plan:
            cache = {}
            for level in args.levels:
                pp, spec = crypto_params.build_pp(vpss, "ec", level)
                row = measure(level, pp, spec["q"], n, k1, m, threads, args, rng, cache)
                writer.writerow({c: row.get(c, "") for c in COLUMNS})
                fh.flush()
                rows.append(row)
                print(f"[bench] {row['curve']} N={n} k1={k1} M={m} threads={threads}: "
                      + ", ".join(f"{op} {row[op + '_s']:.3f} s" for op in OPS if op + "_s" in row),
                      flush=True)
    print_summary(rows, args.ops)
    print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()