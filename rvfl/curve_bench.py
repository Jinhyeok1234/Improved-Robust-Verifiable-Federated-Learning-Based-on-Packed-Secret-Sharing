"""Per-curve costs of the VPSS primitives, for the curve comparison table (run.py kind
"curve_bench"; see experiments/curve_bench.toml).

For a commitment curve (level: 128 = P-256, 192 = P-384, 256 = P-521), a client count N, a
packing k1, a model size M and a thread count, it times the primitives the protocol runs and
records the wire size of what they produce:

  pss_share        first-level sharing of an M-dimensional quantized update: NoneVPSS, no
                   commitments, t1 = N // 3 (the dealing of the enrollment phase)
  pedersen_share   second-level sharing of N values with Pedersen commitments: PedersenVPSS,
                   t2 = N // 3, k2 = 1 (one of the two dealings of the verification phase)
  pedersen_verify  checking one received second-level batch against its commitments
  reconstruct      error-correcting reconstruction of the M-dimensional vector from all N
                   first-level shares (the server's aggregation). Slow at large N and small k1.

Sizes are pickled bytes per element, as the protocol sends them: a first-level share element
({'share': share}, one per receiver) and a commitment group element ({'commitment': ...}).

Why the grid varies what it varies:
  - The curve fixes |q|, hence the size of a share element and the cost of every Z_q
    operation, and the cost of a group operation in the commitments.
  - N and t = N // 3 set how many shares a block yields and the degree each is evaluated at.
  - k1 sets the number of blocks, ceil(M / k1); with k1 = t no random coefficient is left, so
    the shares are deterministic functions of the update and their serialized size can drop
    below that of a random element.
  - M scales the first-level work linearly.
  - The thread count changes the absolute times and, through memory bandwidth, possibly the
    ratios between curves.

The update that is shared is a random direction scaled to the norm the quantization produces:
the multiplier is the largest one allowed by
    N * mult_TS * (mult * ||g_S|| + sqrt(M) / 2) < sqrt(q / 2),
so the quantized vector has norm sqrt(q / 2) / (N * mult_TS) - sqrt(M) / 2 whatever ||g_S||
is. The same bound gives the analytic columns: log2 of the largest multiplier for
||g_S|| <= gs_norm, log2 of the relative l2 quantization error sqrt(M) / (2 * mult * ||g_S||),
and log2 of the smallest entry that is recovered exactly (2^-149, every float32 value, once
mult > 2^149; 1 / mult below that).

Every timing is the mean over `repeats` runs after `warmup` runs that are discarded, with the
standard deviation over the kept runs.
"""
import math
import pickle
import statistics
import time

import vpss_module as vpss

from . import crypto_params

CURVE_NAMES = crypto_params.CURVES          # level -> curve name
OPS = ("pss_share", "pedersen_share", "pedersen_verify", "reconstruct")
DEFAULT_OPS = ("pss_share", "pedersen_share", "pedersen_verify")
FLOAT32_MIN_SUBNORMAL_LOG2 = -149


def make_config(t, k, n):
    """A VPSS configuration as the protocol builds it: message points 1..t, share points n+1..2n."""
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


def measure(level, n, k1, m, threads, ops, repeats, warmup, mult_ts, gs_norm, rng):
    """One CSV row: the timings of `ops` and the element sizes for one curve and setting."""
    pp, spec = crypto_params.build_pp(vpss, "ec", level)
    q = spec["q"]
    t = n // 3
    row = dict(curve=CURVE_NAMES[level], level=level, q_bits=q.bit_length(), N=n, t=t, k1=k1,
               M=m, threads=threads, repeats=repeats)
    row["log2_mult_max"], row["log2_rel_error"], row["log2_min_exact_entry"] = \
        analytic_columns(q, n, mult_ts, m, gs_norm)
    update = random_update(rng, q, n, mult_ts, m)

    first_shares = None
    if "pss_share" in ops or "reconstruct" in ops:
        config1 = make_config(t, k1, n)
        secrets = to_vector(update)
        mean, sd, (first_shares, _) = timed(
            lambda: vpss.NoneVPSS.distribute(pp, config1, secrets, thread_count=threads),
            warmup, repeats)
        if "pss_share" in ops:
            row["pss_share_s"], row["pss_share_sd"] = mean, sd
            row["share_bytes_per_elem"] = share_bytes_per_element(first_shares)

    if "reconstruct" in ops:
        mean, sd, recovered = timed(
            lambda: vpss.NoneVPSS.reconstruct(pp, config1, first_shares, True, thread_count=threads),
            warmup, repeats)
        for i in range(min(8, m)):                      # a reconstruction that is wrong is not a timing
            if recovered[i].to_int() % q != update[i] % q:
                raise RuntimeError(f"reconstruction mismatch at {i} ({CURVE_NAMES[level]}, N={n}, k1={k1})")
        row["reconstruct_s"], row["reconstruct_sd"] = mean, sd
    first_shares = None

    if "pedersen_share" in ops or "pedersen_verify" in ops:
        config2 = make_config(t, 1, n)
        values = to_vector(rng.randrange(q) for _ in range(n))   # stand-ins for first-level shares
        mean, sd, (second_shares, commitments) = timed(
            lambda: vpss.PedersenVPSS.distribute(pp, config2, values, thread_count=threads),
            warmup, repeats)
        if "pedersen_share" in ops:
            row["pedersen_share_s"], row["pedersen_share_sd"] = mean, sd
            row["commit_bytes_per_elem"] = commitment_bytes_per_element(commitments)
        if "pedersen_verify" in ops:
            mean, sd, ok = timed(
                lambda: vpss.PedersenVPSS.verify_batch_share(pp, second_shares[0], commitments,
                                                             thread_count=threads),
                warmup, repeats)
            if not ok:
                raise RuntimeError(f"Pedersen verification failed ({CURVE_NAMES[level]}, N={n})")
            row["pedersen_verify_s"], row["pedersen_verify_sd"] = mean, sd
    return row


COLUMNS = ["curve", "level", "q_bits", "N", "t", "k1", "M", "threads", "repeats",
           "share_bytes_per_elem", "commit_bytes_per_elem",
           "pss_share_s", "pss_share_sd", "pedersen_share_s", "pedersen_share_sd",
           "pedersen_verify_s", "pedersen_verify_sd", "reconstruct_s", "reconstruct_sd",
           "log2_mult_max", "log2_rel_error", "log2_min_exact_entry"]
