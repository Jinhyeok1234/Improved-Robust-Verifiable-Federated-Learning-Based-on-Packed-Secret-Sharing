"""Variable k, plan 3: the (n, k) candidates of a round, the measurement grid they are predicted
from, and the prediction itself. One place for the rules, used by party.py (the server's
choice) and make_cost_model.py (the table read from the measured grid). The privacy bound
(privacy_bound) and the participant floor (filter_floor) live here too, for every mode.

Two bounds come from alpha. c = ceil(alpha * N) is the number of clients that may send wrong
values (robustness). cp = privacy_bound(c) = max(c, 1) is the number of share holders that may
pool their shares: a client always sees its own share, so even with no malicious client one
holder must learn nothing, and with t = k a share carries no randomness and does leak.

Plan 3 moves t with k: t = k + cp, the least t that privacy t - k >= cp allows. A round with n
participants and packing k is valid when
    t = k + cp <= n - 2c         robustness of the server's reconstruction, (n - t) / 2 >= c
    2t - 1 <= n - c              degree reduction: the product of two shares has degree 2(t - 1),
                                 so 2t - 1 points are needed, and up to c clients may fail the
                                 pairwise verification
    k >= 1
There is no k <= n // 3 cap in plan 3.
  e.g. alpha = 0.2 (c = cp = 10), n = 48: k <= min(48 - 10 - 20, (48 - 10 + 1) // 2 - 10) = 9, t = 19
       alpha = 0   (c = 0, cp = 1), n = 48: k <= min(48 - 1, (48 + 1) // 2 - 1) = 23, t = 24

Grid, per c: every n that is a multiple of 3 (and N), at k = 1, 2, 3, 4, then every second k
(6, 8, ...), and the largest valid k; every other n at its largest valid k only (the boundary).
A candidate that was measured uses its measurement. One on a measured row (n) uses the two
measured k around it, linearly in 1/k (a round's time falls roughly as 1/k). Any other is
interpolated from four points: in the nearest rows below and above it (in n) that bracket its
k, the two measured k on either side (linearly in 1/k), then linearly in n between the rows.
"""


def privacy_bound(c):
    """Share holders that may pool their shares, for c = ceil(alpha * N) malicious clients: at
    least one, the holder itself."""
    return max(c, 1)


def filter_floor(k, c):
    """Least participant count n of a filtered round (every mode), for packing k and
    c = ceil(alpha * N) assumed malicious clients, with cp = privacy_bound(c): a threshold t
    with t - k >= cp (privacy), (n - t) / 2 >= c (robustness) and 2t - 1 <= n - c (degree
    reduction) needs n >= k + cp + 2c and n >= 2(k + cp) - 1 + c, and k may not exceed the
    round's filter-off threshold n // 3, i.e. n >= 3k.  k = 1: c >= 1 -> 3c + 1;  c = 0 -> 3."""
    cp = privacy_bound(c)
    return max(k + cp + 2 * c, 2 * (k + cp) - 1 + c, 3 * k)


def k_max(n, c):
    """Largest valid plan-3 k at n participants and c = ceil(alpha * N) (0 if none)."""
    cp = privacy_bound(c)
    return max(0, min(n - cp - 2 * c, (n - c + 1) // 2 - cp))


def valid(n, k, c):
    return 1 <= k <= k_max(n, c)


def grid_points(c, n_total):
    """[(n, k)] to measure for one c (= ceil(alpha * N)), at t = k + privacy_bound(c): the grid
    rows and the boundary."""
    pts = set()
    for n in range(3, n_total + 1):     # 3 is the least round size any mode uses
        km = k_max(n, c)
        if km < 1:
            continue
        if n % 3 == 0 or n == n_total:
            ks = {k for k in (1, 2, 3, 4) if k <= km} | set(range(6, km + 1, 2)) | {km}
        else:
            ks = {km}
        pts |= {(n, k) for k in ks}
    return sorted(pts)


def _row_value(row, k):
    """row: {k: seconds} at one n. The value at k, from k itself or the two measured k around
    it (linear in 1/k); None when k is outside the row."""
    if k in row:
        return row[k]
    lo = [x for x in row if x < k]
    hi = [x for x in row if x > k]
    if not lo or not hi:
        return None
    k0, k1 = max(lo), min(hi)
    w = (1.0 / k - 1.0 / k0) / (1.0 / k1 - 1.0 / k0)
    return row[k0] + w * (row[k1] - row[k0])


def predict(points, n, k):
    """points: {(n, k): seconds} measured for this c. Predicted round seconds at (n, k), or
    None when no row below or above brackets k."""
    if (n, k) in points:
        return points[(n, k)]
    rows = {}
    for (pn, pk), s in points.items():
        rows.setdefault(pn, {})[pk] = s
    if n in rows:                      # a measured row: the two measured k around k
        v = _row_value(rows[n], k)
        if v is not None:
            return v
    below = above = None
    for pn in sorted((x for x in rows if x < n), reverse=True):
        v = _row_value(rows[pn], k)
        if v is not None:
            below = (pn, v)
            break
    for pn in sorted(x for x in rows if x > n):
        v = _row_value(rows[pn], k)
        if v is not None:
            above = (pn, v)
            break
    if below is None or above is None:
        return None
    (n0, v0), (n1, v1) = below, above
    return v0 + (n - n0) / (n1 - n0) * (v1 - v0)


def main():
    """Print the plan-3 grid of one c as the netns runs that measure it: one line per n with
    the RVFL_N and RVFL_E2E_KT values (filter off, t = k + privacy_bound(c))."""
    import argparse
    ap = argparse.ArgumentParser(description="the plan-3 grid of one c, as RVFL_N / RVFL_E2E_KT settings")
    ap.add_argument("--c", type=int, required=True, help="c = ceil(alpha * N)")
    ap.add_argument("--n-total", type=int, required=True, help="N")
    args = ap.parse_args()
    cp = privacy_bound(args.c)
    rows = {}
    for n, k in grid_points(args.c, args.n_total):
        rows.setdefault(n, []).append(k)
    for n, ks in sorted(rows.items()):
        print(f"RVFL_N={n} RVFL_E2E_KT={','.join(f'{k}:{k + cp}' for k in ks)}")
    print(f"# {sum(len(ks) for ks in rows.values())} points, c={args.c}, cp={cp}, N={args.n_total}")


if __name__ == "__main__":
    main()