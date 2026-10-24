"""Per-round participant filter (optional; RVFL protocol only).

A client whose cosine with the server's root update is not positive gets trust score 0 and
contributes exactly zero to the aggregate. Leaving it out of the round leaves the result
unchanged and shrinks the all-to-all share exchange. The filter does this every round: the
clients report their cosines first, the server publishes who takes part, and the VPSS
configurations are rebuilt for that participant list.

Two bounds come from alpha, the assumed fraction of malicious clients among the N clients.
c = ceil(alpha * N) clients may send wrong values (robustness), and cp = max(c, 1) share
holders may pool their shares (privacy): a holder always sees its own share, so even with no
malicious client one holder must learn nothing. A round with n participants, packing k and
threshold t needs

    privacy      t - k >= cp          (t >= k + cp)
    robustness   (n - t) / 2 >= c     (t <= n - 2c)
    degree       2t - 1 <= n - c      (the pairwise verification multiplies two shares, so
                                       2t - 1 points are needed, and up to c clients may fail)

Modes (FilterPolicy.mode):

  "max"             Fixed k1. Every positive-cosine client takes part; when that is fewer
                    than the floor (filter_floor), non-positive ones are added back up to it.
                    t = n // 3, clipped into [k + cp, n - 2c].
  "max_variable_k"  As "max", then the privacy margin left at that n goes into the packing:
                    k1 = max(k1, n // 3 - cp), t = n // 3.
  "optimal_nk"      The round size n is chosen too: non-positive-cosine clients may be added
                    back beyond the floor (they contribute zero) wherever the predicted round
                    time is lower. Two threshold rules:
                      threshold="n_over_3"     t = n // 3 and k1 = max(k1, n // 3 - cp), as in
                                               "max_variable_k"; the time is predicted from a
                                               fit time = A(n) / k + B(n) of measured rounds
                      threshold="lower_bound"  t = k + cp, the least t privacy allows, and n and
                                               k are chosen together (no k <= n // 3 cap); the
                                               time is interpolated from a measured (n, k) grid

Which non-positive clients are added back does not change the aggregate, only who bears the
round's cost, and a deterministic rule could be gamed (with "highest_cos" a client can report
a cosine just below zero and be sure to stay in). The default is therefore a uniform random
draw from a generator seeded per round ("random"); "highest_cos" is the alternative.

The cost models are built from end-to-end runs without the filter:
    python -m rvfl.filtering fit  --runs <dir> --link 10Gbps_0.2ms --n ... --k ... --out cost_model.json
    python -m rvfl.filtering grid --c 0 --n-total 48     # the (n, k) points "lower_bound" needs
    python -m rvfl.filtering fit-grid --runs <dir> --link 10Gbps_0.2ms --c 0 --n-total 48 --out cost_model.json
"""
import json
import math
import random
from pathlib import Path

MODES = ("max", "max_variable_k", "optimal_nk")
THRESHOLDS = ("n_over_3", "lower_bound")
ADDBACK_STRATEGIES = ("random", "highest_cos")
THRESHOLD_DIV = 3


def corrupt_bound(alpha, n_total):
    """c = ceil(alpha * N): the clients that may send wrong values."""
    return math.ceil(alpha * n_total)


def privacy_bound(c):
    """Share holders that may pool their shares: at least one, the holder itself."""
    return max(c, 1)


def filter_floor(k, c):
    """Least participant count n of a filtered round for packing k and c malicious clients: a
    threshold with t - k >= cp, (n - t) / 2 >= c and 2t - 1 <= n - c needs n >= k + cp + 2c and
    n >= 2(k + cp) - 1 + c, and k may not exceed the round's n // 3, i.e. n >= 3k.
      k = 1: c >= 1 -> 3c + 1;  c = 0 -> 3."""
    cp = privacy_bound(c)
    return max(k + cp + 2 * c, 2 * (k + cp) - 1 + c, 3 * k)


def round_threshold(n, k, c):
    """t of a round with n participants and packing k (modes "max", "max_variable_k" and
    threshold="n_over_3"): n // 3 clipped into [k + cp, n - 2c]. Returns (t, valid)."""
    cp = privacy_bound(c)
    base = n // THRESHOLD_DIV
    lo, hi = k + cp, n - 2 * c
    if lo <= hi and k <= base:
        return min(max(max(1, base), lo), hi), True
    return max(max(1, base), k), False


def check_threshold(n, k, t, c):
    """Whether an explicit t (threshold="lower_bound") is valid at n participants and packing k."""
    cp = privacy_bound(c)
    return k + cp <= t <= n - 2 * c and 2 * t - 1 <= n - c


def k_max_lower_bound(n, c):
    """Largest k valid at n participants with t = k + cp (0 if none)."""
    cp = privacy_bound(c)
    return max(0, min(n - cp - 2 * c, (n - c + 1) // 2 - cp))


# --- cost models ---------------------------------------------------------------------------

def predict_fit(fit, n, k):
    """Round seconds at (n, k) from [[n, A, B], ...]: time = A / k + B, with A and B
    interpolated linearly in n between the fitted n and held at the end values outside them."""
    pts = sorted((int(e[0]), float(e[1]), float(e[2])) for e in fit)
    if n <= pts[0][0]:
        a, b = pts[0][1], pts[0][2]
    elif n >= pts[-1][0]:
        a, b = pts[-1][1], pts[-1][2]
    else:
        for (n0, a0, b0), (n1, a1, b1) in zip(pts, pts[1:]):
            if n0 <= n <= n1:
                w = (n - n0) / (n1 - n0)
                a, b = a0 + w * (a1 - a0), b0 + w * (b1 - b0)
                break
    return a / k + b


def _row_value(row, k):
    """row: {k: seconds} at one n. The value at k, measured or from the two measured k around
    it (linear in 1 / k); None outside the row."""
    if k in row:
        return row[k]
    lo = [x for x in row if x < k]
    hi = [x for x in row if x > k]
    if not lo or not hi:
        return None
    k0, k1 = max(lo), min(hi)
    w = (1.0 / k - 1.0 / k0) / (1.0 / k1 - 1.0 / k0)
    return row[k0] + w * (row[k1] - row[k0])


def predict_grid(points, n, k):
    """Round seconds at (n, k) from {(n, k): seconds} measured at t = k + cp, or None when no
    measured row below and above n brackets k. A measured point is used as it is; on a measured
    row the two measured k around k are interpolated in 1 / k; otherwise the rows just below and
    above n are each read at k that way and interpolated linearly in n."""
    if (n, k) in points:
        return points[(n, k)]
    rows = {}
    for (pn, pk), s in points.items():
        rows.setdefault(pn, {})[pk] = s
    if n in rows:
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


def grid_points(c, n_total):
    """The (n, k) points to measure for threshold="lower_bound" at one c: every n that is a
    multiple of 3 (and N itself) at k = 1, 2, 3, 4, every second k from 6, and the largest valid
    k; every other n at its largest valid k only."""
    pts = set()
    for n in range(3, n_total + 1):
        km = k_max_lower_bound(n, c)
        if km < 1:
            continue
        if n % 3 == 0 or n == n_total:
            ks = {k for k in (1, 2, 3, 4) if k <= km} | set(range(6, km + 1, 2)) | {km}
        else:
            ks = {km}
        pts |= {(n, k) for k in ks}
    return sorted(pts)


def load_cost_model(path, link, threshold, c):
    """The part of a cost model file one run needs: the fit ([[n, A, B], ...]) for
    threshold="n_over_3", the measured grid ([[n, k, seconds], ...]) of this c for
    threshold="lower_bound". link None takes the file's only link."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"the optimal_nk filter needs the cost model {path} (python -m rvfl.filtering)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    links = [k for k in doc if not k.startswith("_")]
    if link is None:
        if len(links) != 1:
            raise SystemExit(f"{path} has the links {links}; set cost_link")
        link = links[0]
    entry = doc.get(link, {})
    if threshold == "n_over_3":
        if not entry.get("fit"):
            raise SystemExit(f"{path} has no fit for {link} (it has {links})")
        return [[e["n"], e["A"], e["B"]] for e in entry["fit"]]
    grid = entry.get("grid", {}).get(str(c))
    if not grid or not grid.get("points"):
        raise SystemExit(f"{path} has no grid for {link}, c={c}")
    return grid["points"]


# --- the server's choice -------------------------------------------------------------------

class FilterPolicy:
    """The server's per-round choice of participants, k1 and t."""

    def __init__(self, mode, n_total, k1, k2, alpha, threshold="n_over_3", addback="random",
                 cost_model=None):
        if mode not in MODES:
            raise ValueError(f"unknown filter mode {mode!r}; one of {MODES}")
        if threshold not in THRESHOLDS:
            raise ValueError(f"unknown threshold rule {threshold!r}; one of {THRESHOLDS}")
        if addback not in ADDBACK_STRATEGIES:
            raise ValueError(f"unknown addback strategy {addback!r}; one of {ADDBACK_STRATEGIES}")
        if mode == "optimal_nk" and cost_model is None:
            raise ValueError("the optimal_nk filter needs a cost model")
        self.mode, self.threshold, self.addback = mode, threshold, addback
        self.n_total, self.k1, self.k2 = n_total, k1, k2
        self.c = corrupt_bound(alpha, n_total)
        self.cp = privacy_bound(self.c)
        self.floor = filter_floor(max(k1, k2), self.c)
        self.cost_model = cost_model

    @property
    def lower_bound(self):
        return self.mode == "optimal_nk" and self.threshold == "lower_bound"

    def round_k1(self, n):
        """k1 at n participants: fixed for "max", max(k1, n // 3 - cp) otherwise (t = n // 3)."""
        if self.mode == "max":
            return self.k1
        return max(self.k1, n // THRESHOLD_DIV - self.cp)

    def _choose_size(self, n_lo, n_hi):
        """threshold="n_over_3": the n in [n_lo, n_hi] with the least predicted round time at
        its packing round_k1(n), the smallest on a tie."""
        best_n, best_s, preds = n_lo, None, {}
        for n in range(n_lo, n_hi + 1):
            s = predict_fit(self.cost_model, n, self.round_k1(n))
            preds[n] = round(s, 3)
            if best_s is None or s < best_s - 1e-9:
                best_n, best_s = n, s
        return best_n, None, None, {"n_lo": n_lo, "n": best_n,
                                    "pred_s_lo": preds[n_lo], "pred_s": preds[best_n]}

    def _choose_shape(self, n_lo, n_hi):
        """threshold="lower_bound": the (n, k) with the least predicted round time over n in
        [n_lo, n_hi] and every k valid at n, t = k + cp (the smallest n, then k, on a tie)."""
        points = {(int(e[0]), int(e[1])): float(e[2]) for e in self.cost_model}
        best, lo_best, unpredicted = None, None, 0
        for n in range(n_lo, n_hi + 1):
            for k in range(max(1, self.k1), k_max_lower_bound(n, self.c) + 1):
                s = predict_grid(points, n, k)
                if s is None:
                    unpredicted += 1
                    continue
                key = (s, n, k)
                if best is None or key < best:
                    best = key
                if n == n_lo and (lo_best is None or key < lo_best):
                    lo_best = key
        if best is None:
            raise RuntimeError(f"optimal_nk filter: no predictable (n, k) for n in [{n_lo}, {n_hi}], c={self.c}")
        s, n, k = best
        return n, k, k + self.cp, {"n_lo": n_lo, "n": n, "k": k, "t": k + self.cp,
                                   "pred_s": round(s, 3),
                                   "pred_s_lo": round(lo_best[0], 3) if lo_best else None,
                                   "k_lo": lo_best[2] if lo_best else None,
                                   "unpredicted": unpredicted}

    def _addback(self, rejects, count, cos_by_id, seed):
        if self.addback == "random":
            rng = random.Random(seed) if seed is not None else random.Random()
            return rng.sample(sorted(rejects), count)
        return sorted(rejects, key=lambda cid: (-cos_by_id[cid], cid))[:count]

    def select(self, cos_by_id, seed=None):
        """(participants, k1, t, choice) for a round with these reported cosines. t is None
        unless threshold="lower_bound" (the parties then derive it with round_threshold);
        choice records the predicted round times of "optimal_nk" (None otherwise)."""
        chosen = sorted(cid for cid, c in cos_by_id.items() if c > 0)
        target = max(len(chosen), self.floor)
        k1 = t = choice = None
        if self.mode == "optimal_nk":
            if self.lower_bound:
                target, k1, t, choice = self._choose_shape(target, len(cos_by_id))
            else:
                target, _, _, choice = self._choose_size(target, len(cos_by_id))
        if len(chosen) < target:
            rejects = [cid for cid, c in cos_by_id.items() if c <= 0]
            count = min(target - len(chosen), len(rejects))
            chosen = sorted(chosen + self._addback(rejects, count, cos_by_id, seed))
        if k1 is None:
            k1 = self.round_k1(len(chosen))
        return chosen, k1, t, choice


# --- building the cost models from measured rounds ------------------------------------------

def _round_seconds(point_dir):
    from . import logs
    if logs.read_status(point_dir).get("status") != "ok":
        return None
    return logs.mean_round_seconds(point_dir)


def _fit_line(points):
    """Least squares of T = A x + B with x = 1 / k; points are (k, T)."""
    import statistics
    xs = [1.0 / k for k, _ in points]
    ys = [t for _, t in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    b = my - a * mx
    return a, b, [y - (a * x + b) for x, y in zip(xs, ys)]


def _update_file(out, link, key, value):
    out = Path(out)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    doc.setdefault(link, {})[key] = value
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _find_e2e(runs, link, **settings):
    """The end-to-end point without the filter on the link with this label, or None. The
    config stores the link as "Mbit/s:RTT ms", the cost model and the folders use its label."""
    from . import logs
    found = [d for d in logs.find_points(runs, kind="e2e", filter="none", **settings)
             if logs.link_label(json.loads((d / "config.json").read_text(encoding="utf-8"))["link"]) == link]
    if len(found) > 1:
        raise SystemExit(f"{len(found)} runs under {runs} match {settings} on {link}: {[str(d) for d in found]}")
    return found[0] if found else None


def fit_main(args):
    """time = A / k + B per n, from end-to-end runs without the filter at t = n // 3."""
    from . import logs
    pts_by_n, missing, fits = {}, [], []
    for n in args.n:
        for k in sorted({k for k in args.k if k <= n // 3} | {n // 3}):
            d = _find_e2e(args.runs, args.link, n=n, k1=k, t=None)
            s = _round_seconds(d) if d else None
            if s is None:
                missing.append(f"N{n}_k{k}")
            else:
                pts_by_n.setdefault(n, []).append((k, s))
    for n, at_n in sorted(pts_by_n.items()):
        if len(at_n) < 2:
            continue
        a, b, res = _fit_line(at_n)
        fits.append({"n": n, "A": round(a, 4), "B": round(b, 4), "k": [k for k, _ in at_n],
                     "max_abs_residual_s": round(max(abs(r) for r in res), 3)})
        for (k, s), r in zip(at_n, res):
            print(f"n={n:3d} k={k:3d}  measured {s:8.2f} s  fit {s - r:8.2f} s  residual {r:+7.2f} s")
    if missing:
        print("not found or not finished: " + ", ".join(missing))
    if not fits:
        raise SystemExit("nothing to fit: no n has two finished k values")
    _update_file(args.out, args.link, "fit", fits)
    print(f"-> {args.out} [{args.link}.fit]")


def fit_grid_main(args):
    """The measured (n, k) grid of each c, from end-to-end runs without the filter at t = k + cp."""
    from . import logs
    for c in args.c:
        cp = privacy_bound(c)
        points, missing = [], []
        for n, k in grid_points(c, args.n_total):
            d = _find_e2e(args.runs, args.link, n=n, k1=k, t=k + cp)
            s = _round_seconds(d) if d else None
            if s is None:
                missing.append(f"N{n}_k{k}_t{k + cp}")
            else:
                points.append([n, k, round(s, 3)])
        out = Path(args.out)
        doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        doc.setdefault(args.link, {}).setdefault("grid", {})[str(c)] = {"points": points, "missing": missing}
        out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"c={c}: {len(points)} points" + (f"; not found or not finished: {', '.join(missing)}" if missing else ""))
    print(f"-> {args.out} [{args.link}.grid]")


def grid_main(args):
    """Print the grid of one c as [[run]] entries for run.py (end-to-end, filter off)."""
    cp = privacy_bound(args.c)
    rows = {}
    for n, k in grid_points(args.c, args.n_total):
        rows.setdefault(n, []).append(k)
    print(f"# lower_bound grid, c={args.c}, cp={cp}, N={args.n_total}: "
          f"{sum(len(ks) for ks in rows.values())} points (t = k + {cp})")
    for n, ks in sorted(rows.items()):
        for k in ks:
            print(f'[[run]]\nkind = "e2e"\nname = "grid_c{args.c}"\nn = {n}\nk1 = {k}\nt = {k + cp}\n')


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="python -m rvfl.filtering",
                                 description="Cost models of the optimal_nk participant filter.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help='fit time = A / k + B per n (threshold = "n_over_3")')
    f.add_argument("--runs", required=True, help="output folder of the end-to-end runs")
    f.add_argument("--link", required=True, help="link label, e.g. 10Gbps_0.2ms")
    f.add_argument("--n", type=int, nargs="+", required=True)
    f.add_argument("--k", type=int, nargs="+", required=True, help="packings (n // 3 is added)")
    f.add_argument("--out", default="cost_model.json")
    g = sub.add_parser("grid", help='print the (n, k) grid threshold = "lower_bound" needs, as run.py entries')
    g.add_argument("--c", type=int, required=True, help="c = ceil(alpha * N)")
    g.add_argument("--n-total", type=int, required=True)
    h = sub.add_parser("fit-grid", help='store the measured grid (threshold = "lower_bound")')
    h.add_argument("--runs", required=True)
    h.add_argument("--link", required=True)
    h.add_argument("--c", type=int, nargs="+", required=True)
    h.add_argument("--n-total", type=int, required=True)
    h.add_argument("--out", default="cost_model.json")
    args = ap.parse_args()
    {"fit": fit_main, "grid": grid_main, "fit-grid": fit_grid_main}[args.cmd](args)


if __name__ == "__main__":
    main()
