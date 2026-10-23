#!/usr/bin/env python3
"""
Camera-ready figures from the attack-sweep eval curves (logs/eval/<tag>.jsonl).

Encoding (fixed):  Ours = SOLID line,  FLTrust = DASHED line.  Colour distinguishes the attack
(Okabe-Ito colourblind-safe palette). Metric defaults to test error (1 - test_acc).

    python plot_attacks.py --fig iter               # Fig 1: error vs iteration, fractions {0,20}
    python plot_attacks.py --fig frac               # Fig 2: final error vs malicious fraction {0..95}
    python plot_attacks.py --fig both               # both
    python plot_attacks.py --fig iter --fractions 0,20,80        # pick which fractions appear
    python plot_attacks.py --fig frac --window 100 --band        # last-100-iter mean, +/-std band
    python plot_attacks.py --metric train_acc --eval-dir ./logs/eval --out-dir ./figures
    python plot_attacks.py --fig sweep --N 30 100 --cmap plasma   # one figure, a row pair per N
    python plot_attacks.py --fig scaling --N 30 100 --metric test_acc --cmap plasma

Fig 1 (Exp-A view): one panel, x = iteration. The 0% row is the shared no-attack pair; every other
requested fraction adds each attack's Ours/FLTrust curves at that fraction.
Fig 2 (Exp-B view): one panel per attack, x = malicious fraction, y = final error (mean over the last
`--window` iterations, matching the Table-B convention), Ours vs FLTrust.

Sweep (--fig sweep): x = iteration, every fraction a curve coloured by fraction, the untargeted
attacks only (LF, Trim, Adaptive, Krum). Rows: protocol, and with several --N a row pair (Ours,
FLTrust) per N, each N on its own iteration axis.
Scaling (--fig scaling): the backdoor apart from the others, as its two effects differ by protocol
(FLTrust trains on the de-scaled backdoor update, Ours discards the rejected update). Row 1 = the
--metric curve, row 2 = ASR; columns = (N, Ours), (N, FLTrust) for every --N. The sweep evaluates a
Scaling run with eval_backdoor.py only, so its accuracy is the clean_acc of <tag>_asr.jsonl (the same
full-test-set accuracy eval_checkpoints.py writes as test_acc); row 1 takes test_acc or test_err.
With several --N the figures go to ./runs/figures (and --eval-dir cannot be given); iter, frac and
asr take a single N.

Reads only the accuracy files (skips `_asr` ASR curves and any legacy `_peek`). Missing combos are
skipped with a note, so it runs fine on a partial sweep.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from tabulate_attacks import parse_tag, load        # identical tag parsing as the tables
import run_attack_sweep as R                         # single source of N and the ./runs/N{N} layout

# ---- fixed style ------------------------------------------------------------------------------
ATTACK_ORDER = ["-", "LF", "Trim", "Adaptive", "Krum", "Scaling"]     # "-" == no-attack
ATTACK_COLOR = {                                                       # Okabe-Ito, colourblind-safe
    "-":        "#000000",   # no-attack: black
    "LF":       "#0072B2",   # blue
    "Trim":     "#D55E00",   # vermillion
    "Adaptive": "#009E73",   # bluish green
    "Krum":     "#CC79A7",   # reddish purple
    "Scaling":  "#E69F00",   # orange
}
PROTO_STYLE = {"Ours": "-", "FLTrust": "--"}                          # SOLID vs DASHED
OURS_C, FT_C = "#0072B2", "#D55E00"                                   # Ours = blue, FLTrust = vermillion
NA_OURS_C = "#000000"                                                # no-attack (Ours) baseline: black
NA_FT_C   = "#2C2C2C"                                                # no-attack (FLTrust) baseline: near-black grey (dashed)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight",
})

METRIC_LABEL = {"test_err": "Test error", "test_acc": "Test accuracy",
                "train_err": "Train error", "train_acc": "Train accuracy"}
NOTE = ""     # optional footer stamp (set by --note), e.g. "preliminary" or a data caveat


def _metric(rec, metric):
    if metric == "test_err":
        return 1.0 - rec["test_acc"] if "test_acc" in rec else rec.get("test_err")
    if metric == "train_err":
        return 1.0 - rec["train_acc"] if "train_acc" in rec else None
    return rec.get(metric)                                            # test_acc / train_acc


def load_curves(eval_dir, metric):
    """{(proto, attack, rate): {iteration: value}} from the accuracy jsonls."""
    curves = {}
    for f in sorted(Path(eval_dir).glob("*.jsonl")):
        stem = f.stem
        if stem.endswith("_peek") or stem.endswith("_asr"):          # legacy / ASR -> not error curves
            continue
        pt = parse_tag(stem)
        if not pt:
            continue
        pts = {}
        for r in load(f):
            it, v = r.get("iteration"), _metric(r, metric)
            if it is not None and v is not None:
                pts[it] = v
        if pts:
            curves[pt] = pts
    return curves


def _smooth(xs, ys, w):
    """Centred rolling mean over a window of `w` ITERATIONS (not points). Iteration-based so curves
    sampled at different strides get the same amount of smoothing, and centred so there is no
    trailing lag that would lift a steep curve's smoothed line off its raw values."""
    if w <= 1 or len(ys) < 2:
        return xs, ys
    half, out, lo, hi, n = w / 2.0, [], 0, 0, len(xs)
    for i in range(n):
        while lo < n and xs[lo] < xs[i] - half:
            lo += 1
        while hi < n and xs[hi] <= xs[i] + half:
            hi += 1
        seg = ys[lo:hi]
        out.append(sum(seg) / len(seg))
    return xs, out


def _series(curve):
    xs = sorted(curve)
    return xs, [curve[x] for x in xs]


def _final(curve, window):
    """(mean, std) of the metric over the last `window` iterations of a curve."""
    if not curve:
        return None, None
    hi = max(curve)
    vals = [v for it, v in curve.items() if hi - window < it <= hi]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return m, sd


# ---- drawing helper: faint raw underlay + bold smoothed trend ---------------------------------
def _draw(ax, curve, color, ls, smooth, raw_alpha, lw=1.6, label=None):
    """Plot a curve. When smoothing, underlay the real per-iteration values faintly so genuine
    variability (e.g. Trim's oscillation) stays visible under a clean smoothed trend line."""
    xs, ys = _series(curve)
    if smooth > 1 and raw_alpha > 0:
        ax.plot(xs, ys, color=color, linestyle=ls, lw=0.7, alpha=raw_alpha)
    sx, sy = _smooth(xs, ys, smooth)
    ax.plot(sx, sy, color=color, linestyle=ls, lw=lw, label=label)


# ---- Fig 1: error vs iteration — one panel per attack, Ours vs FLTrust colour-separated -------
def fig_iter(curves, fractions, metric, smooth, raw_alpha, out_dir, fmts, ncols):
    attacks = [a for a in ATTACK_ORDER if a not in ("-", "Scaling")
               and any((p, a, f) in curves for p in ("Ours", "FLTrust") for f in fractions if f != 0)]
    if not attacks:
        print("[plot] no attack curves for the iteration figure."); return None
    nonzero = [f for f in fractions if f != 0]
    inc_na = 0 in fractions
    na = {p: curves.get((p, "-", 0)) for p in ("Ours", "FLTrust")}

    ncols = min(ncols, len(attacks)); nrows = (len(attacks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows),
                             squeeze=False, sharey=True, sharex=True, layout="constrained")
    for k, atk in enumerate(attacks):
        ax = axes[k // ncols][k % ncols]
        if inc_na:                                        # baseline: Ours black solid, FLTrust grey dashed
            na_col = {"Ours": NA_OURS_C, "FLTrust": NA_FT_C}
            for p in ("Ours", "FLTrust"):
                if na[p]:
                    _draw(ax, na[p], na_col[p], PROTO_STYLE[p], smooth, raw_alpha, lw=1.2)
        for f in nonzero:                                 # Ours = blue solid, FLTrust = orange dashed
            for p, col in (("Ours", OURS_C), ("FLTrust", FT_C)):
                c = curves.get((p, atk, f))
                if c:
                    _draw(ax, c, col, PROTO_STYLE[p], smooth, raw_alpha)
        ax.set_title(atk); ax.set_xlabel("Global iteration")
        if k % ncols == 0:
            ax.set_ylabel(METRIC_LABEL[metric])
    for j in range(len(attacks), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles = [Line2D([], [], color=OURS_C, lw=2, ls="-", label="Ours"),
               Line2D([], [], color=FT_C, lw=2, ls="--", label="FLTrust"),
               Line2D([], [], color=NA_OURS_C, lw=1.5, ls="-", label="no-attack (Ours)"),
               Line2D([], [], color=NA_FT_C, lw=1.5, ls="--", label="no-attack (FLTrust)")]
    axes[0][0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{METRIC_LABEL[metric]} vs iteration — Ours vs FLTrust under each attack "
                 f"@ {'/'.join(f'{f}%' for f in nonzero) or '0%'}")
    return _save(fig, out_dir, "fig_iter", fmts)


def load_asr_curves(eval_dir):
    """{(proto, rate): {iteration: asr}} for Scaling, from the <tag>_asr.jsonl files."""
    out = {}
    for f in sorted(Path(eval_dir).glob("*_asr.jsonl")):
        pt = parse_tag(f.stem[:-4])                       # strip trailing "_asr"
        if not pt or pt[1] != "Scaling":
            continue
        proto, _, rate = pt
        pts = {r["iteration"]: r["asr"] for r in load(f) if "asr" in r and "iteration" in r}
        if pts:
            out[(proto, rate)] = pts
    return out


def load_scaling_clean(eval_dir, metric):
    """{(proto, rate): {iteration: value}} of the Scaling runs' clean test accuracy as `metric`.
    run_attack_sweep evaluates a Scaling run with eval_backdoor.py only, which writes the clean
    accuracy into <tag>_asr.jsonl as clean_acc -- the same full-test-set accuracy eval_checkpoints.py
    writes as test_acc -- so there is no <tag>.jsonl for it. Train metrics are not recorded there."""
    if metric not in ("test_acc", "test_err"):
        return {}
    out = {}
    for f in sorted(Path(eval_dir).glob("*_asr.jsonl")):
        pt = parse_tag(f.stem[:-4])                       # strip trailing "_asr"
        if not pt or pt[1] != "Scaling":
            continue
        proto, _, rate = pt
        pts = {r["iteration"]: (r["clean_acc"] if metric == "test_acc" else 1.0 - r["clean_acc"])
               for r in load(f) if "clean_acc" in r and "iteration" in r}
        if pts:
            out[(proto, rate)] = pts
    return out


# ---- Fig 2: breaking point vs malicious fraction — per-attack panels ---------------------------
# Untargeted attacks show final TEST ERROR; Scaling shows final ASR in its own panel (own y-axis
# labelled ASR) -- as in FLTrust, Scaling sits in the same grid with just that panel's axis relabelled.
def fig_frac(curves, asr_curves, fractions, metric, window, band, out_dir, fmts, ncols, fltrust_ref=0.02):
    err_attacks = [a for a in ATTACK_ORDER if a not in ("-", "Scaling")
                   and any((p, a, f) in curves for p in ("Ours", "FLTrust") for f in fractions if f != 0)]
    panels = err_attacks + (["Scaling"] if asr_curves else [])
    if not panels:
        print("[plot] no curves found for the fraction figure."); return None
    ncols = min(ncols, len(panels)); nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows),
                             squeeze=False, layout="constrained")     # not sharey: ASR has its own scale
    na = {p: _final(curves.get((p, "-", 0), {}), window) for p in ("Ours", "FLTrust")}
    err_axes = []

    def _draw_points(ax, get):
        for proto, col in (("Ours", OURS_C), ("FLTrust", FT_C)):
            xs, ys, es = [], [], []
            for f in fractions:
                m, sd = get(proto, f)
                if m is None:
                    continue
                xs.append(f); ys.append(m); es.append(sd or 0.0)
            if not xs:
                continue
            ax.plot(xs, ys, color=col, linestyle=PROTO_STYLE[proto], marker="o", ms=4, lw=1.6, label=proto)
            if band:
                ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                                color=col, alpha=0.15, lw=0)

    for k, atk in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        if atk == "Scaling":
            _draw_points(ax, lambda p, f: (_final(asr_curves[(p, f)], window) if (p, f) in asr_curves else (None, None)))
            if fltrust_ref is not None:
                ax.axhline(fltrust_ref, color="0.5", lw=1.0, ls=":")
            ax.set_ylim(bottom=-0.02)
            ax.set_ylabel(f"Attack success rate (last {window} it.)")
        else:
            _draw_points(ax, lambda p, f, atk=atk: (na[p] if f == 0 else _final(curves.get((p, atk, f), {}), window)))
            if k % ncols == 0:
                ax.set_ylabel(f"{METRIC_LABEL[metric]} (last {window} it.)")
            err_axes.append(ax)
        ax.set_title(atk); ax.set_xlabel("Malicious clients (%)"); ax.legend(loc="best", framealpha=0.9)

    if err_axes:                                          # unify the y-range across the error panels
        lo = min(a.get_ylim()[0] for a in err_axes); hi = max(a.get_ylim()[1] for a in err_axes)
        for a in err_axes:
            a.set_ylim(lo, hi)
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("CIFAR-10 / ResNet20, N=100 — breaking point across malicious fraction")
    return _save(fig, out_dir, "fig_frac", fmts)


# ---- Fig 3: learning curves across the WHOLE fraction sweep -----------------------------------
# FACETED: rows = protocol (a row pair per N when several N are given), cols = attack. Each axes
# holds ONLY one protocol's fraction family, every fraction a curve coloured by fraction (sequential
# cmap + colorbar), test error of the untargeted attacks; Scaling has its own figure (fig_scaling).
# Ours-vs-FLTrust is an upper-vs-lower read within a row pair. Every N keeps its own iteration axis,
# so the same horizontal position is the same share of that N's run. The raw underlay is heavy with
# this many lines, so the sweep defaults --raw-alpha to 0 (pass a value to expose per-iteration wobble).
def fig_sweep(groups, fractions, metric, smooth, raw_alpha, out_dir, fmts, cmap):
    """groups: [(N, curves)] in row order."""
    attacks = [a for a in ATTACK_ORDER if a not in ("-", "Scaling")
               and any((p, a, f) in curves for _, curves in groups
                       for p in ("Ours", "FLTrust") for f in fractions if f != 0)]
    if not attacks:
        print("[plot] no curves found for the sweep figure."); return None
    protocols = ["Ours", "FLTrust"]
    multi = len(groups) > 1
    rows = [(n, curves, proto) for n, curves in groups for proto in protocols]
    norm = Normalize(vmin=min(fractions), vmax=max(fractions))
    cmp = plt.get_cmap(cmap)
    fig, axes = plt.subplots(len(rows), len(attacks),
                             figsize=(3.2 * len(attacks), 3.0 * len(rows)),
                             squeeze=False, sharey=True, layout="constrained")
    for r, (n, curves, proto) in enumerate(rows):
        last_of_n = r == len(rows) - 1 or rows[r + 1][0] != n
        for c, atk in enumerate(attacks):
            ax = axes[r][c]
            first = r - protocols.index(proto)            # the first row of this N
            if r != first:
                ax.sharex(axes[first][c])                 # one iteration axis per N
            for f in fractions:
                cc = curves.get((proto, "-", 0)) if f == 0 else curves.get((proto, atk, f))
                if cc:
                    _draw(ax, cc, cmp(norm(f)), "-", smooth, raw_alpha, lw=1.5)
            if r == 0:
                ax.set_title(atk)
            if last_of_n:
                ax.set_xlabel("Global iteration")
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel((f"N = {n} · " if multi else "") + f"{proto}\n{METRIC_LABEL[metric]}")
            else:
                ax.tick_params(labelleft=False)
    sm = ScalarMappable(norm=norm, cmap=cmp); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, ticks=fractions)
    cbar.set_label("Malicious clients (%)")
    fig.suptitle("Learning curves across malicious fraction  (rows: "
                 + ("N · " if multi else "") + "protocol · cols: attack)")
    return _save(fig, out_dir, "fig_sweep" + ("_" + "_".join(f"N{n}" for n, _ in groups) if multi else ""), fmts)


# ---- Scaling: the metric and the backdoor ASR, apart from the untargeted attacks --------------
# Ours discards a scaled submission (verification fails), so its trace is in the metric row (slow or
# no learning as the fraction grows) while ASR stays flat; FLTrust de-scales and keeps the backdoor
# update, so its metric looks normal while ASR climbs. Row 1 = metric, row 2 = ASR; columns =
# (N, Ours), (N, FLTrust) per N; the two columns of an N share their iteration axis.
def fig_scaling(groups, fractions, metric, smooth, raw_alpha, out_dir, fmts, cmap):
    """groups: [(N, curves, asr_curves, clean_curves)] in column order; clean_curves is the Scaling
    runs' metric from load_scaling_clean, used where there is no <tag>.jsonl."""
    protocols = ["Ours", "FLTrust"]
    cols = [(n, curves, asr, clean, proto) for n, curves, asr, clean in groups for proto in protocols]
    if not any((proto, "Scaling", f) in curves or (proto, f) in asr or (proto, f) in clean
               for _, curves, asr, clean, proto in cols for f in fractions):
        print("[plot] no Scaling curves found for the scaling figure."); return None
    if not any((proto, "Scaling", f) in curves or (proto, f) in clean
               for _, curves, _, clean, proto in cols for f in fractions if f != 0):
        print(f"[plot] no {metric} of the Scaling runs: eval_backdoor.py records clean test accuracy only "
              f"(clean_acc), so row 1 holds just the no-attack curve; use --metric test_acc or test_err")
    norm = Normalize(vmin=min(fractions), vmax=max(fractions))
    cmp = plt.get_cmap(cmap)
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 3.0 * 2),
                             squeeze=False, sharey="row", layout="constrained")
    for c, (n, curves, asr, clean, proto) in enumerate(cols):
        first = c - protocols.index(proto)                # the first column of this N
        for r in range(2):
            ax = axes[r][c]
            if r == 1:
                ax.sharex(axes[0][c])
            elif c != first:
                ax.sharex(axes[0][first])                 # one iteration axis per N
            for f in fractions:
                if r == 1:
                    cc = asr.get((proto, f))
                elif f == 0:
                    cc = curves.get((proto, "-", 0))     # the no-attack run
                else:
                    cc = curves.get((proto, "Scaling", f)) or clean.get((proto, f))
                if cc:
                    _draw(ax, cc, cmp(norm(f)), "-", smooth, raw_alpha, lw=1.5)
            if r == 0:
                ax.set_title(f"N = {n} · {proto}")
                ax.tick_params(labelbottom=False)
            else:
                ax.set_ylim(-0.03, 1.03)                 # ASR is a rate in [0,1]
                ax.set_xlabel("Global iteration")
            if c == 0:
                ax.set_ylabel(METRIC_LABEL[metric] if r == 0 else "Backdoor ASR")
            else:
                ax.tick_params(labelleft=False)
    sm = ScalarMappable(norm=norm, cmap=cmp); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, ticks=fractions)
    cbar.set_label("Malicious clients (%)")
    fig.suptitle(f"Scaling attack — {METRIC_LABEL[metric].lower()} (top) and backdoor ASR (bottom)")
    return _save(fig, out_dir, "fig_scaling" + "".join(f"_N{n}" for n, *_ in groups), fmts)


# ---- ASR figure: Scaling backdoor success rate vs malicious fraction --------------------------
def fig_asr(eval_dir, fractions, window, out_dir, fmts, fltrust_ref=0.02):
    """Scaling is a TARGETED backdoor -> measured by ASR (in <tag>_asr.jsonl), not test error, so it
    gets its own figure (this is what compares against FLTrust's Table III). Lower ASR = better."""
    scal = load_asr_curves(eval_dir)            # (proto, rate) -> {iteration: asr}, Scaling only
    if not scal:
        print("[plot] no scaling *_asr.jsonl found — run the ASR eval first (eval_backdoor.py).")
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    for proto, col in (("Ours", OURS_C), ("FLTrust", FT_C)):
        xs, ys = [], []
        for fr in fractions:
            c = scal.get((proto, fr))
            if not c:
                continue
            hi = max(c); vals = [v for it, v in c.items() if hi - window < it <= hi]
            if vals:
                xs.append(fr); ys.append(sum(vals) / len(vals))
        if xs:
            ax.plot(xs, ys, color=col, linestyle=PROTO_STYLE[proto], marker="o", ms=4, lw=1.6, label=proto)
    if fltrust_ref is not None:
        ax.axhline(fltrust_ref, color="0.5", lw=1.0, ls=":", label=f"FLTrust paper ({fltrust_ref:g})")
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("Malicious clients (%)")
    ax.set_ylabel(f"Attack success rate (last {window} it.)")
    ax.set_title("Scaling backdoor — ASR across malicious fraction (lower = better)")
    ax.legend(loc="best", framealpha=0.9)
    return _save(fig, out_dir, "fig_asr", fmts)


def _save(fig, out_dir, stem, fmts):
    if NOTE:
        fig.text(0.995, 0.005, NOTE, ha="right", va="bottom", fontsize=7, color="0.5", style="italic")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in fmts:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    print(f"[plot] wrote {', '.join(str(p) for p in paths)}")
    return paths


def main(argv):
    ap = argparse.ArgumentParser(description="Camera-ready attack figures from eval curves.")
    ap.add_argument("--fig", choices=["iter", "frac", "asr", "sweep", "scaling", "both"], default="both",
                    help="both = iter+frac; sweep = untargeted attacks across ALL fractions; "
                         "scaling = Scaling metric and ASR; asr = Scaling final ASR vs fraction")
    ap.add_argument("--N", type=int, nargs="+", default=[R.N],
                    help="which N's results to plot (default: run_attack_sweep.N). Sets eval-dir/out-dir "
                         "to ./runs/N{N}/logs/eval and ./runs/N{N}/figures unless those are given explicitly. "
                         "Several N (sweep and scaling only): one figure with all of them, in "
                         "./runs/figures unless --out-dir is given.")
    ap.add_argument("--eval-dir", default=None, help="override the per-N eval dir (default: ./runs/N{N}/logs/eval)")
    ap.add_argument("--out-dir", default=None, help="override the per-N figure dir (default: ./runs/N{N}/figures)")
    ap.add_argument("--metric", default="test_err", choices=list(METRIC_LABEL))
    ap.add_argument("--fractions", default=None, help="comma %% list; default 0,20 (iter) / 0,10,20,40,60,80,90,95 (frac/asr)")
    ap.add_argument("--window", type=int, default=100, help="last-N-iteration window for the 'final' value (Fig 2 / ASR)")
    ap.add_argument("--smooth", type=int, default=15,
                    help="centred smoothing window in ITERATIONS for Fig 1 (1 = raw, no smoothing)")
    ap.add_argument("--raw-alpha", type=float, default=None,
                    help="raw-underlay opacity when --smooth>1 (default: 0.35 for iter, 0 for the dense sweep)")
    ap.add_argument("--band", action="store_true", help="shade +/-std over the window in Fig 2")
    ap.add_argument("--ncols", type=int, default=3, help="panels per row (Fig 1 panels / Fig 2 / sweep)")
    ap.add_argument("--cmap", default="viridis", help="sequential colormap for fraction in --fig sweep")
    ap.add_argument("--fmt", default="pdf,png", help="output formats (comma), e.g. pdf,png")
    ap.add_argument("--note", default="", help="optional footer stamp on every figure")
    args = ap.parse_args(argv)

    global NOTE
    NOTE = args.note
    fmts = [x for x in args.fmt.split(",") if x]

    # Several N: the sweep and the scaling figure put them side by side; each N is read from its own
    # ./runs/N{N}/logs/eval, so --eval-dir (a single folder) cannot be given.
    if len(args.N) > 1:
        if args.fig not in ("sweep", "scaling"):
            ap.error("several --N are for --fig sweep or --fig scaling only")
        if args.eval_dir is not None:
            ap.error("--eval-dir takes a single N; with several --N each is read from ./runs/N{N}/logs/eval")
        out_dir = args.out_dir or str(R.RUN_ROOT.parent / "figures")
        fr = [int(x) for x in (args.fractions or "0,10,20,40,60,80,90,95").split(",")]
        raw = 0.0 if args.raw_alpha is None else args.raw_alpha
        dirs = {n: R.RUN_ROOT.parent / f"N{n}" / "logs" / "eval" for n in args.N}
        print(f"[plot] N={' '.join(map(str, args.N))}: eval-dirs={' '.join(map(str, dirs.values()))}  out-dir={out_dir}")
        if args.fig == "sweep":
            fig_sweep([(n, load_curves(dirs[n], args.metric)) for n in args.N], fr, args.metric,
                      args.smooth, raw, out_dir, fmts, args.cmap)
        else:
            fig_scaling([(n, load_curves(dirs[n], args.metric), load_asr_curves(dirs[n]),
                          load_scaling_clean(dirs[n], args.metric)) for n in args.N],
                        fr, args.metric, args.smooth, raw, out_dir, fmts, args.cmap)
        return

    # Resolve per-N paths: same ./runs/N{N} layout as run_attack_sweep (R.RUN_ROOT.parent == ./runs),
    # so plotting reads exactly where that N's sweep wrote. Explicit --eval-dir/--out-dir still win.
    args.N = args.N[0]
    run_root = R.RUN_ROOT.parent / f"N{args.N}"
    if args.eval_dir is None:
        args.eval_dir = str(run_root / "logs" / "eval")
    if args.out_dir is None:
        args.out_dir = str(run_root / "figures")
    print(f"[plot] N={args.N}: eval-dir={args.eval_dir}  out-dir={args.out_dir}")

    if args.fig == "asr":
        fr = [int(x) for x in (args.fractions or "0,10,20,40,60,80,90,95").split(",")]
        fig_asr(args.eval_dir, fr, args.window, args.out_dir, fmts)
        return

    curves = load_curves(args.eval_dir, args.metric)
    print(f"[plot] loaded {len(curves)} curve(s) from {args.eval_dir}")

    if args.fig in ("sweep", "scaling"):
        fr = [int(x) for x in (args.fractions or "0,10,20,40,60,80,90,95").split(",")]
        raw = 0.0 if args.raw_alpha is None else args.raw_alpha
        if args.fig == "sweep":
            fig_sweep([(args.N, curves)], fr, args.metric, args.smooth, raw, args.out_dir, fmts, args.cmap)
        else:
            fig_scaling([(args.N, curves, load_asr_curves(args.eval_dir),
                          load_scaling_clean(args.eval_dir, args.metric))], fr, args.metric, args.smooth,
                        raw, args.out_dir, fmts, args.cmap)
        return

    if not curves:
        print(f"[plot] no accuracy jsonls in {args.eval_dir} — nothing to plot.")
        return

    if args.fig in ("iter", "both"):
        fr = [int(x) for x in (args.fractions or "0,20").split(",")]
        fig_iter(curves, fr, args.metric, args.smooth,
                 0.35 if args.raw_alpha is None else args.raw_alpha, args.out_dir, fmts, args.ncols)
    if args.fig in ("frac", "both"):
        fr = [int(x) for x in (args.fractions or "0,10,20,40,60,80,90,95").split(",")]
        fig_frac(curves, load_asr_curves(args.eval_dir), fr, args.metric, args.window, args.band,
                 args.out_dir, fmts, args.ncols)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])