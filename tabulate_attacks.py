#!/usr/bin/env python3
"""
Build the discussion draft's attack tables from the eval jsonls — paste-ready markdown + LaTeX.

Reads logs/eval/*.jsonl (untargeted: iteration/test_acc/test_err/train_acc) and *_asr.jsonl
(Scaling: iteration/asr/clean_acc), parses each filename into (protocol, attack, malicious-rate)
exactly as inspect_results.py does (strips _peek/_asr, MODE prefix -> Ours/FLTrust, _m<rate>),
prefers a real (non-peek) eval over a _peek one, and emits three tables:

  TABLE A  Experiment A @ EXP_A_RATE% malicious — error vs. iteration.
           Rows: no-attack + {LF,Trim,Adaptive,Krum} x {FLTrust,Ours}. Columns: iteration
           milestones (snapped to the nearest EVALUATED iteration, since checkpoints are saved
           every round but evaluated on a stride, so 100/250/1000 rarely land exactly).
           Each cell is "test-error / train-error" so both metrics are visible for the
           train-vs-test headline decision.

  TABLE B  Experiment B — final test error vs. malicious fraction.
           Rows: fractions actually present (0 = shared no-attack reference). Columns:
           {LF,Trim,Adaptive,Krum}, each split Ours | FLTrust. "final" = last-K-iteration
           average (robust to an end-of-run wobble).

  TABLE C  Scaling — attack success rate (ASR) + clean accuracy vs. fraction, Ours | FLTrust.

Markers:  * = combo still training (last evaluated iter < DONE) or peek-only -> value NOT final;
          - = no run for that cell yet.

Usage:
    python tabulate_attacks.py [eval_dir]                 # default ./logs/eval; prints md + LaTeX
    python tabulate_attacks.py --milestones 100,250,500,750,1000
    python tabulate_attacks.py --exp-a-rate 20
    python tabulate_attacks.py --write ./logs/tables      # also write attack_tables.{md,tex}
    python tabulate_attacks.py --md-only | --tex-only     # print only one format
"""
import argparse
import json
import re
from pathlib import Path

# ---- knobs (kept consistent with inspect_results.py) --------------------------------
TAIL_K   = 5       # last-K evaluated points averaged for a "final" number
DONE     = 950     # last evaluated iter < DONE  ->  still training (value not final), marked '*'
DATASET, MODEL, N = "CIFAR10", "ResNet20", 100
DEFAULT_MILESTONES = [100, 250, 500, 750, 1000]
EXP_A_RATE_DEFAULT = 20

# untargeted attacks shown in Tables A/B (short names as produced by parse_tag), in draft order.
UNTARGETED = ["LF", "Trim", "Adaptive", "Krum"]
NOATTACK   = "-"
SCALING    = "Scaling"

ATTACK_NAME = {"noattack": "-", "label_flip_fltrust": "LF", "trim_fltrust": "Trim",
               "adaptive_fltrust": "Adaptive", "krum_fltrust": "Krum", "scaling_fltrust": "Scaling"}
_M_RE = re.compile(r"_m(\d+)$")


def parse_tag(stem: str):
    """'<MODE>_<attacktag>[_m<rate>]' (trailing _asr/_peek already stripped) ->
    (proto, attack, rate). proto in {'Ours','FLTrust'}; attack short name; rate in %."""
    if stem.startswith("FLTrust_quantized_perp_"):
        proto, rest = "Ours", stem[len("FLTrust_quantized_perp_"):]
    elif stem.startswith("FLTrust_"):
        proto, rest = "FLTrust", stem[len("FLTrust_"):]
    else:
        return None
    m = _M_RE.search(rest)
    rate = int(m.group(1)) if m else 0
    atag = rest[:m.start()] if m else rest
    return proto, ATTACK_NAME.get(atag, atag), rate


def load(path: Path):
    recs = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    recs.sort(key=lambda r: r.get("iteration", 0))
    return recs


def discover(eval_dir: Path):
    """(proto, attack, rate) -> {recs, is_asr, is_peek, last_iter}. Prefer non-peek over peek."""
    combos = {}
    for f in sorted(eval_dir.glob("*.jsonl")):
        stem = f.stem
        is_peek = stem.endswith("_peek")
        core = stem[:-5] if is_peek else stem
        is_asr = core.endswith("_asr")
        run_tag = core[:-4] if is_asr else core
        pt = parse_tag(run_tag)
        if not pt:
            continue
        recs = load(f)
        if not recs:
            continue
        if pt in combos and not combos[pt]["is_peek"] and is_peek:
            continue                       # keep the real eval already loaded
        combos[pt] = {"recs": recs, "is_asr": is_asr, "is_peek": is_peek,
                      "last_iter": recs[-1].get("iteration", 0)}
    return combos


# ---- value access -------------------------------------------------------------------
def by_iter(recs):
    return {r["iteration"]: r for r in recs if "iteration" in r}


def value_at(recs, target, key, tol):
    """Value of `key` at the evaluated iteration nearest `target` within +/- tol, else None."""
    best, bestd = None, None
    for r in recs:
        it = r.get("iteration")
        if it is None or key not in r:
            continue
        d = abs(it - target)
        if d <= tol and (bestd is None or d < bestd):
            best, bestd = r[key], d
    return best


def tail_avg(recs, key):
    xs = [r[key] for r in recs if key in r][-TAIL_K:]
    return sum(xs) / len(xs) if xs else None


def incomplete(d):
    """True if this combo's value is not final (still training below DONE, or peek-only)."""
    return d["is_peek"] or d["last_iter"] < DONE


# ---- cell formatting ----------------------------------------------------------------
def err_cell(acc, mark):
    if acc is None:
        return "-"
    return f"{1.0 - acc:.3f}" + ("*" if mark else "")


def pair_cell(terr_acc, trerr_acc, mark):
    """'test / train' error cell for Table A (train omitted if unavailable)."""
    if terr_acc is None and trerr_acc is None:
        return "-"
    t = f"{1.0 - terr_acc:.3f}" if terr_acc is not None else "-"
    if trerr_acc is None:
        return t + ("*" if mark else "")
    return f"{t} / {1.0 - trerr_acc:.3f}" + ("*" if mark else "")


def num_cell(v, mark, nd=3):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" + ("*" if mark else "")


WINDOW_DEFAULT = 100    # iteration bin width for the mean±std progress / final stats


def window_stats(recs, lo, hi, key):
    """(mean, std, n) of `key` over records with lo < iteration <= hi (population std);
    (None, None, 0) if empty. Robust to eval stride -- uses whatever samples fall in the window."""
    xs = [r[key] for r in recs if key in r and lo < r.get("iteration", -1) <= hi]
    n = len(xs)
    if n == 0:
        return None, None, 0
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / n) ** 0.5
    return mu, sd, n


def ms_err_cell(stats, mark=False):
    """'errμ±σ' from a test_acc window (mean,std,n): err_mean = 1-acc_mean, err_std = acc_std."""
    mu, sd, _ = stats
    if mu is None:
        return "-"
    return f"{1.0 - mu:.3f}±{sd:.3f}" + ("*" if mark else "")


def ms_pair_cell(t_stats, tr_stats, mark=False):
    """'testμ±σ / trainμ±σ' error cell (train omitted if unavailable)."""
    if t_stats[0] is None and tr_stats[0] is None:
        return "-"
    te = f"{1.0 - t_stats[0]:.3f}±{t_stats[1]:.3f}" if t_stats[0] is not None else "-"
    if tr_stats[0] is None:
        return te + ("*" if mark else "")
    return f"{te} / {1.0 - tr_stats[0]:.3f}±{tr_stats[1]:.3f}" + ("*" if mark else "")


# ---- generic renderers --------------------------------------------------------------
def md_table(headers, rows):
    w = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            w[i] = max(w[i], len(c))
    def line(cells):
        return "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"
    out = [line(headers), "|" + "|".join("-" * (w[i] + 2) for i in range(len(headers))) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def latex_table(headers, rows, caption, label, colspec=None, group_header=None, cmidrules=None):
    ncol = len(headers)
    colspec = colspec or ("l" + "c" * (ncol - 1))
    esc = lambda s: (s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
                     .replace("±", r"$\pm$").replace("≤", r"$\le$"))
    out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    if group_header:
        out.append(" & ".join(group_header) + r" \\")
        if cmidrules:
            out.append(" ".join(cmidrules))
    out.append(" & ".join(esc(h) for h in headers) + r" \\")
    out.append(r"\midrule")
    for r in rows:
        out.append(" & ".join(esc(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


# ---- Table A: error vs iteration @ EXP_A_RATE%, binned into WINDOW-iter intervals ----
def build_table_a(combos, rate, window, with_train):
    """Rows = no-attack + attack x {FLTrust,Ours}; cols = contiguous WINDOW-iter bins. Each cell is
    the error mean +/- std over that bin, so an oscillating (breaking) run shows a large std and a
    stable one a small std. Trailing empty bins (run not there yet) render as '-'."""
    picks = [(NOATTACK, "FLTrust", 0), (NOATTACK, "Ours", 0)]
    for atk in UNTARGETED:
        picks += [(atk, "FLTrust", rate), (atk, "Ours", rate)]
    present = [(a, p, r) for (a, p, r) in picks if (p, a, r) in combos]
    if not present:
        return None, None
    max_iter = max((combos[(p, a, r)]["last_iter"] for (a, p, r) in present), default=0)
    if max_iter <= 0:
        return None, None
    nbins = (max_iter + window - 1) // window
    bins = [(i * window, (i + 1) * window) for i in range(nbins)]        # (lo, hi]  (lo < it <= hi)

    headers = ["method"] + [f"≤{hi}" for (lo, hi) in bins]
    rows = []
    for (atk, proto, r) in present:
        recs = combos[(proto, atk, r)]["recs"]
        label = f"{'no-attack' if atk == NOATTACK else atk} ({proto})"
        cells = [label]
        for (lo, hi) in bins:
            ts = window_stats(recs, lo, hi, "test_acc")
            if with_train:
                cells.append(ms_pair_cell(ts, window_stats(recs, lo, hi, "train_acc")))
            else:
                cells.append(ms_err_cell(ts))
        rows.append(cells)
    return headers, rows


# ---- Table B: final test error vs fraction (mean +/- std over the last WINDOW iters) ----
def build_table_b(combos, window):
    rates = sorted({r for (p, a, r) in combos
                    if a in UNTARGETED and r > 0 and not combos[(p, a, r)]["is_asr"]})
    if not rates:
        return None, None, None
    # columns only for attacks that actually have a swept combo (skip e.g. Krum if it was not run)
    attacks = [a for a in UNTARGETED
               if any((p, a, r) in combos for p in ("Ours", "FLTrust") for r in rates)]
    have_noatk = any((p, NOATTACK, 0) in combos for p in ("Ours", "FLTrust"))
    frac_rows = ([0] if have_noatk else []) + rates

    headers = ["frac"] + [f"{a}·{p}" for a in attacks for p in ("Ours", "FLTrust")]
    group_header = [""] + [rf"\multicolumn{{2}}{{c}}{{{a}}}" for a in attacks]
    cmidrules = [rf"\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(attacks))]
    tex_headers = ["frac"] + ["Ours", "FLTrust"] * len(attacks)
    colspec = "l" + "cc" * len(attacks)

    def final_cell(d):
        if not d:
            return "-"
        hi = d["last_iter"]
        return ms_err_cell(window_stats(d["recs"], hi - window, hi, "test_acc"), incomplete(d))

    rows = []
    for fr in frac_rows:
        cells = ["no-attack" if fr == 0 else f"{fr}%"]
        for atk in attacks:
            for proto in ("Ours", "FLTrust"):
                key = (proto, NOATTACK, 0) if fr == 0 else (proto, atk, fr)
                cells.append(final_cell(combos.get(key)))
        rows.append(cells)
    return (headers, rows), (tex_headers, rows, group_header, cmidrules, colspec), frac_rows


# ---- Table C: Scaling ASR vs fraction -----------------------------------------------
def build_table_c(combos):
    rates = sorted({r for (p, a, r) in combos if a == SCALING and combos[(p, a, r)]["is_asr"]})
    if not rates:
        return None, None
    headers = ["frac", "Ours ASR", "FLTrust ASR", "Ours clean-acc", "FLTrust clean-acc"]
    rows = []
    for fr in rates:
        do, df = combos.get(("Ours", SCALING, fr)), combos.get(("FLTrust", SCALING, fr))
        rows.append([
            f"{fr}%",
            num_cell(tail_avg(do["recs"], "asr"), incomplete(do)) if do else "-",
            num_cell(tail_avg(df["recs"], "asr"), incomplete(df)) if df else "-",
            num_cell(tail_avg(do["recs"], "clean_acc"), incomplete(do)) if do else "-",
            num_cell(tail_avg(df["recs"], "clean_acc"), incomplete(df)) if df else "-",
        ])
    return headers, rows


# ---- coverage summary ---------------------------------------------------------------
def coverage(combos):
    done = sum(1 for d in combos.values() if not incomplete(d))
    peek = sum(1 for d in combos.values() if d["is_peek"])
    return f"{len(combos)} combo(s) evaluated | {done} final, {len(combos)-done} still training/peek ({peek} peek)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", nargs="?", default="./logs/eval")
    ap.add_argument("--exp-a-rate", type=int, default=EXP_A_RATE_DEFAULT)
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT,
                    help="iteration bin width for the mean±std stats (default 100).")
    ap.add_argument("--with-train", action="store_true",
                    help="Table A cells show test AND train error (μ±σ each); default is test only.")
    ap.add_argument("--write", default=None, help="dir to also write attack_tables.{md,tex}")
    ap.add_argument("--md-only", action="store_true")
    ap.add_argument("--tex-only", action="store_true")
    args = ap.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"[tabulate] no eval dir: {eval_dir.resolve()}")
        return 1
    combos = discover(eval_dir)
    if not combos:
        print(f"[tabulate] no parseable eval jsonls under {eval_dir.resolve()}")
        return 0
    a = build_table_a(combos, args.exp_a_rate, args.window, args.with_train)
    b_md, b_tex, _ = build_table_b(combos, args.window)
    c = build_table_c(combos)

    md_parts, tex_parts = [f"<!-- {coverage(combos)} -->"], []

    # Table A
    if a[0]:
        headers, rows = a
        metric = "test / train error" if args.with_train else "test error"
        cap_a = (f"Experiment A — {metric} vs. iteration at {args.exp_a_rate}\\% malicious "
                 f"({DATASET}/{MODEL}, N={N}). Each cell: mean $\\pm$ std over a {args.window}-iteration "
                 f"window; a large std flags an unstable (oscillating) run.")
        md_parts += [f"### Table A — {metric} vs. iteration @ {args.exp_a_rate}% malicious",
                     f"_cell = {metric} as μ±σ over each {args.window}-iter bin; large σ = unstable; "
                     f"`-` = not reached yet._",
                     md_table(headers, rows)]
        tex_parts.append(latex_table(headers, rows, cap_a, "tab:expA"))
    else:
        md_parts.append(f"### Table A — (no {args.exp_a_rate}% combos evaluated yet)")

    # Table B
    if b_md:
        headers, rows = b_md
        th, trows, gh, cmid, cspec = b_tex
        cap_b = (f"Experiment B — final test error (mean $\\pm$ std over the last {args.window} "
                 f"iterations) vs. malicious fraction ({DATASET}/{MODEL}, N={N}). A large std marks an "
                 f"unstable/oscillating run; 0\\% row = shared no-attack reference.")
        md_parts += ["### Table B — final test error vs. malicious fraction",
                     f"_final = μ±σ over the last {args.window} iters; large σ = unstable; `*` = still "
                     f"training; `-` = no run; 0% row = shared no-attack reference._",
                     md_table(headers, rows)]
        tex_parts.append(latex_table(th, trows, cap_b, "tab:expB",
                                     colspec=cspec, group_header=gh, cmidrules=cmid))
    else:
        md_parts.append("### Table B — (no fraction sweep evaluated yet)")

    # Table C
    if c[0]:
        headers, rows = c
        cap_c = (f"Scaling attack — attack success rate and clean accuracy (last-{TAIL_K}-iter avg) "
                 f"vs. malicious fraction ({DATASET}/{MODEL}, N={N}).")
        md_parts += ["### Table C — Scaling ASR vs. malicious fraction",
                     "_ASR/clean = last-5-iter avg; `*` = still training; `-` = no run._",
                     md_table(headers, rows)]
        tex_parts.append(latex_table(headers, rows, cap_c, "tab:scaling"))
    else:
        md_parts.append("### Table C — (no Scaling ASR evaluated yet)")

    md = "\n\n".join(md_parts)
    tex = "\n\n".join(tex_parts)
    if not args.tex_only:
        print(md)
    if not args.md_only:
        print("\n\n% ===================== LaTeX =====================\n")
        print(tex)

    if args.write:
        d = Path(args.write)
        d.mkdir(parents=True, exist_ok=True)
        (d / "attack_tables.md").write_text(md + "\n", encoding="utf-8")
        (d / "attack_tables.tex").write_text(tex + "\n", encoding="utf-8")
        print(f"\n[tabulate] wrote {d/'attack_tables.md'} and {d/'attack_tables.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())