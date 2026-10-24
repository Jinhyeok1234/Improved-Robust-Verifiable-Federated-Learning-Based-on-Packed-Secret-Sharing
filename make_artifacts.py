#!/usr/bin/env python3
"""The paper's tables and figures, from the point folders run.py writes.

    python make_artifacts.py tables                          # every table, LaTeX, from runs/
    python make_artifacts.py tables --format md csv          # the same as Markdown and CSV
    python make_artifacts.py tables --only cost_n brea       # some of them
    python make_artifacts.py figures --fmt pdf png           # every figure
    python make_artifacts.py tables --runs results           # from the published results

Every artifact is written to --out (default artifacts/) as <name>.<ext>. A point enters only
when it finished (status "ok"); the cost and end-to-end figures leave out the first
--drop-first rounds (default 1: the one-time connection set-up and first allocations), unless
a point has no more rounds than that. Which points an artifact reads is given below: the tables
take any point under --runs with those settings, the figures the points of the experiments named
"accuracy" and "attacks" (--accuracy-name, --attacks-name), since both have runs without attack.

Tables
  cost_n          cost, rvfl, k1 = N // 3: bytes sent and computation per phase, by N
  cost_k1         cost, rvfl, N = 50 and N = 100: the same over k1
  brea            cost, k1 = 1, N = 10..40: RVFL next to BREA. BREA logs one phase; its timers
                  are put into the phases as the paper states it (BREA_PHASES)
  e2e_lan         e2e, filter off: per-party computation and communication and the round time,
                  on the 1 Gbps and 10 Gbps links (0.2 ms)
  e2e_wan         the same on the 10, 50 and 100 Mbps links (80 ms)
  comm_breakdown  e2e, filter off: communication time by message (COMM_ROWS) for the
                  --breakdown-points columns, LAN 1 Gbps and WAN 10 Mbps side by side
  curves          curve_bench at N = --curves-n: precision bounds, share size and the time of
                  the first-level sharing and of the Pedersen verification relative to P-256

Figures
  accuracy        train, no attack, N = 10..50: training and test accuracy of RVFL (solid) and
                  FLTrust (dashed), each run up to the evaluation where its training accuracy
                  reaches --accuracy-stop (84 %)
  cosine          train, rvfl, no attack: mean cosine with g_S and share of non-positive
                  cosines per round (adaptive moving average), up to the same evaluation
  attack_sweep    train, N = 30 and 100: test error over rounds for LF, Trim, Adaptive and Krum,
                  one curve per malicious fraction; rows RVFL / FLTrust per N
  scaling         train, Scaling attack: test accuracy (top) and attack success rate (bottom)
  filter          e2e with the participant filter: round time per round, one panel per filter
                  mode, filter-off points of the same N and link as reference lines
                  (--filter-link, --filter-metric, --ref "name=seconds")
"""
import argparse
import csv
import io
import statistics
import sys
from pathlib import Path

from rvfl import logs

# BREA's computation timers in the phases of the paper: Enroll = dealing the updates with
# Pedersen VSS and verifying them; Verify = pairwise distances and Multi-Krum; Agg = aggregating
# the selected clients' updates.
BREA_PHASES = {
    "enrollment": ("calc_share", "verification"),
    "verification": ("calc_pairwise_distance", "reconstruct_d", "dequantize_d", "multi_krum"),
    "aggregation": ("aggregate_shares", "reconstruct_agg", "dequantize_and_update"),
}


# --- tables ----------------------------------------------------------------------------------

class Table:
    """A table with a grouped header.

    header: rows of (text, span) above the column names; columns: (name, unit, fmt) per
    column, fmt turning a cell value into text ("" for None); rows: lists of values; groups:
    the row indices where a block starts (a rule above it, and the first column spanning the
    block in LaTeX)."""

    def __init__(self, name, columns, rows, header=(), groups=(), align=None, caption=None):
        self.name, self.columns, self.rows = name, columns, rows
        self.header, self.groups = list(header), list(groups)
        self.align = align or "".join("l" if i == 0 else "r" for i in range(len(columns)))
        self.caption = caption

    def cells(self, row):
        return [fmt(v) for (_, _, fmt), v in zip(self.columns, row)]

    def flat_names(self):
        names = []
        spans = [[] for _ in self.header]
        for h, hrow in enumerate(self.header):
            for text, span in hrow:
                spans[h] += [text] * span
        for i, (name, unit, _) in enumerate(self.columns):
            parts = [spans[h][i] for h in range(len(self.header)) if spans[h][i]]
            names.append(" ".join(parts + [name] + ([f"({unit})"] if unit else [])))
        return names

    def latex(self):
        out = [f"% {self.caption}" if self.caption else None, f"\\begin{{tabular}}{{{self.align}}}",
               "\\toprule"]
        for hrow in self.header:
            cells, rules, col = [], [], 1
            for text, span in hrow:
                cells.append(f"\\multicolumn{{{span}}}{{c}}{{{text}}}" if span > 1 else text)
                if text and span > 1:
                    rules.append(f"\\cmidrule(lr){{{col}-{col + span - 1}}}")
                col += span
            out.append(" & ".join(cells) + " \\\\")
            if rules:
                out.append(" ".join(rules))
        out.append(" & ".join(name for name, _, _ in self.columns) + " \\\\")
        if any(unit for _, unit, _ in self.columns):
            out.append(" & ".join(f"({unit})" if unit else "" for _, unit, _ in self.columns) + " \\\\")
        out.append("\\midrule")
        starts = sorted(set(self.groups) | {0})
        for i, row in enumerate(self.rows):
            cells = self.cells(row)
            if self.groups:
                if i in starts and i:
                    out.append("\\midrule")
                end = next((s for s in starts if s > i), len(self.rows))
                start = max(s for s in starts if s <= i)
                cells[0] = (f"\\multirow{{{end - start}}}{{*}}{{{cells[0]}}}" if end - start > 1 else cells[0]) \
                    if i == start else ""
            out.append(" & ".join(c.replace("%", "\\%") for c in cells) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(x for x in out if x is not None) + "\n"

    def markdown(self):
        names = self.flat_names()
        rows = [self.cells(r) for r in self.rows]
        widths = [max(len(n), *(len(r[i]) for r in rows)) if rows else len(n) for i, n in enumerate(names)]
        line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
        out = ([f"{self.caption}", ""] if self.caption else []) + [line(names), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        out += [line(r) for r in rows]
        return "\n".join(out) + "\n"

    def csv(self):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.flat_names())
        for row in self.rows:
            w.writerow([getattr(f, "csv", f)(v) for (_, _, f), v in zip(self.columns, row)])
        return buf.getvalue()


def f_num(digits, scale=1.0, dash_zero=False):
    """A cell formatter: the value divided by scale with `digits` decimals and thousands
    separators ("-" for none, and for 0 with dash_zero); fmt.csv gives the plain number."""
    def fmt(v):
        if v is None or v == "":
            return "-"
        if isinstance(v, str):
            return v
        if dash_zero and float(v) == 0:
            return "-"
        return f"{float(v) / scale:,.{digits}f}"

    def plain(v):
        if v is None or v == "" or (dash_zero and not isinstance(v, str) and float(v) == 0):
            return ""
        return v if isinstance(v, str) else f"{float(v) / scale:.{digits}f}"
    fmt.csv = plain
    return fmt


def f_text(v):
    return "" if v is None else str(v)


def done(points):
    return [p for p in points if logs.read_status(p).get("status") == "ok"]


def cost_points(runs, **settings):
    """{(n, k1): folder} of the finished cost points with these settings (filter off, t = n // 3)."""
    out = {}
    for d in done(logs.find_points(runs, kind="cost", filter="none", t=None, cos_reveal="verify",
                                   attack=None, **settings)):
        cfg = logs.read_config(d)
        key = (cfg["n"], cfg["k1"])
        if key in out:
            raise SystemExit(f"two finished cost points for N={key[0]}, k1={key[1]}: {out[key]} and {d}")
        out[key] = d
    return out


def cost_columns(client_bcast_unit="KB", digits=1, total_digits=1):
    scale = {"KB": 1e3, "MB": 1e6}[client_bcast_unit]
    ms = f_num(digits)
    cols = []
    for party, bunit, bscale in (("server", "MB", 1e6), ("client", client_bcast_unit, scale)):
        cols += [("Broadcast", bunit, f_num(2, bscale, dash_zero=True)),
                 ("Unicast", "MB", f_num(2, 1e6, dash_zero=True)),
                 ("Enroll", "ms", ms), ("Verify", "ms", ms), ("Agg", "ms", ms),
                 ("Total", "ms", f_num(total_digits))]
    return cols


def cost_cells(summary):
    row = []
    for party in ("server", "client"):
        row += [summary.get(f"{party}_broadcast_bytes"), summary.get(f"{party}_unicast_bytes")]
        row += [summary.get(f"{party}_comp_{ph}_ms") for ph in ("enrollment", "verification", "aggregation")]
        row.append(summary.get(f"{party}_comp_total_ms"))
    return row


COST_HEADER = [[("", 1), ("Server", 6), ("Client", 6)],
               [("", 1), ("Communication", 2), ("Computation", 4), ("Communication", 2), ("Computation", 4)]]


def table_cost_n(args):
    pts = cost_points(args.runs, protocol="rvfl")
    rows = []
    for (n, k1), d in sorted(pts.items()):
        if k1 != n // 3:
            continue
        rows.append([n] + cost_cells(logs.cost_summary(d, args.drop_first)))
    return Table("cost_n", [("$N$", "", f_text)] + cost_columns("KB"), rows, header=COST_HEADER,
                 caption="Communication volume and computation time per round, k1 = t1 = N // 3")


def table_cost_k1(args):
    pts = cost_points(args.runs, protocol="rvfl")
    rows, groups = [], []
    for spec in args.k1_table:
        n, ks = spec.split(":")
        n = int(n)
        ks = [int(k) for k in ks.split(",") if (n, int(k)) in pts]
        if ks:
            groups.append(len(rows))
        for k in ks:
            rows.append([n, k] + cost_cells(logs.cost_summary(pts[(n, k)], args.drop_first)))
    header = [[("", 2)] + h[1:] for h in COST_HEADER]
    return Table("cost_k1", [("$N$", "", f_text), ("$k_1$", "", f_text)] + cost_columns("KB"), rows,
                 header=header, groups=groups, align="cc" + "r" * 12,
                 caption="Communication volume and computation time per round over k1")


def table_brea(args):
    ours = cost_points(args.runs, protocol="rvfl")
    brea = cost_points(args.runs, protocol="brea")
    rows, groups = [], []
    for n in sorted({n for n, _ in brea} | {n for n, k in ours if k == 1}):
        if (n, 1) not in ours or (n, 1) not in brea:
            continue
        groups.append(len(rows))
        rows.append([n, "Ours"] + cost_cells(logs.cost_summary(ours[(n, 1)], args.drop_first)))
        rows.append([n, "BREA"] + cost_cells(logs.cost_summary(brea[(n, 1)], args.drop_first, BREA_PHASES)))
    header = [[("", 2)] + h[1:] for h in COST_HEADER]
    return Table("brea", [("$N$", "", f_text), ("Scheme", "", f_text)] + cost_columns("MB", 2, 2), rows,
                 header=header, groups=groups, align="cc" + "r" * 12,
                 caption="RVFL (k1 = 1) and BREA: communication volume and computation time per round")


def link_text(label):
    import re
    m = re.match(r"^(\d+)(Gbps|Mbps)_([\d.]+)ms$", label)
    if not m:
        return label
    return f"{m.group(1)} {m.group(2)}, {m.group(3)} ms RTT"


def e2e_points(runs, link):
    out = {}
    for d in done(logs.find_points(runs, kind="e2e", filter="none", t=None, cos_reveal="verify",
                                   attack=None, link=link)):
        cfg = logs.read_config(d)
        key = (cfg["n"], cfg["k1"])
        if key in out:
            raise SystemExit(f"two finished e2e points for N={key[0]}, k1={key[1]} on {link}: {out[key]} and {d}")
        out[key] = d
    return out


def table_e2e(args, name, links, kind):
    per_link = [(lk, e2e_points(args.runs, lk)) for lk in links]
    keys = sorted(set().union(*[set(p) for _, p in per_link])) if per_link else []
    if not args.partial_rows:
        # an (N, k1) enters only when it finished on every link of the table
        keys = [k for k in keys if all(k in pts for _, pts in per_link)]
    rows, groups, last_n = [], [], None
    for n, k1 in keys:
        if n != last_n:
            groups.append(len(rows))
            last_n = n
        row = [n, k1]
        for _, pts in per_link:
            s = logs.e2e_summary(pts[(n, k1)], args.drop_first) if (n, k1) in pts else None
            row += [s[k] if s else None for k in ("s_comp", "s_comm", "c_comp", "c_comm", "round")]
        rows.append(row)
    sec = f_num(2)
    cols = [("$N$", "", f_text), ("$k_1$", "", f_text)]
    for _ in links:
        cols += [("comp", "", sec), ("comm", "", sec), ("comp", "", sec), ("comm", "", sec), ("round", "", sec)]
    header = [[("", 2)] + [(f"{kind} ({link_text(logs.link_label(lk))})", 5) for lk in links],
              [("", 2)] + [x for _ in links for x in (("Server", 2), ("Client", 2), ("", 1))]]
    return Table(name, cols, rows, header=header, groups=groups, align="cc" + "r" * 5 * len(links),
                 caption=f"End-to-end time per round (s), {kind}")


def table_e2e_lan(args):
    return table_e2e(args, "e2e_lan", ["1000:0.2", "10000:0.2"], "LAN")


def table_e2e_wan(args):
    return table_e2e(args, "e2e_wan", ["10:80", "50:80", "100:80"], "WAN")


# The communication-breakdown rows. Each leaf is an exchange, timed from the barrier where it
# starts until the last party is done, a round at a time, then averaged over the rounds:
#   S -> C and C <-> C   the slowest client: the largest sum of the "C:" timers over the clients
#                        (for C <-> C its send and its receive of the message)
#   C -> S               the server's receive ("S:"), until the last client's message is in; a
#                        client's send returns once the bytes are in its socket buffer
# FedAvg: what plain FedAvg sends anyway (the global model; the clients' updates go up as shares
# here). Secret sharing: what sharing the updates adds. Ours-specific: everything else.
COMM_LEAVES = {
    "global": ["C:receive_global_weights"],
    "first": ["C:send_first_level_share", "C:receive_first_level_share"],
    "agg": ["S:receive_agg_share"],
    "gs": ["C:receive_server"],
    "second": ["C:send_second_level_share", "C:receive_second_level_share"],
    "verdict": ["C:send_verify_result", "C:receive_verify_result"],
    "stilde": ["S:receive_s_tilde"],
    "ts": ["C:receive_TS"],
    "cos": ["S:receive_cos"],
    "plist": ["C:receive_participant_list"],
}
COMM_ROWS = [  # (label, leaves or None for a heading, bold, italic, shown only when present)
    ("FedAvg", None, True, False, False),
    ("global model (S $\\to$ C)", ("global",), False, False, False),
    ("Secret sharing (FedAvg + SS)", None, True, False, False),
    ("first-level share (C $\\leftrightarrow$ C)", ("first",), False, False, False),
    ("aggregated share (C $\\to$ S)", ("agg",), False, False, False),
    ("SS subtotal", ("first", "agg"), False, True, False),
    ("Ours-specific", None, True, False, False),
    ("$g_S$ (S $\\to$ C)", ("gs",), False, False, False),
    ("second-level share + commitments (C $\\leftrightarrow$ C)", ("second",), False, False, False),
    ("verdict (C $\\leftrightarrow$ C)", ("verdict",), False, False, False),
    ("$\\tilde{s}$ (C $\\to$ S)", ("stilde",), False, False, False),
    ("TS (S $\\to$ C)", ("ts",), False, False, False),
    ("cosine (C $\\to$ S)", ("cos",), False, False, True),
    ("participant list (S $\\to$ C)", ("plist",), False, False, True),
    ("Ours-subtotal", ("gs", "second", "verdict", "stilde", "ts", "cos", "plist"), False, True, False),
    ("Total communication", tuple(COMM_LEAVES), True, False, False),
]
# the other end of a placed exchange, not counted again
COMM_OTHER_END = {"C:send_agg_share", "C:send_s_tilde", "C:send_cos", "S:send_server_data",
                  "S:send_TS", "S:send_updated_global_weights", "S:send_participant_list"}


def table_comm_breakdown(args):
    links = [("LAN", "1000:0.2"), ("WAN", "10:80")]
    cols_nk = [tuple(int(x) for x in p.split(",")) for p in args.breakdown_points]
    data = {}
    for _, lk in links:
        pts = e2e_points(args.runs, lk)
        for nk in cols_nk:
            data[(lk, nk)] = logs.comm_timers(pts[nk], args.drop_first) if nk in pts else None
    present = set().union(*[d["present"] for d in data.values() if d])
    placed = {m for ms in COMM_LEAVES.values() for m in ms}
    unplaced = sorted(present - placed - COMM_OTHER_END)
    if unplaced:
        raise SystemExit(f"comm_breakdown: timers in no row: {unplaced}")
    rows = []
    for label, leaves, bold, italic, optional in COMM_ROWS:
        if optional and not any(m in present for k in leaves for m in COMM_LEAVES[k]):
            continue
        text = f"\\textbf{{{label}}}" if bold else (f"\\quad \\textit{{{label}}}" if italic else f"\\quad {label}")
        if leaves is None:
            rows.append([text] + [None] * (len(links) * len(cols_nk)))
            continue
        cells = []
        for _, lk in links:
            for nk in cols_nk:
                d = data[(lk, nk)]
                cells.append(sum(logs.exchange_seconds(d, COMM_LEAVES[k]) for k in leaves) if d else None)
        rows.append([text] + cells)
    sec = lambda v: "" if v is None else f"{v:.3f}"
    cols = [("Component", "", f_text)] + [(f"$({n},{k})$", "", sec) for _ in links for n, k in cols_nk]
    header = [[("", 1)] + [(f"{name} ({link_text(logs.link_label(lk))})", len(cols_nk)) for name, lk in links]]
    return Table("comm_breakdown", cols, rows, header=header, align="l" + "r" * len(links) * len(cols_nk),
                 caption="Communication time per round (s) by message")


def table_curves(args):
    from rvfl.crypto_params import CURVES
    pts = done(logs.find_points(args.runs, kind="curve_bench", n=args.curves_n))
    by_level = {}
    for d in pts:
        r = logs.read_json(Path(d) / "result.json")
        by_level.setdefault(r["level"], []).append(r)
    if "128" not in by_level:
        raise SystemExit(f"curves: no P-256 point at N={args.curves_n} under {args.runs}")
    t = args.curves_n // 3

    def keyed(level, key, every_k1=False):
        # k1 = t leaves no random coefficient in the first-level shares, so those rows are left
        # out of the share size and the sharing time; the Pedersen verification of the second-level
        # shares does not depend on k1 and takes every row
        return {r["k1"]: r[key] for r in by_level[level]
                if (every_k1 or r["k1"] < t) and r.get(key) is not None}

    base_share = statistics.mean(keyed("128", "share_bytes_per_elem").values())
    base_pss = keyed("128", "pss_share_s")
    base_ver = statistics.mean(keyed("128", "pedersen_verify_s", True).values())
    rows = []
    for level in sorted(by_level, key=int):
        rs = by_level[level]
        r0 = rs[0]
        share = statistics.mean(keyed(level, "share_bytes_per_elem").values())
        pss = keyed(level, "pss_share_s")
        ratios = [pss[k] / base_pss[k] for k in pss if k in base_pss]
        ver = statistics.mean(keyed(level, "pedersen_verify_s", True).values()) / base_ver
        lo, hi = min(ratios), max(ratios)
        min_exact = r0["log2_min_exact_entry"]
        rows.append([f"NIST {CURVES[level]}", f"{r0['q_bits']} bits ({level}-bit)",
                     f"$\\approx 2^{{{r0['log2_mult_max']:.1f}}}$", f"$\\approx 2^{{{r0['log2_rel_error']:.1f}}}$",
                     "$2^{-149}$" if min_exact <= -149 else f"$\\approx 2^{{{min_exact:.1f}}}$",
                     f"{share:.1f} B (${share / base_share:.2f}\\times$)",
                     f"${lo:.2f}\\times$" if abs(hi - lo) < 0.005 else f"${lo:.2f}$--${hi:.2f}\\times$",
                     f"${ver:.2f}\\times$"])
    cols = [("Curve", "", f_text), ("$|q|$ (Security)", "", f_text), ("$mult$", "", f_text),
            ("$\\varepsilon_{\\mathrm{rel}}$", "", f_text), ("$x_{\\min}$", "", f_text),
            ("Share Size / Elem.", "", f_text), ("PSS Sharing", "", f_text),
            ("Pedersen Verification", "", f_text)]
    return Table("curves", cols, rows, align="lccccccc",
                 caption=f"Elliptic-curve parameters at N = {args.curves_n} (times relative to P-256; "
                         f"PSS sharing as the range over k1 < {t})")


TABLES = {"cost_n": table_cost_n, "cost_k1": table_cost_k1, "brea": table_brea,
          "e2e_lan": table_e2e_lan, "e2e_wan": table_e2e_wan, "comm_breakdown": table_comm_breakdown,
          "curves": table_curves}


def plain(text):
    """LaTeX markup to plain text, for Markdown and CSV."""
    import re
    for a, b in (("$\\to$", "->"), ("$\\leftrightarrow$", "<->"), ("\\quad ", "  "), ("$\\approx ", "~"),
                 ("\\times$", "x"), ("$\\tilde{s}$", "s~"), ("$g_S$", "g_S"), ("$|q|$", "|q|"),
                 ("$\\varepsilon_{\\mathrm{rel}}$", "eps_rel"), ("$x_{\\min}$", "x_min"), ("$k_1$", "k1"),
                 ("$N$", "N"), ("$mult$", "mult")):
        text = text.replace(a, b)
    text = re.sub(r"\\text(?:bf|it)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"2\^\{([^}]*)\}", r"2^\1", text)
    return text.replace("$", "").replace("--", "-")


def write_table(table, fmt_name, out):
    if fmt_name == "latex":
        text, ext = table.latex(), "tex"
    else:
        def wrap(f):
            g = lambda v: plain(f(v))
            if hasattr(f, "csv"):
                g.csv = f.csv
            return g
        cols = [(plain(n), u, wrap(f)) for n, u, f in table.columns]
        rows = [[plain(v) if isinstance(v, str) else v for v in r] for r in table.rows]
        t = Table(table.name, cols, rows, [[(plain(x), s) for x, s in h] for h in table.header],
                  table.groups, table.align, table.caption)
        text, ext = (t.markdown(), "md") if fmt_name == "md" else (t.csv(), "csv")
    path = Path(out) / f"{table.name}.{ext}"
    path.write_text(text, encoding="utf-8")
    return path


# --- figures ---------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# the style of the attack figures
ATTACK_RC = {"font.family": "serif", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
             "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
             "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight"}


def save(fig, args, stem):
    import matplotlib.pyplot as plt
    paths = []
    for ext in args.fmt:
        p = Path(args.out) / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def train_points(runs, name, **settings):
    """{(n, protocol, attack, malicious %): folder} of the finished train points of the
    experiment `name` (the accuracy and the attack experiments both have runs without attack)."""
    out = {}
    for d in done(logs.find_points(runs, kind="train", name=name, **settings)):
        cfg = logs.read_config(d)
        rate = int(round(cfg["malicious_rate"] * 100)) if cfg["attack"] else 0
        key = (cfg["n"], cfg["protocol"], cfg["attack"], rate)
        if key in out:
            raise SystemExit(f"two finished train points of {name} for {key}: {out[key]} and {d}")
        out[key] = d
    return out


def eval_curve(d, key):
    return {r["iteration"]: r[key] for r in logs.read_jsonl(Path(d) / "eval.jsonl") if r.get(key) is not None}


def stop_iteration(d, threshold):
    """The first evaluation at which the training accuracy reaches threshold (None: never)."""
    for r in logs.read_jsonl(Path(d) / "eval.jsonl"):
        if r.get("train_acc") is not None and r["train_acc"] >= threshold:
            return r["iteration"]
    return None


def fig_accuracy(args):
    plt = _plt()
    pts = train_points(args.runs, args.accuracy_name, attack=None)
    ns = sorted({n for n, *_ in pts})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, key, title in ((axes[0][0], "train_acc", "Train accuracy"), (axes[0][1], "test_acc", "Test accuracy")):
        for i, n in enumerate(ns):
            for proto, name, ls in (("rvfl", "Ours", "-"), ("fltrust", "FLTrust", "--")):
                d = pts.get((n, proto, None, 0))
                if d is None:
                    continue
                stop = stop_iteration(d, args.accuracy_stop)
                curve = {it: v for it, v in eval_curve(d, key).items() if stop is None or it <= stop}
                xs = sorted(curve)
                ax.plot(xs, [100 * curve[x] for x in xs], linestyle=ls, color=colors[i % len(colors)],
                        label=f"{name} (N={n})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    return save(fig, args, "accuracy")


def _adaptive_ema(values, its, early=10, late=50, transition=50):
    out, ema = [], None
    for v, it in zip(values, its):
        if ema is None:
            ema = v
        else:
            span = early + (late - early) * it / transition if it < transition else late
            a = 2.0 / (span + 1.0)
            ema = a * v + (1 - a) * ema
        out.append(ema)
    return out


def fig_cosine(args):
    plt = _plt()
    paths = []
    for (n, proto, attack, _), d in sorted(train_points(args.runs, args.accuracy_name, attack=None, protocol="rvfl").items()):
        stop = stop_iteration(d, args.accuracy_stop)
        its, means, ratios = [], [], []
        for r in logs.read_jsonl(Path(d) / "values.jsonl"):
            if stop is not None and r["iteration"] > stop:
                break
            cos = [float(c) for c in (r["values"].get("cos") or {}).values()]
            its.append(r["iteration"])
            means.append(sum(cos) / len(cos) if cos else 0.0)
            ratios.append(sum(1 for c in cos if c <= 0) / len(cos) if cos else 0.0)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        for ax, ys, raw_c, ema_c, title, ylabel in (
                (ax1, means, "green", "blue", f"Cosine Similarity Mean Trend (N={n})", "Cos Mean"),
                (ax2, ratios, "red", "darkred", f"Cosine Similarity Ratio (<=0) Trend (N={n})", "Ratio (<=0)")):
            ax.plot(its, ys, label="Raw Data", color=raw_c, alpha=0.2)
            ax.plot(its, _adaptive_ema(ys, its), label="Adaptive EMA", color=ema_c, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.legend()
        fig.tight_layout()
        paths += save(fig, args, f"cosine_N{n}")
    return paths


ATTACK_LABEL = {"label_flip_fltrust": "LF", "trim_fltrust": "Trim", "adaptive_fltrust": "Adaptive",
                "krum_fltrust": "Krum", "scaling_fltrust": "Scaling"}
PROTOCOLS = (("rvfl", "Ours"), ("fltrust", "FLTrust"))


def _smooth(xs, ys, w):
    """Centred rolling mean over a window of w iterations."""
    if w <= 1 or len(ys) < 2:
        return ys
    half, out, lo, hi, n = w / 2.0, [], 0, 0, len(xs)
    for i in range(n):
        while lo < n and xs[lo] < xs[i] - half:
            lo += 1
        while hi < n and xs[hi] <= xs[i] + half:
            hi += 1
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def _draw(ax, curve, color, smooth):
    xs = sorted(curve)
    ax.plot(xs, _smooth(xs, [curve[x] for x in xs], smooth), color=color, linestyle="-", lw=1.5)


def _fraction_colors(plt, fractions, cmap_name):
    from matplotlib.colors import Normalize
    return Normalize(vmin=min(fractions), vmax=max(fractions)), plt.get_cmap(cmap_name)


def _colorbar(plt, fig, axes, norm, cmap, fractions):
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, ticks=fractions)
    cbar.set_label("Malicious clients (%)")


def fig_attack_sweep(args):
    plt = _plt()
    with plt.rc_context(ATTACK_RC):
        return _fig_attack_sweep(args, plt)


def _fig_attack_sweep(args, plt):
    pts = train_points(args.runs, args.attacks_name)
    ns = [n for n in args.attack_n if any(k[0] == n for k in pts)]
    attacks = [a for a in ATTACK_LABEL if a != "scaling_fltrust" and any(k[2] == a for k in pts)]
    if not ns or not attacks:
        raise SystemExit("attack_sweep: no train points with the untargeted attacks")
    fractions = args.fractions
    norm, cmap = _fraction_colors(plt, fractions, args.cmap)
    rows = [(n, proto, name) for n in ns for proto, name in PROTOCOLS]
    fig, axes = plt.subplots(len(rows), len(attacks), figsize=(3.2 * len(attacks), 3.0 * len(rows)),
                             squeeze=False, sharey=True, layout="constrained")
    for r, (n, proto, name) in enumerate(rows):
        last_of_n = r == len(rows) - 1 or rows[r + 1][0] != n
        first = r - [p for p, _ in PROTOCOLS].index(proto)
        for c, atk in enumerate(attacks):
            ax = axes[r][c]
            if r != first:
                ax.sharex(axes[first][c])
            for f in fractions:
                d = pts.get((n, proto, None, 0)) if f == 0 else pts.get((n, proto, atk, f))
                if d:
                    curve = {it: 1.0 - v for it, v in eval_curve(d, "test_acc").items()}
                    _draw(ax, curve, cmap(norm(f)), args.smooth)
            if r == 0:
                ax.set_title(ATTACK_LABEL[atk])
            if last_of_n:
                ax.set_xlabel("Global iteration")
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel((f"N = {n}, " if len(ns) > 1 else "") + f"{name}\nTest error")
            else:
                ax.tick_params(labelleft=False)
    _colorbar(plt, fig, axes, norm, cmap, fractions)
    fig.suptitle("Learning curves across malicious fraction (rows: " + ("N, " if len(ns) > 1 else "")
                 + "protocol; columns: attack)")
    return save(fig, args, "attack_sweep_" + "_".join(f"N{n}" for n in ns))


def fig_scaling(args):
    plt = _plt()
    with plt.rc_context(ATTACK_RC):
        return _fig_scaling(args, plt)


def _fig_scaling(args, plt):
    pts = train_points(args.runs, args.attacks_name)
    ns = [n for n in args.attack_n if (n, "rvfl", "scaling_fltrust") in {k[:3] for k in pts}
          or (n, "fltrust", "scaling_fltrust") in {k[:3] for k in pts}]
    if not ns:
        raise SystemExit("scaling: no train points with the Scaling attack")
    fractions = args.fractions
    norm, cmap = _fraction_colors(plt, fractions, args.cmap)
    cols = [(n, proto, name) for n in ns for proto, name in PROTOCOLS]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.0), squeeze=False, sharey="row",
                             layout="constrained")
    for c, (n, proto, name) in enumerate(cols):
        first = c - [p for p, _ in PROTOCOLS].index(proto)
        for r, key in ((0, "test_acc"), (1, "asr")):
            ax = axes[r][c]
            if r == 1:
                ax.sharex(axes[0][c])
            elif c != first:
                ax.sharex(axes[0][first])
            for f in fractions:
                d = pts.get((n, proto, None, 0)) if f == 0 else pts.get((n, proto, "scaling_fltrust", f))
                if d is None:
                    continue
                curve = eval_curve(d, key)
                if f == 0 and key == "asr":
                    continue                  # no trigger in a run without attack
                if curve:
                    _draw(ax, curve, cmap(norm(f)), args.smooth)
            if r == 0:
                ax.set_title(f"N = {n}, {name}")
                ax.tick_params(labelbottom=False)
            else:
                ax.set_ylim(-0.03, 1.03)
                ax.set_xlabel("Global iteration")
            if c == 0:
                ax.set_ylabel("Test accuracy" if r == 0 else "Backdoor ASR")
            else:
                ax.tick_params(labelleft=False)
    _colorbar(plt, fig, axes, norm, cmap, fractions)
    fig.suptitle("Scaling attack: test accuracy (top) and backdoor ASR (bottom)")
    return save(fig, args, "scaling_" + "_".join(f"N{n}" for n in ns))


FILTER_ORDER = [("max", "Max filter"), ("max_variable_k", "Max filter, variable k"),
                ("optimal_nk:n_over_3", "Optimal (n, k), t = n // 3"),
                ("optimal_nk:lower_bound", "Optimal (n, k), t = k + cp")]
REF_STYLE = {"off": ("0.35", "--"), "extra": ("C3", ":")}
LABEL_FONTSIZE = 8


def _place_ref_labels(fig, ax, labels):
    """Each reference line's name and value just right of the axes at its height; labels too close
    to read are pushed apart and tied to their line with a short connector."""
    if not labels:
        return
    lo, hi = ax.get_ylim()
    axis_pts = ax.get_window_extent().height * 72.0 / fig.dpi
    gap = 2 * LABEL_FONTSIZE * 1.25 / axis_pts
    items = sorted((((y - lo) / (hi - lo)), kind, name, y) for kind, name, y in labels)
    pos = [it[0] for it in items]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    top = 1.0 + gap / 2
    if pos and pos[-1] > top:
        pos[-1] = top
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - gap)
    for (f, kind, name, y), p in zip(items, pos):
        color = REF_STYLE[kind][0]
        moved = abs(p - f) > 1e-3
        ax.annotate(f"{name}\n({y:.1f} s)", xy=(1.0, f), xycoords="axes fraction",
                    xytext=(1.0 + 0.012 + (0.02 if moved else 0.0), p), textcoords="axes fraction",
                    ha="left", va="center", fontsize=LABEL_FONTSIZE, color=color, annotation_clip=False,
                    arrowprops=(dict(arrowstyle="-", color=color, linewidth=0.5, shrinkA=0, shrinkB=1)
                                if moved else None))


def _moving_average(values, window):
    half = window // 2
    return [statistics.mean(values[max(0, i - half):i + half + 1]) for i in range(len(values))]


def fig_filter(args):
    plt = _plt()
    metric = args.filter_metric
    value = {"round": lambda r: r["seconds"], "n": lambda r: r["n"], "k": lambda r: r["round_k1"],
             "t": lambda r: r["round_t"], "comm": lambda r: r["c_comm_max"]}[metric]
    runs = [d for d in done(logs.find_points(args.runs, kind="e2e", link=args.filter_link))
            if logs.read_config(d)["filter"] != "none"]
    if not runs:
        raise SystemExit(f"filter: no finished filter point on the link {args.filter_link} under {args.runs}")
    cfgs = {d: logs.read_config(d) for d in runs}
    mode_of = lambda c: c["filter"] + (f":{c['threshold']}" if c["filter"] == "optimal_nk" else "")
    present = {mode_of(c) for c in cfgs.values()}
    panels = [(m, t) for m, t in FILTER_ORDER if m in present]
    ns = sorted({c["n"] for c in cfgs.values()})
    alphas = sorted({c["alpha"] for c in cfgs.values()})
    color = {a: f"C{i}" for i, a in enumerate(alphas)}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels) + 1.0, 4.2), sharex=True,
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (mode, title) in zip(axes, panels):
        k1s = set()
        for d, c in sorted(cfgs.items(), key=lambda x: x[1]["alpha"]):
            if mode_of(c) != mode:
                continue
            k1s.add(c["k1"])
            rounds = logs.filter_rounds(d)
            xs = [it for it in sorted(rounds) if value(rounds[it]) is not None]
            ys = [value(rounds[it]) for it in xs]
            if not xs:
                continue
            ax.plot(xs, ys, linewidth=0.6, alpha=0.35, color=color[c["alpha"]])
            ax.plot(xs, _moving_average(ys, args.filter_window), linewidth=1.8, color=color[c["alpha"]],
                    label=f"alpha={c['alpha']:g}")
        ax.set_title(title + (f" (k={min(k1s)})" if mode == "max" else f" (k\u2265{min(k1s)})"), fontsize=10)
        ax.set_xlabel("round")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    labels = []
    if metric in ("round", "comm"):
        lines = []
        for d in done(logs.find_points(args.runs, kind="e2e", link=args.filter_link, filter="none")):
            c = logs.read_config(d)
            if c["n"] not in ns or (c["k1"] >= c["n"] // 3 and not args.show_k_eq_t):
                continue                     # k1 = t: a share carries no randomness
            rounds = logs.filter_rounds(d)
            its = sorted(rounds)[args.drop_first:] if len(rounds) > args.drop_first else sorted(rounds)
            vals = [value(rounds[i]) for i in its]
            if vals:
                lines.append(("off", f"no filter, k={c['k1']}" + (f", N={c['n']}" if len(ns) > 1 else ""),
                              statistics.mean(vals)))
        for item in args.ref:
            name, val = item.rsplit("=", 1)
            lines.append(("extra", name, float(val)))
        for kind, name, y in lines:
            for ax in axes:
                ax.axhline(y, color=REF_STYLE[kind][0], linestyle=REF_STYLE[kind][1], linewidth=0.9)
            labels.append((kind, name, y))
    axes[0].set_ylabel({"round": "time per round [s]", "n": "participants per round",
                        "comm": "client communication per round [s]", "k": "packing k1 per round",
                        "t": "threshold t per round"}[metric])
    handles, names = [], []
    for ax in axes:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    axes[0].legend(handles, names, fontsize=8, loc="best")
    fig.suptitle(f"Participant filter (N={'/'.join(map(str, ns))}, {link_text(logs.link_label(args.filter_link))})")
    fig.tight_layout()
    _place_ref_labels(fig, axes[-1], labels)
    return save(fig, args, f"filter_{metric}_{logs.link_label(args.filter_link)}")


FIGURES = {"accuracy": fig_accuracy, "cosine": fig_cosine, "attack_sweep": fig_attack_sweep,
           "scaling": fig_scaling, "filter": fig_filter}
PAPER_FIGURES = ("accuracy", "cosine", "attack_sweep", "scaling")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["tables", "figures"])
    ap.add_argument("--runs", default="runs", help="folder of the point folders (default: runs)")
    ap.add_argument("--out", default="artifacts", help="output folder (default: artifacts)")
    ap.add_argument("--only", nargs="+", help="names of the tables or figures to make")
    ap.add_argument("--format", nargs="+", default=["latex"], choices=["latex", "csv", "md"],
                    help="table formats (default: latex)")
    ap.add_argument("--fmt", nargs="+", default=["pdf"], help="figure formats (default: pdf)")
    ap.add_argument("--drop-first", type=int, default=1, help="rounds left out at the start (default 1)")
    ap.add_argument("--k1-table", nargs="+", default=["50:1,2,4,8,16", "100:8,16,32"],
                    help="cost_k1: N:k1,k1,... per block (default: the paper's)")
    ap.add_argument("--partial-rows", action="store_true",
                    help="e2e_lan, e2e_wan: also the (N, k1) that did not finish on every link of the table")
    ap.add_argument("--breakdown-points", nargs="+", default=["10,1", "10,3", "50,2", "50,16"],
                    help="comm_breakdown: the (N,k1) columns")
    ap.add_argument("--curves-n", type=int, default=100, help="curves: the N of the table")
    ap.add_argument("--accuracy-name", default="accuracy",
                    help="accuracy, cosine: the experiment name of the runs (default: accuracy)")
    ap.add_argument("--attacks-name", default="attacks",
                    help="attack_sweep, scaling: the experiment name of the runs (default: attacks)")
    ap.add_argument("--accuracy-stop", type=float, default=0.84,
                    help="accuracy, cosine: last evaluation, where the training accuracy reaches this")
    ap.add_argument("--attack-n", type=int, nargs="+", default=[30, 100], help="attack figures: the N values")
    ap.add_argument("--fractions", type=int, nargs="+", default=[0, 10, 20, 40, 60, 80, 90, 95],
                    help="attack figures: malicious fractions (%%)")
    ap.add_argument("--smooth", type=int, default=15, help="attack figures: centred window (iterations)")
    ap.add_argument("--cmap", default="plasma", help="attack figures: colour map of the fractions")
    ap.add_argument("--filter-link", default="10000:0.2", help="filter: the link")
    ap.add_argument("--filter-metric", default="round", choices=["round", "n", "k", "t", "comm"])
    ap.add_argument("--filter-window", type=int, default=10, help="filter: moving-average window (rounds)")
    ap.add_argument("--show-k-eq-t", action="store_true", help="filter: also the no-filter lines with k1 = t")
    ap.add_argument("--ref", action="append", default=[], help='filter: extra line "name=seconds" (repeatable)')
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    makers = TABLES if args.what == "tables" else FIGURES
    names = args.only or (list(TABLES) if args.what == "tables" else list(PAPER_FIGURES))
    unknown = [n for n in names if n not in makers]
    if unknown:
        raise SystemExit(f"unknown {args.what}: {unknown}; one of {list(makers)}")
    failed = []
    for name in names:
        try:
            if args.what == "tables":
                table = makers[name](args)
                if not table.rows:
                    raise SystemExit(f"{name}: no finished points under {args.runs}")
                for f in args.format:
                    print(f"[{name}] {write_table(table, f, args.out)}")
            else:
                for p in makers[name](args):
                    print(f"[{name}] {p}")
        except SystemExit as e:
            print(f"[{name}] {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
