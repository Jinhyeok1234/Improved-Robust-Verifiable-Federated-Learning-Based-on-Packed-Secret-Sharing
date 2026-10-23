#!/usr/bin/env python3
"""
Per-component communication breakdown for representative (N, k1) configs.

Attributes the communication cost to three sources, so each part's influence is explicit:

  (1) FL communication            -- any (server-gradient) FL round incurs it:
        weight update, g_S distribution
  (2) Secret-sharing communication -- any secret-sharing privacy layer incurs it:
        first-level share (plain SS, ~M N / k1), aggregated share
  (3) RVFL protocol-specific       -- the verification machinery this protocol adds,
        decomposed: second-level share (VSS, ~N^2), verify result, s_tilde/TS, other.

The RVFL cost sits on the *verifiable* second-level share (VSS), which is only ~N^2 and
therefore roughly independent of k1; the far larger first-level share (~M N / k1) stays
plain SS and shrinks with packing.

Perspective: the CLIENT's timeline only, per-party MEAN, send + receive both counted.
  A client takes part in EVERY communication of the protocol -- it receives g_S and the
  global weights from the server, exchanges first-/second-level shares with the other
  clients, and sends its aggregated share to the server -- so the client's view already
  covers all components. Reporting client + server would instead double-count each
  server<->client transfer (timed on both the sender and the receiver) and could exceed
  the wall-clock, which is exactly why the main table's columns are not additive. e2e is
  deliberately NOT shown here (it lives in the main table) to avoid inviting that invalid
  comm-vs-e2e comparison. verify_result's receive is a removable no-skip polling artifact
  and is excluded from comm.

Times in SECONDS, 1 decimal. Missing file -> blank cell.

Usage:
    python tabulate_breakdown.py [results_root] [--latex]
"""
import json
import re
import sys
from pathlib import Path

TARGET_COMBOS = [(10, 1), (10, 3), (50, 2), (50, 16)]              # representative configs to compare
EXCLUDE_PHASES = ("train", "plaintext_agg")      # local SGD + plaintext baseline path
COMM_PREFIXES = ("send", "receive", "recv")

# FL / SS comm metrics, classified by exact base name. Verification-phase comm is captured
# by PHASE (below) so new metric names like `receive_TS` are picked up automatically.
NAME_BUCKET = {
    "send_updated_weights": "fl_weight", "send_updated_global_weights": "fl_weight",
    "receive_global_weights": "fl_weight", "receive_local_update": "fl_weight",
    "send_server_data": "fl_gs", "receive_server": "fl_gs",
    "send_first_level_share": "ss_first", "receive_first_level_share": "ss_first",
    "send_agg_share": "ss_agg", "receive_agg_share": "ss_agg",
}

E2E_RE = re.compile(r"iteration\s+\d+\s*:\s*[\d.]+\s*ms\s*\(\s*([\d.]+)\s*s\s*\)", re.I)
LABEL_RE = re.compile(r"^(\d+(?:\.\d+)?)(Gbps|Mbps)_(\d+(?:\.\d+)?)ms$", re.I)

COMM_BUCKETS = ["fl_weight", "fl_gs", "ss_first", "ss_agg",
                "rvfl_second_level", "rvfl_verify_result", "rvfl_stilde", "rvfl_other",
                "other_comm"]


def _is_comm(base: str) -> bool:
    return base.startswith(COMM_PREFIXES)


def classify(phase: str, base: str) -> str:
    if not _is_comm(base):
        return "computation"
    if phase == "verification":                 # everything RVFL-specific lives here
        b = base.lower()
        if "second_level" in b:
            return "rvfl_second_level"
        if "verify_result" in b:
            # verify_result carries only O(N) booleans; its RECEIVE time is a polling
            # artifact (the no-skip 1.0s loop), so count SEND only and divert the receive
            # into a non-component 'artifact' bucket that is excluded from the totals.
            return "rvfl_verify_result" if base.startswith("send") else "artifact_verify_recv"
        if "tilde" in b or "_ts" in b or b.endswith("ts"):
            return "rvfl_stilde"
        return "rvfl_other"
    return NAME_BUCKET.get(base, "other_comm")


def breakdown(path: Path):
    """Return {bucket: seconds}. Comm = worst party (max); comp = per-party mean.
    Also returns the raw metric list per 'other_comm'/'rvfl_other' for transparency."""
    if not path.is_file():
        return None
    # acc[party][(phase, base)] = [values...]  (one per sample / party instance)
    acc = {"server": {}, "client": {}}
    stray = {}   # bucket -> set of base names landing there (for the odd buckets)
    rows = 0
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows += 1
                party = "server" if data.get("party_id", 0) == -1 else "client"
                for phase, metrics in (data.get("time_logs_ms") or {}).items():
                    if phase in EXCLUDE_PHASES or not isinstance(metrics, dict):
                        continue
                    for mk, val in metrics.items():
                        base = mk[:-3] if mk.endswith("_ms") else mk
                        acc[party].setdefault((phase, base), []).append(float(val))
    except OSError:
        return None
    if not rows:
        return None

    totals = {b: 0.0 for b in (["computation"] + COMM_BUCKETS + ["artifact_verify_recv"])}
    # Decompose the CLIENT's timeline only (the busy party; the server mostly idles/waits).
    # Summing client + server per component would double-count each transfer (once at the
    # sender, once at the receiver) and can exceed e2e at low bandwidth. So Total comm here
    # equals the main table's Client-comm (<= e2e). g_S/weight appear as the client's receive,
    # aggregated share as the client's send.
    for party in ("client",):
        for (phase, base), vals in acc[party].items():
            mean_ms = sum(vals) / len(vals)                  # per-party mean
            if base.startswith(("receive", "recv")) and "verify_result" in base.lower():
                # verify_result's receive is a removable no-skip polling artifact: keep it out of
                # comm and subtract its worst-party value from e2e.
                totals["artifact_verify_recv"] = max(totals["artifact_verify_recv"],
                                                     max(vals) / 1000.0)
                continue
            bucket = classify(phase, base)                   # send AND receive both counted
            totals[bucket] += mean_ms / 1000.0
            if bucket in ("other_comm", "rvfl_other"):
                stray.setdefault(bucket, set()).add(base)
    totals["_stray"] = {k: sorted(v) for k, v in stray.items()}
    return totals


def parse_e2e(path: Path):
    if not path.is_file():
        return None
    last = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = E2E_RE.search(line)
            if m:
                last = float(m.group(1))
    except OSError:
        return None
    return last


def label_sort_key(label: str):
    m = LABEL_RE.match(label)
    if not m:
        return (float("inf"), float("inf"), label)
    val, unit, rtt = float(m.group(1)), m.group(2).lower(), float(m.group(3))
    return (val * 1000 if unit == "gbps" else val, rtt, label)


def label_human(label: str) -> str:
    m = LABEL_RE.match(label)
    if not m:
        return label
    unit = "Gbps" if m.group(2).lower() == "gbps" else "Mbps"
    return f"{m.group(1)} {unit}, {m.group(3)} ms RTT"


def discover_labels(root: Path):
    labels = set()
    for N, k in TARGET_COMBOS:
        d = root / f"N{N}_k{k}"
        if d.is_dir():
            labels.update(p.name for p in d.iterdir() if p.is_dir())
    return sorted(labels, key=label_sort_key)


def fmt(v):
    return f"{v:.1f}" if v is not None else ""


# (indent, name, how, hideable) -- a hideable row is dropped when all its cells are blank/0.0
ROWS = [
    (0, "Computation", "computation", False),
    (0, "**Comm — FL (inherent)**", None, False),
    (1, "weight update", "fl_weight", False),
    (1, "$g_S$ distribution", "fl_gs", False),
    (1, "*FL subtotal*", ("sum", ["fl_weight", "fl_gs"]), False),
    (0, "**Comm — Secret sharing**", None, False),
    (1, "first-level share", "ss_first", False),
    (1, "aggregated share", "ss_agg", False),
    (1, "*SS subtotal*", ("sum", ["ss_first", "ss_agg"]), False),
    (0, "**Comm — RVFL-specific (verification)**", None, False),
    (1, "second-level share (VSS)", "rvfl_second_level", False),
    (1, "others", ("sum", ["rvfl_stilde", "rvfl_verify_result", "rvfl_other"]), True),
    (1, "*RVFL subtotal*", ("sum", ["rvfl_second_level", "rvfl_stilde",
                                    "rvfl_verify_result", "rvfl_other"]), False),
    (0, "other comm (unclassified)", "other_comm", True),
    (0, "**Total communication**", ("sum", COMM_BUCKETS), False),
]


def cell_value(bd, e2e, how):
    if how is None:
        return None
    if how == "e2e":
        return e2e
    if how == "e2e_adj":                         # e2e minus the excluded verify_result artifact
        if e2e is None:
            return None
        return e2e - (bd.get("artifact_verify_recv", 0.0) if bd else 0.0)
    if bd is None:
        return None
    if isinstance(how, tuple):
        return sum(bd.get(k, 0.0) for k in how[1])
    return bd.get(how, 0.0)


def render(root: Path, labels, latex: bool):
    cols = TARGET_COMBOS
    strays = {}
    for label in labels:
        data = []
        for (N, k) in cols:
            base = root / f"N{N}_k{k}" / label
            bd = breakdown(base / "processing_time_log.jsonl")
            if bd and bd.get("_stray"):
                for b, names in bd["_stray"].items():
                    strays.setdefault(b, set()).update(names)
            data.append((bd, parse_e2e(base / "terminal_log.txt")))
        header = ["Component"] + [f"({N},{k})" for (N, k) in cols]

        if latex:
            print(f"\n% ---- {label_human(label)} ({label}) ----")
            print("\\begin{tabular}{l" + "r" * len(cols) + "}")
            print("\\toprule")
            print(" & ".join(h.replace("**", "").replace("*", "") for h in header) + " \\\\")
            print("\\midrule")
        else:
            print(f"\n## {label_human(label)}   ({label})\n")
            print("| " + " | ".join(header) + " |")
            print("|" + "|".join(["---"] * len(header)) + "|")

        for indent, name, how, hideable in ROWS:
            cells = [fmt(cell_value(bd, e2e, how)) for (bd, e2e) in data]
            if hideable and all(c in ("", "0.0") for c in cells):
                continue
            if latex:
                nm = ("\\quad " if indent else "") + name.replace("**", "").replace("*", "")
                print(" & ".join([nm] + cells) + " \\\\")
            else:
                nm = ("· " if indent else "") + name
                print("| " + " | ".join([nm] + cells) + " |")

        if latex:
            print("\\bottomrule\n\\end{tabular}")

        arts = [(bd.get("artifact_verify_recv") if bd else None) for (bd, _e) in data]
        if any(a for a in arts):
            print(f"[breakdown] {label}: verify_result RECEIVE excluded (polling artifact) -> "
                  + ", ".join(f"({N},{k})={fmt(a)}s" for (N, k), a in zip(cols, arts)),
                  file=sys.stderr)

    if strays:
        print("\n[breakdown] metrics landing in odd buckets (classify to fix): "
              + "; ".join(f"{b} <- {sorted(n)}" for b, n in strays.items()), file=sys.stderr)


def main(argv):
    latex = "--latex" in argv
    positional = [a for a in argv if not a.startswith("--")]
    root = Path(positional[0]) if positional else Path("e2e_results")
    labels = discover_labels(root)
    if not labels:
        print(f"[breakdown] no result folders for {TARGET_COMBOS} under {root.resolve()}",
              file=sys.stderr)
        return 1
    print(f"[breakdown] configs {TARGET_COMBOS} x {len(labels)} settings under {root.resolve()} "
          f"| comm=send+receive, per-party mean (verify_result recv excluded as artifact)", file=sys.stderr)
    render(root, labels, latex)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))