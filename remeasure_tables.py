#!/usr/bin/env python3
"""Tables of the re-measurement (run_remeasure.py) with the first rounds left out.

    python remeasure_tables.py                         # ec256, first round left out
    python remeasure_tables.py --drop-first 1 --out remeasure_ec256.md
    python remeasure_tables.py --pick 10,1 10,3 50,2 50,16    # (c2) columns of your own
    python remeasure_tables.py --all                   # every (N, k1) on disk, not only the grid
    python remeasure_tables.py --preview               # while run_remeasure.py is running

Only combinations whose logs hold every round (run_remeasure.ROUNDS, brea_rounds(N) for BREA, for
the server and every client) enter the tables; the others are listed with the rounds logged so
far, and --partial puts them in as well, marked with *. A netns combination that is still
running is not listed at all: main.py writes its processing_time_log.jsonl when it ends.
--preview writes *_preview.csv / remeasure_<backend><level>_preview.md, so a look in the middle
of the run never replaces the files the runner's tables step writes. Results of a step that has
not started yet (logs/remeasure/state.json) are the earlier ones, from before the
re-measurement; they are left out and listed apart (--all shows everything on disk).

Every combination of the re-measurement (run_remeasure.COST_GRID, COST_LAST, BREA_N, E2E_GRID,
E2E_LAST)
runs 6 rounds (BREA at N = 20..40 one round, which is then the figure), and round 1 carries one-time costs (the connection set-up to every peer, the first allocation of every
buffer), so the figures are means over rounds 2 to 6. The existing scripts average every round of a log (cost_sweep.py's CSV,
tabulate_e2e.py's breakdown table), so this script reads the same logs, keeps the rounds after
--drop-first, and hands them to the same analyzers (comp_log_analyzer_revised.summarize,
comm_log_analyzer.calculate_communication_averages, tabulate_e2e.round_seconds); nothing is
computed differently, only fewer rounds go in.

Written:
    cost_sweep_<backend><level>_drop<d>.csv   the cost-sweep CSV (same columns as cost_sweep.py's,
                                              "iters" = rounds used), readable by
                                              python tabulate_cost.py --csv <that file>
    brea_<backend><level>_drop<d>.csv         the same columns for the BREA runs
                                              (runs/N{N}/logs/BREA/run_remeasure.BREA_TAG_N{N}_*)
    --out (default remeasure_<backend><level>.md)
        table 4  communication volume and phase-wise computation time at t = k, N = 10..100
                 (a volume of exactly 0, the server's unicast, shows -, as in table 6)
        table 5  the same at N = 50 and N = 100 over k1
        table 6  ours (k1 = 1) and BREA at N = 10..40, laid out as table 4: broadcast and
                 unicast volume, and Enroll / Verify / Agg / Total computation, for the server
                 and for a client. BREA logs a single phase, so its timers are put into the three
                 phases by BREA_PHASES (the paper's rule: Enroll = dealing updates via Pedersen
                 VSS and verifying them; Verify = pairwise distances and Multi-Krum; Agg = the
                 aggregation of the selected clients' updates)
        (c1)     per link: S-comp, S-comm, C-comp, C-comm (per-party means) and the round time
                 (S-comp + slowest client's comp and comm per region, round by round, then the
                 mean and the sd over the rounds used), from e2e_results/<backend><level>/. Runs
                 that did not end well with every round are listed under the tables, not in them
                 (--partial puts them in, N marked *)
        (c2)     the communication time (seconds a round, --c2-digits decimals, default 3) by
                 what causes it, one row per
                 message (C2_ROWS): FedAvg (the global model), secret sharing on FedAvg (the
                 first-level shares, the aggregated share), RVFL-specific (g_S, the second-level
                 shares with their commitments, the verdict, s~, TS). Each row is how long
                 that exchange took, a round at a time: S -> C and C <-> C rows the slowest
                 client (its send and receive of the message), C -> S rows the server's receive
                 (the comment above C2_LEAVES says why); verify_result's receive is counted
                 (comm_records says why); no computation row, the cost tables have it. In
                 one table: LAN (1 Gbps, 0.2 ms) on the left, WAN (10 Mbps, 80 ms) on the right, the
                 same (N, k1) columns in each. The columns: among the N whose every k1 of
                 run_remeasure.E2E_GRID (not E2E_LAST) has a finished run on both links, the lowest and the
                 highest N, and for each its lowest and highest k1 (with every run done,
                 (10,1) (10,3) (50,2) (50,16), the columns of the rebuttal); --pick sets them.
                 Markdown, then the same table in LaTeX
"""
import argparse
import csv
import json
import statistics
import tempfile
from pathlib import Path

import comm_log_analyzer
import comp_log_analyzer_revised
import cost_sweep
import tabulate_breakdown
import tabulate_cost
from tabulate_e2e import render, round_seconds, rounds_in, status_of

DIR_RE_TEXT = r"^N(\d+)_k(\d+)$"

# BREA's computation timers (party.py Server.BREA / Client.BREA, all under the one phase "BREA")
# put into the phases of our protocol, as the paper states it: Enroll corresponds to dealing
# updates via Pedersen VSS and verifying them; Verify involves computing pairwise distances and
# executing Multi-Krum for client selection; Agg represents the aggregation of the selected
# clients' updates. A computation timer that is in none of the lists is reported, not dropped.
BREA_PHASES = {
    "enrollment": ("calc_share", "verification"),                          # client
    "verification": ("calc_pairwise_distance",                             # client
                     "reconstruct_d", "dequantize_d", "multi_krum"),       # server
    "aggregation": ("aggregate_shares",                                    # client
                    "reconstruct_agg", "dequantize_and_update"),           # server
}


def kept_records(path, drop_first):
    """The records of a log with superseded duplicates removed (the last one of each
    (party, round) wins, as cost_sweep.dedup) and the rounds 1..drop_first left out."""
    recs = cost_sweep.dedup(cost_sweep.read_records(Path(path)))
    return [r for r in recs if int(r.get("iteration", 0)) > drop_first]


def full_rounds(time_log, comm_log, n_clients):
    """Rounds logged for the server and every client in both logs."""
    parties = set(range(n_clients)) | {-1}
    done = None
    for path in (time_log, comm_log):
        by = {}
        for rec in cost_sweep.read_records(Path(path)):
            by.setdefault(rec.get("iteration"), set()).add(rec.get("party_id"))
        full = {it for it, ps in by.items() if parties <= ps}
        done = full if done is None else done & full
    return len(done or ())


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda r: (r.get("iteration", 0), r.get("party_id", 0))):
            fh.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------------------------------
# (a)+(b): the cost sweep
# ------------------------------------------------------------------------------------------
def cost_row(n_clients, k1, backend, level, drop_first, tmp, logs=None, phase_map=None):
    """One CSV row from the cost-sweep logs of (N, k1), or from logs = (time log, comm log).
    phase_map (BREA_PHASES) puts the computation timers of a single-phase log into phases."""
    time_log, comm_log = logs or cost_sweep.log_paths(n_clients, k1, backend, level)
    if not (Path(time_log).exists() and Path(comm_log).exists()):
        return None
    t_recs = kept_records(time_log, drop_first)
    c_recs = kept_records(comm_log, drop_first)
    rounds = sorted({r["iteration"] for r in t_recs if r.get("party_id") == -1})
    if not rounds:
        return None
    t_path, c_path = Path(tmp) / "time.jsonl", Path(tmp) / "comm.jsonl"
    write_jsonl(t_recs, t_path)
    write_jsonl(c_recs, c_path)
    comp = comp_log_analyzer_revised.summarize(str(t_path))
    comm = comm_log_analyzer.calculate_communication_averages(str(c_path))
    row = {"N": n_clients, "k1": k1, "t1": n_clients // 3, "k2": 1,
           "backend": backend, "level": level, "iters": len(rounds),
           "rounds_used": " ".join(map(str, rounds)), "first_rounds_dropped": drop_first}
    for party in ("server", "client"):
        for kind in ("comp", "comm"):
            block = comp.get(party, {}).get(kind, {})
            row[f"{party}_{kind}_total_s"] = block.get("total_s", 0.0)
            # the same total in ms, unrounded beyond the per-phase values (total_s is rounded to
            # 1 ms), so a table's Total equals the sum of its phase columns
            row[f"{party}_{kind}_total_ms"] = round(sum(block.get("per_phase_ms", {}).values()), 3)
            for phase, ms in block.get("per_phase_ms", {}).items():
                row[f"{party}_{kind}_{phase}_ms"] = ms
    for key, val in comm.items():
        row[f"{key}_bytes"] = val
    if phase_map:
        where = {metric: phase for phase, metrics in phase_map.items() for metric in metrics}
        for party in ("server", "client"):
            unmapped = []
            for key, ms in comp.get(party, {}).get("comp", {}).get("per_metric_ms", {}).items():
                metric = key.split(" - ", 1)[-1]
                phase = where.get(metric)
                if phase is None:
                    unmapped.append(metric)
                    continue
                col = f"{party}_comp_{phase}_ms"
                row[col] = round(row.get(col, 0.0) + ms, 3)
            if unmapped:
                row[f"{party}_comp_unmapped"] = " ".join(sorted(unmapped))
                print(f"  note: N={n_clients} {party} timers in no phase of BREA_PHASES: {unmapped}")
    return row


def cost_pairs(backend, level):
    """(N, k1) of every cost-sweep log of this backend/level under runs/ (with --all)."""
    import re
    pat = re.compile(rf"^RVFL_noattack_k(\d+)_{re.escape(backend + level)}_N(\d+)_processing_time_log\.jsonl$")
    out = set()
    for f in Path("runs").glob("N*/logs/RVFL/*_processing_time_log.jsonl"):
        m = pat.match(f.name)
        if m:
            out.add((int(m.group(2)), int(m.group(1))))
    return sorted(out)


VOLUME_KEYS = ("server_broadcast_avg_bytes", "server_unicast_avg_bytes",
               "client_broadcast_avg_bytes", "client_unicast_avg_bytes")


def dash_zero_volume(row):
    """The row with every volume that is exactly 0 set to None, so tabulate_cost.table_a prints
    "-" there, as table 6 does (the server unicasts nothing)."""
    out = dict(row)
    for key in VOLUME_KEYS:
        try:
            if float(out.get(key)) == 0:
                out[key] = None
        except (TypeError, ValueError):
            pass
    return out


def write_csv(path, rows):
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def table_six(ours_rows, brea_rows, ns):
    """Ours at k1 = 1 next to BREA, per N, in the layout of the paper's table: per party the
    broadcast and unicast volume (MB) and the Enroll / Verify / Agg / Total computation (ms).
    A cell with nothing behind it (BREA's server has no Enroll timer, no party unicasts from
    the server) shows -."""
    def mb(v):
        return "-" if v in (None, "") or float(v) == 0 else f"{float(v) / 1e6:,.2f}"

    def ms(v):
        return "-" if v in (None, "") else f"{float(v):,.2f}"

    ours = {int(r["N"]): r for r in ours_rows if int(r["k1"]) == 1}
    brea = {int(r["N"]): r for r in brea_rows}
    out, any_row = [], False
    for n in ns:
        for name, src in (("Ours", ours), ("BREA", brea)):
            r = src.get(n)
            if r is None:
                out.append([n, name] + ["-"] * 12)
                continue
            any_row = True
            cells = [n, name + ("" if r.get("complete", 1) else "*")]
            for party in ("server", "client"):
                cells += [mb(r.get(f"{party}_broadcast_avg_bytes")), mb(r.get(f"{party}_unicast_avg_bytes"))]
                cells += [ms(r.get(f"{party}_comp_{ph}_ms")) for ph in ("enrollment", "verification", "aggregation")]
                cells.append(ms(r.get(f"{party}_comp_total_ms")))
            out.append(cells)
    if not any_row:
        return ""
    return render(out, ["N", "Scheme",
                        "S Broadcast (MB)", "S Unicast (MB)", "S Enroll (ms)", "S Verify (ms)", "S Agg (ms)", "S Total (ms)",
                        "C Broadcast (MB)", "C Unicast (MB)", "C Enroll (ms)", "C Verify (ms)", "C Agg (ms)", "C Total (ms)"])


# ------------------------------------------------------------------------------------------
# (c1), (c2): the netns sweep
# ------------------------------------------------------------------------------------------
def e2e_entries(root, picks=None):
    """(N, k1, label, combo_dir) of every combination under root whose log exists."""
    import re
    pat = re.compile(DIR_RE_TEXT)
    out = []
    for n_dir in sorted(Path(root).glob("N*_k*")):
        m = pat.match(n_dir.name)
        if not (m and n_dir.is_dir()):
            continue
        n_clients, k1 = int(m.group(1)), int(m.group(2))
        if picks and (n_clients, k1) not in picks:
            continue
        for combo in sorted(p for p in n_dir.iterdir() if p.is_dir()):
            if (combo / "processing_time_log.jsonl").exists():
                out.append((n_clients, k1, combo.name, combo))
    return out


def link_order(label):
    """10Mbps_80ms -> (10, 80): slowest link first, as tabulate_filter.link_rate_mbit."""
    import re
    m = re.match(r"^(\d+)(Gbps|Mbps)_([\d.]+)ms$", label)
    if not m:
        return (float("inf"), 0.0, label)
    return (int(m.group(1)) * (1000 if m.group(2) == "Gbps" else 1), float(m.group(3)), label)


def used_rounds(combo_dir, drop_first):
    ids = rounds_in(combo_dir)
    return ids[drop_first:] if len(ids) > drop_first else []


def table_lan_wan(entries, drop_first, needed, partial):
    """Per link: S-comp, S-comm, C-comp, C-comm and the round time of every run that ended well
    with every round logged (run_finished). The others are left out (with partial put in, N
    marked *) and returned as lines to list under the tables."""
    by_label, abnormal = {}, []
    for n_clients, k1, label, combo in entries:
        if not run_finished(combo, needed):
            abnormal.append(f"{label} N={n_clients} k1={k1}: STATUS {status_of(combo)}, "
                            f"{len(rounds_in(combo))} of {needed} rounds")
            if not partial:
                continue
        by_label.setdefault(label, []).append((n_clients, k1, combo))
    blocks = []
    for label in sorted(by_label, key=link_order):
        rows = []
        for n_clients, k1, combo in sorted(by_label[label]):
            log = str(combo / "processing_time_log.jsonl")
            ids = used_rounds(combo, drop_first)
            n_cell = f"{n_clients}*" if not run_finished(combo, needed) else n_clients
            if not ids:
                rows.append([n_cell, k1, 0, "-", "-", "-", "-", "-"])
                continue
            mean = comp_log_analyzer_revised.summarize(log, set(ids))
            per_round = []
            for it in ids:
                one = comp_log_analyzer_revised.summarize(log, {it}, client_agg="max")
                per_round.append(round_seconds(one["server"]["comp"]["total_s"],
                                               one["client"]["comp"]["total_s"],
                                               one["client"]["comm"]["total_s"]))
            rt = statistics.mean(per_round)
            rt_cell = f"{rt:.2f} ± {statistics.stdev(per_round):.2f}" if len(per_round) > 1 else f"{rt:.2f}"
            rows.append([n_cell, k1, len(ids),
                         f"{mean['server']['comp']['total_s']:.2f}", f"{mean['server']['comm']['total_s']:.2f}",
                         f"{mean['client']['comp']['total_s']:.2f}", f"{mean['client']['comm']['total_s']:.2f}",
                         rt_cell])
        blocks.append(f"### {label}\n\n" + render(
            rows, ["N", "k1", "rounds", "S-comp s", "S-comm s", "C-comp s", "C-comm s", "round s"]))
    return "\n\n".join(blocks), abnormal


# (c2) rows: (indent, markdown name, LaTeX name, spec, hideable); spec is a key of C2_LEAVES (a
# row of its own), a tuple of keys (a subtotal: the sum of those rows) or None (a heading). A
# message is placed by the job it does, not by its name. FedAvg: what plain FedAvg sends anyway,
# the global model to the clients (the clients' updates go up as shares here, so they are under
# secret sharing). Secret sharing: what putting secret sharing on FedAvg adds, the first-level
# shares and the aggregated share to the server. RVFL-specific: everything else. A hideable row
# is left out when no column has its metrics at all (the filter's messages).
#
# Every row is how long that exchange took, from the _sync where all its parties start it until
# the last party is done, a round at a time, then the mean over the rounds used:
#   S -> C and C <-> C   the slowest client of the round: the largest sum of the row's "C:" timers
#                        over the clients (for C <-> C its send and its receive of that message;
#                        the server sends one client after another, so the mean client would see
#                        about half of the exchange)
#   C -> S               the server's receive ("S:" timer), until the last client's message is in;
#                        a client's send returns once the bytes are in its socket buffer, so it
#                        holds neither the one-way delay nor, for a small message such as s~,
#                        anything at all. Those client sends (C2_REPLACED) are not counted.
C2_LEAVES = {
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
C2_ROWS = [
    (0, "**FedAvg**", "FedAvg", None, False),
    (1, "global model (S → C)", "global model (S $\\to$ C)", "global", False),
    (0, "**Secret sharing (FedAvg + SS)**", "Secret sharing (FedAvg + SS)", None, False),
    (1, "first-level share (C ↔ C)", "first-level share (C $\\leftrightarrow$ C)", "first", False),
    (1, "aggregated share (C → S)", "aggregated share (C $\\to$ S)", "agg", False),
    (1, "*SS subtotal*", "SS subtotal", ("first", "agg"), False),
    (0, "**RVFL-specific**", "RVFL-specific", None, False),
    (1, "$g_S$ (S → C)", "$g_S$ (S $\\to$ C)", "gs", False),
    (1, "second-level share + commitments (C ↔ C)",
     "second-level share + commitments (C $\\leftrightarrow$ C)", "second", False),
    (1, "verdict (C ↔ C)", "verdict (C $\\leftrightarrow$ C)", "verdict", False),
    (1, "$\\tilde{s}$ (C → S)", "$\\tilde{s}$ (C $\\to$ S)", "stilde", False),
    (1, "TS (S → C)", "TS (S $\\to$ C)", "ts", False),
    (1, "cosine (C → S)", "cosine (C $\\to$ S)", "cos", True),
    (1, "participant list (S → C)", "participant list (S $\\to$ C)", "plist", True),
    (1, "*RVFL subtotal*", "RVFL subtotal", ("gs", "second", "verdict", "stilde", "ts", "cos", "plist"), False),
]
C2_PLACED = {m for metrics in C2_LEAVES.values() for m in metrics}
# the clients' sends that the server's receive stands for, and the server's sends of the messages
# whose time the clients' receive gives: the other end of a placed exchange, not counted again
C2_REPLACED = {"C:send_agg_share", "C:send_s_tilde", "C:send_cos"}
C2_OTHER_END = {"S:send_server_data", "S:send_TS", "S:send_updated_global_weights",
                "S:send_participant_list"}


def comm_records(combo, drop_first):
    """The communication timers (ms) of the rounds after drop_first (the ids of terminal_log.txt,
    as (c1)): {"rounds": [...], "C": {round: {client: {"C:<name>": ms}}}, "S": {round:
    {"S:<name>": ms}}, "present": names seen}; the phases of tabulate_breakdown.EXCLUDE_PHASES left
    out, a name in two phases summed. None when no round is left.

    Unlike tabulate_breakdown.py, verify_result's receive is kept: it was a polling artifact when
    the receive loop called every peer again with a 1 s timeout, and party.py's loop now skips the
    peers already received, so the time is that of the last verdict to arrive."""
    ids = used_rounds(combo, drop_first)
    if not ids:
        return None
    wanted = set(ids)
    out = {"rounds": list(ids), "C": {}, "S": {}, "present": set()}
    with open(combo / "processing_time_log.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = rec.get("iteration")
            if it not in wanted:
                continue
            pid = rec.get("party_id", 0)
            if pid == -1:
                slot, prefix = out["S"].setdefault(it, {}), "S:"
            else:
                slot, prefix = out["C"].setdefault(it, {}).setdefault(pid, {}), "C:"
            for phase, metrics in (rec.get("time_logs_ms") or {}).items():
                if phase in tabulate_breakdown.EXCLUDE_PHASES or not isinstance(metrics, dict):
                    continue
                for key, val in metrics.items():
                    base = key[:-3] if key.endswith("_ms") else key
                    if base.startswith(("send", "receive", "recv")):
                        name = prefix + base
                        slot[name] = slot.get(name, 0.0) + float(val)
                        out["present"].add(name)
    return out


def exchange_seconds(recs, metrics):
    """How long the exchange of metrics took (s), the mean over the rounds: per round the largest
    sum of the "C:" metrics over the clients plus the "S:" metrics of the server."""
    c_names = [m for m in metrics if m.startswith("C:")]
    s_names = [m for m in metrics if m.startswith("S:")]
    per_round = []
    for it in recs["rounds"]:
        clients = recs["C"].get(it, {}).values()
        slowest = max((sum(c.get(m, 0.0) for m in c_names) for c in clients), default=0.0) if c_names else 0.0
        per_round.append(slowest + sum(recs["S"].get(it, {}).get(m, 0.0) for m in s_names))
    return sum(per_round) / len(per_round) / 1000.0


def run_finished(combo, needed):
    """True when the netns run of combo ended well with every round logged."""
    return status_of(combo) == "ok" and len(rounds_in(combo)) >= needed


def breakdown_columns(entries, grid, labels, needed, partial, drop_first):
    """The (N, k1) columns of (c2): among the N whose every k1 of grid has a finished run on every
    link of labels, the lowest and the highest N, and for each of them the lowest and the highest
    k1. With partial a run counts once it has a round after the first ones left out."""
    have = {(n, k, label): combo for n, k, label, combo in entries}

    def counts(combo):
        if combo is None:
            return False
        return bool(used_rounds(combo, drop_first)) if partial else run_finished(combo, needed)

    done = [n for n, ks in sorted(grid.items())
            if ks and all(counts(have.get((n, k, label))) for k in ks for label in labels)]
    if not done:
        return []
    cols = []
    for n in sorted({done[0], done[-1]}):
        for k in sorted({min(grid[n]), max(grid[n])}):
            cols.append((n, k))
    return cols


def table_breakdown(entries, cols, groups, drop_first, needed, digits=3):
    """One table, the links of groups side by side (one block of cols each): how long each
    exchange took (s a round), by the job each message does (C2_ROWS; how it is timed, see there),
    with digits decimals
    (3: to the millisecond; a value above 0 that would show as 0 shows as <0.001). Returns the
    markdown and the LaTeX of it."""
    have = {(n, k, label): combo for n, k, label, combo in entries}
    data, marks = {}, {}
    for _, label in groups:
        for col in cols:
            combo = have.get((*col, label))
            data[(label, col)] = comm_records(combo, drop_first) if combo else None
            if combo is not None and not run_finished(combo, needed):
                marks[(label, col)] = "*"
    keys = [(label, col) for _, label in groups for col in cols]
    present = set()
    for recs in data.values():
        present.update(recs["present"] if recs else ())
    unplaced = sorted(present - C2_PLACED - C2_REPLACED - C2_OTHER_END)
    leaves = dict(C2_LEAVES)
    for name in unplaced:                    # each one an exchange of its own
        leaves[name] = [name]

    def seconds(recs, spec):
        keys_ = spec if isinstance(spec, tuple) else (spec,)
        return sum(exchange_seconds(recs, leaves[k]) for k in keys_)

    def cell(recs, spec):
        if spec is None or recs is None:
            return ""
        value = seconds(recs, spec)
        if 0 < value < 0.5 * 10 ** -digits:
            return f"<{10 ** -digits:.{digits}f}"         # not 0, below the last digit shown
        return f"{value:.{digits}f}"

    rows = list(C2_ROWS)
    if unplaced:
        print(f"  note: (c2) metrics with no row, shown as 'not placed': {unplaced}")
        rows.append((0, "not placed: " + ", ".join(unplaced), "not placed", tuple(unplaced), False))
    rows.append((0, "**Total communication**", "Total communication", tuple(leaves), False))
    body = []
    for indent, md_name, tex_name, spec, hideable in rows:
        if hideable and not any(m in present for m in leaves[spec]):
            continue
        body.append((indent, md_name, tex_name, [cell(data[key], spec) for key in keys]))

    col_names = [f"({n},{k}){marks.get((label, (n, k)), '')}" for label, (n, k) in keys]
    head = ["Component"]
    for group, label in groups:
        head += [f"{group} ({tabulate_breakdown.label_human(label)})"] + [""] * (len(cols) - 1)
    md_rows = [["(N, k1)"] + col_names]
    md_rows += [[("· " if indent else "") + md_name] + cells for indent, md_name, _, cells in body]
    markdown = render(md_rows, head)

    width = len(cols)
    tex = ["\\begin{tabular}{l" + ("r" * width) * len(groups) + "}", "\\toprule",
           " & ".join([""] + [f"\\multicolumn{{{width}}}{{c}}{{{group} "
                              f"({tabulate_breakdown.label_human(label)})}}" for group, label in groups])
           + " \\\\",
           "".join(f"\\cmidrule(lr){{{2 + i * width}-{1 + (i + 1) * width}}}" for i in range(len(groups))),
           " & ".join(["Component"] + [f"$({n},{k})${marks.get((label, (n, k)), '')}"
                                       for label, (n, k) in keys]) + " \\\\",
           "\\midrule"]
    for indent, _, tex_name, cells in body:
        tex_cells = [c.replace("<", "$<$", 1) for c in cells]      # a bare < is not a less-than sign in LaTeX text
        tex.append(" & ".join([("\\quad " if indent else "") + tex_name] + tex_cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    return markdown, "\n".join(tex)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="ec")
    ap.add_argument("--level", default="256")
    ap.add_argument("--drop-first", type=int, default=1, help="rounds left out at the start (default 1)")
    ap.add_argument("--c2-digits", type=int, default=3,
                    help="(c2) only: decimals of the seconds (default 3, to the millisecond)")
    ap.add_argument("--pick", nargs="*", default=None,
                    help="(c2) only: its columns, N,k1 pairs such as 10,1 10,3 50,2 50,16 "
                         "(default: the lowest and highest k1 of the lowest and highest finished N)")
    ap.add_argument("--all", action="store_true",
                    help="every (N, k1) found on disk instead of the re-measurement grid")
    ap.add_argument("--csv", default=None, help="default cost_sweep_<backend><level>_drop<d>.csv")
    ap.add_argument("--brea-csv", default=None, help="default brea_<backend><level>_drop<d>.csv")
    ap.add_argument("--out", default=None, help="default remeasure_<backend><level>.md")
    ap.add_argument("--preview", action="store_true",
                    help="a look while the run is going: the default output names get _preview")
    ap.add_argument("--partial", action="store_true",
                    help="also put in the combinations without every round, marked with *")
    args = ap.parse_args()
    config = f"{args.backend}{args.level}"
    tag = "_preview" if args.preview else ""
    csv_path = Path(args.csv or f"cost_sweep_{config}_drop{args.drop_first}{tag}.csv")
    brea_csv = Path(args.brea_csv or f"brea_{config}_drop{args.drop_first}{tag}.csv")
    out_path = Path(args.out or f"remeasure_{config}{tag}.md")
    picks = sorted({tuple(int(x) for x in p.split(",")) for p in args.pick}) if args.pick else None

    import run_remeasure as rq
    cost_grid = {(n, k) for n, ks in rq.COST_GRID.items() for k in ks} | set(rq.COST_LAST)
    e2e_grid = {(n, k) for n, ks in rq.E2E_GRID.items() for k in ks} | set(rq.E2E_LAST)
    pairs = cost_pairs(args.backend, args.level) if args.all else sorted(cost_grid)
    rows, brea_rows, incomplete, earlier, brea_round_one = [], [], [], [], []

    # Results of a step that has not started yet are the earlier ones (before this
    # re-measurement): run_remeasure.py moves them into _set_aside/ only when that step starts.
    # They are left out and listed apart. Without the runner's state file (or with --all) every
    # result on disk counts.
    state = {}
    if not args.all and rq.STATE_FILE.exists():
        state = json.loads(rq.STATE_FILE.read_text(encoding="utf-8"))
    last_steps = {(n, k): f"cost_N{n}_k{k}" for n, k in rq.COST_LAST}

    def started(step):
        return not state or bool(state.get(step, {}).get("set_aside_done"))

    def take(n_clients, k1, logs, needed, what, tmp, phase_map=None, step=None, drop_first=None):
        """The row of one combination if its logs hold every round (or --partial), else None;
        an incomplete one with any round logged is noted in incomplete, one whose step has not
        started yet in earlier."""
        if not (Path(logs[0]).exists() and Path(logs[1]).exists()):
            return None
        if step and not started(step):
            full = full_rounds(logs[0], logs[1], n_clients)
            earlier.append(f"{what} N={n_clients} k1={k1}: {full} round(s), step {step} not started")
            return None
        full = full_rounds(logs[0], logs[1], n_clients)
        complete = full >= needed
        if not complete and full:
            incomplete.append(f"{what} N={n_clients} k1={k1}: {full} of {needed} rounds")
        if not complete and not args.partial:
            return None
        row = cost_row(n_clients, k1, args.backend, args.level,
                       args.drop_first if drop_first is None else drop_first, tmp, logs=logs,
                       phase_map=phase_map)
        if row is not None:
            row["complete"] = int(complete)
        return row

    with tempfile.TemporaryDirectory() as tmp:
        for n_clients, k1 in pairs:
            row = take(n_clients, k1, cost_sweep.log_paths(n_clients, k1, args.backend, args.level),
                       rq.ROUNDS, "cost", tmp, step=last_steps.get((n_clients, k1), f"cost_N{n_clients}"))
            if row is not None:
                rows.append(row)
        if args.backend == rq.BACKEND and args.level == rq.LEVEL:
            for n_clients in rq.BREA_N:
                # where BREA runs a single round (N = 20..40) the tables use that round
                rounds = rq.brea_rounds(n_clients)
                drop = args.drop_first if rounds > args.drop_first else 0
                if drop != args.drop_first:
                    brea_round_one.append(n_clients)
                row = take(n_clients, 1, rq.brea_log_paths(n_clients)[:2], rounds, "BREA", tmp,
                           phase_map=BREA_PHASES, step=f"brea_N{n_clients}", drop_first=drop)
                if row is not None:
                    row["t2"] = 2 * (n_clients // 3) - 1          # BREA: no degree reduction
                    brea_rows.append(row)
    if incomplete:
        print("not complete" + (" (marked *):" if args.partial else " (left out; --partial to add):"))
        for line in incomplete:
            print("  " + line)
    for path, table, what in ((csv_path, rows, "cost-sweep"), (brea_csv, brea_rows, "BREA")):
        if not table:
            print(f"no {'' if args.partial else 'complete '}{what} combination of {config} under runs/")
            continue
        write_csv(path, table)
        print(f"{len(table)} rows -> {path}")

    entries = e2e_entries(Path("e2e_results") / config, None if args.all else e2e_grid)
    for n_clients, k1, label, combo in [e for e in entries if not started(rq.e2e_step_name(e[0], e[1]))]:
        earlier.append(f"e2e N={n_clients} k1={k1} {label}: step {rq.e2e_step_name(n_clients, k1)} not started")
    entries = [e for e in entries if started(rq.e2e_step_name(e[0], e[1]))]
    if earlier:
        print("earlier results, from before this re-measurement (left out; moved to _set_aside/ when "
              "their step starts):")
        for line in earlier:
            print("  " + line)
    parts = [f"# Re-measurement, {config}, rounds after the first {args.drop_first} left out\n"]
    if earlier:
        parts.append("Earlier results still on disk, from before this re-measurement (left out; "
                     "run_remeasure.py moves them to _set_aside/ when their step starts): "
                     + "; ".join(earlier) + ".")
    if incomplete:
        parts.append("Not complete" + (" (in the tables, marked *): " if args.partial else
                                       " (left out of the tables): ") + "; ".join(incomplete) + ".")
    str_rows = {}
    for r in rows:
        cells = {k: str(v) for k, v in r.items()}
        if not r.get("complete", 1):
            cells["N"] = f"{r['N']}*"
        str_rows[(int(r["N"]), int(r["k1"]))] = cells
    note = ("Cost sweep (in-memory transport, so no communication time). S = server, C = client; "
            "per-round means over the rounds used (column iters of the CSV).\n\n")
    t4 = [dash_zero_volume(str_rows[(n, n // 3)]) for n in rq.TABLE4_N if (n, n // 3) in str_rows]
    t5 = [dash_zero_volume(str_rows[(n, k)]) for n, ks in sorted(rq.TABLE5.items()) for k in ks
          if (n, k) in str_rows]
    if t4:
        parts.append("## Table 4: t = k, N = 10..100\n\n" + note + tabulate_cost.table_a(t4))
    if t5:
        missing = [f"N = {n}, k1 = {k}" for n, k in rq.COST_LAST if (n, k) not in str_rows]
        parts.append("## Table 5: N = 50 and N = 100 over k1\n\n" + note + tabulate_cost.table_a(t5)
                     + (f"\n\n({', '.join(missing)}: no complete log)" if missing else ""))
    if args.all and rows:
        parts.append("## Every (N, k1) on disk\n\n" + note + tabulate_cost.table_a([dash_zero_volume(r) for r in str_rows.values()]))
    t6 = table_six(rows, brea_rows, rq.TABLE6_N)
    if t6:
        parts.append("## Table 6: ours (k1 = 1) and BREA\n\n"
                     "S = server, C = client; per party and round, over the rounds used. BREA's timers are put "
                     "into the phases as the paper states it: Enroll = dealing updates via Pedersen VSS and "
                     "verifying them, Verify = pairwise distances and Multi-Krum, Agg = aggregating the "
                     "selected clients' updates (remeasure_tables.BREA_PHASES)."
                     + (f" BREA at N = {', '.join(map(str, brea_round_one))}: one round run, so that round "
                        f"(round 1) is the figure; elsewhere the rounds after the first {args.drop_first}."
                        if brea_round_one else "") + "\n\n" + t6)
    if entries:
        c1, abnormal = table_lan_wan(entries, args.drop_first, rq.ROUNDS, args.partial)
        if abnormal:
            print("network runs that did not end well with every round"
                  + (" (in (c1), N marked *):" if args.partial else " (left out of (c1)):"))
            for line in abnormal:
                print("  " + line)
        parts.append("## (c1) Network runs, per link\n\n"
                     "comp and comm are per-party means; round s = S-comp + slowest client's comp and comm "
                     "per region, round by round, mean ± sd over the rounds used. A combination still "
                     "running is not listed (its log is combined when it ends).\n\n"
                     + (c1 or "(no run that ended well)")
                     + ("\n\nDid not end well with every round"
                        + (" (in the tables above, N marked *):\n\n" if args.partial else
                           " (left out of the tables above):\n\n")
                        + "\n".join(f"- {line}" for line in abnormal) if abnormal else ""))
        groups = (("LAN", rq.link_label(1000, "0.2")), ("WAN", rq.link_label(10, "80")))
        grid = rq.E2E_GRID
        if args.all:
            grid = {}
            for n_clients, k1, _, _ in entries:
                grid.setdefault(n_clients, set()).add(k1)
            grid = {n: sorted(ks) for n, ks in grid.items()}
        cols = picks or breakdown_columns(entries, grid, [label for _, label in groups], rq.ROUNDS,
                                          args.partial, args.drop_first)
        head = ("## (c2) Communication time by what causes it (s per round)\n\n"
                "Each row is how long that exchange took, from the sync where it starts until the last "
                "party is done, the mean over the rounds used. S → C and C ↔ C rows: the slowest client of "
                "the round (for C ↔ C its send and receive of that message together; verify_result's "
                "receive is counted, as party.py's receive loop no longer waits on peers already received). "
                "C → S rows: the server's receive, until the last client's message is in (a client's send "
                "returns once the bytes are in its socket buffer, without the delay). "
                "FedAvg = what plain FedAvg sends anyway (the clients' updates go up as shares here, so "
                "they are under secret sharing); secret sharing = what putting it on FedAvg adds; "
                "RVFL-specific = the rest. S = server, C = client. "
                "Columns: " + ("--pick" if picks else
                               "the lowest and highest k1 of the lowest and highest N "
                               + ("with a round used on both links" if args.partial else "finished on both links"))
                + ("; * = the run did not finish every round" if args.partial or picks else "") + ".\n\n")
        if cols:
            md, tex = table_breakdown(entries, cols, groups, args.drop_first, rq.ROUNDS, args.c2_digits)
            parts.append(head + md + "\n\n```latex\n" + tex + "\n```")
        else:
            parts.append(head + "(no N has every k1 finished on both links yet)")
    else:
        parts.append(f"(no network run under e2e_results/{config}/)")
    out_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"tables -> {out_path}")


if __name__ == "__main__":
    main()