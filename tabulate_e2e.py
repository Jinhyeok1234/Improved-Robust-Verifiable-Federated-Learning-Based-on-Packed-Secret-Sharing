#!/usr/bin/env python3
"""Render the two discussion tables from the netns end-to-end sweep.

    python tabulate_e2e.py                                  # the default backend and level
    python tabulate_e2e.py --backend zp --level 128         # e2e_results/zp128
    python tabulate_e2e.py --all                            # every configuration present
    python tabulate_e2e.py --label 10Mbps_80ms              # one link setting
    python tabulate_e2e.py --table breakdown --pick 50,1 50,4 50,8 50,16

Table 2 is the LAN/WAN table: one row per (N, k1), one block per (bandwidth, RTT) setting,
with server computation, server communication, client computation, client communication and
the end-to-end wall clock.

Table 3 is table 2 narrowed to a few representative (N, k1) and split by which part of the
protocol each communication belongs to -- the federated-learning exchange, the secret-sharing
exchange, and the parts specific to this protocol.

Input is what the netns sweep writes under
e2e_results/{backend}{level}/N{N}_k{k1}/{bandwidth}_{rtt}ms/: processing_time_log.jsonl for the
per-phase timings and terminal_log.txt for the end-to-end wall clock the parent process
measured between the start and end barriers. Point --root at one backend/level folder to render
a single configuration, or leave it at e2e_results to render whatever is there, with the
configuration named in its own column.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

import comp_log_analyzer_revised

# Communication metric -> which part of the protocol pays for it. comp_log_analyzer_revised
# strips the trailing "_ms" from every metric name, so the names here carry no suffix; every
# send/receive key the logs produce appears exactly once below, and unknown() reports any that
# does not, rather than letting it vanish from the total.
BREAKDOWN = [
    ("FL - global weight update", ["send_updated_global_weights", "receive_global_weights"]),
    ("FL - root update g_S",      ["send_server_data", "receive_server"]),
    ("SS - first-level share",    ["send_first_level_share", "receive_first_level_share"]),
    ("SS - aggregated share",     ["send_agg_share", "receive_agg_share"]),
    ("Ours - second-level share", ["send_second_level_share", "receive_second_level_share"]),
    ("Ours - verify / TS",        ["send_verify_result", "receive_verify_result",
                                   "send_s_tilde", "receive_s_tilde",
                                   "send_TS", "receive_TS"]),
]

CLASSIFIED = {m for _, metrics in BREAKDOWN for m in metrics}

E2E_RE = re.compile(r"iteration\s+\d+:\s*([\d.]+)\s*ms")
ROUND_RE = re.compile(r"iteration\s+(\d+):\s*([\d.]+)\s*ms")
DIR_RE = re.compile(r"^N(\d+)_k(\d+)$")


def round_seconds(server_comp_s, client_comp_s, client_comm_s):
    """One round as a deployment would see it.

    The client figures passed in are the slowest client's, per communication region and per
    round (comp_log_analyzer_revised.summarize with client_agg="max"), not the per-client means
    the table's C-comp / C-comm cells show: every region ends when its last client is done,
    and the next region does not start before that, so the round is the sum over regions of
    the slowest client's time. In client-to-client exchanges every client waits for the same
    last arrival and the two agree; where the server delivers to clients one after another
    (receive_server, receive_TS, receive_global_weights) the mean is roughly half the max.

    The netns run puts all parties on one machine and serializes their local computation
    (party._SERIALIZE_COMPUTE), so each party's own computation is measured with the machine
    to itself -- which is what it would have on its own host. _sync() then parks every
    participant until the last one has finished computing, and the timers are re-based after
    each barrier, so the waiting a party does because of that serialization is excluded from
    its communication figures. A party's own numbers are therefore already deployment figures.

    A round is what one client waits through: its own computation, its own communication, and
    the server's computation, which happens while the client is parked and so appears in none
    of the client's own timers. The server's communication is NOT added: every server-client
    transfer is timed once at the sender and once at the receiver, so it is already inside the
    client's figure and adding the server's would count the same wire twice. This follows the
    convention tabulate_breakdown.py states -- report the client's timeline, not the sum of
    both parties'.
    """
    return server_comp_s + client_comp_s + client_comm_s


def rounds_in(combo_dir):
    """Iteration ids recorded in terminal_log.txt, in order."""
    path = combo_dir / "terminal_log.txt"
    if not path.exists():
        return []
    return [int(m.group(1)) for m in ROUND_RE.finditer(path.read_text(encoding="utf-8"))]


def e2e_seconds(combo_dir, iterations=None):
    """Mean wall clock of the experiment, over the rounds recorded in terminal_log.txt.

    This is how long the sweep took, not how long a round takes: every party's serialized
    computation happens inside this window, so it carries roughly N times the client
    computation a deployment would see. Kept as a sanity column, not a result.
    """
    path = combo_dir / "terminal_log.txt"
    if not path.exists():
        return None
    values = [float(m.group(2)) for m in ROUND_RE.finditer(path.read_text(encoding="utf-8"))
              if iterations is None or int(m.group(1)) in iterations]
    return sum(values) / len(values) / 1000.0 if values else None


def status_of(combo_dir):
    path = combo_dir / "terminal_log.txt"
    if not path.exists():
        return "no log"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("STATUS:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def per_metric_means(time_log):
    """party -> metric name -> mean ms per round, for the communication metrics only."""
    summary = comp_log_analyzer_revised.summarize(str(time_log))
    out = {}
    for party in ("server", "client"):
        means = summary.get(party, {}).get("comm", {}).get("per_metric_ms", {})
        # keys arrive as "<phase> - <metric>"; the metric name alone is what BREAKDOWN uses
        flat = {}
        for key, val in means.items():
            metric = key.split(" - ", 1)[-1]
            flat[metric] = flat.get(metric, 0.0) + val
        out[party] = flat
    return out


def walk(root, label_filter):
    """Yield (N, k1, label, combo_dir) for every finished combination under root.

    rglob rather than glob, so that both e2e_results/ and e2e_results/ec256/ work as the root.
    """
    found = []
    for n_dir in sorted(root.rglob("N*_k*")):
        if not n_dir.is_dir():
            continue
        m = DIR_RE.match(n_dir.name)
        if not m:
            continue
        n_clients, k1 = int(m.group(1)), int(m.group(2))
        for combo_dir in sorted(n_dir.iterdir()):
            if not combo_dir.is_dir():
                continue
            if label_filter and combo_dir.name != label_filter:
                continue
            if (combo_dir / "processing_time_log.jsonl").exists():
                # the folder above N{N}_k{k1} names the backend and level when there is one
                config = n_dir.parent.name if n_dir.parent != root else ""
                label = f"{config}/{combo_dir.name}" if config else combo_dir.name
                found.append((n_clients, k1, label, combo_dir))
    found.sort(key=lambda r: (r[2], r[0], r[1]))
    return found


def render(rows, headers):
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def table_lan_wan(entries, skip_first=False):
    """One row per (N, k1). The four comp/comm cells are per-party means over the rounds used;
    'round' is the mean over those rounds of round_seconds(), which takes each region's slowest
    client, and with more than one round it also carries the sample standard deviation over
    rounds. --skip-first leaves round 1 out: it carries the one-time TCP
    connection setup to every peer, which a deployment pays once, not per round."""
    by_label = {}
    for n_clients, k1, label, combo_dir in entries:
        by_label.setdefault(label, []).append((n_clients, k1, combo_dir))

    blocks = []
    for label, items in by_label.items():
        rows = []
        for n_clients, k1, combo_dir in sorted(items):
            log = str(combo_dir / "processing_time_log.jsonl")
            ids = rounds_in(combo_dir)
            if skip_first and len(ids) > 1:
                ids = ids[1:]
            use = set(ids) if ids else None
            summary = comp_log_analyzer_revised.summarize(log, use)
            s_comp = summary['server']['comp']['total_s']
            s_comm = summary['server']['comm']['total_s']
            c_comp = summary['client']['comp']['total_s']
            c_comm = summary['client']['comm']['total_s']
            # round time from the slowest client of each region, round by round
            per_round = []
            for it in (ids if ids else []):
                one = comp_log_analyzer_revised.summarize(log, {it}, client_agg="max")
                per_round.append(round_seconds(one['server']['comp']['total_s'],
                                               one['client']['comp']['total_s'],
                                               one['client']['comm']['total_s']))
            if per_round:
                rt = statistics.mean(per_round)
            else:
                slow = comp_log_analyzer_revised.summarize(log, use, client_agg="max")
                rt = round_seconds(s_comp, slow['client']['comp']['total_s'],
                                   slow['client']['comm']['total_s'])
            # mean over the rounds used, with the sample standard deviation across rounds
            rt_cell = (f"{rt:.2f} \u00b1 {statistics.stdev(per_round):.2f}"
                       if len(per_round) > 1 else f"{rt:.2f}")
            rows.append([
                n_clients, k1, len(ids) if ids else "-",
                f"{s_comp:.2f}", f"{s_comm:.2f}", f"{c_comp:.2f}", f"{c_comm:.2f}",
                rt_cell,
                f"{wall:.2f}" if (wall := e2e_seconds(combo_dir, use)) is not None else "-",
                status_of(combo_dir),
            ])
        blocks.append(f"### {label}\n\n" + render(
            rows, ["N", "k1", "rounds", "S-comp s", "S-comm s", "C-comp s", "C-comm s",
                   "round s", "wall s", "status"]))
    return "\n\n".join(blocks)


def table_per_round(entries, label_filter=None):
    """One row per round of each combination, so a drift across rounds can be traced to the
    column it lives in: computation (the machine slowing under sustained load) or
    communication (the network)."""
    rows = []
    for n_clients, k1, label, combo_dir in entries:
        log = str(combo_dir / "processing_time_log.jsonl")
        for it, wall_ms in [(int(m.group(1)), float(m.group(2))) for m in
                            ROUND_RE.finditer((combo_dir / "terminal_log.txt").read_text(encoding="utf-8"))]:
            one = comp_log_analyzer_revised.summarize(log, {it})
            slow = comp_log_analyzer_revised.summarize(log, {it}, client_agg="max")
            s_comp = one['server']['comp']['total_s']
            s_comm = one['server']['comm']['total_s']
            c_comp = one['client']['comp']['total_s']
            c_comm = one['client']['comm']['total_s']
            rt = round_seconds(s_comp, slow['client']['comp']['total_s'],
                               slow['client']['comm']['total_s'])
            rows.append([label, n_clients, k1, it,
                         f"{s_comp:.2f}", f"{s_comm:.2f}", f"{c_comp:.2f}", f"{c_comm:.2f}",
                         f"{rt:.2f}", f"{wall_ms / 1000:.2f}"])
    if not rows:
        return "(no rounds)"
    return render(rows, ["setting", "N", "k1", "round", "S-comp s", "S-comm s", "C-comp s",
                         "C-comm s", "round s", "wall s"])


def table_breakdown(entries, picks):
    wanted = set(picks) if picks else None
    rows = []
    for n_clients, k1, label, combo_dir in entries:
        if wanted and (n_clients, k1) not in wanted:
            continue
        means = per_metric_means(combo_dir / "processing_time_log.jsonl")
        for party in ("server", "client"):
            row = [label, n_clients, k1, "S" if party == "server" else "C"]
            total = 0.0
            for _, metrics in BREAKDOWN:
                val = sum(means[party].get(m, 0.0) for m in metrics)
                total += val
                row.append(f"{val:.1f}")
            row.append(f"{total:.1f}")
            leftover = sorted(set(means[party]) - CLASSIFIED)
            if leftover:
                print(f"  note: {party} metric(s) not in the breakdown: {leftover}")
            rows.append(row)
    if not rows:
        return "(no combination matched)"
    headers = ["setting", "N", "k1", "party"] + [name for name, _ in BREAKDOWN] + ["total ms"]
    return render(rows, headers)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="explicit results folder; overrides --backend, --level and --all")
    ap.add_argument("--backend", default=None,
                    help="ec or zp (default: the crypto_params default)")
    ap.add_argument("--level", default=None,
                    help="128, 192, 256 or legacy512 (default: the crypto_params default)")
    ap.add_argument("--all", action="store_true",
                    help="render every backend/level folder under e2e_results at once")
    ap.add_argument("--table", choices=["lanwan", "breakdown", "both", "rounds"], default="lanwan",
                    help="rounds: one row per round of each combination")
    ap.add_argument("--label", default=None,
                    help="restrict to one (bandwidth, RTT) folder, e.g. 10Mbit_80ms")
    ap.add_argument("--skip-first", action="store_true",
                    help="leave round 1 out of the means (one-time connection setup)")
    ap.add_argument("--pick", nargs="*", default=None,
                    help="for the breakdown table: N,k1 pairs such as 50,1 50,4 50,8 50,16")
    args = ap.parse_args()

    # The netns sweep writes e2e_results/<backend><level>/..., the same naming cost_sweep.py
    # uses for its CSV, so the folder is rebuilt here from the same two values.
    if args.root is None:
        if args.all:
            args.root = "e2e_results"
        else:
            import crypto_params
            backend = args.backend or crypto_params.DEFAULT_BACKEND
            level = args.level or crypto_params.DEFAULT_LEVEL
            args.root = str(Path("e2e_results") / f"{backend}{level}")

    if not Path(args.root).is_dir():
        print(f"{args.root} not found.")
        here = sorted(p.name for p in Path("e2e_results").iterdir() if p.is_dir()) \
            if Path("e2e_results").is_dir() else []
        if here:
            print("configurations under e2e_results: " + ", ".join(here))
        else:
            print("No e2e_results yet; run the netns sweep first (RVFL_E2E=1).")
        return

    picks = None
    if args.pick:
        picks = []
        for item in args.pick:
            n_clients, k1 = item.split(",")
            picks.append((int(n_clients), int(k1)))

    entries = walk(Path(args.root), args.label)
    if not entries:
        print(f"no finished combination under {args.root}/")
        return

    if args.table in ("lanwan", "both"):
        print("Table 2 -- LAN/WAN timings per (N, k1)")
        print("  comp and comm are per-party, per-round means. 'round' is the deployment round")
        print("  time, S-comp + C-comp + C-comm with the client terms taken from the slowest")
        print("  client of each region (so it is not the sum of the cells); see round_seconds()")
        print("  for that and for why the server's comm is not added. 'wall' is how long the")
        print("  experiment itself took, which includes")
        print("  every party's serialized computation -- a sanity column, not a result.\n")
        print(table_lan_wan(entries, args.skip_first))
        print()

    if args.table == "rounds":
        print("Per round -- comp and comm are per-party means within that round; 'round' uses")
        print("  the slowest client of each region, see round_seconds()")
        print(table_per_round(entries))
        return

    if args.table in ("breakdown", "both"):
        print("Table 3 -- communication time by which part of the protocol pays for it")
        print("  Per-round means in ms. FL is the federated-learning exchange, SS the secret")
        print("  sharing, Ours the parts specific to this protocol.\n")
        print(table_breakdown(entries, picks))


if __name__ == "__main__":
    main()