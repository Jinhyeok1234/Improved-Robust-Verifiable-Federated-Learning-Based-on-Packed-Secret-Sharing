#!/usr/bin/env python3
"""Render the main-text table from the cost-sweep CSV.

    python tabulate_cost.py                      # cost_sweep_<default backend><default level>.csv
    python tabulate_cost.py --backend zp --level 128     # cost_sweep_zp128.csv
    python tabulate_cost.py --csv some/other/path.csv    # an explicit path wins over both

One row per (N, k1): the communication VOLUME, split only into broadcast and unicast for the
server and for the client, and the computation TIME, split by phase with a total. Figures are
per-round means over the rounds in the log.

There is deliberately no communication-time column. cost_sweep.py runs with the in-memory
transport (party.NETWORK_CONFIG backend "inmemory"), where a message is handed over through a
shared-memory queue and no link is crossed, so the send and receive timings of that run
describe the queue and not a network. The timings that belong to a network come from the netns
sweep, where tc shapes each link to a stated bandwidth and RTT; tabulate_e2e.py renders those.
"""
import argparse
import csv
from pathlib import Path


def fmt(value, scale=1.0, digits=2):
    try:
        return f"{float(value) / scale:,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def render(rows, columns, headers):
    widths = [max(len(h), *(len(str(c(r))) for r in rows)) for h, c in zip(headers, columns)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [line, rule]
    for r in rows:
        out.append("| " + " | ".join(str(c(r)).ljust(w) for c, w in zip(columns, widths)) + " |")
    return "\n".join(out)


def total_ms(r, party):
    """The party's total computation in ms: the sum of its phases as remeasure_tables.py writes
    it (*_comp_total_ms), or, in a CSV without that column, total_s, which is rounded to 1 ms."""
    v = r.get(f"{party}_comp_total_ms")
    if v not in (None, ""):
        return v
    return float(r.get(f"{party}_comp_total_s", 0)) * 1000


def table_a(rows):
    columns = [
        lambda r: r["N"],
        lambda r: r["k1"],
        lambda r: fmt(r.get("server_broadcast_avg_bytes"), 1e6),
        lambda r: fmt(r.get("server_unicast_avg_bytes"), 1e6),
        lambda r: fmt(r.get("server_comp_enrollment_ms"), 1, 1),
        lambda r: fmt(r.get("server_comp_verification_ms"), 1, 1),
        lambda r: fmt(r.get("server_comp_aggregation_ms"), 1, 1),
        lambda r: fmt(total_ms(r, "server"), 1, 1),
        lambda r: fmt(r.get("client_broadcast_avg_bytes"), 1e3),
        lambda r: fmt(r.get("client_unicast_avg_bytes"), 1e6),
        lambda r: fmt(r.get("client_comp_enrollment_ms"), 1, 1),
        lambda r: fmt(r.get("client_comp_verification_ms"), 1, 1),
        lambda r: fmt(r.get("client_comp_aggregation_ms"), 1, 1),
        lambda r: fmt(total_ms(r, "client"), 1, 1),
    ]
    headers = ["N", "k1",
               "S bcast MB", "S ucast MB",
               "S enroll ms", "S verify ms", "S agg ms", "S total ms",
               "C bcast KB", "C ucast MB",
               "C enroll ms", "C verify ms", "C agg ms", "C total ms"]
    return render(rows, columns, headers)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="explicit CSV path; overrides --backend and --level")
    ap.add_argument("--backend", default=None,
                    help="ec or zp (default: the crypto_params default)")
    ap.add_argument("--level", default=None,
                    help="128, 192, 256 or legacy512 (default: the crypto_params default)")
    args = ap.parse_args()

    # cost_sweep.py names its output cost_sweep_<backend><level>.csv so that one configuration
    # does not overwrite another, so the same name is rebuilt here from the same two values.
    if args.csv is None:
        import crypto_params
        backend = args.backend or crypto_params.DEFAULT_BACKEND
        level = args.level or crypto_params.DEFAULT_LEVEL
        args.csv = f"cost_sweep_{backend}{level}.csv"

    if not Path(args.csv).exists():
        here = sorted(Path(".").glob("cost_sweep_*.csv"))
        print(f"{args.csv} not found.")
        if here:
            print("CSVs in this directory: " + ", ".join(p.name for p in here))
        else:
            print("No cost_sweep_*.csv here yet; run cost_sweep.py first.")
        return

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: (int(r["N"]), int(r["k1"])))
    if not rows:
        print("no rows")
        return

    head = rows[0]
    print(f"backend={head.get('backend')} level={head.get('level')} "
          f"rounds averaged={head.get('iters')}  ({len(rows)} combinations)\n")

    print("Communication volume and phase-wise computation time")
    print("  S = server, C = client. Per-round means; the transport is in-memory, so there is")
    print("  no communication time here by design -- see tabulate_e2e.py for the network runs.\n")
    print(table_a(rows))


if __name__ == "__main__":
    main()