import json
from typing import Dict, Any, Tuple

# --- classification ---------------------------------------------------------
# A metric is COMMUNICATION if its (｜ _ms-stripped) key starts with any of these;
# everything else is COMPUTATION. Prefix-based, so new compute fields such as
# `recompute_server_share` are picked up automatically without editing this list.
COMM_PREFIXES: Tuple[str, ...] = ("send", "receive", "recv")

# Phases excluded from the protocol comp/comm totals. `train` is local SGD, measured
# separately from the WAN protocol (the E2E "protocol wall-clock" does not include it).
# Set EXCLUDE_PHASES = () to fold training back into the totals.
EXCLUDE_PHASES: Tuple[str, ...] = ("train",)


def _kind(sub_key: str) -> str:
    return "comm" if sub_key.startswith(COMM_PREFIXES) else "comp"


def summarize(file_path: str, iterations=None, client_agg: str = "mean") -> Dict[str, Any]:
    """Per-party comp/comm totals. `iterations`, when given, is the set of iteration ids to
    include; None keeps all.

    `client_agg` says how the clients of one iteration are folded into one figure per metric:
      "mean"  the mean over every (party, iteration) record -- a typical client's own time,
              which is what the per-party table cells report.
      "max"   per iteration the slowest client, then the mean over iterations -- the time an
              exchange actually takes, since every communication region ends when its last
              client is done and the next region cannot start before that. Where every client
              waits for the same last arrival (client-to-client exchanges) the two agree; where
              the server delivers to clients one after another (receive_server, receive_TS,
              receive_global_weights) the mean is roughly half the max.
    The server has one record per iteration, so both give the same server figures."""
    if client_agg not in ("mean", "max"):
        raise ValueError(f"client_agg must be 'mean' or 'max', not {client_agg!r}")
    # party -> kind -> (phase, metric) -> iteration -> [values of every party record]
    acc: Dict[str, Dict[str, Dict[Tuple[str, str], Dict[Any, list]]]] = {
        "server": {"comp": {}, "comm": {}},
        "client": {"comp": {}, "comm": {}},
    }

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data: Dict[str, Any] = json.loads(line)
            it = data.get("iteration")
            if iterations is not None and it not in iterations:
                continue
            party = "server" if data.get("party_id", 0) == -1 else "client"
            time_logs: Dict[str, Any] = data.get("time_logs_ms", {})

            for phase, metrics in time_logs.items():
                if phase in EXCLUDE_PHASES:
                    continue
                for metric_key, val in metrics.items():
                    sub = metric_key[:-3] if metric_key.endswith("_ms") else metric_key
                    acc[party][_kind(sub)].setdefault((str(phase), sub), {}) \
                        .setdefault(it, []).append(float(val))

    def _fold(per_iter: Dict[Any, list]) -> float:
        if client_agg == "mean":
            # pooled mean over every record, as before
            total = sum(sum(v) for v in per_iter.values())
            count = sum(len(v) for v in per_iter.values())
            return total / count
        # slowest party of each iteration, then the mean over iterations
        return sum(max(v) for v in per_iter.values()) / len(per_iter)

    # one figure per metric -> sum by phase -> total in seconds
    out: Dict[str, Any] = {}
    for party in ("server", "client"):
        out[party] = {}
        for kind in ("comp", "comm"):
            means_ms = {pm: _fold(per_iter) for pm, per_iter in acc[party][kind].items()}
            per_phase_ms: Dict[str, float] = {}
            for (phase, _metric), v in means_ms.items():
                per_phase_ms[phase] = per_phase_ms.get(phase, 0.0) + v
            out[party][kind] = {
                "per_metric_ms": {f"{ph} - {m}": round(v, 3) for (ph, m), v in means_ms.items()},
                "per_phase_ms": {ph: round(v, 3) for ph, v in per_phase_ms.items()},
                "total_s": round(sum(per_phase_ms.values()) / 1000.0, 3),
            }
    return out


if __name__ == "__main__":
    # [B6] Path from the command line: `python comp_log_analyzer_revised.py <processing_time_log.jsonl>`.
    import sys
    file_path: str = sys.argv[1] if len(sys.argv) > 1 else "logs/RVFL/RVFL_noattack_netns/processing_time_log.jsonl"
    result = summarize(file_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n=== Table cells (seconds) ===")
    print(f"Server-comp = {result['server']['comp']['total_s']}")
    print(f"Server-comm = {result['server']['comm']['total_s']}")
    print(f"Client-comp = {result['client']['comp']['total_s']}")
    print(f"Client-comm = {result['client']['comm']['total_s']}")