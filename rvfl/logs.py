"""The files one run point writes, and the summaries read from them.

A run point is one setting of one experiment (run.py writes one folder per point):

  config.json    the resolved settings
  status.json    {"status": "ok" | "error: ..." | "timeout ..."}, written when the point ends
  parties.jsonl  one record per (round, party): {"iteration", "party_id", "time_logs_ms":
                 {phase: {metric_ms: ms}}, "comm_logs_bytes": {"unicast": {...}, "broadcast":
                 {...}}} (cost and e2e kinds; e2e adds "lock_wait_ms")
  values.jsonl   one record per round from the server: {"iteration", "values": {...}}
                 (cosines, trust-score checks, filter choices)
  rounds.jsonl   e2e kind: the wall-clock time of every round, {"iteration", "wall_ms"}
  eval.jsonl     train kind: {"iteration", "test_acc", "train_acc", "asr"} per evaluation
  checkpoints/   iter<r>.pt, the global model after round r (when checkpoints are kept)

The published results keep parties.jsonl gzip-compressed (parties.jsonl.gz); read_jsonl reads
either.
"""
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Tuple

SERVER_ID = -1

# A timing metric is communication if its name (without "_ms") starts with one of these;
# everything else is computation.
COMM_PREFIXES: Tuple[str, ...] = ("send", "receive", "recv")
# Phases left out of the protocol totals: local training is not part of the protocol.
EXCLUDE_PHASES: Tuple[str, ...] = ("train",)


def link_label(link):
    """ "10000:0.2" (Mbit/s : RTT ms) -> "10Gbps_0.2ms", the folder name of an end-to-end link."""
    bw, rtt = link.split(":")
    bw = int(float(bw))
    bw_s = f"{bw // 1000}Gbps" if bw >= 1000 and bw % 1000 == 0 else f"{bw}Mbps"
    return f"{bw_s}_{float(rtt):g}ms"


def read_jsonl(path):
    """The records of a JSON-lines file, or of its gzip-compressed copy <path>.gz (as the
    published results keep them); [] when neither exists."""
    path = Path(path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    elif path.with_name(path.name + ".gz").exists():
        import gzip
        text = gzip.decompress(path.with_name(path.name + ".gz").read_bytes()).decode("utf-8")
    else:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(point_dir):
    return read_json(Path(point_dir) / "config.json", {})


def read_status(point_dir):
    return read_json(Path(point_dir) / "status.json", {"status": "not finished"})


def iterations(records):
    return sorted({r["iteration"] for r in records})


def _kind(metric: str) -> str:
    return "comm" if metric.startswith(COMM_PREFIXES) else "comp"


def summarize_time(records, its=None, client_agg: str = "mean") -> Dict[str, Any]:
    """Computation and communication time per party kind (server / client).

    its, when given, is the set of rounds to include. client_agg folds the clients of a round
    into one figure per metric:
      "mean"  the mean over every (client, round) record: a typical client's own time
      "max"   per round the slowest client, then the mean over rounds: the time an exchange
              takes, since a communication region ends when its last client is done. Where the
              server serves the clients one after another (receive_server, receive_TS,
              receive_global_weights) the mean is about half the max.
    The server has one record per round, so both give the same server figures.

    Returns {party: {"comp" | "comm": {"per_metric_ms", "per_phase_ms", "total_s"}}}.
    """
    if client_agg not in ("mean", "max"):
        raise ValueError(f"client_agg must be 'mean' or 'max', not {client_agg!r}")
    acc = {"server": {"comp": {}, "comm": {}}, "client": {"comp": {}, "comm": {}}}
    for rec in records:
        it = rec.get("iteration")
        if its is not None and it not in its:
            continue
        party = "server" if rec.get("party_id", 0) == SERVER_ID else "client"
        for phase, metrics in rec.get("time_logs_ms", {}).items():
            if phase in EXCLUDE_PHASES:
                continue
            for key, val in metrics.items():
                metric = key[:-3] if key.endswith("_ms") else key
                acc[party][_kind(metric)].setdefault((str(phase), metric), {}) \
                    .setdefault(it, []).append(float(val))

    def fold(per_round):
        if client_agg == "mean":
            return sum(sum(v) for v in per_round.values()) / sum(len(v) for v in per_round.values())
        return sum(max(v) for v in per_round.values()) / len(per_round)

    out = {}
    for party in ("server", "client"):
        out[party] = {}
        for kind in ("comp", "comm"):
            means = {pm: fold(per_round) for pm, per_round in acc[party][kind].items()}
            per_phase = {}
            for (phase, _), v in means.items():
                per_phase[phase] = per_phase.get(phase, 0.0) + v
            out[party][kind] = {
                "per_metric_ms": {f"{ph} - {m}": round(v, 3) for (ph, m), v in means.items()},
                "per_phase_ms": {ph: round(v, 3) for ph, v in per_phase.items()},
                "total_s": round(sum(per_phase.values()) / 1000.0, 3),
            }
    return out


def summarize_bytes(records, its=None) -> Dict[str, float]:
    """Mean bytes a party sends per round, split into broadcast and unicast, per party kind. A
    broadcast counts its payload once (the size of one copy), a unicast every copy it sends."""
    tot = {"server": [0, 0, 0], "client": [0, 0, 0]}      # broadcast, unicast, records
    for rec in records:
        if its is not None and rec.get("iteration") not in its:
            continue
        logs = rec.get("comm_logs_bytes", {})
        b = sum(v.get("size_bytes", 0) for v in logs.get("broadcast", {}).values())
        u = sum(m.get("size_bytes", 0) for per_receiver in logs.get("unicast", {}).values()
                for m in per_receiver.values() if isinstance(m, dict))
        t = tot["server" if rec.get("party_id", 0) == SERVER_ID else "client"]
        t[0] += b
        t[1] += u
        t[2] += 1
    out = {}
    for party, (b, u, n) in tot.items():
        n = n or 1
        out[f"{party}_broadcast_avg"] = b / n
        out[f"{party}_unicast_avg"] = u / n
    return out


def round_seconds(server_comp_s, client_comp_s, client_comm_s):
    """One end-to-end round as one client waits through it: its own computation and
    communication, plus the server's computation, which happens while the clients wait and so
    appears in none of their timers.

    The client figures are the slowest client's, per region and per round
    (summarize_time(client_agg="max")): every region ends when its last client is done. Each
    party's computation was measured with the machine to itself (the compute lock serializes
    them) and the waiting for the lock is excluded from every timer, so the figures are what
    separate machines would see. The server's communication is not added: every server-client
    transfer is timed at both ends, so it is already in the client's figure.
    """
    return server_comp_s + client_comp_s + client_comm_s


def mean_round_seconds(point_dir, its=None):
    """Mean of round_seconds over the rounds of an end-to-end point (all rounds by default)."""
    records = read_jsonl(Path(point_dir) / "parties.jsonl")
    rounds = [it for it in iterations(records) if its is None or it in its]
    if not rounds:
        return None
    vals = []
    for it in rounds:
        s = summarize_time(records, {it}, client_agg="max")
        vals.append(round_seconds(s["server"]["comp"]["total_s"], s["client"]["comp"]["total_s"],
                                  s["client"]["comm"]["total_s"]))
    return sum(vals) / len(vals)


def find_points(root, **settings):
    """Every point folder under root whose config.json has all the given settings."""
    found = []
    for cfg_path in sorted(Path(root).rglob("config.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if all(cfg.get(k) == v for k, v in settings.items()):
            found.append(cfg_path.parent)
    return found


def find_point(root, **settings):
    """The one point folder matching settings, or None; several matches are an error."""
    found = find_points(root, **settings)
    if len(found) > 1:
        raise SystemExit(f"{len(found)} runs under {root} match {settings}: {[str(p) for p in found]}")
    return found[0] if found else None


# --- summaries of one point ------------------------------------------------------------------

def used_rounds(records, drop_first):
    """The rounds after the first drop_first; all of them when that would leave none (a run of
    a single round, e.g. BREA at large N)."""
    its = iterations(records)
    return its[drop_first:] if len(its) > drop_first else its


def cost_summary(point_dir, drop_first=1, phase_map=None):
    """Per-party means over the rounds used: bytes sent (broadcast / unicast) and computation
    per phase in ms ({party}_comp_{phase}_ms, {party}_comp_total_ms, {party}_{broadcast,unicast}
    _bytes). phase_map {phase: [metric, ...]} puts the timers of a one-phase log (BREA) into
    phases. None when the point has no records."""
    records = read_jsonl(Path(point_dir) / "parties.jsonl")
    its = set(used_rounds(records, drop_first))
    if not its:
        return None
    time = summarize_time(records, its)
    out = {"rounds_used": sorted(its)}
    for party in ("server", "client"):
        comp = time[party]["comp"]
        out[f"{party}_comp_total_ms"] = round(sum(comp["per_phase_ms"].values()), 3)
        if phase_map:
            where = {m: ph for ph, ms in phase_map.items() for m in ms}
            for key, ms in comp["per_metric_ms"].items():
                phase = where.get(key.split(" - ", 1)[-1])
                if phase is None:
                    raise ValueError(f"{point_dir}: {party} timer {key} is in no phase of the phase map")
                col = f"{party}_comp_{phase}_ms"
                out[col] = round(out.get(col, 0.0) + ms, 3)
        else:
            for phase, ms in comp["per_phase_ms"].items():
                out[f"{party}_comp_{phase}_ms"] = ms
    for key, val in summarize_bytes(records, its).items():
        out[key.replace("_avg", "_bytes")] = val
    return out


def e2e_summary(point_dir, drop_first=1):
    """Per-party mean computation and communication (s) and the mean round time (round_seconds,
    round by round) over the rounds used of an end-to-end point."""
    records = read_jsonl(Path(point_dir) / "parties.jsonl")
    its = used_rounds(records, drop_first)
    if not its:
        return None
    mean = summarize_time(records, set(its))
    per_round = []
    for it in its:
        one = summarize_time(records, {it}, client_agg="max")
        per_round.append(round_seconds(one["server"]["comp"]["total_s"], one["client"]["comp"]["total_s"],
                                       one["client"]["comm"]["total_s"]))
    return {"rounds": len(its),
            "s_comp": mean["server"]["comp"]["total_s"], "s_comm": mean["server"]["comm"]["total_s"],
            "c_comp": mean["client"]["comp"]["total_s"], "c_comm": mean["client"]["comm"]["total_s"],
            "round": statistics.mean(per_round), "round_by_iteration": dict(zip(its, per_round))}


def comm_timers(point_dir, drop_first=1):
    """The communication timers (ms) of the rounds used: {"rounds": [...], "C": {round: {client:
    {"C:<metric>": ms}}}, "S": {round: {"S:<metric>": ms}}, "present": every name seen}. A metric
    found in two phases is summed."""
    records = read_jsonl(Path(point_dir) / "parties.jsonl")
    its = used_rounds(records, drop_first)
    out = {"rounds": its, "C": {}, "S": {}, "present": set()}
    wanted = set(its)
    for rec in records:
        it = rec.get("iteration")
        if it not in wanted:
            continue
        pid = rec.get("party_id", 0)
        if pid == SERVER_ID:
            slot, prefix = out["S"].setdefault(it, {}), "S:"
        else:
            slot, prefix = out["C"].setdefault(it, {}).setdefault(pid, {}), "C:"
        for phase, metrics in rec.get("time_logs_ms", {}).items():
            if phase in EXCLUDE_PHASES:
                continue
            for key, val in metrics.items():
                metric = key[:-3] if key.endswith("_ms") else key
                if _kind(metric) == "comm":
                    name = prefix + metric
                    slot[name] = slot.get(name, 0.0) + float(val)
                    out["present"].add(name)
    return out


def exchange_seconds(timers, metrics):
    """How long an exchange took (s), the mean over the rounds: per round the largest sum of its
    "C:" metrics over the clients plus its "S:" metrics of the server."""
    c_names = [m for m in metrics if m.startswith("C:")]
    s_names = [m for m in metrics if m.startswith("S:")]
    per_round = []
    for it in timers["rounds"]:
        clients = timers["C"].get(it, {}).values()
        slowest = max((sum(c.get(m, 0.0) for m in c_names) for c in clients), default=0.0) if c_names else 0.0
        per_round.append(slowest + sum(timers["S"].get(it, {}).get(m, 0.0) for m in s_names))
    return sum(per_round) / len(per_round) / 1000.0 if per_round else 0.0


def filter_rounds(point_dir):
    """round -> {"n", "round_k1", "round_t", "s_comp", "c_comp_max", "c_comm_max", "seconds"} of an
    end-to-end point, the clients restricted to the round's participants (with the filter off,
    every client): the slowest participant per timer, as in round_seconds."""
    point_dir = Path(point_dir)
    values = {r["iteration"]: r.get("values") or {} for r in read_jsonl(point_dir / "values.jsonl")}
    by_round = {}
    for rec in read_jsonl(point_dir / "parties.jsonl"):
        by_round.setdefault(rec["iteration"], {})[rec["party_id"]] = rec.get("time_logs_ms", {})
    out = {}
    for it, parties in sorted(by_round.items()):
        if SERVER_ID not in parties:
            continue
        vals = values.get(it, {})
        participants = vals.get("participants") or sorted(p for p in parties if p != SERVER_ID)

        def flat(time_logs):
            return {(ph, k[:-3] if k.endswith("_ms") else k): float(v)
                    for ph, ms in time_logs.items() if ph not in EXCLUDE_PHASES for k, v in ms.items()}

        def split(metrics):
            comp = sum(v for (_, m), v in metrics.items() if _kind(m) == "comp")
            return comp, sum(metrics.values()) - comp

        s_comp, _ = split(flat(parties[SERVER_ID]))
        slowest = {}
        for cid in participants:
            for key, v in flat(parties.get(cid, {})).items():
                slowest[key] = max(slowest.get(key, 0.0), v)
        c_comp, c_comm = split(slowest)
        out[it] = {"n": vals.get("num_participants", len(participants)),
                   "round_k1": vals.get("round_k1"), "round_t": vals.get("round_t"),
                   "s_comp": s_comp / 1000.0, "c_comp_max": c_comp / 1000.0, "c_comm_max": c_comm / 1000.0,
                   "seconds": (s_comp + c_comp + c_comm) / 1000.0}
    return out
