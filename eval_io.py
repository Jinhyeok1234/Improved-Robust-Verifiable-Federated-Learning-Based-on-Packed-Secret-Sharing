"""
Single source of truth for evaluation-record I/O.

ONE file per (tag, metric): the accuracy curve lives in `<tag>.jsonl`, the ASR curve in
`<tag>_asr.jsonl`. There is NO separate "peek" file -- peek and the real eval always ran the
*identical* evaluation (same script, dataset, stride, flags), so their results belong in the
same file. Every eval -- on partial or final checkpoints, at any stride, at any time -- MERGES
into that one file (keyed by iteration) and the file is kept SORTED by iteration. "How far is
this evaluated / what's missing" is therefore answered by the file's CONTENTS, never by whether
some marker file exists.

Both eval_checkpoints.py (accuracy) and eval_backdoor.py (ASR) import these helpers, so the
merge/sort logic cannot drift between them. A legacy `<stem>_peek.jsonl` left over from the old
split is folded back in automatically the first time its canonical file is written (see
load_merged); nothing evaluated is lost.
"""
import json
from pathlib import Path


def read_records(path):
    """{iteration: record} parsed from a jsonl file (missing file / bad lines -> skipped)."""
    recs = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                recs[int(r["iteration"])] = r
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return recs


def peek_path(out_path):
    """The legacy throwaway file that used to shadow this one: `<stem>_peek<suffix>`."""
    p = Path(out_path)
    return p.with_name(p.stem + "_peek" + p.suffix)


def load_merged(out_path):
    """Load the canonical file AND fold in a legacy `<stem>_peek.jsonl` (same eval, old split file)
    for iterations the canonical file does not already have. Returns
        (records, absorbed_peek_path_or_None).
    Canonical records win on conflict. The peek file is only READ here; the caller deletes it
    after a successful sorted rewrite, so the one-time migration is crash-safe.
    """
    records = read_records(out_path)
    pk = peek_path(out_path)
    absorbed = None
    if pk.exists():
        for it, rec in read_records(pk).items():
            records.setdefault(it, rec)            # canonical (real) value wins over the old peek
        absorbed = pk
    return records, absorbed


def write_sorted(out_path, records):
    """(Re)write the whole file SORTED by iteration -- merges/orders across differing strides."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in sorted(records):
            f.write(json.dumps(records[it]) + "\n")


def evaluated_iters(out_path):
    """Set of iterations already present in the canonical file (for content-based 'is it done?')."""
    return set(read_records(out_path))  