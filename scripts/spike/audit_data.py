#!/usr/bin/env python3
"""Read private JSONL locally; export only aggregate statistics and one-way hashes.

This is a retrospective audit, not the historical spike-selection script.
No raw messages, user identifiers, credentials, or tool arguments are exported.
"""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FAMILIES = ("Drive", "Docs", "Sheets", "Slides", "Gmail", "Calendar", "Contacts")


def quantiles(values):
    values = sorted(values)
    if not values:
        return {}
    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return round(values[lo] + (values[hi] - values[lo]) * (position - lo), 3)
    return {"p50": percentile(.5), "p90": percentile(.9), "max": max(values)}


def call_name(call):
    if isinstance(call, str):
        call = json.loads(call)
    if "function" in call:
        call = call["function"]
    name = call.get("name", "")
    args = call.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    if name.rsplit(".", 1)[-1] == "proxy_tool":
        if not isinstance(args, dict) or not args.get("tool_name"):
            raise ValueError("proxy_tool has no structured tool_name")
        name = args["tool_name"]
    if not isinstance(name, str) or not name:
        raise ValueError("tool call has no name")
    return name.rsplit(".", 1)[-1]


def feature(row, index):
    messages = row["messages"]
    names = []
    for message in messages:
        if message.get("role") == "tool_call":
            names.append(call_name(message["content"]))
        elif message.get("role") == "assistant":
            names.extend(call_name(c) for c in message.get("tool_calls", []))
    calls = Counter(names)
    families = Counter()
    for name, count in calls.items():
        family = next((f for f in FAMILIES if name.startswith("google_" + f.lower() + "_")), "Other")
        families[family] += count
    assistant_text = "".join(
        (m.get("content") or "") + (m.get("reasoning_content") or "")
        for m in messages if m.get("role") == "assistant"
    )
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    return {
        "source_row": index,
        "canonical_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "assistant_md5": hashlib.md5(assistant_text.encode(), usedforsecurity=False).hexdigest(),
        "first_user_sha256": hashlib.sha256(first_user.encode()).hexdigest(),
        "message_count": len(messages),
        "tool_calls": len(names),
        "families": families,
        "names": calls,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset(path):
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rows.append(feature(json.loads(line), line_number))
            if len(rows) % 200 == 0:
                print(f"{path.name}: {len(rows)} records audited", flush=True)
    return rows, {"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def summarize(rows):
    n = len(rows)
    families = {}
    for family in (*FAMILIES, "Other"):
        records = sum(r["families"][family] > 0 for r in rows)
        families[family] = {
            "records": records,
            "percent": round(100 * records / n, 4) if n else 0,
            "calls": sum(r["families"][family] for r in rows),
        }
    cooccurrence = Counter()
    bins = Counter()
    for row in rows:
        drive, docs = row["families"]["Drive"] > 0, row["families"]["Docs"] > 0
        cooccurrence["Drive+Docs" if drive and docs else "Drive_only" if drive else "Docs_only" if docs else "Neither"] += 1
        calls = row["tool_calls"]
        bins["0" if calls == 0 else "1-5" if calls <= 5 else "6-10" if calls <= 10 else "11-20" if calls <= 20 else "21+"] += 1
    return {
        "records": n,
        "canonical_unique": len({r["canonical_sha256"] for r in rows}),
        "assistant_md5_unique": len({r["assistant_md5"] for r in rows}),
        "first_user_unique": len({r["first_user_sha256"] for r in rows}),
        "families": families,
        "drive_docs_cooccurrence": dict(cooccurrence),
        "tool_calls_total": sum(r["tool_calls"] for r in rows),
        "tool_calls_per_record": quantiles([r["tool_calls"] for r in rows]),
        "tool_call_count_bins": dict(bins),
        "messages_per_record": quantiles([r["message_count"] for r in rows]),
    }


def read_pipeline(path):
    with path.open(encoding="utf-8") as stream:
        meta = json.load(stream)
    return {
        "input": meta["读入"], "kept": meta["留下"],
        "discard_reasons": meta["丢弃原因"],
        "splits": meta["各 split"],
    }


def audit(before_path, after_path, before_meta, after_meta, output):
    with ThreadPoolExecutor(max_workers=2) as pool:
        before_future = pool.submit(read_dataset, before_path)
        after_future = pool.submit(read_dataset, after_path)
        before, before_source = before_future.result()
        after, after_source = after_future.result()
    before_counts = Counter(r["canonical_sha256"] for r in before)
    after_counts = Counter(r["canonical_sha256"] for r in after)
    remaining = before_counts - after_counts
    additions = after_counts - before_counts
    removed = []
    for row in before:
        key = row["canonical_sha256"]
        if remaining[key]:
            removed.append(row)
            remaining[key] -= 1
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Canonical whole-row SHA256 multiset difference; actual tool_call nodes only, proxy_tool unwrapped; tool schemas excluded.",
        "sources": {"before": before_source, "after": after_source},
        "subset_check": {"strict_row_subset": not additions and bool(removed), "added_rows": sum(additions.values()), "removed_rows": len(removed)},
        "before": summarize(before), "after": summarize(after), "removed": summarize(removed),
        "pipeline": {"before": read_pipeline(before_meta), "after": read_pipeline(after_meta)},
        "fingerprint_note": "assistant_md5 is a documented-method reconstruction (per-assistant content then reasoning_content, empty separator); original implementation is unavailable. Canonical SHA256 proves the row difference independently.",
    }
    with (output / "data_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    fields = ["source_row", "canonical_sha256", "assistant_md5", "tool_calls"]
    with (output / "removed_fingerprints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in removed)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("before", "after", "before-meta", "after-meta", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    audit(args.before, args.after, args.before_meta, args.after_meta, args.output)


if __name__ == "__main__":
    main()
