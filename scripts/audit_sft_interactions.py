#!/usr/bin/env python3
"""Read private JSONL and print aggregate interaction counts, never raw content."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def stats(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty dataset")
    return {
        "min": ordered[0],
        "p50": ordered[(len(ordered) - 1) * 50 // 100],
        "p90": ordered[(len(ordered) - 1) * 90 // 100],
        "max": ordered[-1],
        "total": sum(ordered),
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def histogram(values):
    return [{"value": value, "records": count} for value, count in sorted(Counter(values).items())]


def read_rows(path, digest):
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                raise ValueError(f"Invalid JSON at line {line_number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at line {line_number}")
            yield line_number, row


def source(path, digest):
    return {"basename": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def audit_train(path):
    digest = hashlib.sha256()
    features = {name: [] for name in (
        "messages", "assistant_messages", "supervised_assistant_turns",
        "tool_call_turns", "tool_calls", "tool_result_messages", "user_role_messages",
    )}
    roles = Counter()
    unsupervised_assistant = 0
    for line_number, row in read_rows(path, digest):
        messages, supervision = row.get("messages"), row.get("sup")
        if not isinstance(messages, list) or not isinstance(supervision, list):
            raise ValueError(f"Expected messages and sup lists at line {line_number}")
        if len(messages) != len(supervision) or any(type(flag) is not bool for flag in supervision):
            raise ValueError(f"Invalid supervision alignment at line {line_number}")
        counts = Counter(messages=len(messages))
        for message, supervised in zip(messages, supervision):
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"Unsupported message structure at line {line_number}")
            role = message["role"]
            roles[role] += 1
            if supervised and role != "assistant":
                raise ValueError(f"Supervised non-assistant message at line {line_number}")
            if role == "assistant":
                calls = message.get("tool_calls", [])
                if not isinstance(calls, list):
                    raise ValueError(f"Expected tool_calls list at line {line_number}")
                counts["assistant_messages"] += 1
                counts["supervised_assistant_turns"] += supervised
                counts["tool_call_turns"] += bool(calls)
                counts["tool_calls"] += len(calls)
                unsupervised_assistant += not supervised
            counts["tool_result_messages"] += role == "tool"
            counts["user_role_messages"] += role == "user"
        for name, values in features.items():
            values.append(counts[name])
    records = len(features["messages"])
    return {
        "source": source(path, digest),
        "records": records,
        "statistics": {name: stats(values) for name, values in features.items()},
        "histograms": {name: histogram(features[name]) for name in (
            "supervised_assistant_turns", "tool_call_turns", "user_role_messages",
        )},
        "role_counts": dict(sorted(roles.items())),
        "unsupervised_assistant_messages": unsupervised_assistant,
        "records_with_at_least_two_supervised_turns": sum(
            value >= 2 for value in features["supervised_assistant_turns"]
        ),
    }


def audit_rollouts(path):
    digest = hashlib.sha256()
    counts = []
    for line_number, row in read_rows(path, digest):
        count = row.get("n_call_llm_this_turn")
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid n_call_llm_this_turn at line {line_number}")
        counts.append(count)
    return {
        "source": source(path, digest),
        "records": len(counts),
        "recorded_model_calls": stats(counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--rollouts", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "quantile_method": "sorted[floor((n - 1) * p)]; means rounded to six decimals",
        "definitions": {
            "supervised_assistant_turns": "assistant messages with sup=true, including final responses",
            "tool_call_turns": "assistant messages containing at least one tool_call; a batch counts once",
            "tool_calls": "number of tool_calls entries; includes transfer calls without requiring a reply",
            "user_role_messages": "message-role count, not a count of human requests",
        },
        "train": audit_train(args.train),
    }
    if args.rollouts:
        report["upstream_rollouts"] = audit_rollouts(args.rollouts)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
