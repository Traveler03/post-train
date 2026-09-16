#!/usr/bin/env python3
"""Recompute EARLY pc8-minus-pc7 paired results from existing local evidence.

Only reads run JSON and existing verdict caches. Never submits or rejudges cases.
Exports aggregates and run-file identities, not raw cases or item IDs.
"""
import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VOLUMES = ("drive", "docs", "slides")
PAIRS = ((27, 25, "b1"), (54, 50, "b1"), (81, 75, "a1"), (108, 100, "a1"))


def verdict(value):
    value = str(value).strip().upper() if value is not None else ""
    return value if value in ("PASS", "FAIL") else None


def read_run(tag, runs, cache_root):
    files = list(runs.glob(tag + "_-_*_complete.json"))
    # Two a1 runs exist for s108. Freeze the FIRST attempt explicitly, not latest().
    if tag.startswith("pc7_s108_"):
        files = [p for p in files if "_2026-08-27_02_" in p.name]
    if len(files) != 1:
        raise ValueError(f"{tag}: expected one exact run, found {len(files)}")
    path = files[0]
    raw = path.read_bytes()
    cases = json.loads(raw)["cases"]
    source_hash = hashlib.sha256(raw).hexdigest()
    del raw
    ids = {c["item_id"] for c in cases}
    if len(ids) != len(cases):
        raise ValueError("Duplicate item_id")
    values, embedded = {}, {}
    for case in cases:
        for score in case.get("scores") or []:
            if score.get("name") == "评估结果":
                value = verdict(score.get("stringValue")) or verdict(score.get("value"))
                if value:
                    values[case["item_id"]] = value
        er = case.get("eval_result")
        if isinstance(er, dict) and isinstance(er.get("评估结果"), dict):
            value = verdict(er["评估结果"].get("状态"))
            if value:
                embedded[case["item_id"]] = value
    cache = cache_root / "harvest_cache" / (path.name.replace("_complete.json", "") + ".hv.json")
    caches = [cache] if cache.is_file() else list((cache_root / "harvest_out").glob(tag + "__*.json"))
    if tag.startswith("pc7_s108_") and not cache.is_file():
        raise ValueError("Ambiguous s108 cache cannot be assigned by tag alone")
    if len(caches) != 1:
        raise ValueError(f"{tag}: cache missing or ambiguous; no network fallback allowed")
    with caches[0].open(encoding="utf-8") as stream:
        cache_rows = json.load(stream)
    for row in cache_rows:
        key, value = row.get("item_id"), verdict(row.get("span"))
        if key in ids and value:
            if key in values and values[key] != value:
                raise ValueError("Conflicting cached verdict")
            values.setdefault(key, value)
    for key, value in embedded.items():
        if key in values and values[key] != value:
            raise ValueError("Conflicting embedded verdict")
        values.setdefault(key, value)
    info = {"tag": tag, "run_file": path.name, "sha256": source_hash,
            "cases": len(cases), "judged": len(values), "pass": sum(v == "PASS" for v in values.values())}
    print(f"Audited {tag}: {len(values)}/{len(cases)} judged", flush=True)
    return tag, (values, info)


def paired(original, filtered):
    common = set(original) & set(filtered)
    n = len(common)
    if not n:
        raise ValueError("No common judged items")
    op = sum(original[i] == "PASS" for i in common)
    fp = sum(filtered[i] == "PASS" for i in common)
    gains = sum(original[i] == "FAIL" and filtered[i] == "PASS" for i in common)
    losses = sum(original[i] == "PASS" and filtered[i] == "FAIL" for i in common)
    k = gains + losses
    p = min(1., 2 * sum(math.comb(k, j) for j in range(min(gains, losses) + 1)) / 2**k) if k else 1.
    return {"common_items": n, "original_pass": op, "filtered_pass": fp,
            "original_percent": 100 * op / n, "filtered_percent": 100 * fp / n,
            "delta_pp": 100 * (fp - op) / n, "gains": gains, "losses": losses,
            "mcnemar_exact_p": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tags = sorted({t for s7, s8, r8 in PAIRS for v in VOLUMES
                   for t in (f"pc7_s{s7}_{v}_a1", f"pc8_s{s8}_{v}_{r8}")})
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = dict(pool.map(lambda tag: read_run(tag, args.runs, args.cache_root), tags))
    rows = []
    for s7, s8, r8 in PAIRS:
        volumes, all_original, all_filtered = {}, {}, {}
        for volume in VOLUMES:
            original = values[f"pc7_s{s7}_{volume}_a1"][0]
            filtered = values[f"pc8_s{s8}_{volume}_{r8}"][0]
            volumes[volume] = paired(original, filtered)
            all_original.update({(volume, i): v for i, v in original.items()})
            all_filtered.update({(volume, i): v for i, v in filtered.items()})
        rows.append({"pc7_step": s7, "pc8_step": s8, "pc7_epoch": s7 * 8 / 1102,
                     "pc8_epoch": s8 * 8 / 1010, "volumes": volumes,
                     "combined": paired(all_original, all_filtered)})
    running = 0.
    for index, row in enumerate(sorted(rows, key=lambda r: r["combined"]["mcnemar_exact_p"])):
        running = max(running, min(1., (len(rows) - index) * row["combined"]["mcnemar_exact_p"]))
        row["combined"]["holm_p_four_early_pairs"] = running
    result = {"scope": "Early checkpoints only, approximately 0.2-0.8 epoch. Not final model performance.",
              "method": "pc8 minus pc7; matched item_id with PASS/FAIL on both arms; same approximate epoch; pc7 first attempts, s108 explicitly frozen to 02:22 batch.",
              "combined_definition": "Micro-average across common Drive, Docs and Slides items within each checkpoint pair.",
              "mean_definition": "Arithmetic mean of the four checkpoint-pair deltas; not independent replications or a pooled significance test.",
              "mean_delta_pp": sum(r["combined"]["delta_pp"] for r in rows) / len(rows),
              "rows": rows, "sources": [values[tag][1] for tag in tags]}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "early_evaluation.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"mean_delta_pp": result["mean_delta_pp"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
