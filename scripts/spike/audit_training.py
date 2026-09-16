#!/usr/bin/env python3
"""Extract numerical training metrics without exporting private log text."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


def extract(path, label, samples):
    rows = []
    digest = hashlib.sha256()
    completed = False
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            line = raw.decode("utf-8", errors="replace")
            completed |= "[after training is done]" in line
            if "lm loss:" not in line:
                continue
            step = re.search(r"iteration\s+(\d+)\s*/\s*(\d+)", line)
            if not step:
                continue
            def number(name):
                match = re.search(re.escape(name) + r"\s*:\s*([0-9.eE+\-]+)", line)
                if not match:
                    raise ValueError(f"Missing {name} in metric line")
                value = float(match.group(1))
                if not math.isfinite(value):
                    raise ValueError("Non-finite training metric")
                return value
            rows.append({
                "experiment": label, "step": int(step[1]), "planned_steps": int(step[2]),
                "epoch": number("consumed samples") / samples,
                "loss": number("lm loss"), "learning_rate": number("learning rate"),
                "grad_norm": number("grad norm"),
                "skipped": int(number("number of skipped iterations")),
                "nan": int(number("number of nan iterations")),
            })
    if not rows or [r["step"] for r in rows] != list(range(1, rows[-1]["planned_steps"] + 1)):
        raise ValueError("Missing, duplicate, or incomplete training steps")
    if not completed:
        raise ValueError("Training completion marker absent")
    gradients = [r["grad_norm"] for r in rows]
    losses = [r["loss"] for r in rows]
    summary = {
        "experiment": label, "source_directory": path.parent.name,
        "log_sha256": digest.hexdigest(), "train_samples": samples,
        "steps": len(rows), "final_epoch": rows[-1]["epoch"],
        "loss_mean": statistics.mean(losses), "loss_median": statistics.median(losses),
        "loss_first": losses[0], "loss_last": losses[-1],
        "grad_norm_median": statistics.median(gradients), "grad_norm_max": max(gradients),
        "grad_norm_gt_10_steps": sum(v > 10 for v in gradients),
        "grad_norm_gt_10_fraction": sum(v > 10 for v in gradients) / len(rows),
        "max_nan_counter": max(r["nan"] for r in rows),
        "max_skipped_counter": max(r["skipped"] for r in rows),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-log", type=Path, required=True)
    parser.add_argument("--after-log", type=Path, required=True)
    parser.add_argument("--before-samples", type=int, default=1102)
    parser.add_argument("--after-samples", type=int, default=1010)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for key, label, path, samples in [
        ("before", "pc7-v2", args.before_log, args.before_samples),
        ("after", "pc8-v2.1", args.after_log, args.after_samples),
    ]:
        rows, summaries[key] = extract(path, label, samples)
        with (args.output / f"{key}_training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    with (args.output / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
