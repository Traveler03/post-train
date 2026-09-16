#!/usr/bin/env python3
"""Render report figures from public aggregate artifacts, with no private data."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2563eb", "#e87924"


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save(figure, directory, name):
    figure.savefig(directory / name, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def draw_training(artifacts, figures):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, constrained_layout=True)
    for key, color, label in (("before", BLUE, "Original: pc7 / v2"), ("after", ORANGE, "Filtered: pc8 / v2.1")):
        with (artifacts / f"{key}_training_metrics.csv").open() as stream:
            rows = [{k: float(v) for k, v in r.items() if k != "experiment"} for r in csv.DictReader(stream)]
        epoch = np.array([r["epoch"] for r in rows])
        loss = np.array([r["loss"] for r in rows])
        norm = np.array([r["grad_norm"] for r in rows])
        median = np.array([np.median(loss[max(0, i - 7):min(len(loss), i + 8)]) for i in range(len(loss))])
        axes[0].plot(epoch, loss, color=color, alpha=.17, linewidth=.8)
        axes[0].plot(epoch, median, color=color, linewidth=2, label=label)
        axes[1].plot(epoch, norm, color=color, linewidth=.9, alpha=.8, label=label)
    axes[0].set(title="Training loss: raw steps + 15-step rolling median", ylabel="Loss")
    axes[1].set(title="Reported gradient norm (before clipping)", ylabel="Gradient norm (log scale)", xlabel="Epoch = consumed samples / training records")
    axes[1].set_yscale("log")
    axes[1].axhline(10, color="#64748b", linestyle="--", linewidth=1, label="Audit threshold: norm > 10")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.set_xlim(0, 2.04)
    fig.suptitle("Spike filtering: full training curves (about 2 epochs)", fontsize=15, fontweight="bold")
    save(fig, figures, "training_curves.png")


def draw_data(artifacts, figures):
    summary = load_json(artifacts / "data_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    families = ["Drive", "Docs", "Sheets", "Slides", "Gmail"]
    x = np.arange(len(families))
    groups = ["Drive_only", "Drive+Docs", "Docs_only", "Neither"]
    for offset, key, color, label in ((-.18, "before", BLUE, "Original (n=1,149)"), (.18, "after", ORANGE, "Filtered (n=1,057)")):
        values = [summary[key]["families"][f]["percent"] for f in families]
        bars = axes[0].bar(x + offset, values, .34, color=color, label=label)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        values = [100 * summary[key]["drive_docs_cooccurrence"].get(g, 0) / summary[key]["records"] for g in groups]
        bars = axes[1].bar(np.arange(4) + offset, values, .34, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    axes[0].set(title="Actual tool-call coverage (multi-label)", xticks=x, xticklabels=families, ylim=(0, 110), ylabel="Share of source records (%)")
    axes[1].set(title="Drive / Docs co-occurrence (exclusive groups)", xticks=np.arange(4), xticklabels=["Drive only", "Drive + Docs", "Docs only", "Neither"], ylim=(0, 70))
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Data composition before and after removing 92 records", fontsize=15, fontweight="bold")
    save(fig, figures, "data_distribution.png")


def draw_early(artifacts, figures):
    report = load_json(artifacts / "early_evaluation.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    rows = report["rows"]
    labels = [f"~{r['pc8_epoch']:.1f} ep\npc7 s{r['pc7_step']} / pc8 s{r['pc8_step']}" for r in rows]
    delta = [r["combined"]["delta_pp"] for r in rows]
    bars = axes[0].bar(labels, delta, color=ORANGE, width=.58)
    axes[0].bar_label(bars, labels=[f"+{v:.2f}" for v in delta], padding=4, fontsize=11)
    axes[0].axhline(report["mean_delta_pp"], color=BLUE, linestyle="--", label=f"Four-pair mean: +{report['mean_delta_pp']:.2f} pp")
    axes[0].set(title="Combined paired pass-rate change", ylabel="Filtered minus original (percentage points)", ylim=(0, 7.5))
    axes[0].legend(frameon=False, fontsize=9)
    x = np.arange(4)
    for volume, color, marker in (("drive", BLUE, "o"), ("docs", ORANGE, "s"), ("slides", "#64748b", "^")):
        axes[1].plot(x, [r["volumes"][volume]["delta_pp"] for r in rows], marker=marker, color=color, label=volume.title())
    axes[1].set(title="Domain changes are not uniformly positive", xticks=x, xticklabels=[f"~{r['pc8_epoch']:.1f} ep" for r in rows], ylabel="Paired change (percentage points)")
    axes[1].axhline(0, color="#94a3b8", linewidth=1)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("EARLY evaluation only: approximately 0.2-0.8 epoch", fontsize=15, fontweight="bold")
    fig.supxlabel("Single-run observations; matched judged item IDs; four pairs are not independent replications.", fontsize=9)
    save(fig, figures, "early_gain.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    draw_training(args.artifacts, args.figures)
    draw_data(args.artifacts, args.figures)
    draw_early(args.artifacts, args.figures)


if __name__ == "__main__":
    main()
