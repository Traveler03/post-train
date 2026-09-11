#!/usr/bin/env python3
"""Generate the first Docs/Drive SkillBank from saved benchmark rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo.offline_skills import GenerationOptions, run_generation  # noqa: E402

RUNTIME = ROOT.parent / "runtime/rollout_traces"
DEFAULT_DOCS_TRACES = RUNTIME / "benchmark_docs_single_qwen35_4b_full_b8_g8_20260904_102132"
DEFAULT_DRIVE_TRACES = RUNTIME / "benchmark_drive_single_qwen35_4b_full_b8_g8_20260904_191849"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-trace-dir", type=Path, default=DEFAULT_DOCS_TRACES)
    parser.add_argument("--drive-trace-dir", type=Path, default=DEFAULT_DRIVE_TRACES)
    parser.add_argument("--docs-dataset", type=Path, default=ROOT / "eval/datasets/google_docs/items.jsonl")
    parser.add_argument("--drive-dataset", type=Path, default=ROOT / "eval/datasets/google_drive/items.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/offline_skillbank_v1")
    parser.add_argument(
        "--api-config",
        "--codex-config",
        dest="api_config",
        type=Path,
        default=Path("~/.codex/config.toml"),
        help="TOML file supplying the direct Responses API endpoint, model, and bearer key",
    )
    parser.add_argument("--model", help="Override the model in the API config")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        help="Override model_reasoning_effort from the API config",
    )
    parser.add_argument("--api-concurrency", type=int, default=3)
    parser.add_argument("--api-timeout-s", type=float, default=300.0)
    parser.add_argument("--api-max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--positive-examples-per-family", type=int, default=8)
    parser.add_argument("--negative-examples-per-family", type=int, default=4)
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Build deterministic placeholder packages without calling the Responses API",
    )
    parser.add_argument(
        "--reuse-generations-from",
        type=Path,
        help="Reuse generated family packages from another SkillBank and rebuild only deterministic artifacts",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite known generated files in output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.no_api and args.reuse_generations_from:
        raise SystemExit("--no-api and --reuse-generations-from are mutually exclusive")
    options = GenerationOptions(
        docs_trace_dir=args.docs_trace_dir,
        drive_trace_dir=args.drive_trace_dir,
        docs_dataset=args.docs_dataset,
        drive_dataset=args.drive_dataset,
        output_dir=args.output_dir,
        api_config=args.api_config,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        api_timeout_s=args.api_timeout_s,
        api_max_attempts=args.api_max_attempts,
        api_concurrency=args.api_concurrency,
        max_output_tokens=args.max_output_tokens,
        positive_examples_per_family=args.positive_examples_per_family,
        negative_examples_per_family=args.negative_examples_per_family,
        use_api=not args.no_api and args.reuse_generations_from is None,
        force=args.force,
    )
    package_generator = None
    if args.reuse_generations_from:
        source = args.reuse_generations_from.expanduser().resolve()
        reused: dict[str, dict] = {}
        for path in sorted((source / "generations").glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            reused[str(value["family"])] = {
                "content": value["content"],
                "source": str(path),
            }

        def package_generator(blueprint, _prompt):
            try:
                value = reused[blueprint.family]
            except KeyError as exc:
                raise ValueError(f"no reusable generation for {blueprint.family} in {source}") from exc
            return value["content"], {"mode": "reused_generation", "source": value["source"]}

    manifest = run_generation(options, package_generator=package_generator)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "cases": manifest["totals"]["cases"],
                "rollouts": manifest["totals"]["matched_rollouts"],
                "verified_successes": manifest["totals"]["verified_successes"],
                "families": manifest["classification"]["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
