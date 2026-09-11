#!/usr/bin/env python3
"""Validate SkillBank routing and case instantiation against offline benchmark data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algo.offline_skills.audit import AuditOptions, run_audit  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skillbank-dir",
        type=Path,
        default=ROOT / "artifacts/offline_skillbank_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/offline_skillbank_v1/audit",
    )
    parser.add_argument(
        "--api-config",
        type=Path,
        default=Path("~/.codex/config.toml"),
        help="TOML file supplying the direct Responses API endpoint, model, and bearer key",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--api-concurrency", type=int, default=3)
    parser.add_argument("--api-timeout-s", type=float, default=300.0)
    parser.add_argument("--api-max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_audit(
        AuditOptions(
            skillbank_dir=args.skillbank_dir,
            output_dir=args.output_dir,
            api_config=args.api_config,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            use_api=not args.no_api,
            batch_size=args.batch_size,
            api_concurrency=args.api_concurrency,
            api_timeout_s=args.api_timeout_s,
            api_max_attempts=args.api_max_attempts,
            max_output_tokens=args.max_output_tokens,
            force=args.force,
        )
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "structural": report["structural"],
                "semantic": report["semantic"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
