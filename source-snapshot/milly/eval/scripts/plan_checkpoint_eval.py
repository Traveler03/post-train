#!/usr/bin/env python3
"""Resolve adapters/collections and generate an immutable manifest + launcher."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MILLY = REPO.parent
SKILL_DIR = MILLY / "skills/start-evaluation"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
from resolve_eval_set import load_catalog, resolve  # noqa: E402


CATALOG = SKILL_DIR / "references/eval_catalog.json"


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:80]


def adapter_step(path: Path) -> int | None:
    for pattern in (r"checkpoint[-_](\d+)$", r"(?:^|[_-])s(?:tep)?(\d+)$", r"(\d+)$"):
        match = re.search(pattern, path.name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def discover_adapters(root: Path) -> dict[int, Path]:
    candidates = [root] if root.is_dir() else []
    if root.is_dir():
        candidates.extend(path for path in root.iterdir() if path.is_dir())
    found: dict[int, Path] = {}
    for path in candidates:
        config = path / "adapter_config.json"
        weights = path / "adapter_model.safetensors"
        step = adapter_step(path)
        if step is not None and config.is_file() and weights.is_file() and weights.stat().st_size:
            if step in found:
                raise ValueError(f"duplicate adapter step {step}: {found[step]} and {path}")
            found[step] = path.resolve()
    if not found:
        raise ValueError(
            f"no complete PEFT adapters under {root}; convert checkpoints first"
        )
    return found


def select_steps(expression: str, available: dict[int, Path]) -> list[int]:
    expression = expression.strip().lower()
    if expression in {"all", "全部", "*"}:
        return sorted(available)
    selected: set[int] = set()
    for token in re.split(r"[,\s]+", expression):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", token)
        if match:
            low, high = sorted((int(match.group(1)), int(match.group(2))))
            selected.update(step for step in available if low <= step <= high)
        elif token.isdigit():
            selected.add(int(token))
        else:
            raise ValueError(f"invalid checkpoint selector: {token}")
    missing = sorted(selected - available.keys())
    if missing:
        raise ValueError(f"selected checkpoints have no complete adapter: {missing}")
    if not selected:
        raise ValueError("checkpoint selection is empty")
    return sorted(selected)


def read_item_ids(path: Path) -> list[str]:
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("id"):
                raise ValueError(f"item without id: {path}")
            ids.append(str(row["id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate item IDs: {path}")
    return ids


def launcher_text(run_id: str, manifest: Path) -> str:
    session = safe(f"eval_{run_id}")
    controller = REPO / "eval/scripts/run_checkpoint_eval.py"
    watcher = REPO / "eval/scripts/watch_eval_manifest.py"
    manifest_q = shlex.quote(str(manifest))
    controller_q = shlex.quote(str(controller))
    watcher_q = shlex.quote(str(watcher))
    repo_q = shlex.quote(str(REPO))
    return f'''#!/usr/bin/env bash
set -euo pipefail
MANIFEST={manifest_q}
SESSION="${{EVAL_SESSION:-{session}}}"
case "${{1:-start}}" in
  start)
    tmux has-session -t "$SESSION" 2>/dev/null && {{ echo "session exists: $SESSION" >&2; exit 1; }}
    tmux new-session -d -s "$SESSION" -c {repo_q} \
      "exec python3 {controller_q} --manifest {manifest_q}"
    tmux new-window -t "$SESSION" -n progress -c {repo_q} \
      "exec python3 {watcher_q} --manifest {manifest_q} --interval 300"
    echo "started session=$SESSION manifest=$MANIFEST"
    ;;
  dry-run) python3 {controller_q} --manifest "$MANIFEST" --dry-run ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux=$SESSION running" || echo "tmux=$SESSION not-running"
    [[ -f "$(dirname "$MANIFEST")/controller.log" ]] && tail -n 1 "$(dirname "$MANIFEST")/controller.log"
    ;;
  logs) tail -n "${{EVAL_LOG_LINES:-80}}" "$(dirname "$MANIFEST")/controller.log" ;;
  *) echo "Usage: $0 {{start|dry-run|status|logs}}" >&2; exit 2 ;;
esac
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--checkpoints", required=True, help="all, CSV, or inclusive range like 46-462")
    parser.add_argument("--collections", nargs="+", required=True, help="catalog IDs, aliases, or comma-separated group")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--tier", default="low")
    parser.add_argument("--thinking", choices=("true", "false"), default="true")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--vllm-host", default="10.251.209.7")
    parser.add_argument("--vllm-port", type=int, default=8010)
    parser.add_argument("--worker-python", default="/home/work/migoo_ai_public/posttrain/dsv4_run/env/venv/bin/python")
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--worker-concurrency", type=int, default=3)
    parser.add_argument("--dataset-parallelism", type=int, default=3)
    parser.add_argument("--case-retries", type=int, default=2)
    parser.add_argument("--judge-wait", type=int, default=480)
    parser.add_argument("--user-email", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "eval/runs")
    parser.add_argument("--launcher-dir", type=Path, default=REPO / "eval/example/generated")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.case_retries < 0:
        parser.error("rounds must be >=1 and case-retries >=0")

    adapters = discover_adapters(args.adapter_root.resolve())
    steps = select_steps(args.checkpoints, adapters)
    catalog = load_catalog(CATALOG)
    collections, _ = resolve(catalog, args.collections)
    if not collections:
        raise ValueError("collection selection is empty")
    if any(row["family"] != "pro" for row in collections):
        raise ValueError("checkpoint matrix currently supports Pro collections only")
    for row in collections:
        item_file = REPO / row["item_file"]
        count = len(read_item_ids(item_file))
        if count != int(row["case_count"]):
            raise ValueError(
                f"{row['dataset_name']} count {count} != catalog {row['case_count']}"
            )

    run_id = safe(args.run_id or f"{args.adapter_root.name}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_root = args.output_root.resolve() / run_id
    launcher_dir = args.launcher_dir.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    launcher_dir.mkdir(parents=True, exist_ok=True)
    extra_body = {
        "chat_template_kwargs": {
            "thinking": args.thinking == "true",
            "reasoning_effort": args.reasoning_effort,
        },
        "temperature": 0,
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "live_smoke": False,
        "model": {"kind": "local_lora", "tier": args.tier, "extra_body": extra_body},
        "checkpoints": [
            {
                "step": step,
                "label": f"s{step:03d}",
                "adapter_path": str(adapters[step]),
                "model_name": safe(f"{run_id}-s{step:03d}"),
                "lane": safe(f"ais-relay-{run_id}-s{step:03d}"),
                "worker_id": safe(f"{run_id}-s{step:03d}"),
            }
            for step in steps
        ],
        "collections": collections,
        "rounds": args.rounds,
        "task_concurrency": args.task_concurrency,
        "dataset_parallelism": min(args.dataset_parallelism, len(collections)),
        "user_email": args.user_email,
        "retry": {
            "case_retries": args.case_retries,
            "judge_wait_seconds": args.judge_wait,
            "poll_interval_seconds": 30,
            "task_timeout_seconds": 43200,
            "trace_workers": 8,
        },
        "serving": {
            "host": args.vllm_host,
            "port": args.vllm_port,
            "worker_python": args.worker_python,
            "worker_concurrency": args.worker_concurrency,
            "request_timeout_seconds": 900,
            "relay_url": "http://10.187.175.247:18000/inbeeai/api/v1/chat_adk/ais_model_relay",
            "hub_status_url": "http://10.187.175.247:18000/inbeeai/api/v1/chat_adk/ais_model_relay/status",
        },
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    launcher = launcher_dir / f"start_{run_id}.sh"
    launcher.write_text(launcher_text(run_id, manifest_path), encoding="utf-8")
    launcher.chmod(0o755)
    print(json.dumps({"run_id": run_id, "manifest": str(manifest_path), "launcher": str(launcher)}, ensure_ascii=False, indent=2))
    if args.start:
        preflight = subprocess.run([str(launcher), "dry-run"], cwd=REPO, check=False)
        if preflight.returncode:
            print("错误: generated launcher dry-run failed; evaluation not started", file=sys.stderr)
            return preflight.returncode
        return subprocess.run([str(launcher), "start"], cwd=REPO, check=False).returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
