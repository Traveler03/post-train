"""Streaming readers for benchmark datasets, rollouts, and rewards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from .evidence import extract_tool_events
from .models import CaseRecord, Corpus, JsonObject, RolloutRecord


def iter_jsonl(path: Path) -> Iterator[tuple[int, JsonObject]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_cases(path: Path, domain: str) -> tuple[list[str], dict[str, JsonObject]]:
    order: list[str] = []
    result: dict[str, JsonObject] = {}
    for _, item in iter_jsonl(path):
        item_input = item.get("input") if isinstance(item.get("input"), dict) else {}
        if item_input.get("multi_turn") is True:
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            raise ValueError(f"dataset item in {path} has no id")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        item_domain = str(metadata.get("skill_domain") or domain).lower()
        if item_domain != domain:
            continue
        if item_id in result:
            raise ValueError(f"duplicate dataset item id {item_id!r} in {path}")
        result[item_id] = {
            "query": str(item_input.get("question") or "").strip(),
            "metadata": dict(metadata),
            "input_id": item_input.get("id"),
            "dataset_name": item.get("datasetName"),
        }
        order.append(item_id)
    return order, result


def _load_rewards(path: Path) -> tuple[dict[str, JsonObject], list[str]]:
    rewards: dict[str, JsonObject] = {}
    duplicates: list[str] = []
    for _, reward in iter_jsonl(path):
        request_id = str(reward.get("request_id") or "").strip()
        if not request_id:
            raise ValueError(f"reward row in {path} has no request_id")
        if request_id in rewards:
            duplicates.append(request_id)
        rewards[request_id] = reward
    return rewards, duplicates


def _as_object(value: Any) -> JsonObject:
    return value if isinstance(value, dict) else {}


def load_corpus(trace_dir: Path, dataset_path: Path, domain: str) -> Corpus:
    """Join only reward-backed rollout attempts and group them by official case."""

    trace_dir = trace_dir.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    rollout_path = trace_dir / "rollouts.jsonl"
    reward_path = trace_dir / "rewards.jsonl"
    for path in (rollout_path, reward_path, dataset_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset_order, dataset_items = load_dataset_cases(dataset_path, domain)
    dataset_positions = {item_id: position for position, item_id in enumerate(dataset_order)}
    rewards, duplicate_reward_ids = _load_rewards(reward_path)
    cases: dict[str, CaseRecord] = {}
    matched = 0
    unmatched = 0

    for line_number, raw in iter_jsonl(rollout_path):
        request_id = str(raw.get("request_id") or "").strip()
        reward = rewards.get(request_id)
        if reward is None:
            unmatched += 1
            continue
        matched += 1
        case_id = str(raw.get("official_dataset_item_id") or reward.get("official_dataset_item_id") or "").strip()
        if not case_id:
            iid_suffix = str(raw.get("iid") or "").partition(":")[2]
            case_id = iid_suffix or str(raw.get("iid") or request_id)
        dataset_item = dataset_items.get(case_id, {})
        raw_metadata = _as_object(raw.get("metadata"))
        metadata = {
            **_as_object(dataset_item.get("metadata")),
            **raw_metadata,
            "dataset_input_id": dataset_item.get("input_id"),
            "dataset_name": dataset_item.get("dataset_name"),
        }
        query = str(dataset_item.get("query") or raw.get("question") or "").strip()
        actual_outcome = _as_object(raw.get("actual_outcome"))
        answer_value = actual_outcome.get("respond", raw.get("answer", raw.get("message_text", "")))
        answer = (
            answer_value if isinstance(answer_value, str) else json.dumps(answer_value, ensure_ascii=False, default=str)
        )
        seed_summary = _as_object(raw.get("seed_summary"))
        rollout = RolloutRecord(
            domain=domain,
            case_id=case_id,
            iid=str(raw.get("iid") or reward.get("iid") or case_id),
            request_id=request_id,
            trace_id=str(raw.get("trace_id") or reward.get("trace_id") or ""),
            query=query,
            answer=answer,
            metadata=metadata,
            seed_summary=seed_summary,
            reward=reward,
            events=extract_tool_events(actual_outcome),
            rollout_path=rollout_path,
            rollout_line=line_number,
        )
        case = cases.get(case_id)
        if case is None:
            case = CaseRecord(
                domain=domain,
                case_id=case_id,
                iid=rollout.iid,
                query=query,
                metadata=metadata,
                seed_summary=seed_summary,
                dataset_position=dataset_positions.get(case_id, len(dataset_positions) + len(cases)),
            )
            cases[case_id] = case
        case.rollouts.append(rollout)

    unused_rewards = set(rewards) - {rollout.request_id for case in cases.values() for rollout in case.rollouts}
    if unused_rewards:
        raise ValueError(f"{len(unused_rewards)} reward rows have no matching rollout in {trace_dir}")

    ordered_cases = sorted(cases.values(), key=lambda case: (case.dataset_position, case.case_id))
    return Corpus(
        domain=domain,
        trace_dir=trace_dir,
        dataset_path=dataset_path,
        cases=ordered_cases,
        matched_rollouts=matched,
        unmatched_rollouts=unmatched,
        reward_rows=len(rewards),
        duplicate_reward_ids=duplicate_reward_ids,
    )
