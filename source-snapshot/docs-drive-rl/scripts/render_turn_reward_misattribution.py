#!/usr/bin/env python3
# ruff: noqa: E501
"""Render a self-contained audit of outcome and milestone reward attribution by turn."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EPSILON = 1e-12


@dataclass
class Trajectory:
    run_index: int
    update_index: int
    request_id: str
    iid: str
    score: float
    overall_pass: bool
    turn_count: int
    events_by_turn: dict[int, tuple[str, ...]]


@dataclass
class TurnRow:
    trajectory: Trajectory
    turn_index: int
    state_ids: tuple[str, ...]
    process_reward: float = 0.0
    global_advantage: float = 0.0
    trajectory_equal_advantage: float = 0.0
    local_advantage: float = 0.0
    segment_id: int = 0
    distance_to_milestone: int | None = None

    @property
    def is_final(self) -> bool:
        return self.turn_index == self.trajectory.turn_count - 1

    @property
    def total_advantage(self) -> float:
        return self.global_advantage + self.local_advantage


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _reward_events(record: dict[str, Any]) -> dict[int, tuple[str, ...]]:
    raw_events = record.get("milestone_state_events_json") or "[]"
    events = json.loads(raw_events) if isinstance(raw_events, str) else raw_events
    result: dict[int, tuple[str, ...]] = {}
    for event in events or []:
        turn_index = int(event["turn_index"])
        state_ids = tuple(str(value) for value in event.get("state_ids") or [])
        if turn_index in result:
            raise ValueError(f"duplicate milestone event turn {turn_index} for request {record.get('request_id')}")
        result[turn_index] = state_ids
    return result


def _train_step_sizes(trace_dir: Path) -> list[int]:
    path = trace_dir / "monitoring" / "events.jsonl"
    if not path.is_file():
        return []
    sizes: list[int] = []
    for event in _read_jsonl(path):
        if event.get("event_type") != "train_step":
            continue
        metrics = event.get("metrics") or {}
        value = metrics.get("reward_quality/effective_pair_count")
        if value is not None:
            sizes.append(int(round(float(value))))
    return sizes


def _assign_updates(
    rewards: list[dict[str, Any]],
    turn_counts: dict[str, int],
    expected_sizes: list[int],
) -> list[int]:
    if not expected_sizes:
        return [0] * len(rewards)

    result: list[int] = []
    reward_position = 0
    for update_index, expected_turns in enumerate(expected_sizes):
        observed_turns = 0
        while reward_position < len(rewards) and observed_turns < expected_turns:
            request_id = str(rewards[reward_position].get("request_id") or "")
            observed_turns += turn_counts[request_id]
            result.append(update_index)
            reward_position += 1
        if observed_turns != expected_turns:
            raise ValueError(
                "reward records cannot be aligned to monitoring train step: "
                f"update={update_index + 1}, expected_turns={expected_turns}, "
                f"observed_turns={observed_turns}"
            )
    if reward_position != len(rewards):
        raise ValueError(f"{len(rewards) - reward_position} reward records remain after train-step alignment")
    return result


def load_trajectories(trace_dirs: Iterable[Path]) -> tuple[list[Trajectory], dict[str, Any]]:
    trajectories: list[Trajectory] = []
    seen_request_ids: set[str] = set()
    source_summaries: list[dict[str, Any]] = []

    for run_index, raw_trace_dir in enumerate(trace_dirs):
        trace_dir = raw_trace_dir.expanduser().resolve()
        rewards_path = trace_dir / "rewards.jsonl"
        rollouts_path = trace_dir / "rollouts.jsonl"
        if not rewards_path.is_file() or not rollouts_path.is_file():
            raise FileNotFoundError(f"trace directory lacks rewards/rollouts JSONL: {trace_dir}")

        rewards = _read_jsonl(rewards_path)
        reward_ids = [str(value.get("request_id") or "") for value in rewards]
        if any(not request_id for request_id in reward_ids):
            raise ValueError(f"reward record without request_id in {rewards_path}")
        if len(set(reward_ids)) != len(reward_ids):
            raise ValueError(f"duplicate reward request_id in {rewards_path}")

        reward_id_set = set(reward_ids)
        rollout_by_id: dict[str, dict[str, Any]] = {}
        all_rollout_count = 0
        for value in _iter_jsonl(rollouts_path):
            all_rollout_count += 1
            request_id = str(value.get("request_id") or "")
            if request_id in reward_id_set:
                rollout_by_id[request_id] = {
                    "iid": value.get("iid"),
                    "model_calls": value.get("model_calls"),
                    "training_samples": value.get("training_samples"),
                }
        missing_rollouts = set(reward_ids) - set(rollout_by_id)
        if missing_rollouts:
            raise ValueError(f"{len(missing_rollouts)} rewarded request IDs have no rollout in {rollouts_path}")

        turn_counts: dict[str, int] = {}
        for request_id, rollout in rollout_by_id.items():
            model_calls = int(rollout.get("model_calls") or 0)
            training_samples = int(rollout.get("training_samples") or 0)
            if model_calls <= 0 or training_samples != model_calls:
                raise ValueError(
                    f"request {request_id} has model_calls={model_calls}, training_samples={training_samples}"
                )
            turn_counts[request_id] = training_samples

        step_sizes = _train_step_sizes(trace_dir)
        update_indices = _assign_updates(rewards, turn_counts, step_sizes)
        for reward, update_index in zip(rewards, update_indices, strict=True):
            request_id = str(reward["request_id"])
            if request_id in seen_request_ids:
                raise ValueError(f"request ID appears in multiple trace directories: {request_id}")
            seen_request_ids.add(request_id)
            rollout = rollout_by_id[request_id]
            iid = str(reward.get("iid") or rollout.get("iid") or "")
            if not iid:
                raise ValueError(f"request {request_id} has no iid")
            events_by_turn = _reward_events(reward)
            turn_count = turn_counts[request_id]
            invalid_event_turns = [turn for turn in events_by_turn if not 0 <= turn < turn_count]
            if invalid_event_turns:
                raise ValueError(
                    f"request {request_id} has milestone turns outside [0, {turn_count}): {invalid_event_turns}"
                )
            trajectories.append(
                Trajectory(
                    run_index=run_index,
                    update_index=update_index,
                    request_id=request_id,
                    iid=iid,
                    score=float(reward["score"]),
                    overall_pass=bool(float(reward.get("benchmark_overall_pass") or 0.0)),
                    turn_count=turn_count,
                    events_by_turn=events_by_turn,
                )
            )

        source_summaries.append(
            {
                "trace_dir": str(trace_dir),
                "rewarded_trajectories": len(rewards),
                "rewarded_turn_rows": sum(turn_counts.values()),
                "unrewarded_rollouts_excluded": all_rollout_count - len(rollout_by_id),
                "train_step_sizes": step_sizes,
            }
        )

    return trajectories, {
        "sources": source_summaries,
        "unique_request_ids": len(seen_request_ids),
    }


def _sign(value: float) -> str:
    if value < -EPSILON:
        return "negative"
    if value > EPSILON:
        return "positive"
    return "zero"


def build_summary(
    trajectories: list[Trajectory],
    *,
    gamma: float = 0.95,
    local_weight: float = 1.0,
    source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("no rewarded trajectories")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if local_weight < 0.0:
        raise ValueError(f"local_weight must be non-negative, got {local_weight}")

    rows: list[TurnRow] = []
    rows_by_trajectory: dict[str, list[TurnRow]] = defaultdict(list)
    rows_by_group: dict[tuple[int, int, str], list[TurnRow]] = defaultdict(list)
    for trajectory in trajectories:
        for turn_index in range(trajectory.turn_count):
            row = TurnRow(
                trajectory=trajectory,
                turn_index=turn_index,
                state_ids=trajectory.events_by_turn.get(turn_index, ()),
            )
            rows.append(row)
            rows_by_trajectory[trajectory.request_id].append(row)
            rows_by_group[(trajectory.run_index, trajectory.update_index, trajectory.iid)].append(row)

    group_baselines: list[dict[str, Any]] = []
    for (run_index, update_index, iid), group_rows in rows_by_group.items():
        group_trajectories = {row.trajectory.request_id: row.trajectory for row in group_rows}
        row_baseline = sum(row.trajectory.score for row in group_rows) / len(group_rows)
        trajectory_baseline = sum(trajectory.score for trajectory in group_trajectories.values()) / len(
            group_trajectories
        )
        for row in group_rows:
            row.global_advantage = row.trajectory.score - row_baseline
            row.trajectory_equal_advantage = row.trajectory.score - trajectory_baseline
        group_baselines.append(
            {
                "run": run_index + 1,
                "step": update_index + 1,
                "iid": iid,
                "trajectories": len(group_trajectories),
                "turn_rows": len(group_rows),
                "row_weighted_baseline": row_baseline,
                "trajectory_equal_baseline": trajectory_baseline,
                "baseline_delta": row_baseline - trajectory_baseline,
            }
        )

    rows_by_segment: dict[tuple[int, int, str, int], list[TurnRow]] = defaultdict(list)
    completed_segment_lengths: list[int] = []
    for trajectory in trajectories:
        trajectory_rows = rows_by_trajectory[trajectory.request_id]
        segment_start = 0
        segment_id = 0
        for end_position, row in enumerate(trajectory_rows):
            row.segment_id = segment_id
            rows_by_segment[(trajectory.run_index, trajectory.update_index, trajectory.iid, segment_id)].append(row)
            if not row.state_ids:
                continue
            completed_segment_lengths.append(end_position - segment_start + 1)
            for position in range(segment_start, end_position + 1):
                segment_row = trajectory_rows[position]
                distance = end_position - position
                segment_row.process_reward = 10.0 * gamma**distance
                segment_row.distance_to_milestone = distance
            segment_start = end_position + 1
            segment_id += 1

    for segment_rows in rows_by_segment.values():
        baseline = sum(row.process_reward for row in segment_rows) / len(segment_rows)
        for row in segment_rows:
            row.local_advantage = local_weight * (row.process_reward - baseline)

    final_rows = [row for row in rows if row.is_final]
    nonfinal_rows = [row for row in rows if not row.is_final]
    event_rows = [row for row in rows if row.state_ids]
    prefix_rows = [row for row in rows if row.process_reward > 0.0]
    backfilled_rows = [row for row in prefix_rows if not row.state_ids]
    unshaped_rows = [row for row in rows if row.process_reward <= 0.0]
    failed_trajectories = [trajectory for trajectory in trajectories if not trajectory.overall_pass]

    global_signs = Counter(_sign(row.global_advantage) for row in rows)
    nonfinal_global_signs = Counter(_sign(row.global_advantage) for row in nonfinal_rows)
    event_global_signs = Counter(_sign(row.global_advantage) for row in event_rows)
    event_total_signs = Counter(_sign(row.total_advantage) for row in event_rows)
    prefix_global_signs = Counter(_sign(row.global_advantage) for row in prefix_rows)
    prefix_total_signs = Counter(_sign(row.total_advantage) for row in prefix_rows)
    backfilled_global_signs = Counter(_sign(row.global_advantage) for row in backfilled_rows)
    backfilled_local_signs = Counter(_sign(row.local_advantage) for row in backfilled_rows)
    backfilled_total_signs = Counter(_sign(row.total_advantage) for row in backfilled_rows)
    event_local_signs = Counter(_sign(row.local_advantage) for row in event_rows)
    unshaped_global_signs = Counter(_sign(row.global_advantage) for row in unshaped_rows)
    unshaped_local_signs = Counter(_sign(row.local_advantage) for row in unshaped_rows)
    unshaped_total_signs = Counter(_sign(row.total_advantage) for row in unshaped_rows)
    current_vs_equal_sign_changes = sum(
        _sign(row.global_advantage) != _sign(row.trajectory_equal_advantage) for row in rows
    )
    current_vs_equal_trajectory_changes = len(
        {
            row.trajectory.request_id
            for row in rows
            if _sign(row.global_advantage) != _sign(row.trajectory_equal_advantage)
        }
    )

    distance_counts = Counter(
        int(row.distance_to_milestone) for row in prefix_rows if row.distance_to_milestone is not None
    )
    distance_reward_sums: Counter[int] = Counter()
    for row in prefix_rows:
        if row.distance_to_milestone is not None:
            distance_reward_sums[int(row.distance_to_milestone)] += row.process_reward
    distance_buckets = [
        {
            "distance": distance,
            "turn_rows": distance_counts[distance],
            "process_reward": 10.0 * gamma**distance,
            "process_reward_sum": distance_reward_sums[distance],
            "direct_milestone_event": distance == 0,
        }
        for distance in sorted(distance_counts)
    ]
    segment_length_counts = Counter(completed_segment_lengths)
    completed_segment_count = len(completed_segment_lengths)
    segment_length_buckets = [
        {
            "turn_count": turn_count,
            "segments": segment_length_counts[turn_count],
            "share": segment_length_counts[turn_count] / completed_segment_count,
        }
        for turn_count in sorted(segment_length_counts)
    ]

    updates: list[dict[str, Any]] = []
    updates_by_key: dict[tuple[int, int], list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        updates_by_key[(trajectory.run_index, trajectory.update_index)].append(trajectory)
    for (run_index, update_index), update_trajectories in sorted(updates_by_key.items()):
        turn_rows = sum(trajectory.turn_count for trajectory in update_trajectories)
        final_count = len(update_trajectories)
        update_groups = [
            group for group in group_baselines if group["run"] == run_index + 1 and group["step"] == update_index + 1
        ]
        update_rows = [
            row for row in rows if row.trajectory.run_index == run_index and row.trajectory.update_index == update_index
        ]
        updates.append(
            {
                "run": run_index + 1,
                "step": update_index + 1,
                "label": f"R{run_index + 1} · S{update_index + 1}",
                "trajectories": final_count,
                "turn_rows": turn_rows,
                "nonfinal_rows": turn_rows - final_count,
                "wrong_rate": (turn_rows - final_count) / turn_rows,
                "mean_abs_baseline_delta": sum(abs(group["baseline_delta"]) for group in update_groups)
                / len(update_groups),
                "max_abs_baseline_delta": max(abs(group["baseline_delta"]) for group in update_groups),
                "advantage_sign_changed_rows": sum(
                    _sign(row.global_advantage) != _sign(row.trajectory_equal_advantage) for row in update_rows
                ),
                "milestone_boundary_turns": sum(bool(row.state_ids) for row in update_rows),
                "backfilled_non_boundary_turns": sum(
                    row.process_reward > 0.0 and not row.state_ids for row in update_rows
                ),
                "unfinished_segment_turns": sum(row.process_reward <= 0.0 for row in update_rows),
                "process_reward_mean": sum(row.process_reward for row in update_rows) / len(update_rows),
                "local_advantage_abs_mean": sum(abs(row.local_advantage) for row in update_rows) / len(update_rows),
                "negative_local_advantage_turns": sum(row.local_advantage < -EPSILON for row in update_rows),
            }
        )

    score_buckets: list[dict[str, Any]] = []
    for score in sorted({trajectory.score for trajectory in trajectories}):
        bucket_trajectories = [trajectory for trajectory in trajectories if trajectory.score == score]
        bucket_nonfinal_rows = sum(trajectory.turn_count - 1 for trajectory in bucket_trajectories)
        score_buckets.append(
            {
                "score": score,
                "trajectories": len(bucket_trajectories),
                "nonfinal_rows": bucket_nonfinal_rows,
                "excess_score_sum": score * bucket_nonfinal_rows,
            }
        )

    terminal_score_sum = sum(trajectory.score for trajectory in trajectories)
    implemented_score_sum = sum(row.trajectory.score for row in rows)
    nonfinal_excess_score_sum = implemented_score_sum - terminal_score_sum
    multi_turn_trajectories = [trajectory for trajectory in trajectories if trajectory.turn_count > 1]
    query_groups = {(trajectory.run_index, trajectory.update_index, trajectory.iid) for trajectory in trajectories}
    affected_query_groups = {
        (trajectory.run_index, trajectory.update_index, trajectory.iid) for trajectory in multi_turn_trajectories
    }

    failed_with_any_milestone = sum(bool(trajectory.events_by_turn) for trajectory in failed_trajectories)
    failed_with_early_milestone = sum(
        any(turn < trajectory.turn_count - 1 for turn in trajectory.events_by_turn)
        for trajectory in failed_trajectories
    )

    baseline_deltas = [float(group["baseline_delta"]) for group in group_baselines]
    affected_baseline_groups = [delta for delta in baseline_deltas if abs(delta) > EPSILON]
    group_size_counts = Counter(int(group["trajectories"]) for group in group_baselines)
    typical_group_size = max(group_size_counts, key=lambda size: (group_size_counts[size], size))
    trajectory_lengths: list[dict[str, Any]] = []
    for turn_count in sorted({trajectory.turn_count for trajectory in trajectories}):
        bucket = [trajectory for trajectory in trajectories if trajectory.turn_count == turn_count]
        trajectory_lengths.append(
            {
                "turn_count": turn_count,
                "trajectories": len(bucket),
                "overall_pass": sum(trajectory.overall_pass for trajectory in bucket),
                "mean_score": sum(trajectory.score for trajectory in bucket) / len(bucket),
            }
        )

    verifier_errors = 0
    judge_errors = 0
    judge_completed = 0
    tool_mapping_coverages: list[float] = []
    if source_info:
        for source in source_info.get("sources") or []:
            rewards_path = Path(source["trace_dir"]) / "rewards.jsonl"
            for reward in _read_jsonl(rewards_path):
                verifier_errors += bool(str(reward.get("milestone_verifier_error") or "").strip())
                judge_errors += bool(str(reward.get("benchmark_judge_error") or "").strip())
                judge_completed += reward.get("milestone_judge_status") == "completed"
                coverage = reward.get("milestone_tool_mapping_coverage")
                if coverage is not None:
                    tool_mapping_coverages.append(float(coverage))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "The audit distinguishes terminal-score fan-out, the row-weighted GRPO baseline it creates, "
            "and process rewards backfilled to non-boundary turns inside completed milestone segments."
        ),
        "configuration": {
            "gamma": gamma,
            "local_weight": local_weight,
            "normalize_global_by_std": False,
            "normalize_segment_by_std": False,
            "unit": "model-call row (not token weighted)",
        },
        "overview": {
            "query_groups": len(query_groups),
            "affected_query_groups": len(affected_query_groups),
            "trajectories": len(trajectories),
            "multi_turn_trajectories": len(multi_turn_trajectories),
            "single_turn_trajectories": len(trajectories) - len(multi_turn_trajectories),
            "turn_rows": len(rows),
            "final_rows": len(final_rows),
            "nonfinal_rows": len(nonfinal_rows),
            "terminal_score_fanout_rate": len(nonfinal_rows) / len(rows),
            # Kept for compatibility with the first report schema.
            "wrong_turn_rate": len(nonfinal_rows) / len(rows),
            "overall_pass_trajectories": sum(trajectory.overall_pass for trajectory in trajectories),
            "overall_fail_trajectories": len(failed_trajectories),
            "terminal_score_sum": terminal_score_sum,
            "implemented_score_sum": implemented_score_sum,
            "nonfinal_excess_score_sum": nonfinal_excess_score_sum,
            "score_attribution_factor": implemented_score_sum / terminal_score_sum,
            "nonfinal_rows_from_failed_outcomes": sum(not row.trajectory.overall_pass for row in nonfinal_rows),
        },
        "global_advantage": {
            "all_turns": dict(global_signs),
            "nonfinal_turns": dict(nonfinal_global_signs),
            "query_groups": len(group_baselines),
            "groups_with_row_weighting_bias": len(affected_baseline_groups),
            "mean_abs_baseline_delta": (sum(abs(delta) for delta in baseline_deltas) / len(baseline_deltas)),
            "max_abs_baseline_delta": max(abs(delta) for delta in baseline_deltas),
            "advantage_sign_changed_rows": current_vs_equal_sign_changes,
            "advantage_sign_changed_trajectories": current_vs_equal_trajectory_changes,
            "typical_group_size": typical_group_size,
            "group_size_distribution": dict(sorted(group_size_counts.items())),
            "group_baselines": sorted(
                group_baselines,
                key=lambda value: abs(float(value["baseline_delta"])),
                reverse=True,
            ),
        },
        "segment_attribution": {
            "completed_segments": completed_segment_count,
            "completed_segment_turns": sum(completed_segment_lengths),
            "mean_completed_segment_turns": (
                sum(completed_segment_lengths) / completed_segment_count if completed_segment_count else 0.0
            ),
            "median_completed_segment_turns": (
                statistics.median(completed_segment_lengths) if completed_segment_lengths else 0.0
            ),
            "single_turn_segments": segment_length_counts[1],
            "single_turn_segment_share": (
                segment_length_counts[1] / completed_segment_count if completed_segment_count else 0.0
            ),
            "at_most_two_turn_segments": segment_length_counts[1] + segment_length_counts[2],
            "at_most_two_turn_segment_share": (
                (segment_length_counts[1] + segment_length_counts[2]) / completed_segment_count
                if completed_segment_count
                else 0.0
            ),
            "max_completed_segment_turns": max(completed_segment_lengths, default=0),
            "segment_length_buckets": segment_length_buckets,
            "shaped_turns": len(prefix_rows),
            "milestone_boundary_turns": len(event_rows),
            "backfilled_non_boundary_turns": len(backfilled_rows),
            "unshaped_turns": len(unshaped_rows),
            "backfilled_share_of_shaped": len(backfilled_rows) / len(prefix_rows) if prefix_rows else 0.0,
            "backfilled_share_of_all": len(backfilled_rows) / len(rows),
            "boundary_process_reward_sum": sum(row.process_reward for row in event_rows),
            "backfilled_process_reward_sum": sum(row.process_reward for row in backfilled_rows),
            "max_backfill_distance": max(
                (int(row.distance_to_milestone or 0) for row in backfilled_rows),
                default=0,
            ),
            "backfilled_global_sign": dict(backfilled_global_signs),
            "backfilled_local_sign": dict(backfilled_local_signs),
            "backfilled_total_sign": dict(backfilled_total_signs),
            "event_local_sign": dict(event_local_signs),
            "unshaped_global_sign": dict(unshaped_global_signs),
            "unshaped_local_sign": dict(unshaped_local_signs),
            "unshaped_total_sign": dict(unshaped_total_signs),
            "distance_buckets": distance_buckets,
            "semantic_error_label_available": False,
            "interpretation": (
                "A backfilled non-boundary turn received process reward because a later turn completed the segment. "
                "It is an attribution-risk row, not proof that the turn itself was semantically wrong."
            ),
        },
        "milestone_impact": {
            "event_turns": len(event_rows),
            "completed_states": sum(len(row.state_ids) for row in event_rows),
            "early_event_turns": sum(not row.is_final for row in event_rows),
            "early_completed_states": sum(len(row.state_ids) for row in event_rows if not row.is_final),
            "failed_trajectories_with_any_milestone": failed_with_any_milestone,
            "failed_trajectories_with_early_milestone": failed_with_early_milestone,
            "event_turn_global_sign": dict(event_global_signs),
            "event_turn_total_sign": dict(event_total_signs),
            "shaped_prefix_turns": len(prefix_rows),
            "prefix_global_sign": dict(prefix_global_signs),
            "prefix_total_sign": dict(prefix_total_signs),
        },
        "updates": updates,
        "score_buckets": score_buckets,
        "trajectory_lengths": trajectory_lengths,
        "validation": {
            "rewarded_request_ids": len(trajectories),
            "unique_request_ids": source_info.get("unique_request_ids") if source_info else len(trajectories),
            "judge_completed": judge_completed,
            "benchmark_judge_errors": judge_errors,
            "milestone_verifier_errors": verifier_errors,
            "mean_tool_mapping_coverage": (
                sum(tool_mapping_coverages) / len(tool_mapping_coverages) if tool_mapping_coverages else None
            ),
            "tool_mapping_coverage_below_one": sum(coverage < 1.0 for coverage in tool_mapping_coverages),
        },
        "sources": (source_info or {}).get("sources") or [],
    }
    return summary


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float | int, digits: int = 2) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _bar(value: float, maximum: float, *, color: str, label: str) -> str:
    width = 0.0 if maximum <= 0 else 100.0 * value / maximum
    return (
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{width:.3f}%;background:{color}"></div>'
        f"<span>{html.escape(label)}</span></div>"
    )


def render_simple_html(summary: dict[str, Any], title: str) -> str:
    overview = summary["overview"]
    outcome = summary["global_advantage"]
    segment = summary["segment_attribution"]
    payload = json.dumps(summary, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#172033;--muted:#657087;--line:#dce2ec;--bg:#f3f6fa;--panel:#fff;--navy:#102b46;--red:#d94f4f;--red-soft:#fde8e7;--green:#238866;--green-soft:#e4f5ef;--amber:#b46b12;--amber-soft:#fff3dd;--blue:#3274b9;--blue-soft:#e8f1fb}}
*{{box-sizing:border-box;letter-spacing:0}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif}}
header{{padding:30px max(20px,calc((100% - 1080px)/2));background:var(--navy);color:#fff;border-bottom:5px solid #ec7063}} h1{{margin:0;font-size:28px}} header p{{margin:6px 0 0;color:#d4e1eb}}
main{{max-width:1080px;margin:auto;padding:22px}} section{{margin-bottom:18px}} h2{{margin:0 0 10px;font-size:19px}} h3{{margin:0 0 7px;font-size:15px}}
.panel,.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px}} .panel{{padding:18px}} .cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}} .card{{padding:15px}}
.answer{{padding:18px;border-radius:8px;background:var(--red-soft);border:1px solid #f1b9b5;font-size:17px}} .answer strong{{color:#a62e2e}}
.flow{{display:grid;grid-template-columns:1fr 34px 1fr 34px 1fr;align-items:center;gap:6px}} .node{{height:100%;padding:13px;border:1px solid var(--line);border-radius:7px;background:#fff}} .node b{{display:block;margin-bottom:3px}} .arrow{{text-align:center;color:var(--red);font-size:22px;font-weight:800}}
.value{{display:block;margin:2px 0;font-size:29px;font-weight:760}} .muted{{color:var(--muted);font-size:13px}} .bad{{color:var(--red)}} .warn{{color:var(--amber)}} .good{{color:var(--green)}}
.compare{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .lane{{padding:14px;border-radius:8px}} .lane.current{{background:var(--red-soft);border:1px solid #f1b9b5}} .lane.target{{background:var(--green-soft);border:1px solid #acdccc}}
table{{width:100%;min-width:620px;border-collapse:collapse}} th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-size:12px}} .table-wrap{{overflow-x:auto}}
.bar-row{{display:grid;grid-template-columns:190px 1fr 110px;align-items:center;gap:10px;margin:10px 0}} .track{{height:22px;background:#edf1f6;border-radius:5px;overflow:hidden}} .fill{{height:100%}} .count{{font-weight:700;text-align:right}}
.callout{{margin-top:12px;padding:12px 14px;border-left:4px solid var(--blue);background:var(--blue-soft)}} .decision{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .decision div{{padding:14px;border-radius:8px}} .confirmed{{background:var(--red-soft);border:1px solid #f1b9b5}} .risk{{background:var(--amber-soft);border:1px solid #ead09f}}
code{{font:13px ui-monospace,SFMono-Regular,Consolas,monospace}} a{{color:var(--blue)}} footer{{color:var(--muted);font-size:12px;margin:20px 0}}
@media(max-width:760px){{.cards,.compare,.decision{{grid-template-columns:1fr}}.flow{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}.bar-row{{grid-template-columns:1fr}}.count{{text-align:left}}}}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p>{_num(overview["trajectories"])} 条 Docs trajectory，{_num(len(summary["updates"]))} 个训练 step，共 {_num(overview["turn_rows"])} 个 model turn。</p></header>
<main>
<section class="answer"><strong>一句话结论：</strong>同一个 query 的 {_num(outcome["typical_group_size"])} 条 rollout 本应“一条 trajectory 一票”计算 outcome baseline；当前实现是展开以后“一个 turn 一票”，所以 turn 越多的 rollout 对 baseline 影响越大。</section>

<section class="panel"><h2>问题发生在哪里</h2>
<div class="compare">
  <div class="lane current"><h3>当前实现</h3><div class="flow"><div class="node"><b>先展开</b>每条 trajectory 变成多个 turn row</div><div class="arrow">→</div><div class="node"><b>再求均值</b><code>Σ(Tᵢ × Rᵢ) / ΣTᵢ</code></div></div><p>10 个 turn 的 trajectory 相当于投了 10 票。</p></div>
  <div class="lane target"><h3>应使用的对照</h3><div class="flow"><div class="node"><b>先比较</b>{_num(outcome["typical_group_size"])} 条 trajectory 各投 1 票</div><div class="arrow">→</div><div class="node"><b>再广播</b>把算好的 <code>A_global</code> 给该轨迹所有 turn</div></div><p>多轮 action 仍能得到终局 credit，但不会改变组 baseline。</p></div>
</div></section>

<section class="panel"><h2>一个会翻转奖励方向的例子</h2><div class="table-wrap"><table><thead><tr><th>Trajectory</th><th>Score</th><th>Turn 数</th><th>trajectory 等权 advantage</th><th>当前 row 加权 advantage</th></tr></thead><tbody>
<tr><td>A</td><td>1.00</td><td>3</td><td>+0.433</td><td>+0.260</td></tr>
<tr><td><strong>B</strong></td><td>0.70</td><td>1</td><td class="good"><strong>+0.133</strong></td><td class="bad"><strong>-0.040</strong></td></tr>
<tr><td>C</td><td>0.00</td><td>1</td><td>-0.567</td><td>-0.740</td></tr>
</tbody></table></div><div class="callout">B 自己的 score 和内容都没变，仅因为高分 A 有 3 个 turn，B 就从正 advantage 变成负 advantage。</div></section>

<section class="cards">
  <div class="card"><span class="muted">发生 baseline 偏移的 query 组</span><span class="value bad">{_num(outcome["groups_with_row_weighting_bias"])} / {_num(outcome["query_groups"])}</span><span class="muted">平均绝对偏移 {outcome["mean_abs_baseline_delta"]:.4f}</span></div>
  <div class="card"><span class="muted">outcome advantage 方向翻转</span><span class="value bad">{_num(outcome["advantage_sign_changed_rows"])} turns</span><span class="muted">涉及 {_num(outcome["advantage_sign_changed_trajectories"])} 条 trajectory</span></div>
  <div class="card"><span class="muted">终局 score 出现在非 final turn</span><span class="value warn">{_num(overview["nonfinal_rows"])}</span><span class="muted">这本身不全是错，错误点是先按这些 row 求 baseline</span></div>
</section>

<section class="panel"><h2>Milestone segment 又做了什么</h2>
<div class="bar-row"><span>直接完成 milestone 的 turn</span><div class="track"><div class="fill" style="width:{segment["milestone_boundary_turns"] / overview["turn_rows"]:.1%};background:var(--blue)"></div></div><span class="count">{_num(segment["milestone_boundary_turns"])}</span></div>
<div class="bar-row"><span>向前衰减回填的 turn</span><div class="track"><div class="fill" style="width:{segment["backfilled_non_boundary_turns"] / overview["turn_rows"]:.1%};background:var(--amber)"></div></div><span class="count warn">{_num(segment["backfilled_non_boundary_turns"])}</span></div>
<div class="bar-row"><span>没有完成当前 segment</span><div class="track"><div class="fill" style="width:{segment["unshaped_turns"] / overview["turn_rows"]:.1%};background:#9aa7b8"></div></div><span class="count">{_num(segment["unshaped_turns"])}</span></div>
<div class="callout">{_num(segment["backfilled_non_boundary_turns"])} 个回填 turn 没有在当下关闭 milestone，只是后面的 turn 完成了 milestone，所以它们也获得 <code>10 × 0.95^distance</code>。其中可能有必要的搜索/读取，也可能有绕路或错误；现有数据没有逐 turn 正误标签，不能把 {_num(segment["backfilled_non_boundary_turns"])} 个全部判错。</div>
<p>另外，{_num(segment["unshaped_turns"])} 个未完成当前 segment 的 turn 以 process reward 0 参与比较，其中 <strong>{_num(segment["unshaped_local_sign"].get("negative", 0))}</strong> 个获得负 <code>A_local</code>。</p></section>

<section class="panel"><h2>应该如何解读</h2><div class="decision">
  <div class="confirmed"><h3>已确认需要修</h3><p><strong>Outcome baseline 必须先按 trajectory 等权计算，再广播到 turn。</strong>真实数据中已有 {_num(outcome["advantage_sign_changed_rows"])} 个 turn 被改变了训练方向。</p></div>
  <div class="risk"><h3>需要进一步判断</h3><p><strong>Segment 回填的 {_num(segment["backfilled_non_boundary_turns"])} 个 turn 是风险集合，不是 {_num(segment["backfilled_non_boundary_turns"])} 个确定错误。</strong>若要只惩罚具体坏 turn，需要新增逐 turn checklist/judge，而不是只改 gamma。</p></div>
</div></section>

<section class="panel"><h2>不要误读的数字</h2><p><strong>{_pct(overview["terminal_score_fanout_rate"])} 并不表示 {_pct(overview["terminal_score_fanout_rate"])} 的 turn 都错了。</strong>它只表示这些 turn 在 final 之前也携带相同 outcome score。报告确认的直接影响是 {_num(outcome["advantage_sign_changed_rows"])} 个 turn 的 advantage 符号翻转。</p><p><a href="details.html">查看详细统计附录</a> · <a href="summary.json">查看机器可读数据</a> · <a href="overview.png">查看总览图</a></p></section>
<footer>Generated {html.escape(summary["generated_at"])}.</footer>
<script id="report-data" type="application/json">{payload}</script>
</main></body></html>"""


def render_html(summary: dict[str, Any], title: str) -> str:
    overview = summary["overview"]
    impact = summary["milestone_impact"]
    segment = summary["segment_attribution"]
    global_advantage = summary["global_advantage"]
    validation = summary["validation"]
    updates = summary["updates"]
    score_buckets = summary["score_buckets"]
    max_update_rows = max(update["turn_rows"] for update in updates)
    max_bucket_rows = max(bucket["nonfinal_rows"] for bucket in score_buckets)
    max_distance_rows = max(bucket["turn_rows"] for bucket in segment["distance_buckets"])

    update_rows = "".join(
        "<tr>"
        f"<td>{html.escape(update['label'])}</td>"
        f"<td class='num'>{_num(update['trajectories'])}</td>"
        f"<td class='num'>{_num(update['turn_rows'])}</td>"
        f"<td class='num risk'>{_num(update['nonfinal_rows'])}</td>"
        f"<td>{_bar(update['nonfinal_rows'], max_update_rows, color='#e05252', label=_pct(update['wrong_rate']))}</td>"
        f"<td class='num'>{update['mean_abs_baseline_delta']:.4f}</td>"
        f"<td class='num risk'>{_num(update['advantage_sign_changed_rows'])}</td>"
        "</tr>"
        for update in updates
    )
    bucket_rows = "".join(
        "<tr>"
        f"<td class='num'>{bucket['score']:.2f}</td>"
        f"<td class='num'>{_num(bucket['trajectories'])}</td>"
        f"<td class='num risk'>{_num(bucket['nonfinal_rows'])}</td>"
        f"<td>{_bar(bucket['nonfinal_rows'], max_bucket_rows, color='#ef8a62', label=_num(bucket['nonfinal_rows']))}</td>"
        f"<td class='num'>{_num(bucket['excess_score_sum'])}</td>"
        "</tr>"
        for bucket in score_buckets
    )
    distance_rows = "".join(
        "<tr>"
        f"<td>{'milestone boundary' if bucket['distance'] == 0 else str(bucket['distance']) + ' turn before'}</td>"
        f"<td class='num'>{_num(bucket['turn_rows'])}</td>"
        f"<td class='num'>{bucket['process_reward']:.4f}</td>"
        f"<td>{_bar(bucket['turn_rows'], max_distance_rows, color='#3274b9' if bucket['distance'] == 0 else '#d68b22', label=_num(bucket['turn_rows']))}</td>"
        "</tr>"
        for bucket in segment["distance_buckets"]
    )
    top_group_rows = "".join(
        "<tr>"
        f"<td>R{group['run']} · S{group['step']}</td>"
        f"<td><code>{html.escape(group['iid'])}</code></td>"
        f"<td class='num'>{_num(group['trajectories'])}</td>"
        f"<td class='num'>{_num(group['turn_rows'])}</td>"
        f"<td class='num'>{group['trajectory_equal_baseline']:.4f}</td>"
        f"<td class='num'>{group['row_weighted_baseline']:.4f}</td>"
        f"<td class='num risk'>{group['baseline_delta']:+.4f}</td>"
        "</tr>"
        for group in global_advantage["group_baselines"][:10]
    )

    event_global_negative = impact["event_turn_global_sign"].get("negative", 0)
    event_total_negative = impact["event_turn_total_sign"].get("negative", 0)
    backfilled_global_negative = segment["backfilled_global_sign"].get("negative", 0)
    backfilled_total_negative = segment["backfilled_total_sign"].get("negative", 0)
    source_rows = "".join(
        "<li>"
        f"<code>{html.escape(source['trace_dir'])}</code> — "
        f"{_num(source['rewarded_trajectories'])} rewarded trajectories, "
        f"{_num(source['unrewarded_rollouts_excluded'])} unrewarded rollouts excluded"
        "</li>"
        for source in summary["sources"]
    )
    payload = json.dumps(summary, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#172033;--muted:#657087;--line:#dce2ec;--panel:#fff;--bg:#f3f6fa;--red:#d94f4f;--red-soft:#fde8e7;--blue:#3274b9;--blue-soft:#e8f1fb;--green:#238866;--green-soft:#e4f5ef;--amber:#d68b22;--navy:#102b46}}
*{{box-sizing:border-box;letter-spacing:0}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif}}
header{{padding:34px max(24px,calc((100% - 1240px)/2));background:var(--navy);color:#fff;border-bottom:5px solid #ec7063}}
h1{{margin:0;font-size:29px}} header p{{max-width:900px;margin:8px 0 0;color:#d4e1eb;font-size:15px}}
main{{max-width:1240px;margin:auto;padding:24px}} section{{margin:0 0 22px}} h2{{margin:0 0 12px;font-size:19px}} h3{{margin:0 0 8px;font-size:15px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 2px 8px #243b5320}} .card{{padding:16px}} .panel{{padding:18px;overflow-x:auto}}
.label{{color:var(--muted);font-size:12px}} .value{{margin-top:2px;font-size:27px;font-weight:750}} .detail{{margin-top:4px;color:var(--muted);font-size:12px}}
.bad{{color:var(--red)!important}} .risk{{color:#b46b12!important}} .good{{color:var(--green)!important}} .warn{{color:var(--amber)!important}} code{{font:12px ui-monospace,SFMono-Regular,Consolas,monospace}}
.flow{{display:grid;grid-template-columns:1fr 38px 1fr 38px 1fr 38px 1fr;align-items:stretch;gap:8px}} .node{{border:1px solid var(--line);border-radius:8px;padding:14px;background:#fff}} .node strong{{display:block;margin-bottom:6px}} .node code{{display:block;padding:7px;background:#f6f8fb;border-radius:5px;white-space:normal}} .arrow{{display:flex;align-items:center;justify-content:center;color:var(--red);font-size:25px;font-weight:800}}
.actual-expected{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}} .lane{{border-radius:8px;padding:13px}} .lane.actual{{background:var(--red-soft);border:1px solid #f3b8b4}} .lane.expected{{background:var(--green-soft);border:1px solid #a9decd}} .turns{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}} .turn{{padding:9px;text-align:center;border-radius:6px;background:#fff;border:1px solid #ffffffaa}} .turn b{{display:block;font-size:18px}}
.bar-track{{position:relative;height:22px;background:#edf1f6;border-radius:5px;overflow:hidden;min-width:130px}} .bar-fill{{position:absolute;inset:0 auto 0 0;border-radius:5px}} .bar-track span{{position:absolute;inset:1px 7px;text-align:right;font-weight:650;font-variant-numeric:tabular-nums}}
.compare{{display:grid;grid-template-columns:160px 1fr 70px;gap:8px;align-items:center;margin:10px 0}} .compare-label{{font-size:13px}} .compare-value{{text-align:right;font-weight:700;font-variant-numeric:tabular-nums}}
.insight{{padding:13px 15px;border-left:4px solid var(--red);background:var(--red-soft);border-radius:0 7px 7px 0}} .note{{padding:13px 15px;border-left:4px solid var(--blue);background:var(--blue-soft);border-radius:0 7px 7px 0}}
table{{width:100%;min-width:620px;border-collapse:collapse}} th,td{{padding:8px 7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}} th{{color:var(--muted);font-size:12px;font-weight:650}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.checks{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}} .check{{padding:11px;border-radius:7px;background:var(--green-soft);border:1px solid #b7e3d5}} .check b{{display:block;color:var(--green);font-size:18px}} .metric-stack{{display:grid;gap:9px}} .metric-row{{padding:11px;border:1px solid var(--line);border-radius:7px;background:#f8fafc}} .metric-row b{{display:block;font-size:22px}}
ul{{margin:8px 0 0;padding-left:20px}} footer{{color:var(--muted);font-size:12px;margin-top:22px}}
@media(max-width:900px){{.grid,.checks{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}.flow{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg);height:22px}}}}
@media(max-width:560px){{main{{padding:14px}}header{{padding:26px 16px}}.grid,.checks,.actual-expected{{grid-template-columns:1fr}}.value{{font-size:23px}}}}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p>基于真实训练 row 复算 outcome 与 milestone reward，区分终局 score fan-out、turn 数引入的 baseline 偏移，以及 segment 内的衰减回填。</p></header>
<main>
<section class="grid">
  <div class="card"><div class="label">非 final score fan-out</div><div class="value risk">{_num(overview["nonfinal_rows"])}</div><div class="detail">占全部 {_num(overview["turn_rows"])} 个 turn 的 {_pct(overview["terminal_score_fanout_rate"])}</div></div>
  <div class="card"><div class="label">baseline 被 turn 数改变的组</div><div class="value bad">{_num(global_advantage["groups_with_row_weighting_bias"])} / {_num(global_advantage["query_groups"])}</div><div class="detail">平均绝对偏移 {global_advantage["mean_abs_baseline_delta"]:.4f}</div></div>
  <div class="card"><div class="label">advantage 符号变化</div><div class="value bad">{_num(global_advantage["advantage_sign_changed_rows"])}</div><div class="detail">涉及 {_num(global_advantage["advantage_sign_changed_trajectories"])} 条 trajectory</div></div>
  <div class="card"><div class="label">segment 回填的非边界 turn</div><div class="value warn">{_num(segment["backfilled_non_boundary_turns"])}</div><div class="detail">占全部 shaped turn 的 {_pct(segment["backfilled_share_of_shaped"])}</div></div>
</section>

<section class="panel">
<h2>Outcome reward 的 turn 权重问题</h2>
<div class="flow">
  <div class="node"><strong>① 一条 ADK trajectory</strong><code>turn 0 → turn 1 → turn 2(final)</code><span class="detail">共享同一个 request_id</span></div><div class="arrow">→</div>
  <div class="node"><strong>② 展开为独立训练 row</strong><code>AgentLoopOutput × 3</code><span class="detail">每个 model call 都成为一个样本</span></div><div class="arrow">→</div>
  <div class="node"><strong>③ Judge 只评一次终局</strong><code>results[request_id] = score</code><span class="detail">一条 trajectory 一个 outcome</span></div><div class="arrow">→</div>
  <div class="node"><strong>④ 在所有 turn row 上求 baseline</strong><code>b = Σ(TᵢRᵢ) / ΣTᵢ</code><span class="detail">长 trajectory 在组均值中权重更大</span></div>
</div>
<div class="actual-expected">
  <div class="lane actual"><strong>当前：row 等权求 baseline</strong><div class="turns"><div class="turn">turn 0<b>R</b></div><div class="turn">turn 1<b>R</b></div><div class="turn">final<b>R</b></div></div></div>
  <div class="lane expected"><strong>对照：trajectory 等权后再广播</strong><div class="turns"><div class="turn">turn 0<b>A</b></div><div class="turn">turn 1<b>A</b></div><div class="turn">final<b>A</b></div></div></div>
</div>
<div class="insight" style="margin-top:14px"><strong>可确认的问题：</strong>不是“终局回报能否传到早期 action”本身，而是组 baseline 在展开后按 turn 数加权。本数据中 {_num(global_advantage["groups_with_row_weighting_bias"])} 个组发生偏移，最大绝对偏移 {global_advantage["max_abs_baseline_delta"]:.4f}，导致 {_num(global_advantage["advantage_sign_changed_rows"])} 个训练 row 的 outcome advantage 符号改变。</div>
</section>

<section class="two">
<div class="panel"><h2>每个有效更新批次</h2><table><thead><tr><th>批次</th><th class="num">Trajectory</th><th class="num">Turn</th><th class="num">非 final</th><th>Fan-out</th><th class="num">|Δbaseline|</th><th class="num">符号变更</th></tr></thead><tbody>{update_rows}</tbody></table><p class="detail">R1 为首批训练；R2 为 seedfix 续训。7 个 optimizer update 均纳入统计。</p></div>
<div class="panel"><h2>raw score fan-out 规模</h2>
  <div class="compare"><span class="compare-label">每 trajectory 计一次</span>{_bar(overview["terminal_score_sum"], overview["implemented_score_sum"], color="#238866", label=f"{overview['terminal_score_sum']:.2f}")}<span class="compare-value">1.00×</span></div>
  <div class="compare"><span class="compare-label">当前每 turn 回填</span>{_bar(overview["implemented_score_sum"], overview["implemented_score_sum"], color="#d94f4f", label=f"{overview['implemented_score_sum']:.2f}")}<span class="compare-value">{overview["score_attribution_factor"]:.2f}×</span></div>
  <div class="note" style="margin-top:15px">raw score 总和从 <strong>{overview["terminal_score_sum"]:.2f}</strong> 变为 <strong>{overview["implemented_score_sum"]:.2f}</strong>。这不表示 policy loss 简单放大 {overview["score_attribution_factor"]:.2f} 倍；真正需要关注的是 trajectory 等权 baseline 和后续 credit assignment。</div>
</div>
</section>

<section class="panel"><h2>baseline 偏移最大的 query 组</h2><table><thead><tr><th>批次</th><th>IID</th><th class="num">Trajectory</th><th class="num">Turn</th><th class="num">trajectory baseline</th><th class="num">row baseline</th><th class="num">Δ</th></tr></thead><tbody>{top_group_rows}</tbody></table><p class="detail">同一 query 的 rollout 数相同，但每条 rollout 的 turn 数不同；当 score 与长度相关时，row baseline 会偏离 8 条 trajectory 等权均值。</p></section>

<section class="two">
<div class="panel"><h2>Segment 衰减回填</h2>
  <div class="compare"><span class="compare-label">直接完成 milestone</span>{_bar(segment["milestone_boundary_turns"], segment["shaped_turns"], color="#3274b9", label=_num(segment["milestone_boundary_turns"]))}<span class="compare-value">{_pct(segment["milestone_boundary_turns"] / segment["shaped_turns"])}</span></div>
  <div class="compare"><span class="compare-label">非边界回填 turn</span>{_bar(segment["backfilled_non_boundary_turns"], segment["shaped_turns"], color="#d68b22", label=_num(segment["backfilled_non_boundary_turns"]))}<span class="compare-value">{_pct(segment["backfilled_share_of_shaped"])}</span></div>
  <div class="note">这 {_num(segment["backfilled_non_boundary_turns"])} 个 turn 没有在该 turn 关闭任何 milestone，但因为后续 turn 完成 segment，获得了 <code>10 × 0.95^distance</code>。它们是需要审计的归因风险，不等于已证明的错误 turn。</div>
  <div class="metric-row" style="margin-top:10px"><span class="label">未完成当前 segment 的 zero-reward turn</span><b>{_num(segment["unshaped_turns"])}</b><span>{_num(segment["unshaped_local_sign"].get("negative", 0))} 个因同组其他 rollout 完成该 segment 而获得负 A_local；其余 baseline 为 0。</span></div>
</div>
<div class="panel"><h2>距 milestone 的回填距离</h2><table><thead><tr><th>位置</th><th class="num">Turn</th><th class="num">单 turn reward</th><th>规模</th></tr></thead><tbody>{distance_rows}</tbody></table><p class="detail">最远回填 {_num(segment["max_backfill_distance"])} 个 turn；边界 reward 总和 {segment["boundary_process_reward_sum"]:.2f}，非边界回填总和 {segment["backfilled_process_reward_sum"]:.2f}。</p></div>
</section>

<section class="two">
<div class="panel"><h2>Milestone 修正前后：负 advantage turn</h2>
  <h3>正好完成 milestone 的 turn（共 {_num(impact["event_turns"])}）</h3>
  <div class="compare"><span class="compare-label">仅 A_global</span>{_bar(event_global_negative, impact["event_turns"], color="#d94f4f", label=f"{event_global_negative} negative")}<span class="compare-value">{_pct(event_global_negative / impact["event_turns"])}</span></div>
  <div class="compare"><span class="compare-label">+ A_local 后</span>{_bar(event_total_negative, impact["event_turns"], color="#238866", label=f"{event_total_negative} negative")}<span class="compare-value">{_pct(event_total_negative / impact["event_turns"])}</span></div>
  <h3 style="margin-top:18px">没有直接完成 milestone 的回填 turn（共 {_num(segment["backfilled_non_boundary_turns"])}）</h3>
  <div class="compare"><span class="compare-label">仅 A_global</span>{_bar(backfilled_global_negative, segment["backfilled_non_boundary_turns"], color="#d94f4f", label=f"{backfilled_global_negative} negative")}<span class="compare-value">{_pct(backfilled_global_negative / segment["backfilled_non_boundary_turns"])}</span></div>
  <div class="compare"><span class="compare-label">+ A_local 后</span>{_bar(backfilled_total_negative, segment["backfilled_non_boundary_turns"], color="#238866", label=f"{backfilled_total_negative} negative")}<span class="compare-value">{_pct(backfilled_total_negative / segment["backfilled_non_boundary_turns"])}</span></div>
</div>
<div class="panel"><h2>失败轨迹里的信号冲突</h2>
  <div class="metric-stack"><div class="metric-row"><span class="label">最终失败 trajectory</span><b>{_num(overview["overall_fail_trajectories"])}</b></div>
  <div class="metric-row"><span class="label">失败但曾完成任意 milestone</span><b class="risk">{_num(impact["failed_trajectories_with_any_milestone"])}</b></div>
  <div class="metric-row"><span class="label">在 final 前已经完成 milestone</span><b class="bad">{_num(impact["failed_trajectories_with_early_milestone"])}</b><span class="detail">这些轨迹同时存在“局部完成”与“最终失败”信号。</span></div></div>
  <p class="detail">Milestone 局部信号能抵消大部分反向信号，但它只是相加，不能修正 row-weighted outcome baseline。</p>
</div>
</section>

<section class="panel"><h2>fan-out 按终局 score 分桶</h2><table><thead><tr><th class="num">Score</th><th class="num">Trajectory</th><th class="num">非 final turn</th><th>规模</th><th class="num">重复 score 和</th></tr></thead><tbody>{bucket_rows}</tbody></table></section>

<section class="panel"><h2>排除“串奖励”</h2><div class="checks">
  <div class="check"><span class="label">Rewarded request ID</span><b>{_num(validation["rewarded_request_ids"])}</b><span>全部唯一</span></div>
  <div class="check"><span class="label">Judge completed</span><b>{_num(validation["judge_completed"])}</b><span>无 benchmark judge error</span></div>
  <div class="check"><span class="label">Verifier error</span><b>{_num(validation["milestone_verifier_errors"])}</b><span>不是 verifier 异常导致</span></div>
  <div class="check"><span class="label">Tool mapping mean</span><b>{_pct(validation["mean_tool_mapping_coverage"])}</b><span>{_num(validation["tool_mapping_coverage_below_one"])} 条低于 100%</span></div>
</div><p class="detail">request/reward/rollout 没有跨 trajectory 串线。本报告中的偏移来自同一 request 展开后的 turn 权重，以及 milestone segment 的反向衰减回填。</p></section>

<section class="panel"><h2>统计范围</h2><ul>{source_rows}</ul><p class="detail">只统计同时存在 reward 记录、并实际进入 train_step 的 rollout；中断或未产生 reward 的 rollout 已排除。统计单位为 model-call row，不按 token 加权。</p></section>
<footer>Generated {html.escape(summary["generated_at"])}. Embedded machine-readable summary is available below.</footer>
<script id="report-data" type="application/json">{payload}</script>
</main></body></html>"""


def render_png(summary: dict[str, Any], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    overview = summary["overview"]
    impact = summary["milestone_impact"]
    segment = summary["segment_attribution"]
    global_advantage = summary["global_advantage"]
    updates = summary["updates"]
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    figure.patch.set_facecolor("#f3f6fa")
    figure.suptitle(title, fontsize=20, fontweight="bold", color="#172033")

    axis = axes[0, 0]
    labels = [update["label"] for update in updates]
    final = [update["trajectories"] for update in updates]
    fanout = [update["nonfinal_rows"] for update in updates]
    axis.bar(labels, fanout, color="#d68b22", label="Non-final score fan-out")
    axis.bar(labels, final, bottom=fanout, color="#3274b9", label="Final rows")
    for index, update in enumerate(updates):
        axis.text(index, update["turn_rows"] + 5, _pct(update["wrong_rate"]), ha="center", fontsize=9)
    axis.set_title("Turn rows carrying terminal score by optimizer update")
    axis.set_ylabel("Model-call rows")
    axis.legend(frameon=False, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[0, 1]
    baseline_deltas = [abs(group["baseline_delta"]) for group in global_advantage["group_baselines"]]
    axis.hist(baseline_deltas, bins=12, color="#d94f4f", edgecolor="white")
    axis.axvline(
        global_advantage["mean_abs_baseline_delta"],
        color="#102b46",
        linestyle="--",
        linewidth=2,
        label=f"mean = {global_advantage['mean_abs_baseline_delta']:.4f}",
    )
    axis.set_title("Row-weighted vs trajectory-equal baseline")
    axis.set_xlabel("Absolute baseline delta per query group")
    axis.set_ylabel("Query groups")
    axis.legend(frameon=False)
    axis.text(
        0.98,
        0.92,
        f"{global_advantage['advantage_sign_changed_rows']:,} rows change advantage sign",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
    )
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[1, 0]
    categories = ["Milestone boundary", "Backfilled non-boundary", "Unfinished segment"]
    before = [
        impact["event_turn_global_sign"].get("negative", 0),
        segment["backfilled_global_sign"].get("negative", 0),
        segment["unshaped_global_sign"].get("negative", 0),
    ]
    after = [
        impact["event_turn_total_sign"].get("negative", 0),
        segment["backfilled_total_sign"].get("negative", 0),
        segment["unshaped_total_sign"].get("negative", 0),
    ]
    totals = [impact["event_turns"], segment["backfilled_non_boundary_turns"], segment["unshaped_turns"]]
    positions = range(len(categories))
    axis.barh(
        [position + 0.18 for position in positions], before, height=0.32, color="#d94f4f", label="Negative A_global"
    )
    axis.barh(
        [position - 0.18 for position in positions],
        after,
        height=0.32,
        color="#238866",
        label="Negative after +A_local",
    )
    for position, value, total in zip(positions, before, totals, strict=True):
        axis.text(value + 5, position + 0.18, f"{value} ({value / total:.1%})", va="center")
    for position, value, total in zip(positions, after, totals, strict=True):
        axis.text(value + 5, position - 0.18, f"{value} ({value / total:.1%})", va="center")
    axis.set_yticks(list(positions), categories)
    axis.set_title("Negative advantage before and after local shaping")
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right", "left"]].set_visible(False)

    axis = axes[1, 1]
    axis.axis("off")
    box = dict(boxstyle="round,pad=0.65", facecolor="white", edgecolor="#dce2ec")
    axis.text(0.02, 0.94, "Measured attribution risks", fontsize=15, fontweight="bold", va="top")
    axis.text(
        0.03,
        0.78,
        "8 trajectories\n(one query group)",
        bbox=box,
        ha="left",
        va="center",
    )
    axis.text(0.34, 0.78, "→", color="#d94f4f", fontsize=24, fontweight="bold", va="center")
    axis.text(0.43, 0.78, "different turn counts\nper trajectory", bbox=box, ha="left", va="center")
    axis.text(0.76, 0.78, "→", color="#d94f4f", fontsize=24, fontweight="bold", va="center")
    axis.text(0.84, 0.78, "row-weighted\ngroup baseline", bbox=box, ha="center", va="center")
    axis.text(
        0.03,
        0.48,
        f"{overview['nonfinal_rows']:,} / {overview['turn_rows']:,} rows receive score before final ({overview['terminal_score_fanout_rate']:.1%})\n"
        f"{global_advantage['groups_with_row_weighting_bias']:,} / {global_advantage['query_groups']:,} group baselines shift\n"
        f"{segment['backfilled_non_boundary_turns']:,} non-boundary turns receive decayed process reward",
        fontsize=13,
        linespacing=1.65,
        va="top",
        color="#172033",
    )
    axis.text(
        0.03,
        0.13,
        "Outcome comparison should be trajectory-equal before broadcasting advantage to turns.\n"
        "Segment backfill is a risk set: no independent per-turn semantic error label exists.",
        fontsize=10.5,
        color="#657087",
        va="bottom",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dirs", nargs="+", help="Reward-backed GRPO trace directories")
    parser.add_argument(
        "--output-dir",
        default="artifacts/turn_reward_misattribution",
        help="Directory for report.html, overview.png, and summary.json",
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--title", default="Turn Reward 问题简报 · Docs Milestone v4")
    parser.add_argument("--png-title", default="Turn Reward Attribution Audit | Docs Milestone v4")
    args = parser.parse_args()

    trace_dirs = [Path(value) for value in args.trace_dirs]
    trajectories, source_info = load_trajectories(trace_dirs)
    summary = build_summary(
        trajectories,
        gamma=args.gamma,
        local_weight=args.local_weight,
        source_info=source_info,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "report.html"
    details_path = output_dir / "details.html"
    png_path = output_dir / "overview.png"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_simple_html(summary, args.title), encoding="utf-8")
    details_path.write_text(render_html(summary, f"{args.title} · 详细附录"), encoding="utf-8")
    render_png(summary, png_path, args.png_title)
    print(
        f"Rendered {html_path}, {details_path}, {png_path}, and {summary_path} "
        f"from {len(trajectories)} rewarded trajectories"
    )


if __name__ == "__main__":
    main()
