"""Milestone-guided GRPO advantage for expanded multi-turn ADK rollouts."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class MilestoneAdvantageResult:
    advantages: torch.Tensor
    returns: torch.Tensor
    process_rewards: torch.Tensor
    local_advantages: torch.Tensor
    trajectory_advantages: torch.Tensor
    completed_segments: int


_MILESTONE_REWARD_SCALE = 1.0


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _parse_state_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, np.ndarray):
        parsed = value.tolist()
    else:
        parsed = value
    if parsed is None:
        return ()
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"milestone state ids must be a JSON list, got {type(parsed)!r}")
    state_ids = tuple(str(item).strip() for item in parsed if str(item).strip())
    if len(set(state_ids)) != len(state_ids):
        raise ValueError(f"duplicate milestone state id in one turn: {state_ids}")
    return state_ids


def _group_trajectory_rows(
    prompt_ids: list[Any],
    trajectory_ids: list[str],
    turn_indices: list[int],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    rows_by_trajectory: dict[str, list[int]] = defaultdict(list)
    prompt_by_trajectory: dict[str, Any] = {}
    for row_index, (prompt_id, trajectory_id) in enumerate(zip(prompt_ids, trajectory_ids, strict=True)):
        if not trajectory_id:
            raise ValueError(f"row {row_index} has an empty milestone trajectory id")
        previous_prompt = prompt_by_trajectory.setdefault(trajectory_id, prompt_id)
        if previous_prompt != prompt_id:
            raise ValueError(
                f"trajectory {trajectory_id!r} appears under multiple prompt groups: {previous_prompt!r}, {prompt_id!r}"
            )
        rows_by_trajectory[trajectory_id].append(row_index)

    for trajectory_id, rows in rows_by_trajectory.items():
        rows.sort(key=lambda row: (turn_indices[row], row))
        ordered_turns = [turn_indices[row] for row in rows]
        if len(set(ordered_turns)) != len(ordered_turns):
            raise ValueError(f"trajectory {trajectory_id!r} contains duplicate turn indices: {ordered_turns}")
    return dict(rows_by_trajectory), prompt_by_trajectory


def _compute_baseline_outcome_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_ids: list[Any],
    *,
    normalize_by_std: bool,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact baseline-GRPO token and row outcome advantages."""

    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    token_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.asarray(prompt_ids, dtype=object),
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=normalize_by_std,
    )
    valid_tokens = response_mask.sum(dim=-1)
    row_advantages = token_advantages.sum(dim=-1) / valid_tokens.clamp_min(1)
    row_advantages = torch.where(valid_tokens > 0, row_advantages, torch.zeros_like(row_advantages))
    return token_advantages, row_advantages


def _build_boundary_rewards(
    prompt_by_trajectory: dict[str, Any],
    rows_by_trajectory: dict[str, list[int]],
    state_ids_by_row: list[tuple[str, ...]],
) -> tuple[dict[tuple[Any, int], list[tuple[int | None, float]]], torch.Tensor, int]:
    """Build one binary observation per trajectory and milestone position.

    Only the first-completion boundary row receives raw process reward. Missing
    milestone positions remain zero-valued comparison observations, but their
    trajectory turns are never assigned local credit or blame.
    """

    row_count = len(state_ids_by_row)
    process_rewards = torch.zeros(row_count, dtype=torch.float32)
    boundary_rows_by_trajectory: dict[str, list[int]] = {}
    for trajectory_id, rows in rows_by_trajectory.items():
        boundaries = [row_index for row_index in rows if state_ids_by_row[row_index]]
        boundary_rows_by_trajectory[trajectory_id] = boundaries
        process_rewards[boundaries] = _MILESTONE_REWARD_SCALE

    trajectories_by_prompt: dict[Any, list[str]] = defaultdict(list)
    for trajectory_id, prompt_id in prompt_by_trajectory.items():
        trajectories_by_prompt[prompt_id].append(trajectory_id)
    observations: dict[tuple[Any, int], list[tuple[int | None, float]]] = defaultdict(list)
    for prompt_id, trajectory_ids in trajectories_by_prompt.items():
        max_completed = max(
            (len(boundary_rows_by_trajectory[trajectory_id]) for trajectory_id in trajectory_ids),
            default=0,
        )
        for progress_index in range(max_completed):
            for trajectory_id in trajectory_ids:
                boundaries = boundary_rows_by_trajectory[trajectory_id]
                boundary_row = boundaries[progress_index] if progress_index < len(boundaries) else None
                observations[(prompt_id, progress_index)].append(
                    (boundary_row, _MILESTONE_REWARD_SCALE if boundary_row is not None else 0.0)
                )
    completed_segments = sum(len(rows) for rows in boundary_rows_by_trajectory.values())
    return dict(observations), process_rewards, completed_segments


def _compute_local_advantages(
    observations: dict[tuple[Any, int], list[tuple[int | None, float]]],
    process_rewards: torch.Tensor,
    *,
    normalize_by_std: bool,
    epsilon: float,
) -> torch.Tensor:
    """Compute completion advantage and apply it only to boundary turns."""

    local = torch.zeros_like(process_rewards)
    for values in observations.values():
        rewards = torch.tensor([reward for _, reward in values], dtype=torch.float32)
        baseline = rewards.mean()
        scale = rewards.std(unbiased=True) + epsilon if normalize_by_std and len(values) > 1 else 1.0
        for boundary_row, reward in values:
            if boundary_row is not None:
                local[boundary_row] = (reward - baseline) / scale
    return local


def compute_milestone_grpo_advantage(
    *,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_ids: Iterable[Any],
    trajectory_ids: Iterable[Any],
    turn_indices: Iterable[Any],
    state_ids_by_row: Iterable[Any],
    gamma: float = 0.95,
    step_advantage_weight: float = 1.0,
    normalize_global_by_std: bool = False,
    normalize_segment_by_std: bool = False,
    epsilon: float = 1e-6,
) -> MilestoneAdvantageResult:
    """Add boundary-only milestone local advantages to baseline GRPO.

    The outcome component is computed by the same row-level implementation as
    the non-milestone GRPO path. ``gamma`` remains accepted for configuration
    compatibility, but no within-segment decay or credit backfill is performed.
    """

    if token_level_rewards.ndim != 2 or response_mask.shape != token_level_rewards.shape:
        raise ValueError("token_level_rewards and response_mask must be aligned 2-D tensors")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"milestone gamma must be in [0, 1], got {gamma}")
    if step_advantage_weight < 0.0:
        raise ValueError(f"milestone step_advantage_weight must be non-negative, got {step_advantage_weight}")

    row_count = token_level_rewards.shape[0]
    prompt_values = _as_list(prompt_ids)
    trajectory_values = [str(value) for value in _as_list(trajectory_ids)]
    turn_values = [int(value) for value in _as_list(turn_indices)]
    state_values = [_parse_state_ids(value) for value in _as_list(state_ids_by_row)]
    lengths = {
        "prompts": len(prompt_values),
        "trajectories": len(trajectory_values),
        "turns": len(turn_values),
        "states": len(state_values),
    }
    if any(length != row_count for length in lengths.values()):
        raise ValueError(f"milestone annotations do not align with reward rows: rows={row_count}, {lengths}")

    rows_by_trajectory, prompt_by_trajectory = _group_trajectory_rows(
        prompt_values,
        trajectory_values,
        turn_values,
    )
    baseline_advantages, baseline_row_advantages = _compute_baseline_outcome_advantages(
        token_level_rewards,
        response_mask,
        prompt_values,
        normalize_by_std=normalize_global_by_std,
        epsilon=epsilon,
    )
    observations, process_rewards_cpu, completed_segments = _build_boundary_rewards(
        prompt_by_trajectory,
        rows_by_trajectory,
        state_values,
    )
    local_advantages_cpu = _compute_local_advantages(
        observations,
        process_rewards_cpu,
        normalize_by_std=normalize_segment_by_std,
        epsilon=epsilon,
    )

    device = token_level_rewards.device
    process_rewards = process_rewards_cpu.to(device=device)
    local_advantages = local_advantages_cpu.to(device=device)
    advantages = baseline_advantages + (float(step_advantage_weight) * local_advantages.unsqueeze(-1) * response_mask)
    return MilestoneAdvantageResult(
        advantages=advantages,
        returns=advantages,
        process_rewards=process_rewards,
        local_advantages=local_advantages,
        trajectory_advantages=baseline_row_advantages,
        completed_segments=completed_segments,
    )
