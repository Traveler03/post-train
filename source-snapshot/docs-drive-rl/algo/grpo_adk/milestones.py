"""Runtime milestone annotations from instantiated SkillBank state graphs."""

from __future__ import annotations

import fnmatch
import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from algo.offline_skills.authorization import EXTERNAL_MUTATION_TOOL_PATTERNS
from algo.offline_skills.evidence import extract_tool_events, normalize_tool_name
from algo.offline_skills.models import ToolEvent

from .milestone_policy import (
    AuthorizationContext,
    conversation_history,
    prepare_policy_aware_instance,
)

SUPPORTED_CHECKS = {"tool_called", "forbidden_tool_absent", "response_nonempty"}


@dataclass(frozen=True)
class MilestoneTrace:
    """Incremental verifier result for one ADK trajectory."""

    state_ids_by_turn: tuple[tuple[str, ...], ...]
    required_state_count: int
    shapeable_state_count: int
    completed_state_count: int
    tool_mapping_coverage: float
    verifier_version: str
    semantic_judge_status: str = "not_run"
    semantic_judge_model: str = ""
    semantic_judge_response_id: str = ""
    semantic_judge_duration_s: float = 0.0
    semantic_judge_attempts: int = 0
    semantic_judge_result_json: str = ""
    authorization_state: str = "not_required"
    authorization_source: str = "task_has_no_external_mutation"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


@lru_cache(maxsize=8)
def load_state_instances(skillbank_dir: str) -> dict[str, dict[str, Any]]:
    """Load instantiated case graphs keyed by the dataset ``iid``."""

    root = Path(skillbank_dir).expanduser().resolve()
    instance_dir = root / "case_instances"
    paths = sorted(instance_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no SkillBank case instances found under {instance_dir}")

    instances: dict[str, dict[str, Any]] = {}
    for path in paths:
        for instance in _iter_jsonl(path):
            iid = str(instance.get("iid") or "").strip()
            if not iid:
                raise ValueError(f"SkillBank instance in {path} has no iid")
            if iid in instances:
                raise ValueError(f"duplicate SkillBank iid {iid!r}")
            instances[iid] = instance
    return instances


def _response_message(record: dict[str, Any]) -> dict[str, Any]:
    response = _as_dict(record.get("response"))
    if not response:
        response = _as_dict(_as_dict(record.get("reward_record")).get("response"))
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    return _as_dict(choices[0].get("message"))


def _tool_call_name(call: dict[str, Any]) -> str:
    function = _as_dict(call.get("function"))
    name = normalize_tool_name(function.get("name"))
    raw_args = function.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    else:
        args = _as_dict(raw_args)
    if name in {"execute_tool", "proxy_tool"}:
        wrapped_name = args.get("tool_name") or args.get("name")
        if wrapped_name:
            return normalize_tool_name(wrapped_name)
    return name


def _tool_call_arguments(call: dict[str, Any]) -> Any:
    raw_args = _as_dict(call.get("function")).get("arguments")
    if not isinstance(raw_args, str):
        return raw_args if raw_args is not None else {}
    try:
        return json.loads(raw_args)
    except json.JSONDecodeError:
        return raw_args


def _target_tool_slots(relay_records: Iterable[Any]) -> list[tuple[int, str]]:
    slots: list[tuple[int, str]] = []
    for turn_index, raw_record in enumerate(relay_records):
        record = _as_dict(raw_record)
        message = _response_message(record)
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            name = _tool_call_name(raw_call)
            if name:
                slots.append((turn_index, name))
    return slots


def _is_delegation_tool(name: str) -> bool:
    return name in {"agent", "load_skill", "transfer_to_agent"} or name.endswith("_agent")


def map_tool_events_to_turns(
    events: Iterable[ToolEvent],
    relay_records: Iterable[Any],
) -> tuple[list[tuple[ToolEvent, int]], float]:
    """Align official tool evidence with the model action that caused it.

    The official trace stores top-level events and delegated sandbox events in
    separate lists, so their concatenated order is not the target model-call
    order. Match each tool name independently, then use those matches as
    anchors for the remaining events in the same delegated execution chain.
    """

    slots = _target_tool_slots(relay_records)
    slots_by_name: dict[str, deque[int]] = defaultdict(deque)
    delegation_turns: list[int] = []
    for turn_index, name in slots:
        slots_by_name[name].append(turn_index)
        if _is_delegation_tool(name):
            delegation_turns.append(turn_index)

    candidates = [event for event in events if event.name != "transfer_to_agent"]
    mapped_turns: dict[int, int] = {}
    for event_position, event in enumerate(candidates):
        matching_slots = slots_by_name[event.name]
        if matching_slots:
            mapped_turns[event_position] = matching_slots.popleft()

    positions_by_source: dict[str, list[int]] = defaultdict(list)
    for event_position, event in enumerate(candidates):
        if event.source != "actual_outcome.tool":
            positions_by_source[event.source].append(event_position)

    fallback_delegation_turn = delegation_turns[-1] if delegation_turns else None
    for positions in positions_by_source.values():
        anchors = [position for position in positions if position in mapped_turns]
        for position in positions:
            if position in mapped_turns:
                continue
            preceding = [anchor for anchor in anchors if anchor < position]
            following = [anchor for anchor in anchors if anchor > position]
            if preceding:
                mapped_turns[position] = mapped_turns[preceding[-1]]
            elif following:
                mapped_turns[position] = mapped_turns[following[0]]
            elif fallback_delegation_turn is not None:
                mapped_turns[position] = fallback_delegation_turn

    mapped = [(event, mapped_turns[position]) for position, event in enumerate(candidates) if position in mapped_turns]
    coverage = len(mapped) / len(candidates) if candidates else 1.0
    return mapped, coverage


def _event_matches(name: str, patterns: Iterable[Any]) -> bool:
    return any(fnmatch.fnmatchcase(name, str(pattern)) for pattern in patterns)


def _shapeable(node: dict[str, Any]) -> bool:
    checks = node.get("deterministic_checks") or []
    has_judge_rules = any(node.get(key) for key in ("checklist", "semantic_checks", "hard_failures"))
    return (bool(checks) or has_judge_rules) and all(
        str(check.get("kind") or "") in SUPPORTED_CHECKS for check in checks
    )


def _effective_dependencies(
    node_id: str,
    nodes_by_id: dict[str, dict[str, Any]],
    shapeable_ids: set[str],
    stack: tuple[str, ...] = (),
) -> set[str]:
    if node_id in stack:
        raise ValueError(f"cycle in SkillBank state graph: {' -> '.join((*stack, node_id))}")
    result: set[str] = set()
    node = nodes_by_id[node_id]
    for raw_dependency in node.get("depends_on") or []:
        dependency = str(raw_dependency)
        if dependency not in nodes_by_id:
            raise ValueError(f"state {node_id!r} depends on missing state {dependency!r}")
        if dependency in shapeable_ids:
            result.add(dependency)
        else:
            result.update(
                _effective_dependencies(
                    dependency,
                    nodes_by_id,
                    shapeable_ids,
                    (*stack, node_id),
                )
            )
    return result


def _checks_pass_at_turn(
    checks: Iterable[dict[str, Any]],
    mapped_events: list[tuple[ToolEvent, int]],
    *,
    dependency_boundary: int,
    turn_index: int,
    terminal_turn: int,
    final_answer: str,
) -> bool:
    for check in checks:
        kind = str(check.get("kind") or "")
        patterns = check.get("tool_patterns") or []
        if kind == "tool_called":
            count = sum(
                event.successful
                and dependency_boundary <= event_turn <= turn_index
                and _event_matches(event.name, patterns)
                for event, event_turn in mapped_events
            )
            if count < int(check.get("min_count") or 1):
                return False
        elif kind == "forbidden_tool_absent":
            if turn_index != terminal_turn:
                return False
            if any(event.successful and _event_matches(event.name, patterns) for event, _ in mapped_events):
                return False
        elif kind == "response_nonempty":
            if turn_index != terminal_turn or not final_answer.strip():
                return False
        else:
            return False
    return True


def verify_trajectory_milestones(
    instance: dict[str, Any],
    extra_info: dict[str, Any],
    *,
    approved_state_turns: dict[str, int] | None = None,
    authorization_context: AuthorizationContext | None = None,
) -> MilestoneTrace:
    """Find the first target-agent turn that completes each verifiable state."""

    turn_count = int(extra_info.get("adk_model_calls") or 0)
    relay_records = extra_info.get("adk_relay_records") or []
    if turn_count <= 0:
        turn_count = len(relay_records)
    if turn_count <= 0:
        raise ValueError("ADK trajectory has no model turns")

    actual_outcome = _as_dict(extra_info.get("adk_trace_actual_outcome"))
    events = extract_tool_events(actual_outcome)
    mapped_events, mapping_coverage = map_tool_events_to_turns(events, relay_records)
    final_value = actual_outcome.get("respond", extra_info.get("adk_actual_outcome", ""))
    final_answer = final_value if isinstance(final_value, str) else json.dumps(final_value, ensure_ascii=False)

    nodes = [node for node in instance.get("state_graph") or [] if bool(node.get("required"))]
    nodes_by_id = {str(node.get("id")): node for node in nodes}
    if len(nodes_by_id) != len(nodes):
        raise ValueError(f"duplicate or empty state id in SkillBank instance {instance.get('iid')!r}")
    shapeable_ids = {node_id for node_id, node in nodes_by_id.items() if _shapeable(node)}
    dependencies = {node_id: _effective_dependencies(node_id, nodes_by_id, shapeable_ids) for node_id in shapeable_ids}
    if approved_state_turns is not None:
        unknown_ids = set(approved_state_turns) - shapeable_ids
        if unknown_ids:
            raise ValueError(f"semantic judge approved unknown or unshapeable states: {sorted(unknown_ids)}")
        invalid_turns = {
            node_id: turn
            for node_id, turn in approved_state_turns.items()
            if not isinstance(turn, int) or not 0 <= turn < turn_count
        }
        if invalid_turns:
            raise ValueError(f"semantic judge approved invalid milestone turns: {invalid_turns}")

    completed_at: dict[str, int] = {}
    state_ids_by_turn: list[list[str]] = [[] for _ in range(turn_count)]
    terminal_turn = turn_count - 1
    authorization_violation_turns = {
        event_turn
        for event, event_turn in mapped_events
        if authorization_context is not None
        and authorization_context.authorization_state
        not in {"confirmed", "preauthorized", "not_required"}
        and _event_matches(event.name, EXTERNAL_MUTATION_TOOL_PATTERNS)
    }
    for turn_index in range(turn_count):
        if turn_index in authorization_violation_turns:
            continue
        changed = True
        while changed:
            changed = False
            for node in nodes:
                node_id = str(node["id"])
                if node_id not in shapeable_ids or node_id in completed_at:
                    continue
                if approved_state_turns is not None:
                    approved_turn = approved_state_turns.get(node_id)
                    if approved_turn is None or turn_index < approved_turn:
                        continue
                required_dependencies = dependencies[node_id]
                if not required_dependencies.issubset(completed_at):
                    continue
                dependency_boundary = max(
                    (completed_at[dependency] for dependency in required_dependencies),
                    default=0,
                )
                if not _checks_pass_at_turn(
                    node.get("deterministic_checks") or [],
                    mapped_events,
                    dependency_boundary=dependency_boundary,
                    turn_index=turn_index,
                    terminal_turn=terminal_turn,
                    final_answer=final_answer,
                ):
                    continue
                completed_at[node_id] = turn_index
                state_ids_by_turn[turn_index].append(node_id)
                changed = True

    return MilestoneTrace(
        state_ids_by_turn=tuple(tuple(values) for values in state_ids_by_turn),
        required_state_count=len(nodes),
        shapeable_state_count=len(shapeable_ids),
        completed_state_count=len(completed_at),
        tool_mapping_coverage=mapping_coverage,
        verifier_version=str(instance.get("verifier_version") or ""),
        authorization_state=(authorization_context.authorization_state if authorization_context else "not_required"),
        authorization_source=(
            authorization_context.source if authorization_context else "task_has_no_external_mutation"
        ),
    )


def _build_judge_evidence(
    extra_info: dict[str, Any],
    authorization_context: AuthorizationContext,
) -> dict[str, Any]:
    turn_count = int(extra_info.get("adk_model_calls") or 0)
    relay_records = extra_info.get("adk_relay_records") or []
    if turn_count <= 0:
        turn_count = len(relay_records)
    actual_outcome = _as_dict(extra_info.get("adk_trace_actual_outcome"))
    events = extract_tool_events(actual_outcome)
    mapped_events, mapping_coverage = map_tool_events_to_turns(events, relay_records)
    events_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    mapped_event_ids: set[int] = set()
    for event, turn_index in mapped_events:
        events_by_turn[turn_index].append(event.to_dict())
        mapped_event_ids.add(id(event))

    turns: list[dict[str, Any]] = []
    for turn_index in range(turn_count):
        record = _as_dict(relay_records[turn_index]) if turn_index < len(relay_records) else {}
        message = _response_message(record)
        tool_calls = []
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            tool_calls.append(
                {
                    "name": _tool_call_name(raw_call),
                    "arguments": _tool_call_arguments(raw_call),
                }
            )
        turns.append(
            {
                "turn_index": turn_index,
                "assistant_content": message.get("content") or "",
                "assistant_tool_calls": tool_calls,
                "tool_results": events_by_turn.get(turn_index, []),
            }
        )

    final_value = actual_outcome.get("respond", extra_info.get("adk_actual_outcome", ""))
    final_answer = final_value if isinstance(final_value, str) else json.dumps(final_value, ensure_ascii=False)
    result = {
        "question": extra_info.get("adk_user_query_with_msg_time") or extra_info.get("question") or "",
        "authorization_context": authorization_context.to_dict(),
        "conversation_history": conversation_history(extra_info),
        "active_account": extra_info.get("adk_account") or extra_info.get("account") or "",
        "seed_summary": extra_info.get("adk_seed_summary") or "",
        "turns": turns,
        "unmapped_tool_results": [event.to_dict() for event in events if id(event) not in mapped_event_ids],
        "tool_mapping_coverage": mapping_coverage,
        "final_answer": final_answer,
    }
    if turns:
        result["terminal_turn_index"] = int(turns[-1]["turn_index"])
        turns[-1]["final_answer"] = final_answer
    return result


def _judge_trajectory(
    semantic_judge: Any,
    instance: dict[str, Any],
    info: dict[str, Any],
    authorization_context: AuthorizationContext,
) -> MilestoneTrace:
    required_nodes = [node for node in instance.get("state_graph") or [] if bool(node.get("required"))]
    missing_checklists = sorted(
        str(node.get("id") or "<empty>")
        for node in required_nodes
        if not isinstance(node.get("checklist"), list) or not node.get("checklist")
    )
    if missing_checklists:
        raise ValueError(f"required milestone states have no checklist: {missing_checklists}")
    state_ids = [str(node["id"]) for node in required_nodes if _shapeable(node)]
    unshapeable_ids = sorted(str(node.get("id") or "<empty>") for node in required_nodes if not _shapeable(node))
    if unshapeable_ids:
        raise ValueError(f"required milestone states cannot be verified: {unshapeable_ids}")
    if not state_ids:
        return replace(
            verify_trajectory_milestones(
                instance,
                info,
                authorization_context=authorization_context,
            ),
            semantic_judge_status="skipped_no_shapeable_states",
        )
    outcome = semantic_judge.judge(
        instance=instance,
        trajectory_evidence=_build_judge_evidence(info, authorization_context),
        state_ids=state_ids,
    )
    trace = verify_trajectory_milestones(
        instance,
        info,
        approved_state_turns=outcome.approved_state_turns,
        authorization_context=authorization_context,
    )
    return replace(
        trace,
        semantic_judge_status="completed",
        semantic_judge_model=outcome.model,
        semantic_judge_response_id=outcome.response_id,
        semantic_judge_duration_s=outcome.duration_s,
        semantic_judge_attempts=outcome.attempts,
        semantic_judge_result_json=outcome.result_json,
    )


def annotate_milestone_rows(
    extra_infos: Iterable[Any],
    *,
    skillbank_dir: str,
    strict: bool = True,
    semantic_judge: Any | None = None,
    judge_concurrency: int = 1,
) -> list[dict[str, Any]]:
    """Return one stable milestone annotation for every expanded model-call row."""

    infos = [_as_dict(value) for value in extra_infos]
    instances = load_state_instances(skillbank_dir)
    traces: dict[str, MilestoneTrace] = {}
    errors: dict[str, str] = {}
    jobs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    for row_index, info in enumerate(infos):
        trajectory_id = str(info.get("adk_request_id") or "").strip()
        if not trajectory_id:
            message = f"row {row_index} has no adk_request_id"
            if strict:
                raise ValueError(message)
            errors[f"__row_{row_index}"] = message
            continue
        if trajectory_id in jobs or trajectory_id in errors:
            continue
        iid = str(info.get("iid") or "").strip()
        instance = instances.get(iid)
        if instance is None:
            message = f"no SkillBank state instance for iid {iid!r}"
            if strict:
                raise KeyError(message)
            errors[trajectory_id] = message
            continue
        jobs[trajectory_id] = (instance, info)

    def record_result(trajectory_id: str, instance: dict[str, Any], info: dict[str, Any]) -> None:
        try:
            active_instance, authorization_context = prepare_policy_aware_instance(instance, info)
            if semantic_judge is None:
                traces[trajectory_id] = verify_trajectory_milestones(
                    active_instance,
                    info,
                    authorization_context=authorization_context,
                )
            else:
                traces[trajectory_id] = _judge_trajectory(
                    semantic_judge,
                    active_instance,
                    info,
                    authorization_context,
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors[trajectory_id] = message

    if semantic_judge is not None and len(jobs) > 1:
        max_workers = max(1, min(len(jobs), judge_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(record_result, trajectory_id, instance, info): trajectory_id
                for trajectory_id, (instance, info) in jobs.items()
            }
            for future in as_completed(futures):
                future.result()
    else:
        for trajectory_id, (instance, info) in jobs.items():
            record_result(trajectory_id, instance, info)

    if strict and errors:
        examples = "; ".join(f"{trajectory_id}: {error}" for trajectory_id, error in list(errors.items())[:3])
        raise RuntimeError(f"milestone verification failed for {len(errors)}/{len(jobs)} trajectories: {examples}")

    annotations: list[dict[str, Any]] = []
    for row_index, info in enumerate(infos):
        trajectory_id = str(info.get("adk_request_id") or "").strip()
        turn_index = int(info.get("adk_turn_index") or 0)
        trace = traces.get(trajectory_id)
        error = errors.get(trajectory_id) or errors.get(f"__row_{row_index}") or ""
        if trace is None:
            state_ids: tuple[str, ...] = ()
            required_count = 0
            shapeable_count = 0
            completed_count = 0
            mapping_coverage = 0.0
            verifier_version = ""
            judge_status = "error" if error and semantic_judge is not None else "not_run"
            judge_model = ""
            judge_response_id = ""
            judge_duration_s = 0.0
            judge_attempts = 0
            judge_result_json = ""
            authorization_state = "error" if error else "unknown"
            authorization_source = ""
        else:
            if turn_index < 0 or turn_index >= len(trace.state_ids_by_turn):
                raise ValueError(
                    f"turn index {turn_index} is outside trajectory {trajectory_id} "
                    f"with {len(trace.state_ids_by_turn)} turns"
                )
            state_ids = trace.state_ids_by_turn[turn_index]
            required_count = trace.required_state_count
            shapeable_count = trace.shapeable_state_count
            completed_count = trace.completed_state_count
            mapping_coverage = trace.tool_mapping_coverage
            verifier_version = trace.verifier_version
            judge_status = trace.semantic_judge_status
            judge_model = trace.semantic_judge_model
            judge_response_id = trace.semantic_judge_response_id
            judge_duration_s = trace.semantic_judge_duration_s
            judge_attempts = trace.semantic_judge_attempts
            judge_result_json = trace.semantic_judge_result_json
            authorization_state = trace.authorization_state
            authorization_source = trace.authorization_source
        annotations.append(
            {
                "milestone_trajectory_id": trajectory_id,
                "milestone_turn_index": turn_index,
                "milestone_state_ids_json": json.dumps(state_ids, ensure_ascii=True, separators=(",", ":")),
                "milestone_event_count": len(state_ids),
                "milestone_required_state_count": required_count,
                "milestone_shapeable_state_count": shapeable_count,
                "milestone_completed_state_count": completed_count,
                "milestone_state_coverage": (completed_count / shapeable_count if shapeable_count else 0.0),
                "milestone_tool_mapping_coverage": mapping_coverage,
                "milestone_verifier_version": verifier_version,
                "milestone_verifier_error": error,
                "milestone_judge_status": judge_status,
                "milestone_judge_model": judge_model,
                "milestone_judge_response_id": judge_response_id,
                "milestone_judge_duration_s": judge_duration_s,
                "milestone_judge_attempts": judge_attempts,
                "milestone_judge_result_json": judge_result_json,
                "milestone_authorization_state": authorization_state,
                "milestone_authorization_source": authorization_source,
            }
        )
    return annotations
