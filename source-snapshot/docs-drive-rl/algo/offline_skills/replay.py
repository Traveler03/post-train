"""Deterministic replay of saved rollouts against instantiated state graphs."""

from __future__ import annotations

import fnmatch
from collections import Counter, defaultdict

from .authorization import infer_instance_authorization_policy, materialize_authorization_graph
from .models import JsonObject, RolloutRecord


def _event_matches(event_name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(event_name, pattern) for pattern in patterns)


def _check_state(checks: list[JsonObject], rollout: RolloutRecord) -> tuple[bool, list[JsonObject]]:
    details: list[JsonObject] = []
    for check in checks:
        kind = str(check.get("kind") or "")
        patterns = [str(pattern) for pattern in check.get("tool_patterns") or []]
        if kind == "tool_called":
            count = sum(event.successful and _event_matches(event.name, patterns) for event in rollout.events)
            passed = count >= int(check.get("min_count") or 1)
            details.append({"kind": kind, "passed": passed, "match_count": count})
        elif kind == "forbidden_tool_absent":
            matched = [
                event.name for event in rollout.events if event.successful and _event_matches(event.name, patterns)
            ]
            passed = not matched
            details.append({"kind": kind, "passed": passed, "matched_tools": matched})
        elif kind == "response_nonempty":
            passed = bool(rollout.answer.strip())
            details.append({"kind": kind, "passed": passed})
        else:
            passed = False
            details.append({"kind": kind, "passed": False, "error": "unsupported deterministic check"})
        if not passed:
            return False, details
    return True, details


def replay_rollout(instance: JsonObject, rollout: RolloutRecord) -> JsonObject:
    authorization_policy = infer_instance_authorization_policy(instance)
    authorization_state = "confirmation_required" if authorization_policy is not None else "not_required"
    instance = materialize_authorization_graph(instance, authorization_state)
    nodes = instance.get("state_graph") or []
    results: dict[str, JsonObject] = {}
    required_by_id = {str(node["id"]): bool(node.get("required")) for node in nodes}
    for node in nodes:
        node_id = str(node["id"])
        checks_pass, check_details = _check_state(node.get("deterministic_checks") or [], rollout)
        required_dependencies = [
            dependency for dependency in node.get("depends_on") or [] if required_by_id.get(str(dependency), False)
        ]
        dependencies_pass = all(results.get(str(dependency), {}).get("passed") for dependency in required_dependencies)
        results[node_id] = {
            "passed": bool(checks_pass and dependencies_pass),
            "required": bool(node.get("required")),
            "dependencies_pass": dependencies_pass,
            "checks": check_details,
        }
    required_nodes = [node_id for node_id, value in results.items() if value["required"]]
    passed_required = sum(results[node_id]["passed"] for node_id in required_nodes)
    predicted_pass = bool(required_nodes) and passed_required == len(required_nodes)
    return {
        "case_id": instance["case_id"],
        "family": instance["family"],
        "request_id": rollout.request_id,
        "trace_id": rollout.trace_id,
        "benchmark_pass": rollout.benchmark_pass,
        "benchmark_score": rollout.score,
        "deterministic_pass": predicted_pass,
        "required_state_coverage": passed_required / len(required_nodes) if required_nodes else 0.0,
        "states": results,
        "semantic_judge_status": "not_run",
        "authorization_state": authorization_state,
    }


def build_validation_report(results: list[JsonObject], *, cases: list[JsonObject]) -> JsonObject:
    rollouts_per_case = Counter(str(result["case_id"]) for result in results)

    def summarize(rows: list[JsonObject]) -> JsonObject:
        total = len(rows)
        benchmark_positive = sum(bool(row["benchmark_pass"]) for row in rows)
        deterministic_positive = sum(bool(row["deterministic_pass"]) for row in rows)
        agreement = sum(bool(row["benchmark_pass"]) == bool(row["deterministic_pass"]) for row in rows)
        true_positive = sum(bool(row["benchmark_pass"]) and bool(row["deterministic_pass"]) for row in rows)
        return {
            "rollouts": total,
            "benchmark_pass": benchmark_positive,
            "deterministic_pass": deterministic_positive,
            "label_agreement_rate": agreement / total if total else 0.0,
            "benchmark_pass_coverage": true_positive / benchmark_positive if benchmark_positive else None,
            "mean_required_state_coverage": (
                sum(float(row["required_state_coverage"]) for row in rows) / total if total else 0.0
            ),
        }

    by_domain: dict[str, list[JsonObject]] = defaultdict(list)
    by_family: dict[str, list[JsonObject]] = defaultdict(list)
    case_domain = {str(case["case_id"]): str(case["domain"]) for case in cases}
    for result in results:
        by_domain[case_domain[str(result["case_id"])]].append(result)
        by_family[str(result["family"])].append(result)

    count_distribution = Counter(rollouts_per_case.values())
    return {
        "schema": "migoo-task-state-replay-report-v1",
        "replay_mode": "deterministic_checks_only",
        "semantic_judge": {
            "status": "not_run",
            "reason": (
                "The generated LLM rubric is saved for online/version-pinned judging; offline replay uses "
                "existing benchmark judge labels."
            ),
        },
        "overall": summarize(results),
        "by_domain": {domain: summarize(rows) for domain, rows in sorted(by_domain.items())},
        "by_family": {family: summarize(rows) for family, rows in sorted(by_family.items())},
        "rollouts_per_case_distribution": {
            str(count): case_count for count, case_count in sorted(count_distribution.items())
        },
        "cases_not_replayed_eight_times": [
            {"case_id": case_id, "rollouts": count}
            for case_id, count in sorted(rollouts_per_case.items())
            if count != 8
        ],
        "result_count": len(results),
    }
