"""Prompt and strict output schema for category-level skill extraction."""

from __future__ import annotations

import json

from .models import FamilyBlueprint, JsonObject, skill_name_for_family

SYSTEM_INSTRUCTIONS = """你是离线轨迹 Skill 提取器。目标是从经过独立证据校验的成功轨迹中总结可复用的策略 Skill，
并为任务中间状态补充 checklist、可接受证据、hard failures 和 LLM judge 语义标准。

必须遵守：
1. 中间状态描述任务语义是否达成，不描述某一种固定工具路径；等价工具只是同一状态的证据。
2. 只从 verified_success 学习策略。regression_negative 只用于写防回归规则，禁止模仿其动作路径。
3. 抽象具体文件名、账号、ID、链接、日期和业务值；不要把单个 case 写成通用规则。
4. 保留给定 state blueprint 的全部 state_id，且不得增加、删除或改名。依赖和确定性检查由代码维护。
5. 策略 Skill 与 verifier 语义分开：skill 讲如何行动，state_guidance/judge_rubric 讲如何验收。
6. 不把 benchmark judge 的 PASS 当成唯一依据；以提供的经过筛选的动作证据和任务结果为准。
7. 输出简洁、具体、可执行的中文。JSON 之外不要输出任何文本。
"""


def skill_package_schema() -> JsonObject:
    workflow_item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "instruction": {"type": "string"},
            "applies_when": {"type": "string"},
        },
        "required": ["id", "instruction", "applies_when"],
        "additionalProperties": False,
    }
    annotation = {
        "type": "object",
        "properties": {
            "state_id": {"type": "string"},
            "objective": {"type": "string"},
            "checklist": {"type": "array", "items": {"type": "string"}},
            "accepted_evidence": {"type": "array", "items": {"type": "string"}},
            "hard_failures": {"type": "array", "items": {"type": "string"}},
            "semantic_checks": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "state_id",
            "objective",
            "checklist",
            "accepted_evidence",
            "hard_failures",
            "semantic_checks",
        ],
        "additionalProperties": False,
    }
    criterion = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "question": {"type": "string"},
            "pass_condition": {"type": "string"},
            "fail_condition": {"type": "string"},
            "severity": {"type": "string", "enum": ["hard", "soft"]},
        },
        "required": ["id", "name", "question", "pass_condition", "fail_condition", "severity"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "skill": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "purpose": {"type": "string"},
                    "workflow": {"type": "array", "items": workflow_item},
                    "decision_rules": {"type": "array", "items": {"type": "string"}},
                    "stop_conditions": {"type": "array", "items": {"type": "string"}},
                    "failure_recovery": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "name",
                    "description",
                    "purpose",
                    "workflow",
                    "decision_rules",
                    "stop_conditions",
                    "failure_recovery",
                ],
                "additionalProperties": False,
            },
            "state_guidance": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "state_annotations": {"type": "array", "items": annotation},
                },
                "required": ["summary", "state_annotations"],
                "additionalProperties": False,
            },
            "judge_rubric": {
                "type": "object",
                "properties": {
                    "overview": {"type": "string"},
                    "pass_rule": {"type": "string"},
                    "criteria": {"type": "array", "items": criterion},
                    "hard_failures": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["overview", "pass_rule", "criteria", "hard_failures"],
                "additionalProperties": False,
            },
        },
        "required": ["skill", "state_guidance", "judge_rubric"],
        "additionalProperties": False,
    }


def build_family_prompt(
    *,
    blueprint: FamilyBlueprint,
    case_catalog: list[JsonObject],
    experience_cards: list[JsonObject],
) -> str:
    positives = [card for card in experience_cards if card.get("label") == "verified_success"]
    negatives = [card for card in experience_cards if card.get("label") == "regression_negative"]
    payload = {
        "task_family": blueprint.family,
        "required_skill_name": skill_name_for_family(blueprint.family),
        "state_blueprint": blueprint.to_dict(),
        "case_catalog": case_catalog,
        "verified_success_examples": positives,
        "regression_negative_examples": negatives,
        "generation_requirements": {
            "workflow": "总结跨 case 共性的最短可靠策略；复杂 case 可并行收集证据，但不要绑定单一工具名。",
            "state_annotations": "为 blueprint 中每个 state_id 恰好输出一项，顺序保持一致。",
            "judge": "同时检查动作结果、最终回复、未授权写入、事实证据和部分完成；hard failure 一票否决。",
            "low_data": "若成功样本不足，只写 blueprint 和 case 能支持的保守规则，不从失败路径反推成功操作。",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def validate_generated_package(generated: JsonObject, blueprint: FamilyBlueprint) -> None:
    skill = generated.get("skill") if isinstance(generated.get("skill"), dict) else {}
    expected_name = skill_name_for_family(blueprint.family)
    if skill.get("name") != expected_name:
        raise ValueError(f"generated skill name must be {expected_name!r}, found {skill.get('name')!r}")
    guidance = generated.get("state_guidance") if isinstance(generated.get("state_guidance"), dict) else {}
    annotations = guidance.get("state_annotations") if isinstance(guidance.get("state_annotations"), list) else []
    actual_ids = [str(annotation.get("state_id")) for annotation in annotations if isinstance(annotation, dict)]
    expected_ids = [state.id for state in blueprint.states]
    if actual_ids != expected_ids:
        raise ValueError(f"state annotations must be exactly {expected_ids}, found {actual_ids}")
    for annotation in annotations:
        state_id = str(annotation.get("state_id") or "")
        for field in ("checklist", "accepted_evidence", "hard_failures", "semantic_checks"):
            values = annotation.get(field)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"state annotation {state_id!r} must have a non-empty {field}")
    if not skill.get("workflow"):
        raise ValueError("generated skill workflow is empty")
    rubric = generated.get("judge_rubric") if isinstance(generated.get("judge_rubric"), dict) else {}
    if not rubric.get("criteria"):
        raise ValueError("generated judge rubric has no criteria")
