import json
from types import SimpleNamespace

import pytest

from algo.grpo_adk.milestone_judge import ChecklistMilestoneJudge


def _instance():
    return {
        "iid": "benchmark-docs:case-1",
        "family": "docs.read",
        "query": "Read the requested document",
        "bindings": {"requested_resource": "Plan"},
        "completion_rule": "All required states pass.",
        "state_graph": [
            {
                "id": "inspect",
                "required": True,
                "depends_on": [],
                "objective": "Read the correct document.",
                "applies_when": "always",
                "checklist": ["The target identity matches.", "The requested content is present."],
                "semantic_checks": ["The evidence supports the answer."],
                "hard_failures": ["The wrong document was read."],
                "deterministic_checks": [
                    {
                        "kind": "tool_called",
                        "tool_patterns": ["google_docs_cat"],
                        "min_count": 1,
                    }
                ],
            }
        ],
    }


def _evidence():
    return {
        "turns": [
            {
                "turn_index": 0,
                "assistant_content": "",
                "assistant_tool_calls": [{"name": "google_docs_cat", "arguments": {"name": "Plan"}}],
                "tool_results": [{"name": "google_docs_cat", "response": {"title": "Plan", "text": "ok"}}],
            }
        ],
        "final_answer": "The document says ok.",
    }


def _completed_response(*, failed_checklist_indices=None):
    return {
        "state_results": [
            {
                "state_id": "inspect",
                "status": "PASS",
                "completed_turn": 0,
                "failed_checklist_indices": failed_checklist_indices or [],
                "failed_semantic_check_indices": [],
                "triggered_hard_failure_indices": [],
                "evidence": ["Turn 0 returned title Plan and requested content."],
                "reason": "The complete checklist is supported.",
            }
        ],
        "overall_reason": "The state is complete.",
    }


def _skillbank(tmp_path):
    rubric_dir = tmp_path / "skills" / "docs-read"
    rubric_dir.mkdir(parents=True)
    (rubric_dir / "judge_rubric.md").write_text("# Strict docs.read rubric\n", encoding="utf-8")
    return tmp_path


def test_checklist_judge_sends_instantiated_rules_and_approves_valid_pass(tmp_path):
    calls = []

    class FakeClient:
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=_completed_response(),
                response_id="response-1",
                model="judge-model",
                attempts=1,
            )

    judge = ChecklistMilestoneJudge(
        skillbank_dir=_skillbank(tmp_path),
        client=FakeClient(),
        validation_attempts=1,
    )
    outcome = judge.judge(
        instance=_instance(),
        trajectory_evidence=_evidence(),
        state_ids=["inspect"],
    )

    assert outcome.approved_state_turns == {"inspect": 0}
    payload = json.loads(calls[0]["prompt"])
    assert payload["family_judge_rubrics"]["docs.read"].startswith("# Strict docs.read rubric")
    assert payload["state_nodes"][0]["checklist"] == _instance()["state_graph"][0]["checklist"]
    assert calls[0]["schema"]["properties"]["state_results"]["minItems"] == 1
    completion_schema = calls[0]["schema"]["properties"]["state_results"]["items"]["properties"][
        "completed_turn"
    ]
    assert completion_schema["anyOf"][0] == {"type": "integer", "minimum": 0, "maximum": 0}


def test_checklist_judge_rejects_internally_inconsistent_pass(tmp_path):
    class FakeClient:
        def generate_json(self, **_kwargs):
            return SimpleNamespace(
                content=_completed_response(failed_checklist_indices=[0]),
                response_id="response-2",
                model="judge-model",
                attempts=1,
            )

    judge = ChecklistMilestoneJudge(
        skillbank_dir=_skillbank(tmp_path),
        client=FakeClient(),
        validation_attempts=1,
    )

    with pytest.raises(RuntimeError, match="PASS with failed criteria"):
        judge.judge(
            instance=_instance(),
            trajectory_evidence=_evidence(),
            state_ids=["inspect"],
        )


def test_checklist_judge_rejects_state_without_checklist(tmp_path):
    instance = _instance()
    instance["state_graph"][0]["checklist"] = []
    judge = ChecklistMilestoneJudge(
        skillbank_dir=_skillbank(tmp_path),
        client=object(),
        validation_attempts=1,
    )

    with pytest.raises(ValueError, match="states have no checklist"):
        judge.judge(
            instance=instance,
            trajectory_evidence=_evidence(),
            state_ids=["inspect"],
        )


def test_checklist_judge_loads_every_component_skill_rubric(tmp_path):
    calls = []

    class FakeClient:
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=_completed_response(),
                response_id="response-components",
                model="judge-model",
                attempts=1,
            )

    skillbank = _skillbank(tmp_path)
    drive_rubric_dir = skillbank / "skills" / "drive-retrieve"
    drive_rubric_dir.mkdir(parents=True)
    (drive_rubric_dir / "judge_rubric.md").write_text("# Strict drive.retrieve rubric\n", encoding="utf-8")
    instance = _instance()
    instance["component_families"] = ["docs.read", "drive.retrieve"]
    judge = ChecklistMilestoneJudge(
        skillbank_dir=skillbank,
        client=FakeClient(),
        validation_attempts=1,
    )

    judge.judge(instance=instance, trajectory_evidence=_evidence(), state_ids=["inspect"])

    payload = json.loads(calls[0]["prompt"])
    assert list(payload["family_judge_rubrics"]) == ["docs.read", "drive.retrieve"]


def test_checklist_judge_prunes_dependency_violation_after_validation_retry(tmp_path, caplog):
    calls = []

    class FakeClient:
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content={
                    "state_results": [
                        {
                            "state_id": "inspect",
                            "status": "FAIL",
                            "completed_turn": None,
                            "failed_checklist_indices": [0],
                            "failed_semantic_check_indices": [],
                            "triggered_hard_failure_indices": [],
                            "evidence": [],
                            "reason": "The source was not inspected.",
                        },
                        {
                            "state_id": "deliver",
                            "status": "PASS",
                            "completed_turn": 0,
                            "failed_checklist_indices": [],
                            "failed_semantic_check_indices": [],
                            "triggered_hard_failure_indices": [],
                            "evidence": ["Turn 0 claimed delivery."],
                            "reason": "The response claimed completion.",
                        },
                    ],
                    "overall_reason": "Delivery was claimed without inspection.",
                },
                response_id=f"response-{len(calls)}",
                model="judge-model",
                attempts=1,
            )

    instance = _instance()
    instance["state_graph"].append(
        {
            "id": "deliver",
            "required": True,
            "depends_on": ["inspect"],
            "objective": "Deliver the grounded result.",
            "applies_when": "always",
            "checklist": ["The grounded result was delivered."],
            "semantic_checks": [],
            "hard_failures": [],
            "deterministic_checks": [],
        }
    )
    judge = ChecklistMilestoneJudge(
        skillbank_dir=_skillbank(tmp_path),
        client=FakeClient(),
        validation_attempts=2,
    )

    with caplog.at_level("WARNING"):
        outcome = judge.judge(
            instance=instance,
            trajectory_evidence=_evidence(),
            state_ids=["inspect", "deliver"],
        )

    assert len(calls) == 2
    assert "Correct this error" in calls[1]["prompt"]
    assert outcome.approved_state_turns == {}
    assert "pruned 1 dependency-inconsistent milestone approvals" in caplog.text
