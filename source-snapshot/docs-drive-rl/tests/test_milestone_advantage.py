import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from algo.grpo_adk.milestone_advantage import compute_milestone_grpo_advantage
from algo.grpo_adk.milestone_judge import MilestoneJudgeOutcome
from algo.grpo_adk.milestones import annotate_milestone_rows, map_tool_events_to_turns
from algo.offline_skills.models import ToolEvent
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_grpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage


def _terminal_rewards(values):
    rewards = torch.zeros(len(values), 2, dtype=torch.float32)
    rewards[:, -1] = torch.tensor(values, dtype=torch.float32)
    return rewards


def test_milestone_advantage_rewards_only_completion_boundary():
    result = compute_milestone_grpo_advantage(
        token_level_rewards=_terminal_rewards([0.0, 1.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 2),
        prompt_ids=["query"] * 4,
        trajectory_ids=["a", "b", "a", "b"],
        turn_indices=[1, 1, 0, 0],
        state_ids_by_row=['["m1"]', "[]", "[]", '["m1"]'],
        gamma=0.5,
        step_advantage_weight=1.0,
    )

    # Both trajectories complete m1, so its group-relative local advantage is
    # zero. Earlier turns receive neither decayed credit nor local blame.
    assert result.process_rewards.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
    assert result.local_advantages.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert result.trajectory_advantages.tolist() == pytest.approx([-0.5, 0.5, -0.5, 0.5])
    assert result.advantages[:, 0].tolist() == pytest.approx([-0.5, 0.5, -0.5, 0.5])
    assert result.completed_segments == 2


def test_incomplete_trajectory_is_baseline_only_and_does_not_receive_local_blame():
    result = compute_milestone_grpo_advantage(
        token_level_rewards=_terminal_rewards([0.0, 0.0, 0.0, 0.0]),
        response_mask=torch.ones(4, 2),
        prompt_ids=["query"] * 4,
        trajectory_ids=["failed", "failed", "success", "success"],
        turn_indices=[0, 1, 0, 1],
        state_ids_by_row=["[]", "[]", "[]", '["m1"]'],
        gamma=0.5,
    )

    assert result.process_rewards.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert result.local_advantages.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.5])
    assert result.completed_segments == 1


def test_milestone_advantage_does_not_shield_failed_prefix():
    result = compute_milestone_grpo_advantage(
        token_level_rewards=_terminal_rewards([0.0, 0.0, 1.0, 1.0]),
        response_mask=torch.ones(4, 2),
        prompt_ids=["query"] * 4,
        trajectory_ids=["failed", "failed", "success", "success"],
        turn_indices=[0, 1, 0, 1],
        state_ids_by_row=["[]", '["m1"]', "[]", '["m1"]'],
        gamma=0.5,
    )

    # Failure-prefix protection is intentionally absent. Both failed turns keep
    # the same negative trajectory-level advantage, including the completed prefix.
    assert result.trajectory_advantages.tolist() == pytest.approx([-0.5, -0.5, 0.5, 0.5])


def test_global_advantage_matches_baseline_turn_weighting():
    result = compute_milestone_grpo_advantage(
        token_level_rewards=_terminal_rewards([0.0, 0.0, 0.0, 1.0]),
        response_mask=torch.ones(4, 2),
        prompt_ids=["query"] * 4,
        trajectory_ids=["long", "long", "long", "short"],
        turn_indices=[0, 1, 2, 0],
        state_ids_by_row=["[]"] * 4,
    )

    assert result.trajectory_advantages.tolist() == pytest.approx([-0.25, -0.25, -0.25, 0.75])
    assert result.local_advantages.tolist() == pytest.approx([0.0] * 4)


@pytest.mark.parametrize("normalize_by_std", [False, True])
def test_zero_local_weight_exactly_matches_baseline_grpo(normalize_by_std):
    token_level_rewards = _terminal_rewards([0.0, 0.0, 0.0, 1.0])
    response_mask = torch.ones(4, 2)
    prompt_ids = np.array(["query"] * 4, dtype=object)
    expected, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=prompt_ids,
        norm_adv_by_std_in_grpo=normalize_by_std,
    )

    result = compute_milestone_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        prompt_ids=prompt_ids,
        trajectory_ids=["long", "long", "long", "short"],
        turn_indices=[0, 1, 2, 0],
        state_ids_by_row=["[]", '["m1"]', "[]", '["m1"]'],
        step_advantage_weight=0.0,
        normalize_global_by_std=normalize_by_std,
    )

    torch.testing.assert_close(result.advantages, expected)
    torch.testing.assert_close(result.returns, expected)


def test_state_labels_at_same_progress_index_share_completion_group():
    result = compute_milestone_grpo_advantage(
        token_level_rewards=_terminal_rewards([1.0, 1.0, 1.0]),
        response_mask=torch.ones(3, 2),
        prompt_ids=["query"] * 3,
        trajectory_ids=["left-path", "left-path", "right-path"],
        turn_indices=[0, 1, 0],
        state_ids_by_row=["[]", '["left"]', '["right"]'],
        gamma=0.5,
    )

    assert result.process_rewards.tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert result.local_advantages.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert result.completed_segments == 2


def test_multiple_state_ids_on_one_turn_produce_one_boundary_reward():
    result = compute_milestone_grpo_advantage(
        token_level_rewards=_terminal_rewards([1.0, 0.0]),
        response_mask=torch.ones(2, 2),
        prompt_ids=["query", "query"],
        trajectory_ids=["completed", "incomplete"],
        turn_indices=[0, 0],
        state_ids_by_row=['["resolve", "inspect"]', "[]"],
    )

    assert result.process_rewards.tolist() == pytest.approx([1.0, 0.0])
    assert result.local_advantages.tolist() == pytest.approx([0.5, 0.0])
    assert result.completed_segments == 1


def test_trainer_grpo_path_uses_milestone_annotations_when_enabled():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": _terminal_rewards([0.0, 0.0, 1.0, 1.0]),
            "response_mask": torch.ones(4, 2),
        },
        non_tensors={
            "uid": np.array(["query"] * 4, dtype=object),
            "milestone_trajectory_id": np.array(["failed", "failed", "success", "success"], dtype=object),
            "milestone_turn_index": np.array([0, 1, 0, 1]),
            "milestone_state_ids_json": np.array(["[]", '["m1"]', "[]", '["m1"]'], dtype=object),
            "milestone_verifier_error": np.array([""] * 4, dtype=object),
        },
    )
    config = OmegaConf.create(
        {
            "milestone": {
                "enabled": True,
                "gamma": 0.5,
                "step_advantage_weight": 1.0,
                "norm_segment_by_std": False,
                "strict": True,
            }
        }
    )

    output = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
        config=config,
    )

    assert output.batch["milestone_process_rewards"].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert output.batch["milestone_trajectory_advantages"].tolist() == pytest.approx([-0.5, -0.5, 0.5, 0.5])
    assert output.batch["advantages"].shape == (4, 2)
    assert output.meta_info["milestone_completed_segments"] == 2


def _relay_record(tool_name=None):
    message = {"role": "assistant", "content": ""}
    if tool_name:
        message["tool_calls"] = [
            {
                "id": f"call-{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ]
    return {"response": {"choices": [{"message": message}]}}


def _tool_event(index, name, source="actual_outcome.tool"):
    return ToolEvent(
        index=index,
        name=name,
        args={},
        response={"ok": True},
        source=source,
        successful=True,
    )


def test_tool_mapping_uses_per_source_anchors_for_delegated_chain():
    events = [
        _tool_event(1, "google_drive_search"),
        _tool_event(2, "file_read"),
        _tool_event(1, "toolsearch", "actual_outcome.sandbox_execution_chain"),
        _tool_event(2, "google_drive_download", "actual_outcome.sandbox_execution_chain"),
        _tool_event(3, "upload_file", "actual_outcome.sandbox_execution_chain"),
    ]
    relay_records = [
        _relay_record("google_drive_search"),
        _relay_record("google_drive_download"),
        _relay_record("file_read"),
    ]

    mapped, coverage = map_tool_events_to_turns(events, relay_records)

    assert [(event.name, turn) for event, turn in mapped] == [
        ("google_drive_search", 0),
        ("file_read", 2),
        ("toolsearch", 1),
        ("google_drive_download", 1),
        ("upload_file", 1),
    ]
    assert coverage == pytest.approx(1.0)


@pytest.mark.parametrize("delegation_tool", ["transfer_to_agent", "office_agent", "load_skill"])
def test_tool_mapping_attributes_unanchored_chain_to_last_delegation(delegation_tool):
    events = [
        _tool_event(1, "google_drive_download", "actual_outcome.sandbox_execution_chain"),
        _tool_event(2, "upload_file", "actual_outcome.sandbox_execution_chain"),
    ]

    mapped, coverage = map_tool_events_to_turns(
        events,
        [_relay_record(delegation_tool)],
    )

    assert [(event.name, turn) for event, turn in mapped] == [
        ("google_drive_download", 0),
        ("upload_file", 0),
    ]
    assert coverage == pytest.approx(1.0)


def test_runtime_annotation_incrementally_completes_state_graph(tmp_path):
    instance_dir = tmp_path / "case_instances"
    instance_dir.mkdir()
    instance = {
        "iid": "benchmark-docs:case-1",
        "verifier_version": "test-v1",
        "state_graph": [
            {
                "id": "locate",
                "required": True,
                "depends_on": [],
                "checklist": ["The requested resource is located."],
                "deterministic_checks": [
                    {
                        "kind": "tool_called",
                        "tool_patterns": ["google_drive_search"],
                        "min_count": 1,
                    }
                ],
            },
            {
                "id": "semantic_only",
                "required": True,
                "depends_on": ["locate"],
                "checklist": ["The located resource is semantically matched to the request."],
                "deterministic_checks": [],
            },
            {
                "id": "inspect",
                "required": True,
                "depends_on": ["semantic_only"],
                "checklist": ["The requested content is inspected."],
                "deterministic_checks": [
                    {
                        "kind": "tool_called",
                        "tool_patterns": ["google_docs_cat"],
                        "min_count": 1,
                    }
                ],
            },
            {
                "id": "report",
                "required": True,
                "depends_on": ["inspect"],
                "checklist": ["The grounded result is reported."],
                "deterministic_checks": [{"kind": "response_nonempty", "min_count": 1}],
            },
        ],
    }
    (instance_dir / "docs.jsonl").write_text(json.dumps(instance, ensure_ascii=False) + "\n", encoding="utf-8")
    shared = {
        "iid": instance["iid"],
        "adk_request_id": "request-1",
        "adk_account": "current-user@example.test",
        "adk_model_calls": 3,
        "adk_relay_records": [
            _relay_record("google_drive_search"),
            _relay_record("google_docs_cat"),
            _relay_record(),
        ],
        "adk_trace_actual_outcome": {
            "respond": "grounded answer",
            "tool": [
                {"index": 1, "tool_name": "transfer_to_agent", "args": {}, "response": {}},
                {"index": 2, "tool_name": "google_drive_search", "args": {}, "response": {"ok": True}},
                {"index": 3, "tool_name": "google_docs_cat", "args": {}, "response": {"ok": True}},
            ],
        },
    }
    infos = [{**shared, "adk_turn_index": turn_index} for turn_index in range(3)]

    class FakeChecklistJudge:
        def judge(self, *, instance, trajectory_evidence, state_ids):
            assert instance["iid"] == "benchmark-docs:case-1"
            assert len(trajectory_evidence["turns"]) == 3
            assert trajectory_evidence["active_account"] == "current-user@example.test"
            assert state_ids == ["locate", "semantic_only", "inspect", "report"]
            return MilestoneJudgeOutcome(
                # The judge can recognize the reporting intent before the final
                # response exists; the deterministic check completes it later.
                approved_state_turns={"locate": 0, "semantic_only": 0, "inspect": 1, "report": 1},
                result={"state_results": [], "overall_reason": "test"},
                response_id="judge-response-1",
                model="test-judge",
                duration_s=0.25,
                attempts=1,
            )

    annotations = annotate_milestone_rows(
        infos,
        skillbank_dir=str(tmp_path),
        strict=True,
        semantic_judge=FakeChecklistJudge(),
    )

    assert [json.loads(item["milestone_state_ids_json"]) for item in annotations] == [
        ["locate", "semantic_only"],
        ["inspect"],
        ["report"],
    ]
    assert [item["milestone_event_count"] for item in annotations] == [2, 1, 1]
    assert annotations[0]["milestone_required_state_count"] == 4
    assert annotations[0]["milestone_shapeable_state_count"] == 4
    assert annotations[0]["milestone_completed_state_count"] == 4
    assert annotations[0]["milestone_tool_mapping_coverage"] == pytest.approx(1.0)
    assert annotations[0]["milestone_verifier_version"] == "test-v1"
    assert annotations[0]["milestone_judge_status"] == "completed"
    assert annotations[0]["milestone_judge_model"] == "test-judge"


def test_runtime_checklist_approval_without_deterministic_evidence_stays_incomplete(tmp_path):
    instance_dir = tmp_path / "case_instances"
    instance_dir.mkdir()
    instance = {
        "iid": "benchmark-docs:case-missing-evidence",
        "verifier_version": "test-v1",
        "state_graph": [
            {
                "id": "inspect",
                "required": True,
                "depends_on": [],
                "checklist": ["The requested document is inspected."],
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
    (instance_dir / "docs.jsonl").write_text(json.dumps(instance) + "\n", encoding="utf-8")
    info = {
        "iid": instance["iid"],
        "adk_request_id": "request-missing-evidence",
        "adk_model_calls": 1,
        "adk_turn_index": 0,
        "adk_relay_records": [_relay_record()],
        "adk_trace_actual_outcome": {"respond": "done", "tool": []},
    }

    class ApprovingChecklistJudge:
        def judge(self, **_kwargs):
            return MilestoneJudgeOutcome(
                approved_state_turns={"inspect": 0},
                result={"state_results": [], "overall_reason": "claimed inspection"},
                response_id="judge-response-missing-evidence",
                model="test-judge",
                duration_s=0.1,
                attempts=1,
            )

    annotations = annotate_milestone_rows(
        [info],
        skillbank_dir=str(tmp_path),
        strict=True,
        semantic_judge=ApprovingChecklistJudge(),
    )

    assert json.loads(annotations[0]["milestone_state_ids_json"]) == []
    assert annotations[0]["milestone_completed_state_count"] == 0
    assert annotations[0]["milestone_verifier_error"] == ""
    assert annotations[0]["milestone_judge_status"] == "completed"


def test_runtime_checklist_judge_can_reject_tool_call_candidate(tmp_path):
    instance_dir = tmp_path / "case_instances"
    instance_dir.mkdir()
    instance = {
        "iid": "benchmark-docs:case-2",
        "verifier_version": "test-v1",
        "state_graph": [
            {
                "id": "locate",
                "required": True,
                "depends_on": [],
                "checklist": ["The returned resource is the requested document."],
                "deterministic_checks": [
                    {
                        "kind": "tool_called",
                        "tool_patterns": ["google_drive_search"],
                        "min_count": 1,
                    }
                ],
            }
        ],
    }
    (instance_dir / "docs.jsonl").write_text(json.dumps(instance) + "\n", encoding="utf-8")
    info = {
        "iid": instance["iid"],
        "adk_request_id": "request-2",
        "adk_model_calls": 1,
        "adk_turn_index": 0,
        "adk_relay_records": [_relay_record("google_drive_search")],
        "adk_trace_actual_outcome": {
            "respond": "I found it",
            "tool": [
                {
                    "index": 1,
                    "tool_name": "google_drive_search",
                    "args": {"query": "wrong document"},
                    "response": {"ok": True, "files": []},
                }
            ],
        },
    }

    class RejectingChecklistJudge:
        def judge(self, **_kwargs):
            return MilestoneJudgeOutcome(
                approved_state_turns={},
                result={"state_results": [], "overall_reason": "wrong resource"},
                response_id="judge-response-2",
                model="test-judge",
                duration_s=0.1,
                attempts=1,
            )

    annotations = annotate_milestone_rows(
        [info],
        skillbank_dir=str(tmp_path),
        semantic_judge=RejectingChecklistJudge(),
    )

    assert json.loads(annotations[0]["milestone_state_ids_json"]) == []
    assert annotations[0]["milestone_completed_state_count"] == 0
    assert annotations[0]["milestone_judge_status"] == "completed"
