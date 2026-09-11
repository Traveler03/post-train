import pytest

from scripts.render_segment_decay_report import render_html as render_segment_decay_html
from scripts.render_turn_reward_misattribution import Trajectory, build_summary, render_simple_html


def _trajectory(
    request_id: str,
    *,
    score: float,
    turn_count: int,
    events_by_turn: dict[int, tuple[str, ...]],
    overall_pass: bool = False,
) -> Trajectory:
    return Trajectory(
        run_index=0,
        update_index=0,
        request_id=request_id,
        iid="benchmark-docs:test",
        score=score,
        overall_pass=overall_pass,
        turn_count=turn_count,
        events_by_turn=events_by_turn,
    )


def test_turn_reward_summary_separates_baseline_bias_and_segment_backfill():
    summary = build_summary(
        [
            _trajectory(
                "long-high",
                score=1.0,
                turn_count=3,
                events_by_turn={2: ("m1",)},
                overall_pass=True,
            ),
            _trajectory("short-mid", score=0.7, turn_count=1, events_by_turn={}),
            _trajectory("short-low", score=0.0, turn_count=1, events_by_turn={}),
        ],
        gamma=0.95,
        local_weight=1.0,
    )

    overview = summary["overview"]
    assert overview["turn_rows"] == 5
    assert overview["nonfinal_rows"] == 2
    assert overview["terminal_score_fanout_rate"] == pytest.approx(0.4)

    outcome = summary["global_advantage"]
    assert outcome["query_groups"] == 1
    assert outcome["groups_with_row_weighting_bias"] == 1
    assert outcome["group_baselines"][0]["row_weighted_baseline"] == pytest.approx(0.74)
    assert outcome["group_baselines"][0]["trajectory_equal_baseline"] == pytest.approx(1.7 / 3)
    assert outcome["advantage_sign_changed_rows"] == 1
    assert outcome["advantage_sign_changed_trajectories"] == 1

    segment = summary["segment_attribution"]
    assert segment["completed_segments"] == 1
    assert segment["mean_completed_segment_turns"] == 3
    assert segment["single_turn_segment_share"] == 0
    assert segment["segment_length_buckets"] == [{"turn_count": 3, "segments": 1, "share": 1.0}]
    assert segment["milestone_boundary_turns"] == 1
    assert segment["backfilled_non_boundary_turns"] == 2
    assert segment["unshaped_turns"] == 2
    assert segment["unshaped_local_sign"] == {"negative": 2}
    assert segment["max_backfill_distance"] == 2
    assert [bucket["distance"] for bucket in segment["distance_buckets"]] == [0, 1, 2]
    assert [bucket["process_reward"] for bucket in segment["distance_buckets"]] == pytest.approx([10.0, 9.5, 9.025])

    report = render_simple_html(summary, "Turn Reward Test")
    assert "一条 trajectory 一票" in report
    assert "1 turns" in report
    assert "2 个回填 turn" in report

    segment_report = render_segment_decay_html(summary, "Segment Test")
    assert "真正的问题不是公式算错" in segment_report
    assert "段内向前回填（风险集合）" in segment_report
    assert "Segment 长度分布" in segment_report
    assert "0.95^distance" in segment_report


def test_segment_report_surfaces_confirmed_error_turns():
    summary = build_summary(
        [_trajectory("trace", score=1.0, turn_count=2, events_by_turn={1: ("m1",)}, overall_pass=True)],
        gamma=0.95,
        local_weight=1.0,
    )
    summary["confirmed_turn_audits"] = [
        {
            "iid": "benchmark-docs:test",
            "query": "test query",
            "milestone_turn": 1,
            "judge_evidence": "tool result proves the call failed",
            "diagnosis": "the failed call did not contribute to the milestone",
            "turns": [
                {
                    "turn_index": 0,
                    "action": "called a missing tool",
                    "classification": "tool_error",
                    "label": "错误",
                    "process_reward": 9.5,
                    "local_advantage": 1.0,
                },
                {
                    "turn_index": 1,
                    "action": "completed the milestone",
                    "classification": "necessary",
                    "label": "正确完成 milestone",
                    "process_reward": 10.0,
                    "local_advantage": 1.5,
                },
            ],
        }
    ]

    report = render_segment_decay_html(summary, "Segment Test")
    assert "真正的问题不是公式算错" in report
    assert "called a missing tool" in report
    assert "错误且 A_local 为正" in report
    assert "9.500" in report
