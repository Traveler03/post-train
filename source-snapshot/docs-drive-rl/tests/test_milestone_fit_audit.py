from scripts.audit_milestone_fit import (
    _render_report,
    _turn_credit_roles,
    aggregate_summaries,
    build_progress_payload,
    compare_annotations,
    consensus_progress_results,
)


def test_progress_payload_attaches_final_answer_to_terminal_turn():
    rollout = {
        "iid": "benchmark-docs:test",
        "question": "read it",
        "model_calls": 2,
        "relay_records": [],
        "actual_outcome": {"respond": "The grounded final answer."},
    }
    instance = {
        "iid": "benchmark-docs:test",
        "family": "docs.read",
        "classification": {"expected_operations": ["read_content"]},
        "component_families": ["docs.read"],
        "query": "read it",
        "bindings": {},
        "state_graph": [],
    }

    payload = build_progress_payload(rollout, instance)

    assert payload["trajectory"]["terminal_turn_index"] == 1
    assert payload["trajectory"]["turns"][1]["final_answer"] == "The grounded final answer."


def _annotation(request_id, labels):
    return {
        "request_id": request_id,
        "result": {
            "turns": [
                {"turn_index": index, "label": label, "credit_eligible": label in {"progress", "necessary_preparation"}}
                for index, label in enumerate(labels)
            ]
        },
    }


def test_compare_annotations_distinguishes_boundary_and_backfill_quality():
    annotations = [
        _annotation("a", ["error", "necessary_preparation", "progress", "neutral"]),
        _annotation("b", ["progress", "error"]),
    ]
    rewards = {
        "a": {"milestone_state_events_json": '[{"turn_index":2,"state_ids":["m1"]}]'},
        "b": {"milestone_state_events_json": "[]"},
    }

    result = compare_annotations(annotations, rewards)

    assert result["true_progress_turns"] == 2
    assert result["milestone_boundary_turns"] == 1
    assert result["backfilled_turns"] == 2
    assert result["boundary_label_counts"] == {"progress": 1}
    assert result["backfill_label_counts"] == {"error": 1, "necessary_preparation": 1}
    assert result["progress_boundary_recall"] == 0.5
    assert result["segment_credit_eligible_precision"] == 2 / 3
    assert result["miscredited_turns"] == 1


def test_progress_consensus_votes_each_turn_and_uses_conservative_tie_break():
    results = []
    for labels, status in [
        (["progress", "neutral"], "complete"),
        (["progress", "error"], "complete"),
        (["neutral", "necessary_preparation"], "partial"),
    ]:
        result = _annotation("unused", labels)["result"]
        result.update(
            {
                "task_status": status,
                "ideal_state_sequence": [],
                "overall_reason": status,
            }
        )
        for turn in result["turns"]:
            turn["progress_step"] = turn["label"] if turn["label"] == "progress" else None
            turn["evidence"] = turn["label"]
        results.append(result)

    consensus, agreement = consensus_progress_results(results)

    assert [turn["label"] for turn in consensus["turns"]] == ["progress", "error"]
    assert consensus["task_status"] == "complete"
    assert agreement["repeat_count"] == 3
    assert agreement["turn_label_agreements"] == [2 / 3, 1 / 3]


def test_aggregate_summaries_uses_micro_averages():
    result = aggregate_summaries(
        [
            {
                "comparison": {
                    "trajectories": 2,
                    "true_progress_turns": 4,
                    "milestone_boundary_turns": 2,
                    "backfilled_turns": 2,
                    "shaped_turns": 4,
                    "miscredited_turns": 1,
                    "boundary_label_counts": {"progress": 2},
                    "all_shaped_label_counts": {"progress": 2, "necessary_preparation": 1, "error": 1},
                },
                "milestone_fit": {
                    "verdict": "major_revision",
                    "node_assessments": [{"status": "irrelevant"}],
                },
            },
            {
                "comparison": {
                    "trajectories": 1,
                    "true_progress_turns": 1,
                    "milestone_boundary_turns": 1,
                    "backfilled_turns": 0,
                    "shaped_turns": 1,
                    "miscredited_turns": 0,
                    "boundary_label_counts": {"progress": 1},
                    "all_shaped_label_counts": {"progress": 1},
                },
                "milestone_fit": {
                    "verdict": "suitable",
                    "node_assessments": [{"status": "appropriate"}],
                },
            },
        ]
    )

    assert result["cases"] == 2
    assert result["trajectories"] == 3
    assert result["fit_verdict_counts"] == {"major_revision": 1, "suitable": 1}
    assert result["boundary_progress_precision"] == 1
    assert result["progress_boundary_recall"] == 3 / 5
    assert result["segment_credit_eligible_precision"] == 4 / 5


def test_turn_credit_roles_distinguishes_backfill_and_boundary():
    reward = {"milestone_state_events_json": '[{"turn_index":2},{"turn_index":4}]'}

    assert _turn_credit_roles({}, reward) == {
        0: "backfill",
        1: "backfill",
        2: "boundary",
        3: "backfill",
        4: "boundary",
    }


def test_case_report_shows_independent_turn_annotations():
    summary = {
        "iid": "benchmark-docs:test",
        "comparison": {
            "trajectories": 1,
            "true_progress_turns": 1,
            "miscredited_turns": 1,
            "segment_credit_eligible_precision": 0.5,
            "boundary_label_counts": {"progress": 1},
            "backfill_label_counts": {"error": 1},
            "all_shaped_label_counts": {"progress": 1, "error": 1},
        },
        "milestone_fit": {
            "verdict": "major_revision",
            "overall_reason": "Dependency is wrong.",
            "recommended_graph": ["resolve", "read"],
            "missing_states": [],
            "node_assessments": [
                {"state_id": "read", "status": "wrong_dependency", "reason": "Must resolve first."}
            ],
        },
    }
    annotation = {
        "request_id": "request-123456789",
        "result": {
            "task_status": "partial",
            "overall_reason": "Only located the document.",
            "turns": [
                {
                    "turn_index": 0,
                    "label": "error",
                    "progress_step": None,
                    "evidence": "Search failed.",
                },
                {
                    "turn_index": 1,
                    "label": "progress",
                    "progress_step": "Located the document",
                    "evidence": "Search returned one exact match.",
                },
            ],
        },
    }
    rewards = {"request-123456789": {"milestone_state_events_json": '[{"turn_index":1}]'}}

    report = _render_report(summary, [annotation], rewards)

    assert "逐轨迹真正推进步骤" in report
    assert "Located the document" in report
    assert "段内回填" in report
    assert "Milestone 边界" in report
