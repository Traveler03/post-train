import pytest

from algo.offline_skills.evolution import (
    EVOLUTION_SOURCE_MIGRATION_SUBTYPES,
    FAMILY_SUBTYPE_DEFINITIONS,
    TASK_ORACLE_ANNOTATION_SCHEMA_VERSION,
    EvolutionThresholds,
    attribute_error_families,
    candidate_states_fingerprint,
    case_component_subtypes,
    compose_active_candidate_states,
    consensus_prediction,
    dependent_same_turn_stats,
    evaluate_acceptance,
    evaluate_turn_predictions,
    feedback_subtypes_for_error,
    normalize_task_oracle_annotation,
    outcome_diverse_case_sample,
    round_robin_case_sample,
    stable_case_split,
    stable_three_way_split,
    validate_proposal,
)
from scripts.evolve_milestone_skillbank import (
    _candidate_task_payload,
    _mark_credit_excluded_turns,
    _preserve_unaffected_states,
    _proposal_set_fingerprint,
    _select_regression_anchors,
    _targeted_evolution_contract,
    _unaffected_state_preservation_errors,
    _validate_candidate_judgment,
    _validate_dev_gate_report,
    build_parser,
)


def _state(state_id, kind, subtypes, depends_on=()):
    return {
        "id": state_id,
        "kind": kind,
        "objective": state_id,
        "applies_when": "the subtype is active",
        "applies_to_subtypes": list(subtypes),
        "depends_on": list(depends_on),
        "source_state_ids": [],
        "checklist": ["objective is established"],
        "accepted_evidence": ["observable evidence"],
        "hard_failures": ["objective is not established"],
        "semantic_checks": ["evidence matches the request"],
    }


def _branched_proposal(family):
    subtypes = list(FAMILY_SUBTYPE_DEFINITIONS[family])
    states = [_state("resolve_target", "evidence", subtypes)]
    for index, subtype in enumerate(subtypes):
        states.append(
            _state(f"finish_branch_{index}", "terminal", [subtype], ["resolve_target"])
        )
    return {"family": family, "covered_subtypes": subtypes, "states": states}


def _row(request_id, iid, family, labels):
    return {
        "request_id": request_id,
        "iid": iid,
        "family": family,
        "annotation": {
            "result": {
                "turns": [
                    {"turn_index": index, "label": label}
                    for index, label in enumerate(labels)
                ]
            }
        },
    }


def test_case_split_keeps_all_rollouts_of_iid_together_and_stratifies():
    values = {
        "docs.read": ["r1", "r1", "r2", "r3", "r4"],
        "docs.edit": ["e1", "e2"],
    }

    first = stable_case_split(values, holdout_fraction=0.25, seed="fixed")
    second = stable_case_split(values, holdout_fraction=0.25, seed="fixed")

    assert first == second
    assert {first["r1"], first["r2"], first["r3"], first["r4"]} == {"proposal", "holdout"}
    assert {first["e1"], first["e2"]} == {"proposal", "holdout"}


def test_three_way_split_is_stable_disjoint_and_keeps_proposal_cases():
    values = {
        "docs.read:content": [f"r{index}" for index in range(6)],
        "docs.comments:add": ["a1", "a2", "a3"],
        "docs.comments:reply": ["only"],
    }

    first = stable_three_way_split(values, seed="fixed")
    second = stable_three_way_split(values, seed="fixed")

    assert first == second
    assert set(first.values()) == {"proposal", "dev", "test"}
    assert first["only"] == "proposal"
    for stratum_iids in values.values():
        assert any(first[iid] == "proposal" for iid in stratum_iids)


def test_three_way_split_rejects_iid_in_conflicting_strata():
    values = {
        "family:a": ["same", "a2", "a3"],
        "family:b": ["same", "b2", "b3"],
    }

    try:
        stable_three_way_split(values, seed="fixed")
    except ValueError as exc:
        assert "multiple split strata" in str(exc)
    else:
        raise AssertionError("conflicting IID assignment was accepted")


def test_proposal_split_can_be_replayed_but_test_still_requires_dev_gate():
    choices = next(
        action.choices
        for action in build_parser()._actions
        if action.dest == "eval_split"
    )

    assert tuple(choices) == ("proposal", "dev", "test")


def test_rollout_limit_samples_each_case_before_repeating():
    rows = [
        {"iid": iid, "request_id": f"{iid}-{rollout}"}
        for iid in ("a", "b", "c")
        for rollout in range(3)
    ]

    selected = round_robin_case_sample(rows, limit=5)

    assert [row["request_id"] for row in selected] == ["a-0", "b-0", "c-0", "a-1", "b-1"]


def test_replay_sampling_prefers_terminal_positive_and_then_negative_per_case():
    rows = [
        _row("a-neutral", "a", "docs.read", ["progress", "error"]),
        _row("a-terminal", "a", "docs.read", ["progress", "progress"]),
        _row("a-empty", "a", "docs.read", ["neutral"]),
        _row("b-neutral", "b", "docs.read", ["progress", "neutral"]),
    ]

    one_each = outcome_diverse_case_sample(rows, per_case=1)
    two_each = outcome_diverse_case_sample(rows, per_case=2)

    assert [row["request_id"] for row in one_each] == ["a-terminal", "b-neutral"]
    assert [row["request_id"] for row in two_each] == [
        "a-terminal",
        "a-neutral",
        "b-neutral",
    ]


def test_turn_metrics_deduplicate_multiple_states_and_measure_terminal_recall():
    rows = [
        _row("a", "i1", "docs.read", ["progress", "neutral", "progress"]),
        _row("b", "i2", "docs.read", ["error", "progress"]),
    ]
    predictions = {
        "a": {0: ["resolve", "inspect"], 1: ["bad"]},
        "b": {1: ["answer"]},
    }

    metrics = evaluate_turn_predictions(rows, predictions)

    assert metrics["micro"]["predicted"] == 3
    assert metrics["micro"]["tp"] == 2
    assert metrics["micro"]["fp"] == 1
    assert metrics["micro"]["fn"] == 1
    assert metrics["micro"]["terminal_recall"] == 0.5


def test_turn_metrics_exclude_preparation_that_did_not_reach_a_task_state():
    row = _row("a", "i1", "docs.read", ["necessary_preparation", "error"])
    row["annotation"]["result"]["turns"][0]["credit_eligible"] = True
    row["annotation"]["result"]["turns"][1]["credit_eligible"] = False

    metrics = evaluate_turn_predictions([row], {"a": {0: ["request_resource_id"]}})

    assert metrics["micro"]["gold"] == 0
    assert metrics["micro"]["tp"] == 0
    assert metrics["micro"]["fp"] == 1


def test_dependent_same_turn_rate_only_counts_dependency_pairs():
    rows = [_row("a", "i1", "docs.read", ["progress", "progress"])]
    predictions = {"a": {0: ["resolve", "inspect"], 1: ["answer"]}}
    dependencies = {"docs.read": {"inspect": ["resolve"], "answer": ["inspect"]}}

    result = dependent_same_turn_stats(rows, predictions, dependencies)

    assert result["predicted_turns"] == 2
    assert result["dependent_cofire_turns"] == 1
    assert result["dependent_same_turn_rate"] == 0.5


def test_dependent_same_turn_rate_accepts_request_scoped_composite_dependencies():
    rows = [_row("a", "i1", "docs.comments", ["progress"])]
    predictions = {"a": {0: ["docs_comments__resolve", "docs_comments__add"]}}
    dependencies = {"a": {"docs_comments__add": ["docs_comments__resolve"]}}

    result = dependent_same_turn_stats(rows, predictions, dependencies)

    assert result["dependent_cofire_turns"] == 1


def test_proposal_validation_rejects_generic_terminal_and_cycles():
    state = {
        "kind": "terminal",
        "objective": "x",
        "applies_when": "always",
        "source_state_ids": [],
        "checklist": ["x"],
        "accepted_evidence": ["x"],
        "hard_failures": ["x"],
        "semantic_checks": ["x"],
    }
    proposal = {
        "family": "docs.read",
        "states": [
            {**state, "id": "report_task_result", "depends_on": ["inspect_document"]},
            {
                **state,
                "id": "inspect_document",
                "kind": "evidence",
                "depends_on": ["report_task_result"],
            },
        ],
    }

    errors = validate_proposal(proposal, expected_family="docs.read")

    assert any("generic terminal" in error for error in errors)
    assert any("cycle" in error for error in errors)


def test_proposal_validation_rejects_terminal_without_evidence_dependency():
    proposal = {
        "family": "docs.read",
        "states": [
            {
                "id": "inspect_document",
                "kind": "evidence",
                "objective": "inspect",
                "applies_when": "always",
                "depends_on": [],
                "source_state_ids": [],
                "checklist": ["x"],
                "accepted_evidence": ["x"],
                "hard_failures": ["x"],
                "semantic_checks": ["x"],
            },
            {
                "id": "deliver_document_answer",
                "kind": "terminal",
                "objective": "answer",
                "applies_when": "always",
                "depends_on": [],
                "source_state_ids": [],
                "checklist": ["x"],
                "accepted_evidence": ["x"],
                "hard_failures": ["x"],
                "semantic_checks": ["x"],
            },
        ],
    }

    errors = validate_proposal(proposal, expected_family="docs.read")

    assert any("terminal state deliver_document_answer" in error for error in errors)


def test_acceptance_requires_each_family_to_pass():
    metrics = {
        "micro": {"precision": 0.95, "recall": 0.95, "terminal_recall": 1.0},
        "by_family": {
            "docs.read": {"gold": 10, "precision": 0.95, "recall": 0.95},
            "docs.edit": {"gold": 10, "precision": 0.80, "recall": 0.95},
        },
    }
    cofire = {"dependent_same_turn_rate": 0.01}

    result = evaluate_acceptance(metrics, cofire, thresholds=EvolutionThresholds())

    assert result["accepted"] is False
    assert result["failed_checks"] == ["family_precision:docs.edit"]


def test_comment_add_case_activates_add_branch_only():
    instance = {
        "family": "docs.comments",
        "component_families": ["docs.comments"],
        "classification": {"expected_operations": ["locate", "comment"]},
        "bindings": {"comment_constraints": {"operation": "add"}},
    }
    component_subtypes = case_component_subtypes(instance)
    states = compose_active_candidate_states(
        {
            "docs.comments": _branched_proposal("docs.comments"),
            "boundary_or_negative": _branched_proposal("boundary_or_negative"),
        },
        component_subtypes,
    )

    assert component_subtypes == {
        "docs.comments": ("comments_add",),
        "boundary_or_negative": (
            "missing_resource",
            "ambiguous_request",
            "unavailable_required_evidence",
        ),
    }
    assert [state["template_state_id"] for state in states] == [
        "resolve_target",
        "finish_branch_1",
        "resolve_target",
        "finish_branch_0",
        "finish_branch_1",
        "finish_branch_5",
    ]
    comment_state_ids = {
        state["template_state_id"]
        for state in states
        if state["template_family"] == "docs.comments"
    }
    assert "finish_branch_0" not in comment_state_ids
    assert "finish_branch_3" not in comment_state_ids


def test_composite_case_activates_each_component_family_with_namespaced_ids():
    instance = {
        "family": "docs.replace_or_edit",
        "component_families": ["docs.read", "docs.replace_or_edit"],
        "classification": {"expected_operations": ["locate", "read_content", "replace"]},
        "bindings": {"edit_constraints": {"replacements": [{"from": "a", "to": "b"}]}},
    }
    component_subtypes = case_component_subtypes(instance)
    states = compose_active_candidate_states(
        {
            "docs.read": _branched_proposal("docs.read"),
            "docs.replace_or_edit": _branched_proposal("docs.replace_or_edit"),
            "boundary_or_negative": _branched_proposal("boundary_or_negative"),
        },
        component_subtypes,
    )

    assert component_subtypes == {
        "docs.read": ("document_content",),
        "docs.replace_or_edit": ("text_replace",),
        "boundary_or_negative": (
            "missing_resource",
            "ambiguous_request",
            "unavailable_required_evidence",
        ),
    }
    assert {state["template_family"] for state in states} == {
        "docs.read",
        "docs.replace_or_edit",
        "boundary_or_negative",
    }
    assert len({state["id"] for state in states}) == len(states)


def test_candidate_state_fingerprint_covers_semantics_not_only_ids():
    states = [_state("resolve_target", "evidence", ["document_content"])]
    same_content = [{**states[0], "checklist": list(states[0]["checklist"])}]
    changed = [{**states[0], "checklist": ["different evidence contract"]}]

    assert candidate_states_fingerprint(states) == candidate_states_fingerprint(same_content)
    assert candidate_states_fingerprint(states) != candidate_states_fingerprint(changed)


def test_validator_requires_every_boundary_subtype():
    proposal = _branched_proposal("boundary_or_negative")
    proposal["covered_subtypes"] = proposal["covered_subtypes"][:-1]

    errors = validate_proposal(proposal, expected_family="boundary_or_negative")

    assert any("covered_subtypes must be exactly" in error for error in errors)


def test_legacy_candidate_migration_allows_only_newly_added_subtype_to_be_absent():
    proposal = _branched_proposal("boundary_or_negative")
    proposal["states"].extend(
        [
            _state("check_locator_context", "evidence", ["missing_resource"]),
            _state("expand_locator_scope", "evidence", ["missing_resource"]),
        ]
    )
    removed = "unavailable_required_evidence"
    removed_index = list(FAMILY_SUBTYPE_DEFINITIONS["boundary_or_negative"]).index(removed)
    proposal["covered_subtypes"].remove(removed)
    proposal["states"] = [
        state
        for state in proposal["states"]
        if state["id"] != f"finish_branch_{removed_index}"
    ]
    for state in proposal["states"]:
        if removed in state["applies_to_subtypes"]:
            state["applies_to_subtypes"].remove(removed)

    strict_errors = validate_proposal(
        proposal, expected_family="boundary_or_negative"
    )
    migration_errors = validate_proposal(
        proposal,
        expected_family="boundary_or_negative",
        allowed_missing_subtypes=EVOLUTION_SOURCE_MIGRATION_SUBTYPES[
            "boundary_or_negative"
        ],
    )

    assert any("covered_subtypes must be exactly" in error for error in strict_errors)
    assert migration_errors == []


def test_explicit_boundary_case_also_activates_missing_resource_fallback():
    instance = {
        "family": "boundary_or_negative",
        "component_families": ["boundary_or_negative"],
        "classification": {"signals": ["out-of-range"]},
        "query": "read comment 999",
    }

    assert case_component_subtypes(instance) == {
        "boundary_or_negative": ("out_of_range", "missing_resource")
    }


def test_legacy_instance_recovers_query_declared_composite_components():
    instance = {
        "family": "boundary_or_negative",
        "component_families": ["boundary_or_negative"],
        "classification": {
            "expected_operations": ["establish_boundary"],
            "signals": ["boundary term: 全员通知"],
        },
        "query": "把《员工身份证汇总》导出并在评论里@全员通知大家来下载",
        "bindings": {"benchmark_description": "边-docs/comments/add + docs/export"},
    }

    assert case_component_subtypes(instance) == {
        "boundary_or_negative": ("missing_resource",),
        "docs.comments": ("comments_add",),
        "docs.create_or_export": ("artifact_export",),
    }


def test_seed_oracle_normalization_rejects_false_missing_terminal():
    instance = {
        "bindings": {
            "requested_resources": ["数据报告"],
            "seed_resource_names": ["数据报告"],
        }
    }
    annotation = {
        "annotation_schema_version": "independent-progress-v6-task-oracle",
        "result": {
            "turns": [
                {
                    "turn_index": 0,
                    "label": "necessary_preparation",
                    "credit_eligible": True,
                    "progress_step": "Searched for the document.",
                    "evidence": "The search returned no files.",
                },
                {
                    "turn_index": 1,
                    "label": "progress",
                    "credit_eligible": True,
                    "progress_step": "Reported that searches had not surfaced the document.",
                    "evidence": "Requested additional locator information.",
                },
            ],
            "ideal_state_sequence": [],
            "task_status": "failed",
        },
    }

    normalized = normalize_task_oracle_annotation(instance, annotation)

    assert normalized["annotation_schema_version"] == TASK_ORACLE_ANNOTATION_SCHEMA_VERSION
    assert normalized["result"]["turns"][-1]["label"] == "error"
    assert normalized["result"]["turns"][-1]["credit_eligible"] is False
    assert normalized["task_oracle_corrections"] == [
        {"turn_index": 1, "rule": "seeded_requested_resource_is_not_missing"}
    ]


def test_validator_reserves_capacity_for_missing_resource_coverage_increments():
    proposal = _branched_proposal("boundary_or_negative")

    errors = validate_proposal(proposal, expected_family="boundary_or_negative")

    assert any(
        "subtype missing_resource must activate 4-6 states" in error for error in errors
    )

    proposal["states"].extend(
        [
            _state("check_available_locator_context", "evidence", ["missing_resource"]),
            _state(
                "expand_distinct_discovery_scope",
                "evidence",
                ["missing_resource"],
                ["resolve_target"],
            ),
        ]
    )

    errors = validate_proposal(proposal, expected_family="boundary_or_negative")

    assert not any("subtype missing_resource must activate" in error for error in errors)


def test_validator_rejects_dependency_inactive_for_child_subtype():
    proposal = _branched_proposal("docs.comments")
    proposal["states"][0]["applies_to_subtypes"] = ["comments_read"]

    errors = validate_proposal(proposal, expected_family="docs.comments")

    assert any("inactive for some child subtypes" in error for error in errors)


def test_ambiguous_edit_can_represent_candidate_discovery_before_inspection():
    subtypes = list(FAMILY_SUBTYPE_DEFINITIONS["docs.replace_or_edit"])
    proposal = {
        "family": "docs.replace_or_edit",
        "covered_subtypes": subtypes,
        "states": [
            _state("candidate_set_found", "evidence", subtypes),
            _state("each_candidate_inspected", "evidence", subtypes, ["candidate_set_found"]),
            _state("report_edit_outcome", "terminal", subtypes, ["each_candidate_inspected"]),
        ],
    }

    assert validate_proposal(proposal, expected_family="docs.replace_or_edit") == []


def test_feedback_attributes_false_positive_to_component_state_owner():
    error = {
        "family": "docs.create_or_export",
        "false_positive_turns": [0],
        "false_negative_turns": [],
    }
    judgment = {
        "result": {
            "state_results": [
                {
                    "state_id": "docs_replace_or_edit__establish_concrete_document_basis",
                    "status": "PASS",
                    "completed_turn": 0,
                }
            ]
        }
    }

    families = attribute_error_families(
        error,
        judgment,
        ["docs.create_or_export", "docs.replace_or_edit"],
    )

    assert families == ("docs.replace_or_edit",)


def test_feedback_attributes_false_negative_to_primary_family():
    error = {
        "family": "docs.read",
        "false_positive_turns": [],
        "false_negative_turns": [3],
    }

    families = attribute_error_families(error, {}, ["docs.read"])

    assert families == ("docs.read",)


def test_feedback_attributes_content_access_blocker_to_boundary_family():
    error = {
        "family": "docs.replace_or_edit",
        "false_positive_turns": [],
        "false_negative_turns": [2],
    }
    annotation = {
        "turns": [
            {
                "turn_index": 0,
                "label": "progress",
                "progress_step": "Located the requested document and recovered its exact ID.",
            },
            {"turn_index": 1, "label": "error", "evidence": "Document read failed."},
            {
                "turn_index": 2,
                "label": "progress",
                "progress_step": "Communicated the grounded content-access blocker.",
                "evidence": "The content could not be read and the user was asked to provide it.",
            },
        ],
        "ideal_state_sequence": [
            {
                "name": "Locate requested document",
                "objective": "Find and identify the concrete target.",
                "achieved_turn": 0,
            },
            {
                "name": "Communicate grounded read blocker",
                "objective": "Request the minimum content needed to continue.",
                "achieved_turn": 2,
            },
        ],
        "task_status": "blocked",
    }

    families = attribute_error_families(
        error,
        {},
        ["docs.replace_or_edit", "boundary_or_negative"],
        annotation,
    )
    subtypes = feedback_subtypes_for_error(
        {"family": "docs.replace_or_edit"},
        "boundary_or_negative",
        error,
        annotation,
    )

    assert families == ("boundary_or_negative",)
    assert subtypes == ("unavailable_required_evidence",)


def test_failed_read_turn_is_not_attributed_as_boundary_progress():
    error = {
        "family": "docs.replace_or_edit",
        "false_positive_turns": [],
        "false_negative_turns": [1],
    }
    annotation = {
        "turns": [
            {
                "turn_index": 0,
                "label": "progress",
                "progress_step": "Located the requested document.",
            },
            {
                "turn_index": 1,
                "label": "error",
                "evidence": "The required document read failed.",
            },
        ],
        "task_status": "blocked",
    }

    families = attribute_error_families(
        error,
        {},
        ["docs.replace_or_edit", "boundary_or_negative"],
        annotation,
    )

    assert families == ("docs.replace_or_edit",)


def test_consensus_uses_majority_completion_and_median_pass_turn():
    prediction, agreement = consensus_prediction(
        [
            {1: ["resolve"], 4: ["answer"]},
            {2: ["resolve"], 4: ["answer"]},
            {2: ["resolve"]},
        ],
        state_ids=["resolve", "answer", "missing"],
    )

    assert prediction == {2: ["resolve"], 4: ["answer"]}
    assert agreement["repeat_count"] == 3
    assert agreement["unanimous"] is False
    assert agreement["mean_completion_agreement"] == 8 / 9


def test_candidate_task_payload_includes_benchmark_oracle_fields():
    instance = {
        "query": "read seeded doc",
        "classification": {"outcome": "read"},
        "bindings": {
            "requested_resources": ["seeded doc"],
            "seed_resource_names": ["seeded doc"],
        },
    }

    task = _candidate_task_payload(
        instance,
        iid="benchmark-docs:test",
        family="docs.read",
        component_subtypes={"docs.read": ["document_content"]},
        authorization={"authorization_state": "not_required"},
    )

    assert task["classification"] == {"outcome": "read"}
    assert task["bindings"]["requested_resources"] == ["seeded doc"]
    assert task["bindings"]["seed_resource_names"] == ["seeded doc"]


def test_credit_excluded_turn_cannot_complete_candidate_state():
    state = _state("resolve_target", "evidence", ["document_content"])
    judgment = {
        "state_results": [
            {
                "state_id": "resolve_target",
                "status": "PASS",
                "completed_turn": 0,
                "failed_checklist_indices": [],
                "failed_semantic_check_indices": [],
                "triggered_hard_failure_indices": [],
                "evidence": ["target found"],
                "reason": "found",
            }
        ]
    }

    try:
        _validate_candidate_judgment(
            judgment,
            {"resolve_target": state},
            2,
            credit_excluded_turns={0},
        )
    except ValueError as exc:
        assert "credit-excluded turn 0" in str(exc)
    else:
        raise AssertionError("credit-excluded completion was accepted")


def test_terminal_candidate_state_cannot_complete_before_final_turn():
    state = _state("deliver_answer", "terminal", ["document_content"])
    judgment = {
        "state_results": [
            {
                "state_id": "deliver_answer",
                "status": "PASS",
                "completed_turn": 0,
                "failed_checklist_indices": [],
                "failed_semantic_check_indices": [],
                "triggered_hard_failure_indices": [],
                "evidence": ["final answer"],
                "reason": "answered",
            }
        ]
    }

    with pytest.raises(ValueError, match="expected terminal turn 1"):
        _validate_candidate_judgment(judgment, {"deliver_answer": state}, 2)


def test_credit_exclusions_are_exposed_in_judge_evidence():
    evidence = {"turns": [{"turn_index": 0}, {"turn_index": 1}]}

    _mark_credit_excluded_turns(evidence, {0})

    assert evidence["credit_excluded_turns"] == [0]
    assert evidence["turns"][0]["credit_eligible"] is False
    assert evidence["turns"][0]["credit_exclusion_reason"] == "authorization_violation"
    assert evidence["turns"][1]["credit_eligible"] is True


def test_regression_anchors_preserve_terminal_positive_and_negative_examples():
    def anchor(request_id, labels, prediction):
        return {
            "request_id": request_id,
            "independent_annotation": {
                "turns": [
                    {"turn_index": index, "label": label}
                    for index, label in enumerate(labels)
                ]
            },
            "candidate_prediction": prediction,
        }

    candidates = [
        anchor("terminal-positive", ["progress", "progress"], {0: ["read"], 1: ["answer"]}),
        anchor("terminal-negative", ["progress", "error"], {0: ["read"]}),
        anchor("empty-negative", ["error"], {}),
        anchor("ordinary", ["progress", "neutral"], {0: ["read"]}),
    ]

    selected = _select_regression_anchors(candidates, limit=4)

    assert {value["request_id"] for value in selected} == {
        "terminal-positive",
        "terminal-negative",
        "empty-negative",
        "ordinary",
    }


def test_targeted_evolution_rejects_changes_to_unaffected_subtype_states():
    current = {
        "states": [
            _state("missing", "terminal", ["missing_resource"], ["shared"]),
            _state("shared", "evidence", ["missing_resource", "out_of_range"]),
            _state("range", "terminal", ["out_of_range"], ["shared"]),
        ]
    }
    candidate = {
        "states": [
            {**current["states"][0], "objective": "rewritten unrelated state"},
            {**current["states"][1], "objective": "allowed shared repair"},
            {**current["states"][2], "objective": "allowed range repair"},
        ]
    }

    errors = _unaffected_state_preservation_errors(
        current,
        candidate,
        affected_subtypes={"out_of_range"},
    )

    assert errors == [
        "unaffected state 'missing' must be preserved exactly for targeted evolution",
    ]

    merged = _preserve_unaffected_states(
        current,
        candidate,
        affected_subtypes={"out_of_range"},
    )

    assert merged["states"][0] == current["states"][0]
    assert merged["states"][1]["objective"] == "allowed shared repair"
    assert merged["states"][1]["applies_to_subtypes"] == current["states"][1][
        "applies_to_subtypes"
    ]
    assert {state["id"] for state in merged["states"]} == {"missing", "shared", "range"}
    assert next(state for state in merged["states"] if state["id"] == "range")[
        "objective"
    ] == "allowed range repair"


def test_targeted_evolution_rejects_new_states_covering_unaffected_subtypes():
    current = {
        "states": [
            _state("shared", "evidence", ["text_replace", "structural_edit"]),
        ]
    }
    candidate = {
        "states": [
            current["states"][0],
            _state("new_shared", "evidence", ["text_replace", "structural_edit"]),
            _state("new_structural", "evidence", ["structural_edit"], ["shared"]),
        ]
    }

    errors = _unaffected_state_preservation_errors(
        current,
        candidate,
        affected_subtypes={"structural_edit"},
    )
    merged = _preserve_unaffected_states(
        current,
        candidate,
        affected_subtypes={"structural_edit"},
    )

    assert errors == ["targeted evolution added unrelated state 'new_shared'"]
    assert [state["id"] for state in merged["states"]] == ["shared", "new_structural"]


def test_targeted_evolution_restores_shared_state_graph_structure():
    current = {
        "states": [
            _state("shared", "boundary", ["text_replace", "structural_edit"], ["resolve"]),
            _state("resolve", "evidence", ["text_replace", "structural_edit"]),
        ]
    }
    rewritten = {
        **current["states"][0],
        "kind": "evidence",
        "applies_to_subtypes": ["text_replace"],
        "depends_on": [],
        "objective": "stricter shared semantics",
    }
    candidate = {"states": [rewritten, current["states"][1]]}

    merged = _preserve_unaffected_states(
        current,
        candidate,
        affected_subtypes={"text_replace"},
    )
    shared = next(state for state in merged["states"] if state["id"] == "shared")

    assert shared["objective"] == "stricter shared semantics"
    assert shared["kind"] == "boundary"
    assert shared["applies_to_subtypes"] == ["text_replace", "structural_edit"]
    assert shared["depends_on"] == ["resolve"]


def test_targeted_evolution_contract_accounts_for_immutable_shared_state_budget():
    current = {
        "states": [
            _state("resolve", "evidence", ["text_replace", "structural_edit"]),
            _state("preview", "boundary", ["text_replace", "structural_edit"], ["resolve"]),
            _state("confirm", "boundary", ["text_replace", "structural_edit"], ["preview"]),
            _state("apply", "action", ["structural_edit"], ["confirm"]),
            _state("verify", "terminal", ["structural_edit"], ["apply"]),
        ]
    }

    contract = _targeted_evolution_contract(
        current,
        affected_subtypes={"structural_edit"},
    )

    assert contract is not None
    budget = contract["subtype_state_budgets"]["structural_edit"]
    assert budget["immutable_active_state_ids"] == ["resolve", "preview", "confirm"]
    assert budget["maximum_editable_active_states"] == 3
    assert budget["current_editable_state_ids"] == ["apply", "verify"]


def test_sealed_test_gate_requires_complete_passing_dev_with_same_proposals():
    proposals = {"docs.read": _branched_proposal("docs.read")}
    split = {"proposal": "proposal", "dev": "dev", "test": "test"}
    report = {
        "evaluation_split": "dev",
        "evaluation_request_filter": [],
        "evaluated_rollouts": 64,
        "holdout_rollouts": 64,
        "split_by_iid": split,
        "proposal_set_fingerprint": _proposal_set_fingerprint(proposals),
        "judge_consistency": {"repeat_count": 3},
        "evaluated_annotation_schema_versions": [TASK_ORACLE_ANNOTATION_SCHEMA_VERSION],
        "acceptance": {
            "checks": {
                "micro_precision": True,
                "micro_recall": True,
                "terminal_recall": True,
                "sealed_test_gate": False,
            }
        },
    }

    assert _validate_dev_gate_report(report, proposals=proposals, split=split) == []

    report["evaluation_request_filter"] = ["only-one"]
    report["acceptance"]["checks"]["micro_recall"] = False
    errors = _validate_dev_gate_report(report, proposals=proposals, split=split)

    assert "gate report used a request-level evaluation filter" in errors
    assert any("micro_recall" in error for error in errors)
