import json
from pathlib import Path

import pytest

from algo.offline_skills.audit import (
    _semantic_summary,
    _validate_semantic_batch,
    structural_audit,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_structural_audit_accepts_catalog_domain_fallback_and_sanitized_query(tmp_path):
    family = "docs.create_or_export"
    skill_dir = tmp_path / "skills" / "docs-create-or-export"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Test\n", encoding="utf-8")
    _write_json(skill_dir / "state_template.json", {"states": [{"id": "materialize"}]})
    (skill_dir / "judge_rubric.md").write_text("# Test\n", encoding="utf-8")
    (skill_dir / "tests.jsonl").write_text("", encoding="utf-8")

    classification = {
        "domain": "docs",
        "family": family,
        "expected_operations": ["create", "edit"],
        "cardinality": "single",
    }
    catalog = {
        "case_id": "case-1",
        "query": "Replace <URL_1> with <URL_2>",
        "classification": classification,
    }
    instance = {
        "domain": "docs",
        "case_id": "case-1",
        "query": "Replace https://old.example/v1 with https://new.example/v2",
        "family": family,
        "classification": classification,
        "bindings": {"requested_resources": ["resource"]},
        "state_graph": [
            {
                "id": "materialize",
                "template_state_id": "materialize",
                "required": True,
                "depends_on": [],
                "checklist": ["The requested artifact is materialized."],
                "accepted_evidence": ["A successful materialization result."],
                "hard_failures": ["No artifact was created."],
                "semantic_checks": ["The artifact matches the request."],
            }
        ],
    }
    _write_json(tmp_path / "manifest.json", {"totals": {"cases": 1}, "families": {family: {}}})
    _write_json(tmp_path / "case_catalog.jsonl", catalog)
    _write_json(tmp_path / "case_instances" / f"{family}.jsonl", instance)
    _write_json(
        tmp_path / "validation_report.json",
        {"overall": {"rollouts": 8}, "cases_not_replayed_eight_times": []},
    )

    results, summary = structural_audit(tmp_path)

    assert summary["global_errors"] == []
    assert summary["structural_pass"] == 1
    assert summary["operation_coverage_warning_cases"] == 1
    assert results[0]["case_ref"] == "docs:case-1"
    assert results[0]["uncovered_expected_operations"] == ["edit"]


def test_semantic_summary_reports_reroutes_and_compositions():
    results = [
        {
            "case_ref": "docs:1",
            "assigned_family": "docs.read",
            "classification_status": "pass",
            "suggested_family": "docs.read",
            "instantiation_status": "pass",
            "required_skill_components": ["docs.read"],
        },
        {
            "case_ref": "docs:2",
            "assigned_family": "docs.create_or_export",
            "classification_status": "fail",
            "suggested_family": "docs.replace_or_edit",
            "instantiation_status": "fail",
            "required_skill_components": ["docs.create_or_export", "docs.replace_or_edit"],
        },
        {
            "case_ref": "drive:1",
            "assigned_family": "drive.retrieve",
            "classification_status": "uncertain",
            "suggested_family": "drive.retrieve",
            "instantiation_status": "pass",
            "required_skill_components": ["drive.retrieve"],
        },
    ]

    summary = _semantic_summary(results)

    assert summary["classification"] == {"fail": 1, "pass": 1, "uncertain": 1}
    assert summary["instantiation"] == {"fail": 1, "pass": 2}
    assert summary["classification_pass_rate"] == pytest.approx(1 / 3)
    assert summary["instantiation_pass_rate"] == pytest.approx(2 / 3)
    assert summary["primary_family_matches"] == 2
    assert summary["primary_family_match_rate"] == pytest.approx(2 / 3)
    assert summary["classification_fail_same_family"] == 0
    assert summary["suggested_reroutes"] == [
        {"from": "docs.create_or_export", "to": "docs.replace_or_edit", "cases": 1}
    ]
    assert summary["common_compositions"] == [
        {"components": ["docs.create_or_export", "docs.replace_or_edit"], "cases": 1}
    ]


def test_semantic_batch_requires_every_case_in_input_order():
    with pytest.raises(ValueError, match="case refs"):
        _validate_semantic_batch(
            {"assessments": [{"case_ref": "docs:2"}, {"case_ref": "docs:1"}]},
            ["docs:1", "docs:2"],
        )
