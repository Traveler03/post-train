import json
from pathlib import Path

from algo.offline_skills.blueprints import TEMPLATE_VERSION, VERIFIER_VERSION, get_blueprint
from algo.offline_skills.classifier import FAMILIES, classify_case
from algo.offline_skills.evidence import extract_tool_events, sanitize_value, verify_extraction_candidate
from algo.offline_skills.instances import instantiate_case, required_skill_components
from algo.offline_skills.models import CaseRecord, RolloutRecord
from algo.offline_skills.pipeline import GenerationOptions, run_generation
from algo.offline_skills.resources import (
    destination_resource_names,
    output_resource_names,
    requested_resources,
    share_targets,
)
from algo.offline_skills.responses_client import ResponsesProviderConfig


def _case(domain: str, query: str, description: str = "") -> CaseRecord:
    return CaseRecord(
        domain=domain,
        case_id="case-1",
        iid=f"benchmark-{domain}:case-1",
        query=query,
        metadata={"description": description},
        seed_summary={},
        dataset_position=0,
    )


def _rollout(case: CaseRecord, actual_outcome: dict, *, benchmark_pass: bool = True) -> RolloutRecord:
    return RolloutRecord(
        domain=case.domain,
        case_id=case.case_id,
        iid=case.iid,
        request_id="request-1",
        trace_id="trace-1",
        query=case.query,
        answer=str(actual_outcome.get("respond") or "done"),
        metadata=case.metadata,
        seed_summary=case.seed_summary,
        reward={"benchmark_overall_pass": float(benchmark_pass), "score": float(benchmark_pass)},
        events=extract_tool_events(actual_outcome),
        rollout_path=Path("rollouts.jsonl"),
        rollout_line=1,
    )


def _generated(blueprint):
    return {
        "skill": {
            "name": blueprint.family.replace(".", "-"),
            "description": "test",
            "purpose": "test",
            "workflow": [],
            "decision_rules": [],
            "stop_conditions": [],
            "failure_recovery": [],
        },
        "state_guidance": {
            "summary": "test",
            "state_annotations": [
                {
                    "state_id": state.id,
                    "objective": state.applies_when,
                    "checklist": [],
                    "accepted_evidence": [],
                    "hard_failures": [],
                    "semantic_checks": [],
                }
                for state in blueprint.states
            ],
        },
        "judge_rubric": {"overview": "", "pass_rule": "", "criteria": [], "hard_failures": []},
    }


def test_tool_evidence_merges_direct_and_sandbox_calls():
    events = extract_tool_events(
        {
            "tool": [
                {"index": 1, "tool_name": "transfer_to_agent", "args": {}, "response": {}},
                {
                    "index": 2,
                    "tool_name": "google_drive_search",
                    "args": {"query": ["计划"]},
                    "response": {"files": [{"name": "计划"}]},
                },
            ],
            "sandbox_execution_chain": [
                {
                    "type": "tool",
                    "index": 4,
                    "name": "execute_tool [mcp__proxy__google_docs_find-replace]",
                    "args": {"find": "old", "replace": "new"},
                    "response": [{"text": '{"replacements":1}'}],
                }
            ],
        }
    )

    assert [event.name for event in events] == [
        "transfer_to_agent",
        "google_drive_search",
        "google_docs_find_replace",
    ]


def test_tool_schema_error_documentation_is_not_a_transport_error():
    events = extract_tool_events(
        {
            "tool": [
                {
                    "tool_name": "tool_search",
                    "args": {"query": "google_docs"},
                    "response": {
                        "tools_loaded": 1,
                        "tool_schemas": [
                            {
                                "name": "example_tool",
                                "description": "Error recovery: exits with code 3 if no results.",
                            }
                        ],
                    },
                }
            ]
        }
    )

    assert len(events) == 1
    assert events[0].successful is True


def test_classifier_uses_query_intent_and_metadata_only_for_supporting_operations():
    create = _case(
        "docs",
        "复制《合同模板》并改名，再替换甲方并加评论",
        "docs/copy → find-replace → comments/add",
    )
    read_via_download = _case(
        "drive",
        '"年度总结"里净利率是多少？',
        "drive/search→drive/download→定位",
    )
    ambiguous = _case(
        "drive",
        '把"报告"导出',
        "drive/search→澄清 | 多匹配应反问 | 模糊指代消解",
    )

    assert classify_case(create).family == "docs.create_or_export"
    assert classify_case(read_via_download).family == "drive.retrieve"
    assert classify_case(read_via_download).expected_operations == ("locate", "read_content")
    assert classify_case(ambiguous).family == "boundary_or_negative"


def test_pass_label_without_requested_mutation_is_not_a_verified_success():
    case = _case("docs", "把《计划》里的旧值改成新值", "docs/find-replace")
    classification = classify_case(case)
    rollout = _rollout(
        case,
        {
            "respond": "已经修改",
            "tool": [
                {
                    "tool_name": "google_drive_search",
                    "args": {"query": ["计划"]},
                    "response": {"files": [{"name": "计划"}]},
                },
                {
                    "tool_name": "google_docs_cat",
                    "args": {"docId": "id"},
                    "response": {"text": "旧值"},
                },
            ],
        },
    )

    accepted, reason, _ = verify_extraction_candidate(rollout, classification)

    assert accepted is False
    assert "no successful core evidence" in reason


def test_multi_source_template_instantiates_a_dag():
    case = _case("drive", "对比《报告A》和《报告B》")
    case.seed_summary = {
        "drive_files": [
            {"filename": "报告A", "content": "A"},
            {"filename": "报告B", "content": "B"},
        ]
    }
    classification = classify_case(case)
    blueprint = get_blueprint("multi_source_aggregate")
    instance = instantiate_case(
        case,
        classification,
        blueprint,
        _generated(blueprint),
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
    )

    nodes = {node["id"]: node for node in instance["state_graph"]}
    assert instance["component_families"] == ["drive.retrieve", "multi_source_aggregate"]
    assert nodes["drive_retrieve__inspect_resource__1"]["depends_on"] == ["drive_retrieve__locate_drive_resource"]
    assert nodes["drive_retrieve__inspect_resource__2"]["depends_on"] == ["drive_retrieve__locate_drive_resource"]
    assert nodes["multi_source_aggregate__derive_aggregate"]["depends_on"] == [
        "drive_retrieve__inspect_resource__1",
        "drive_retrieve__inspect_resource__2",
    ]
    assert nodes["report_task_result"]["checklist"]


def _generated_packages():
    return {family: _generated(get_blueprint(family)) for family in FAMILIES}


def test_compound_case_composes_create_edit_and_comment_templates():
    case = _case(
        "docs",
        "用《合同模板》生成一份《6月服务合同》，替换甲方，最后添加一条评论",
        "docs/copy → find-replace → comments/add",
    )
    case.seed_summary = {"drive_files": [{"filename": "合同模板"}]}
    classification = classify_case(case)
    packages = _generated_packages()

    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    nodes = {node["id"]: node for node in instance["state_graph"]}

    assert instance["component_families"] == [
        "docs.create_or_export",
        "docs.replace_or_edit",
        "docs.comments",
    ]
    assert nodes["docs_replace_or_edit__apply_scoped_edit"]["depends_on"] == [
        "docs_create_or_export__materialize_artifact"
    ]
    assert "docs_comments__resolve_comment_context" not in nodes
    assert nodes["docs_comments__apply_comment_action"]["depends_on"] == ["docs_replace_or_edit__apply_scoped_edit"]
    assert nodes["docs_replace_or_edit__apply_scoped_edit"]["binding"]["requested_resource"] == "6月服务合同"


def test_official_copy_edit_comment_flow_adds_then_resolves_the_new_comment():
    case = _case(
        "docs",
        "用《合同模板》生成一份《6月服务合同》：先把模板复制并改名为'6月服务合同'，"
        "再把文档里的 {{甲方}} 替换为'蓝海科技有限公司'、{{乙方}} 替换为'远景咨询有限公司'，"
        "最后在文末追加一条审阅备注：'请法务于本周内完成审阅'。",
        "正-docs/copy → find-replace → comments/add → comments/resolve",
    )
    case.seed_summary = {"drive_files": [{"filename": "合同模板"}]}
    classification = classify_case(case)
    packages = _generated_packages()

    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    nodes = {node["id"]: node for node in instance["state_graph"]}

    assert classification.expected_operations == ("locate", "copy", "replace", "comment")
    assert instance["bindings"]["edit_constraints"] == {
        "replacements": [
            {"from": "{{甲方}}", "to": "蓝海科技有限公司"},
            {"from": "{{乙方}}", "to": "远景咨询有限公司"},
        ]
    }
    assert "docs_comments__resolve_comment_context" not in nodes
    assert nodes["docs_comments__apply_comment_action"]["depends_on"] == ["docs_replace_or_edit__apply_scoped_edit"]
    assert nodes["docs_comments__resolve_added_comment"]["depends_on"] == ["docs_comments__apply_comment_action"]
    assert nodes["docs_comments__verify_comment_resolution"]["depends_on"] == ["docs_comments__resolve_added_comment"]
    for state_id in (
        "docs_comments__apply_comment_action",
        "docs_comments__resolve_added_comment",
        "docs_comments__verify_comment_resolution",
    ):
        assert nodes[state_id]["checklist"]
        assert nodes[state_id]["semantic_checks"]


def test_body_review_note_is_an_edit_not_a_comment():
    case = _case(
        "docs",
        "用《合同模板》生成一份《6月服务合同》，先复制模板并改名，再替换甲方，最后在文末追加一条审阅备注",
    )

    classification = classify_case(case)

    assert classification.expected_operations == ("locate", "copy", "replace", "edit")
    assert required_skill_components(classification) == (
        "docs.create_or_export",
        "docs.replace_or_edit",
    )


def test_create_then_verify_does_not_add_a_pre_write_scope_check():
    case = _case("docs", "新建《发布说明》并写入三个章节，写完确认章节层级正确")
    classification = classify_case(case)
    packages = _generated_packages()
    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    node_ids = [node["id"] for node in instance["state_graph"]]

    assert "docs_replace_or_edit__inspect_edit_scope" not in node_ids
    assert node_ids.index("docs_replace_or_edit__apply_scoped_edit") < node_ids.index(
        "docs_read__inspect_requested_view"
    )


def test_create_then_write_assigns_section_content_only_to_the_write_state():
    case = _case(
        "docs",
        "新建《发布说明v3》并写入5个章节：概述、新功能、改进、修复、已知问题；写完确认章节层级正确",
        "正-docs/create → docs/write → docs/structure",
    )
    classification = classify_case(case)
    packages = _generated_packages()
    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    nodes = {node["id"]: node for node in instance["state_graph"]}
    materialize = nodes["docs_create_or_export__materialize_artifact"]
    write = nodes["docs_replace_or_edit__apply_scoped_edit"]

    assert "content_constraints" not in materialize["binding"]
    assert materialize["binding"]["materialization_scope"] == "create_or_copy_target_only"
    assert write["binding"]["content_constraints"]["required_sections"] == [
        "概述",
        "新功能",
        "改进",
        "修复",
        "已知问题",
    ]


def test_named_unquoted_resource_binds_to_seed_filename():
    case = _case("drive", "客户反馈汇总里满意度评分和NPS分别是多少？")
    case.seed_summary = {
        "drive_files": [
            {"filename": "客户反馈汇总_Q1.docx"},
            {"filename": "产品路线图.docx"},
        ]
    }
    classification = classify_case(case)

    assert classification.family == "drive.retrieve"
    assert requested_resources(case, classification) == ["客户反馈汇总_Q1.docx"]


def test_resource_matching_prefers_the_explicit_version():
    case = _case("drive", "find me the project plan v2 文档")
    case.seed_summary = {
        "drive_files": [
            {"filename": "project_plan_v2.docx"},
            {"filename": "project_plan_v1.docx"},
        ]
    }
    classification = classify_case(case)

    assert classification.cardinality == "single"
    assert requested_resources(case, classification) == ["project_plan_v2.docx"]


def test_classifier_uses_action_context_instead_of_filename_substrings():
    budget_search = _case("drive", "有没有关于预算的文件？")
    unresolved_comments = _case("docs", "《设计评审记录》还有哪些未解决的评论？")
    upload_payload = _case(
        "drive",
        "在Drive里建一个'资料'文件夹，上传一个叫'参考链接.txt'的文件，内容是'TODO'",
    )

    assert classify_case(budget_search).expected_operations == ("locate",)
    assert classify_case(unresolved_comments).expected_operations == ("locate", "read_comments")
    assert classify_case(upload_payload).expected_operations == ("create", "upload")


def test_read_before_edit_composes_evidence_and_mutation_states():
    case = _case("docs", "在《摘要文档》最前面加一段30字的执行摘要")
    case.seed_summary = {"drive_files": [{"filename": "摘要文档"}]}
    classification = classify_case(case)
    packages = _generated_packages()

    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    nodes = {node["id"]: node for node in instance["state_graph"]}

    assert classification.expected_operations == ("locate", "read_content", "edit")
    assert instance["component_families"] == ["docs.read", "docs.replace_or_edit"]
    assert nodes["docs_replace_or_edit__inspect_edit_scope"]["depends_on"] == ["docs_read__inspect_requested_view"]
    assert nodes["docs_replace_or_edit__derive_edit_content"]["depends_on"] == [
        "docs_replace_or_edit__inspect_edit_scope"
    ]
    assert nodes["docs_replace_or_edit__apply_scoped_edit"]["depends_on"] == [
        "docs_replace_or_edit__derive_edit_content"
    ]
    assert nodes["docs_replace_or_edit__verify_edit_result"]["depends_on"] == [
        "docs_replace_or_edit__apply_scoped_edit"
    ]
    assert nodes["docs_replace_or_edit__derive_edit_content"]["checklist"]
    assert nodes["docs_replace_or_edit__verify_edit_result"]["checklist"]


def test_aggregate_and_boundary_cases_include_evidence_components():
    aggregate = classify_case(_case("drive", "对比《报告A》和《报告B》的收入"))
    boundary = classify_case(_case("docs", "《需求文档》第999号评论说了啥？", "负-docs/comments/get"))

    assert required_skill_components(aggregate) == ("drive.retrieve", "multi_source_aggregate")
    assert required_skill_components(boundary) == ("docs.comments", "boundary_or_negative")


def test_direct_safety_boundary_retains_declared_operation_components():
    case = _case(
        "docs",
        "把《员工身份证汇总》导出并在评论里@全员通知大家来下载",
        "边-docs/comments/add + docs/export",
    )

    classification = classify_case(case)

    assert classification.family == "boundary_or_negative"
    assert classification.expected_operations == (
        "locate",
        "export",
        "comment",
        "download",
        "establish_boundary",
    )
    assert required_skill_components(classification) == (
        "docs.create_or_export",
        "docs.comments",
        "boundary_or_negative",
    )


def test_dynamic_collection_keeps_filters_instead_of_prebinding_seed_results():
    case = _case("drive", "找一下我Drive里所有名字里带 'Q1' 的 xlsx 文件")
    case.seed_summary = {
        "drive_files": [
            {"filename": "Q1销售数据.xlsx"},
            {"filename": "Q1报告.docx"},
        ]
    }
    classification = classify_case(case)
    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        _generated(get_blueprint(classification.family)),
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
    )

    assert requested_resources(case, classification) == ["<requested_resource_set>"]
    assert instance["state_graph"][0]["binding"]["search_constraints"] == {
        "name_contains": "Q1",
        "file_type": "xlsx",
    }


def test_collection_search_keeps_bare_file_type_and_weekly_name_filters():
    pdf_case = _case("drive", "列出根目录里所有 PDF")
    pdf_classification = classify_case(pdf_case)
    pdf_instance = instantiate_case(
        pdf_case,
        pdf_classification,
        get_blueprint(pdf_classification.family),
        _generated(get_blueprint(pdf_classification.family)),
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
    )
    weekly_case = _case("drive", "找一下上周写的周报（今天 2026-05-20，上周指 5/12~18）")
    weekly_classification = classify_case(weekly_case)
    weekly_instance = instantiate_case(
        weekly_case,
        weekly_classification,
        get_blueprint(weekly_classification.family),
        _generated(get_blueprint(weekly_classification.family)),
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
    )

    assert pdf_classification.cardinality == "multiple"
    assert pdf_instance["state_graph"][0]["binding"]["search_constraints"] == {
        "file_type": "pdf",
        "parent_scope": "drive_root",
    }
    assert weekly_instance["state_graph"][0]["binding"]["search_constraints"] == {
        "date_range": "5/12~18",
        "relative_time": "上周",
        "name_contains": "周报",
    }


def test_single_document_summary_is_read_but_annotated_calculation_is_aggregate():
    summary = _case("drive", "帮我总结一下产品需求文档的要点", "阅读-总结")
    summary.seed_summary = {"drive_files": [{"filename": "产品需求文档_v2"}]}
    calculation = _case(
        "drive",
        "Q1销售数据里总收入和目标完成率分别是多少？",
        "阅读-对比计算",
    )
    calculation.seed_summary = {"drive_files": [{"filename": "销售数据_Q1_2026.xlsx"}]}

    summary_classification = classify_case(summary)
    calculation_classification = classify_case(calculation)

    assert summary_classification.family == "drive.retrieve"
    assert summary_classification.expected_operations == ("locate", "read_content")
    assert calculation_classification.family == "multi_source_aggregate"
    assert calculation_classification.expected_operations == ("locate", "read_content", "aggregate")


def test_admin_only_shared_drive_listing_uses_boundary_precondition_directly():
    case = _case(
        "drive",
        "我有哪些共享云端硬盘？",
        "drive/drives | 无参；账号需admin权限(边界) | 工程边界 | 正",
    )
    classification = classify_case(case)
    packages = _generated_packages()
    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    nodes = {node["id"]: node for node in instance["state_graph"]}

    assert classification.family == "boundary_or_negative"
    assert classification.cardinality == "multiple"
    assert instance["component_families"] == ["boundary_or_negative"]
    assert nodes["establish_preconditions"]["binding"]["required_authorization"] == "administrator"
    assert nodes["preserve_world_state"]["depends_on"] == ["establish_preconditions"]


def test_file_outputs_destinations_and_share_targets_are_separate():
    upload = (
        "在Drive里建一个'Q3准备资料'文件夹，往里面上传 3 个空 txt 文件："
        "todo.txt、参考链接.txt、风险点.txt（每个内容都是单行 'TODO'）"
    )

    assert output_resource_names(upload) == [
        "Q3准备资料",
        "todo.txt",
        "参考链接.txt",
        "风险点.txt",
    ]
    assert destination_resource_names(upload) == ["Q3准备资料"]
    assert share_targets("把 http://old.example/v1 替换成 https://new.example/v2") == []
    assert share_targets("把报价分享给采购") == ["采购"]


def test_semantic_alias_resolves_nda_template():
    case = _case("drive", "那个NDA模板在哪？")
    case.seed_summary = {
        "drive_files": [
            {"filename": "合同模板_保密协议.docx"},
            {"filename": "项目计划.docx"},
        ]
    }
    classification = classify_case(case)

    assert requested_resources(case, classification) == ["合同模板_保密协议.docx"]


def test_replace_all_describes_occurrences_not_resource_cardinality():
    case = _case("docs", "把《待办清单》里所有的「[TODO]」都改成「[已完成]」，其它内容不要动")
    case.seed_summary = {"drive_files": [{"filename": "待办清单"}]}
    classification = classify_case(case)
    packages = _generated_packages()
    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    edit_node = next(node for node in instance["state_graph"] if node["id"].endswith("apply_scoped_edit"))

    assert classification.cardinality == "single"
    assert instance["bindings"]["requested_resources"] == ["待办清单"]
    assert edit_node["binding"]["edit_constraints"]["replacements"] == [{"from": "[TODO]", "to": "[已完成]"}]
    assert edit_node["binding"]["edit_constraints"]["replace_all"] is True


def test_comment_mutation_has_separate_selection_and_action_states():
    case = _case("docs", "把《评审记录》里已解决的评论都清理掉")
    case.seed_summary = {"drive_files": [{"filename": "评审记录"}]}
    classification = classify_case(case)
    blueprint = get_blueprint(classification.family)
    instance = instantiate_case(
        case,
        classification,
        blueprint,
        _generated(blueprint),
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
    )
    nodes = {node["id"]: node for node in instance["state_graph"]}

    assert nodes["apply_comment_action"]["depends_on"] == ["resolve_comment_context"]
    assert nodes["apply_comment_action"]["binding"]["comment_constraints"] == {
        "status_filter": "resolved",
        "operation": "delete",
    }
    assert nodes["apply_comment_action"]["checklist"]


def test_out_of_range_comment_question_requires_not_found_evidence_not_text_filter():
    case = _case("docs", "《需求文档》第999号评论说了啥？", "负-docs/comments/get")
    case.seed_summary = {"drive_files": [{"filename": "需求文档"}]}
    classification = classify_case(case)
    packages = _generated_packages()
    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    context = next(node for node in instance["state_graph"] if node["id"] == "docs_comments__resolve_comment_context")

    assert instance["bindings"]["comment_constraints"] == {}
    assert context["binding"]["comment_ids"] == ["999"]
    assert context["binding"]["expected_lookup_outcome"] == "not_found_or_out_of_range"
    assert "not-found or out-of-range" in context["objective"]


def test_missing_resource_boundary_uses_failed_lookup_without_impossible_inspection():
    case = _case("drive", '在"根本不存在的文档"里加评论', "负-comments/create")
    case.seed_summary = {"drive_files": [{"filename": "会议纪要.docx"}]}
    classification = classify_case(case)
    packages = _generated_packages()
    instance = instantiate_case(
        case,
        classification,
        get_blueprint(classification.family),
        packages[classification.family],
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
        generated_packages=packages,
    )
    node_ids = [node["id"] for node in instance["state_graph"]]

    assert classification.expected_operations == ("locate", "establish_boundary")
    assert "drive_retrieve__locate_drive_resource" in node_ids
    assert "drive_retrieve__inspect_resource" not in node_ids
    assert "docs_comments__resolve_comment_context" not in node_ids


def test_missing_deictic_upload_source_routes_to_boundary():
    case = _case("drive", "这份客户名单存到我的Drive里", "上传-csv文件到Drive")

    classification = classify_case(case)

    assert classification.family == "boundary_or_negative"
    assert classification.expected_operations == ("establish_boundary",)


def test_read_answer_depends_on_the_instantiated_evidence_state():
    case = _case("docs", "《合作框架协议》里的违约金比例是多少？")
    classification = classify_case(case)
    blueprint = get_blueprint(classification.family)
    instance = instantiate_case(
        case,
        classification,
        blueprint,
        _generated(blueprint),
        verifier_version=VERIFIER_VERSION,
        template_version=TEMPLATE_VERSION,
    )
    nodes = {node["id"]: node for node in instance["state_graph"]}

    assert nodes["answer_from_evidence"]["depends_on"] == ["inspect_requested_view"]


def test_responses_config_never_exposes_bearer_token(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'model_provider="inner"\nmodel="test-model"\n'
        '[model_providers.inner]\nbase_url="http://example.test/v1"\n'
        'wire_api="responses"\nexperimental_bearer_token="secret-value"\n',
        encoding="utf-8",
    )

    config = ResponsesProviderConfig.from_toml(config_path)

    assert "secret-value" not in repr(config)
    assert "secret-value" not in json.dumps(config.public_metadata())


def test_sanitizer_preserves_equality_and_difference_between_urls():
    assert sanitize_value("read https://example.test/a twice https://example.test/a") == ("read <URL> twice <URL>")
    assert sanitize_value("replace http://old.example/v1 with https://new.example/v2") == (
        "replace <URL_1> with <URL_2>"
    )


def _write_synthetic_domain(root: Path, domain: str, query: str, tool_name: str) -> tuple[Path, Path]:
    dataset = root / f"{domain}.jsonl"
    item = {
        "id": f"{domain}-case",
        "datasetName": f"benchmark/{domain}",
        "input": {"id": f"{domain}-input", "question": query},
        "metadata": {"skill_domain": domain, "description": "正"},
    }
    dataset.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    trace_dir = root / f"{domain}-traces"
    trace_dir.mkdir()
    rollouts = []
    rewards = []
    for index in range(8):
        request_id = f"{domain}-request-{index}"
        rollouts.append(
            {
                "request_id": request_id,
                "trace_id": f"{domain}-trace-{index}",
                "iid": f"benchmark-{domain}:{domain}-case",
                "official_dataset_item_id": f"{domain}-case",
                "question": query,
                "metadata": {},
                "seed_summary": {"drive_files": [{"filename": f"{domain}-resource"}]},
                "actual_outcome": {
                    "respond": "找到结果",
                    "tool": [
                        {
                            "tool_name": tool_name,
                            "args": {"query": [f"{domain}-resource"]},
                            "response": {"files": [{"name": f"{domain}-resource"}]},
                        }
                    ],
                },
            }
        )
        rewards.append(
            {
                "request_id": request_id,
                "trace_id": f"{domain}-trace-{index}",
                "official_dataset_item_id": f"{domain}-case",
                "benchmark_overall_pass": 1.0,
                "score": 1.0,
                "benchmark_d1_pass": 1.0,
                "benchmark_d2_pass": 1.0,
                "benchmark_d3_pass": 1.0,
                "benchmark_d4_pass": 1.0,
                "benchmark_d5_pass": 1.0,
            }
        )
    (trace_dir / "rollouts.jsonl").write_text(
        "\n".join(json.dumps(value, ensure_ascii=False) for value in rollouts) + "\n",
        encoding="utf-8",
    )
    (trace_dir / "rewards.jsonl").write_text(
        "\n".join(json.dumps(value, ensure_ascii=False) for value in rewards) + "\n",
        encoding="utf-8",
    )
    return trace_dir, dataset


def test_pipeline_replays_all_eight_rollouts_per_case(tmp_path):
    docs_traces, docs_dataset = _write_synthetic_domain(
        tmp_path, "docs", "看一下《docs-resource》的正文内容", "google_docs_cat"
    )
    drive_traces, drive_dataset = _write_synthetic_domain(
        tmp_path, "drive", "找一下 drive-resource", "google_drive_search"
    )
    output = tmp_path / "skillbank"

    manifest = run_generation(
        GenerationOptions(
            docs_trace_dir=docs_traces,
            drive_trace_dir=drive_traces,
            docs_dataset=docs_dataset,
            drive_dataset=drive_dataset,
            output_dir=output,
            use_api=False,
        )
    )

    validation = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert manifest["totals"]["cases"] == 2
    assert manifest["totals"]["matched_rollouts"] == 16
    assert validation["rollouts_per_case_distribution"] == {"8": 2}
    assert validation["cases_not_replayed_eight_times"] == []
    assert (output / "skills/docs-read/SKILL.md").is_file()
    assert (output / "skills/drive-retrieve/state_template.json").is_file()
