"""Offline structural and semantic audit for a generated SkillBank."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .classifier import FAMILIES, FAMILY_CAPABILITIES
from .evidence import sanitize_value
from .models import JsonObject, skill_name_for_family
from .responses_client import DirectResponsesClient, ResponsesProviderConfig
from .storage import write_json, write_jsonl, write_text

AUDIT_SCHEMA_VERSION = "migoo-offline-skillbank-audit-v1"

FAMILY_DEFINITIONS = {
    "docs.read": "读取单个 Docs 文档的正文、结构或元数据，不执行写入。",
    "docs.replace_or_edit": "在已有 Docs 文档中执行替换、插入、删除或局部更新。",
    "docs.comments": "读取、添加、回复、解决或删除 Docs 评论。",
    "docs.create_or_export": "创建、复制或导出 Docs 文档及交付物。",
    "drive.retrieve": "搜索、列举、读取 Drive 资源、元数据或链接。",
    "drive.file_operation": "创建目录、复制、移动、上传、下载、删除或导出 Drive 文件。",
    "drive.transfer_or_share": "查看或修改 Drive 访问权限、共享对象和角色。",
    "multi_source_aggregate": "从多个来源取证后执行比较、计算、分组或综合，可带后续物化动作。",
    "boundary_or_negative": "目标不存在、歧义、参数缺失、越权、安全或能力不支持，应停止有害动作。",
}

_OPERATIONS = (
    "locate",
    "read_content",
    "read_metadata",
    "read_comments",
    "replace",
    "edit",
    "comment",
    "create",
    "copy",
    "export",
    "download",
    "upload",
    "move",
    "delete",
    "permissions",
    "share",
    "aggregate",
    "establish_boundary",
)


@dataclass(frozen=True)
class AuditOptions:
    skillbank_dir: Path
    output_dir: Path
    api_config: Path = Path("~/.codex/config.toml")
    model: str | None = None
    reasoning_effort: str | None = "medium"
    use_api: bool = True
    batch_size: int = 10
    api_concurrency: int = 3
    api_timeout_s: float = 300.0
    api_max_attempts: int = 3
    max_output_tokens: int = 10000
    force: bool = False


def _iter_jsonl(path: Path) -> list[JsonObject]:
    values: list[JsonObject] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            values.append(value)
    return values


def _case_domain(value: JsonObject) -> str:
    domain = str(value.get("domain") or "").strip()
    if domain:
        return domain
    classification = value.get("classification")
    if isinstance(classification, dict):
        domain = str(classification.get("domain") or "").strip()
    if not domain:
        raise ValueError(f"case {value.get('case_id')!r} has no domain")
    return domain


def _case_ref(value: JsonObject) -> str:
    return f"{_case_domain(value)}:{value['case_id']}"


def _load_instances(root: Path) -> dict[str, JsonObject]:
    instances: dict[str, JsonObject] = {}
    for path in sorted((root / "case_instances").glob("*.jsonl")):
        for instance in _iter_jsonl(path):
            ref = _case_ref(instance)
            if ref in instances:
                raise ValueError(f"duplicate case instance: {ref}")
            instances[ref] = instance
    return instances


def _topological_errors(nodes: list[JsonObject]) -> list[str]:
    node_ids = {str(node.get("id")) for node in nodes}
    errors: list[str] = []
    unknown = sorted(
        {
            str(dependency)
            for node in nodes
            for dependency in node.get("depends_on") or []
            if str(dependency) not in node_ids
        }
    )
    if unknown:
        errors.append(f"unknown dependencies: {unknown}")
        return errors
    indegree = {node_id: 0 for node_id in node_ids}
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        node_id = str(node["id"])
        for dependency in node.get("depends_on") or []:
            dependency = str(dependency)
            indegree[node_id] += 1
            children[dependency].append(node_id)
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for child in children[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(node_ids):
        errors.append("state graph contains a cycle")
    return errors


def structural_audit(root: Path) -> tuple[list[JsonObject], JsonObject]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    catalog = _iter_jsonl(root / "case_catalog.jsonl")
    instances = _load_instances(root)
    results: list[JsonObject] = []
    global_errors: list[str] = []

    catalog_refs = [_case_ref(case) for case in catalog]
    if len(catalog_refs) != len(set(catalog_refs)):
        global_errors.append("case catalog contains duplicate domain:case_id values")
    if set(catalog_refs) != set(instances):
        missing = sorted(set(catalog_refs) - set(instances))
        extra = sorted(set(instances) - set(catalog_refs))
        global_errors.append(f"catalog/instance mismatch: missing={missing}, extra={extra}")
    if len(catalog) != int(manifest.get("totals", {}).get("cases") or -1):
        global_errors.append("manifest case total does not match case catalog")

    for family in FAMILIES:
        skill_dir = root / "skills" / skill_name_for_family(family)
        for name in ("SKILL.md", "state_template.json", "judge_rubric.md", "tests.jsonl"):
            if not (skill_dir / name).is_file() and family in manifest.get("families", {}):
                global_errors.append(f"missing {family} artifact: {name}")

    for case in catalog:
        ref = _case_ref(case)
        instance = instances.get(ref)
        errors: list[str] = []
        warnings: list[str] = []
        classification = case.get("classification") if isinstance(case.get("classification"), dict) else {}
        family = str(classification.get("family") or "")
        if family not in FAMILIES:
            errors.append(f"unknown family: {family}")
        if instance is None:
            errors.append("case has no instance")
            results.append(
                {
                    "case_ref": ref,
                    "domain": _case_domain(case),
                    "case_id": case["case_id"],
                    "family": family,
                    "structure_status": "fail",
                    "errors": errors,
                    "warnings": warnings,
                    "uncovered_expected_operations": [],
                }
            )
            continue
        if instance.get("family") != family:
            errors.append("instance family differs from catalog classification")
        instance_query = sanitize_value(str(instance.get("query") or ""), max_string=700)
        if instance_query != case.get("query"):
            errors.append("instance query differs from catalog query")
        nodes = instance.get("state_graph") if isinstance(instance.get("state_graph"), list) else []
        if not nodes:
            errors.append("instance has no state nodes")
        if not any(bool(node.get("required")) for node in nodes):
            errors.append("instance has no required state")
        if len({str(node.get("id")) for node in nodes}) != len(nodes):
            errors.append("instance state ids are not unique")
        errors.extend(_topological_errors(nodes))
        required_nodes = [node for node in nodes if bool(node.get("required"))]
        for field in ("checklist", "accepted_evidence", "hard_failures", "semantic_checks"):
            missing = sorted(
                str(node.get("id") or "<empty>")
                for node in required_nodes
                if not isinstance(node.get(field), list) or not node.get(field)
            )
            if missing:
                errors.append(f"required states missing {field}: {missing}")

        operations = {str(operation) for operation in classification.get("expected_operations") or []}
        component_families = [str(item) for item in instance.get("component_families") or [family]]
        component_capabilities = set().union(
            *(FAMILY_CAPABILITIES.get(component, frozenset()) for component in component_families)
        )
        uncovered = sorted(operations - component_capabilities)
        if uncovered:
            warnings.append("composed templates do not cover every extracted operation")
        requested = instance.get("bindings", {}).get("requested_resources") or []
        mentioned = instance.get("bindings", {}).get("mentioned_resources") or []
        if (
            family != "boundary_or_negative"
            and classification.get("cardinality") != "multiple"
            and requested == ["<requested_resource_set>"]
            and mentioned
        ):
            warnings.append("explicit or single-resource case retains an unresolved resource placeholder")

        unknown_template_refs: list[str] = []
        for node in nodes:
            if node.get("synthetic"):
                continue
            template_family = str(node.get("template_family") or family)
            template_state_id = str(node.get("template_state_id") or "")
            template_path = root / "skills" / skill_name_for_family(template_family) / "state_template.json"
            if not template_path.is_file():
                unknown_template_refs.append(f"{template_family}:{template_state_id}")
                continue
            template = json.loads(template_path.read_text(encoding="utf-8"))
            template_ids = {str(state["id"]) for state in template.get("states") or []}
            if template_state_id not in template_ids:
                unknown_template_refs.append(f"{template_family}:{template_state_id}")
        if unknown_template_refs:
            errors.append(f"instance references unknown template states: {sorted(unknown_template_refs)}")

        results.append(
            {
                "case_ref": ref,
                "domain": _case_domain(case),
                "case_id": case["case_id"],
                "family": family,
                "structure_status": "fail" if errors else "pass",
                "errors": errors,
                "warnings": warnings,
                "uncovered_expected_operations": uncovered,
            }
        )

    replay_path = root / "validation_report.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.is_file() else {}
    if replay and replay.get("cases_not_replayed_eight_times"):
        global_errors.append("some cases were not replayed exactly eight times")
    summary = {
        "global_errors": global_errors,
        "cases": len(results),
        "structural_pass": sum(result["structure_status"] == "pass" for result in results),
        "structural_fail": sum(result["structure_status"] == "fail" for result in results),
        "operation_coverage_warning_cases": sum(bool(result["uncovered_expected_operations"]) for result in results),
        "binding_warning_cases": sum(
            any("placeholder" in warning for warning in result["warnings"]) for result in results
        ),
        "replay": replay.get("overall", {}),
        "cases_not_replayed_eight_times": replay.get("cases_not_replayed_eight_times", []),
    }
    return results, summary


def _semantic_schema() -> JsonObject:
    assessment = {
        "type": "object",
        "properties": {
            "case_ref": {"type": "string"},
            "classification_status": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
            "suggested_family": {"type": "string", "enum": list(FAMILIES)},
            "required_skill_components": {
                "type": "array",
                "items": {"type": "string", "enum": list(FAMILIES)},
            },
            "required_operations": {
                "type": "array",
                "items": {"type": "string", "enum": list(_OPERATIONS)},
            },
            "classification_reason": {"type": "string"},
            "instantiation_status": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
            "missing_task_states": {"type": "array", "items": {"type": "string"}},
            "unnecessary_required_states": {"type": "array", "items": {"type": "string"}},
            "binding_issues": {"type": "array", "items": {"type": "string"}},
            "dependency_issues": {"type": "array", "items": {"type": "string"}},
            "instantiation_reason": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": [
            "case_ref",
            "classification_status",
            "suggested_family",
            "required_skill_components",
            "required_operations",
            "classification_reason",
            "instantiation_status",
            "missing_task_states",
            "unnecessary_required_states",
            "binding_issues",
            "dependency_issues",
            "instantiation_reason",
            "confidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "assessments": {"type": "array", "items": assessment},
        },
        "required": ["assessments"],
        "additionalProperties": False,
    }


_SEMANTIC_INSTRUCTIONS = """你是独立的离线 Skill 路由和任务状态实例审计员。只根据用户 query、benchmark 描述提示、
分类结果、资源绑定和实例状态图判断，不参考 reward 标签，也不假设线上环境状态。

分类规则：
- suggested_family 是最合适的主 family；复合任务仍可有一个主 family。
- required_skill_components 必须列出完成整个请求所需的全部类别组件，不能只列主 family。
- benchmark 描述中的 download/search 可能只是实现路径，主分类以用户意图为准。
- 文件名中出现“汇总/报告”等词不等于 aggregate；只有请求要求比较、计算、分组、排序或综合才算。
- 不存在、歧义、缺参数、越权、安全风险或不支持能力，应归 boundary_or_negative。

实例化规则：
- pass 要求状态图覆盖 query 的全部必要中间状态、写入/读取结果和关键依赖。
- 复合请求缺少任一次级操作状态应 fail，例如“复制→替换→评论”不能只有复制状态。
- 节点是任务语义状态，不要求绑定特定工具。
- 集合查询使用 <requested_resource_set> 可以合理；明确命名的资源仍为占位符则是 binding issue。
- conditional 节点只有在该 case 确实需要时才应 required。
- 不要因为措辞偏好判 fail；只报告会影响路由、覆盖、依赖或验收的实质问题。

每个输入 case 必须且只能输出一项，case_ref 原样返回。输出严格 JSON。
"""


def _compact_semantic_case(case: JsonObject, instance: JsonObject) -> JsonObject:
    return {
        "case_ref": _case_ref(case),
        "query": case.get("query"),
        "benchmark_description": case.get("description"),
        "assigned_classification": case.get("classification"),
        "component_families": instance.get("component_families") or [instance.get("family")],
        "bindings": {
            "requested_resources": instance.get("bindings", {}).get("requested_resources") or [],
            "mentioned_resources": instance.get("bindings", {}).get("mentioned_resources") or [],
            "output_resources": instance.get("bindings", {}).get("output_resources") or [],
            "destination_resources": instance.get("bindings", {}).get("destination_resources") or [],
            "share_targets": instance.get("bindings", {}).get("share_targets") or [],
            "comment_targets": instance.get("bindings", {}).get("comment_targets") or [],
            "output_format": instance.get("bindings", {}).get("output_format"),
            "search_constraints": instance.get("bindings", {}).get("search_constraints") or {},
            "requested_view": instance.get("bindings", {}).get("requested_view") or {},
            "edit_constraints": instance.get("bindings", {}).get("edit_constraints") or {},
            "content_constraints": instance.get("bindings", {}).get("content_constraints") or {},
            "comment_constraints": instance.get("bindings", {}).get("comment_constraints") or {},
            "share_constraints": instance.get("bindings", {}).get("share_constraints") or {},
            "seed_resource_names": (instance.get("bindings", {}).get("seed_resource_names") or [])[:20],
        },
        "instantiated_states": [
            {
                "id": node.get("id"),
                "objective": node.get("objective"),
                "required": bool(node.get("required")),
                "depends_on": node.get("depends_on") or [],
                "binding": node.get("binding") or {},
            }
            for node in instance.get("state_graph") or []
        ],
    }


def _semantic_prompt(cases: list[JsonObject]) -> str:
    return json.dumps(
        {
            "family_definitions": FAMILY_DEFINITIONS,
            "cases": cases,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _validate_semantic_batch(result: JsonObject, expected_refs: list[str]) -> None:
    assessments = result.get("assessments") if isinstance(result.get("assessments"), list) else []
    actual_refs = [str(item.get("case_ref")) for item in assessments if isinstance(item, dict)]
    if actual_refs != expected_refs:
        raise ValueError(f"semantic audit case refs must be {expected_refs}, found {actual_refs}")


def semantic_audit(
    root: Path,
    output_dir: Path,
    options: AuditOptions,
) -> tuple[list[JsonObject], JsonObject]:
    catalog = _iter_jsonl(root / "case_catalog.jsonl")
    instances = _load_instances(root)
    compact = [_compact_semantic_case(case, instances[_case_ref(case)]) for case in catalog]
    batches = [compact[index : index + options.batch_size] for index in range(0, len(compact), options.batch_size)]
    config = ResponsesProviderConfig.from_toml(
        options.api_config,
        model_override=options.model,
        reasoning_effort_override=options.reasoning_effort,
    )

    def judge(batch_index: int, batch: list[JsonObject]) -> tuple[int, list[JsonObject], JsonObject]:
        prompt = _semantic_prompt(batch)
        write_json(
            output_dir / "prompts" / f"batch_{batch_index:03d}.json",
            {
                "schema": "migoo-offline-skillbank-semantic-audit-prompt-v1",
                "instructions": _SEMANTIC_INSTRUCTIONS,
                "input": json.loads(prompt),
                "structured_output_schema": _semantic_schema(),
            },
        )
        client = DirectResponsesClient(
            config,
            timeout_s=options.api_timeout_s,
            max_attempts=options.api_max_attempts,
        )
        expected_refs = [str(item["case_ref"]) for item in batch]
        response = None
        validation_error: ValueError | None = None
        validation_attempt = 0
        for validation_attempt in range(1, options.api_max_attempts + 1):
            response = client.generate_json(
                instructions=_SEMANTIC_INSTRUCTIONS,
                prompt=prompt,
                schema_name="offline_skillbank_semantic_audit",
                schema=_semantic_schema(),
                max_output_tokens=options.max_output_tokens,
            )
            try:
                _validate_semantic_batch(response.content, expected_refs)
                validation_error = None
                break
            except ValueError as exc:
                validation_error = exc
        if validation_error is not None:
            raise RuntimeError(
                f"semantic audit batch {batch_index} remained invalid after "
                f"{options.api_max_attempts} attempts: {validation_error}"
            ) from validation_error
        assert response is not None
        metadata = {
            "response_id": response.response_id,
            "model": response.model,
            "status": response.status,
            "usage": response.usage,
            "elapsed_s": response.elapsed_s,
            "attempts": response.attempts,
            "validation_attempts": validation_attempt,
        }
        write_json(
            output_dir / "responses" / f"batch_{batch_index:03d}.json",
            {"content": response.content, "metadata": metadata},
        )
        return batch_index, list(response.content["assessments"]), metadata

    by_batch: dict[int, list[JsonObject]] = {}
    metadata_by_batch: dict[str, JsonObject] = {}
    workers = max(1, min(options.api_concurrency, len(batches)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="skill-audit") as executor:
        futures = {executor.submit(judge, index, batch): index for index, batch in enumerate(batches, start=1)}
        for future in as_completed(futures):
            batch_index, assessments, metadata = future.result()
            by_batch[batch_index] = assessments
            metadata_by_batch[str(batch_index)] = metadata
            print(f"audited batch {batch_index}/{len(batches)}", flush=True)
    results = [assessment for index in sorted(by_batch) for assessment in by_batch[index]]
    return results, {
        "provider": config.public_metadata(),
        "batches": metadata_by_batch,
    }


def _semantic_summary(results: list[JsonObject]) -> JsonObject:
    classification = Counter(str(result["classification_status"]) for result in results)
    instantiation = Counter(str(result["instantiation_status"]) for result in results)
    by_family: dict[str, list[JsonObject]] = defaultdict(list)
    reroutes = Counter()
    component_sets = Counter()
    primary_family_matches = 0
    for result in results:
        assigned = str(result.get("assigned_family") or "")
        if assigned:
            by_family[assigned].append(result)
        if assigned == str(result["suggested_family"]):
            primary_family_matches += 1
        elif result["classification_status"] == "fail":
            reroutes[(assigned, str(result["suggested_family"]))] += 1
        components = tuple(sorted(str(item) for item in result["required_skill_components"]))
        if len(components) > 1:
            component_sets[components] += 1
    return {
        "cases": len(results),
        "classification": dict(sorted(classification.items())),
        "instantiation": dict(sorted(instantiation.items())),
        "classification_pass_rate": classification["pass"] / len(results) if results else 0.0,
        "instantiation_pass_rate": instantiation["pass"] / len(results) if results else 0.0,
        "primary_family_matches": primary_family_matches,
        "primary_family_mismatches": len(results) - primary_family_matches,
        "primary_family_match_rate": primary_family_matches / len(results) if results else 0.0,
        "classification_fail_same_family": sum(
            result["classification_status"] == "fail" and result.get("assigned_family") == result["suggested_family"]
            for result in results
        ),
        "issue_case_counts": {
            "missing_task_states": sum(bool(result.get("missing_task_states")) for result in results),
            "unnecessary_required_states": sum(bool(result.get("unnecessary_required_states")) for result in results),
            "binding_issues": sum(bool(result.get("binding_issues")) for result in results),
            "dependency_issues": sum(bool(result.get("dependency_issues")) for result in results),
        },
        "by_family": [
            {
                "family": family,
                "cases": len(rows),
                "primary_family_matches": sum(row.get("assigned_family") == row["suggested_family"] for row in rows),
                "classification_pass": sum(row["classification_status"] == "pass" for row in rows),
                "instantiation_pass": sum(row["instantiation_status"] == "pass" for row in rows),
            }
            for family, rows in sorted(by_family.items())
        ],
        "suggested_reroutes": [
            {"from": source, "to": target, "cases": count} for (source, target), count in reroutes.most_common()
        ],
        "common_compositions": [
            {"components": list(components), "cases": count} for components, count in component_sets.most_common()
        ],
    }


def _render_audit_report(report: JsonObject, semantic_results: list[JsonObject]) -> str:
    structural = report["structural"]
    semantic = report.get("semantic")
    lines = [
        "# Offline SkillBank Audit",
        "",
        "## Structural Audit",
        "",
        f"- Cases: {structural['cases']}",
        f"- Structural pass: {structural['structural_pass']}",
        f"- Structural fail: {structural['structural_fail']}",
        f"- Operation-coverage warning cases: {structural['operation_coverage_warning_cases']}",
        f"- Binding warning cases: {structural['binding_warning_cases']}",
        f"- Cases not replayed exactly 8 times: {len(structural['cases_not_replayed_eight_times'])}",
    ]
    if semantic is None:
        lines.extend(["", "Semantic audit was not run."])
        return "\n".join(lines)
    lines.extend(
        [
            "",
            "## Independent Semantic Audit",
            "",
            f"- Primary-family match: {semantic['primary_family_matches']}/{semantic['cases']} "
            f"({semantic['primary_family_match_rate']:.1%})",
            f"- Classification pass: {semantic['classification'].get('pass', 0)}/{semantic['cases']} "
            f"({semantic['classification_pass_rate']:.1%})",
            f"- Classification fail: {semantic['classification'].get('fail', 0)}",
            f"- Classification uncertain: {semantic['classification'].get('uncertain', 0)}",
            f"- Instantiation pass: {semantic['instantiation'].get('pass', 0)}/{semantic['cases']} "
            f"({semantic['instantiation_pass_rate']:.1%})",
            f"- Instantiation fail: {semantic['instantiation'].get('fail', 0)}",
            f"- Instantiation uncertain: {semantic['instantiation'].get('uncertain', 0)}",
            f"- Missing-state cases: {semantic['issue_case_counts']['missing_task_states']}",
            f"- Dependency-issue cases: {semantic['issue_case_counts']['dependency_issues']}",
            f"- Binding-issue cases: {semantic['issue_case_counts']['binding_issues']}",
            f"- Unnecessary-required-state cases: {semantic['issue_case_counts']['unnecessary_required_states']}",
            "",
            "## Failed Cases",
            "",
            "| Case | Classification | Suggested family | Instantiation | Main issue |",
            "|---|---|---|---|---|",
        ]
    )
    for result in semantic_results:
        if result["classification_status"] == "pass" and result["instantiation_status"] == "pass":
            continue
        issue = result["instantiation_reason"] or result["classification_reason"]
        issue = str(issue).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| `{result['case_ref']}` | {result['classification_status']} | "
            f"`{result['suggested_family']}` | {result['instantiation_status']} | {issue} |"
        )
    return "\n".join(lines)


def run_audit(options: AuditOptions) -> JsonObject:
    root = options.skillbank_dir.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(root / "manifest.json")
    if (output_dir / "audit_report.json").exists() and not options.force:
        raise FileExistsError(f"{output_dir} already contains an audit report; pass --force to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    structural_results, structural_summary = structural_audit(root)
    write_jsonl(output_dir / "structural_results.jsonl", structural_results)

    semantic_results: list[JsonObject] = []
    generation_metadata: JsonObject = {"mode": "not_run"}
    if options.use_api:
        semantic_results, generation_metadata = semantic_audit(root, output_dir, options)
        catalog = {_case_ref(case): case for case in _iter_jsonl(root / "case_catalog.jsonl")}
        for result in semantic_results:
            case = catalog[str(result["case_ref"])]
            result["assigned_family"] = case["classification"]["family"]
            result["query"] = case["query"]
        write_jsonl(output_dir / "semantic_results.jsonl", semantic_results)

    semantic_summary = _semantic_summary(semantic_results) if semantic_results else None
    report = {
        "schema": AUDIT_SCHEMA_VERSION,
        "skillbank_dir": str(root),
        "structural": structural_summary,
        "semantic": semantic_summary,
        "semantic_generation": generation_metadata,
        "verdict": {
            "primary_routing_usable": bool(semantic_summary and semantic_summary["primary_family_match_rate"] >= 0.95),
            "classification_usable": bool(semantic_summary and semantic_summary["classification_pass_rate"] >= 0.95),
            "instantiation_usable": bool(semantic_summary and semantic_summary["instantiation_pass_rate"] >= 0.95),
            "threshold": 0.95,
        },
    }
    write_json(output_dir / "audit_report.json", report)
    write_text(output_dir / "AUDIT.md", _render_audit_report(report, semantic_results))
    return report
