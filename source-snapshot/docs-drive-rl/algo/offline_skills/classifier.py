"""Deterministic benchmark-oriented task classification.

The classifier intentionally uses explicit benchmark metadata and lexical rules. It does
not cluster embeddings: categories must remain stable and auditable between training runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import CaseClassification, CaseRecord
from .resources import (
    destination_resource_names,
    infer_resource_cardinality,
    match_seed_resources,
    quoted_mentions,
    seed_resource_names,
)

FAMILIES = (
    "docs.read",
    "docs.replace_or_edit",
    "docs.comments",
    "docs.create_or_export",
    "drive.retrieve",
    "drive.file_operation",
    "drive.transfer_or_share",
    "multi_source_aggregate",
    "boundary_or_negative",
)

FAMILY_CAPABILITIES: dict[str, frozenset[str]] = {
    "docs.read": frozenset({"locate", "read_content", "read_metadata"}),
    "docs.replace_or_edit": frozenset({"locate", "read_content", "replace", "edit"}),
    "docs.comments": frozenset({"locate", "read_comments", "comment", "delete"}),
    "docs.create_or_export": frozenset({"locate", "create", "copy", "export", "download", "upload"}),
    "drive.retrieve": frozenset({"locate", "read_content", "read_metadata"}),
    "drive.file_operation": frozenset({"locate", "create", "copy", "export", "download", "upload", "move", "delete"}),
    "drive.transfer_or_share": frozenset({"locate", "permissions", "share"}),
    "multi_source_aggregate": frozenset(
        {
            "locate",
            "read_content",
            "read_metadata",
            "read_comments",
            "aggregate",
            "create",
            "copy",
            "export",
            "download",
            "upload",
            "share",
            "permissions",
        }
    ),
    "boundary_or_negative": frozenset(
        {
            "establish_boundary",
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
        }
    ),
}


def family_for_operation(domain: str, operation: str) -> str | None:
    if operation in {"read_comments", "comment"}:
        return "docs.comments"
    if operation in {"replace", "edit"}:
        return "docs.replace_or_edit"
    if operation in {"permissions", "share"}:
        return "drive.transfer_or_share"
    if operation == "aggregate":
        return "multi_source_aggregate"
    if operation in {"read_content", "read_metadata"}:
        return "docs.read" if domain == "docs" else "drive.retrieve"
    if operation in {"create", "copy", "export", "download", "upload", "move", "delete"}:
        if domain == "docs" and operation not in {"move", "delete"}:
            return "docs.create_or_export"
        return "drive.file_operation"
    return None


_BOUNDARY_TERMS = (
    "不存在",
    "seed不含",
    "seed无",
    "没说",
    "pii",
    "敏感",
    "公开到网上",
    "全员通知",
    "owner 权限",
    "owner权限",
    "不支持",
    "unsupported",
    "越界",
    "歧义",
    "应反问",
    "多匹配",
    "模糊指代",
    "ambiguous",
    "hallucination",
    "failure_notfound",
)

_VAGUE_QUERIES = (
    "帮我更新一下",
    "写得不好的部分改改",
    '把"报告"移走',
    "把“报告”移走",
    '把"报告"导出',
    "把“报告”导出",
)

_MULTIPLE_TERMS = (
    "所有文件",
    "全部文件",
    "三个文件",
    "三份文件",
    "两个文件",
    "两份文件",
    "多文件",
)

_COMPOUND_TERMS = (
    "然后",
    "最后",
    "顺便",
    "再把",
    "再将",
    "对比",
    "比较",
    "环比",
    "总结",
    "提取",
    "并写入",
    "并分享",
    "并公开",
)


_OPERATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "locate": (
        r"drive/search",
        r"drive/ls",
        r"\bsearch\b",
        r"\bfind\b",
        r"找一下",
        r"找到",
        r"搜一下",
        r"搜一下|搜索",
        r"在哪",
        r"有没有",
        r"列出",
        r"列一下",
        r"有哪些",
        r"看看.*drive",
        r"看下.*云盘",
    ),
    "read_content": (
        r"docs/cat",
        r"docs/structure",
        r"sheets/get",
        r"正文",
        r"(?:正文|文档|里面|里).{0,12}内容",
        r"哪几个章节",
        r"段落结构",
        r"是多少",
        r"怎么处理",
        r"怎么调",
        r"要点",
        r"待办事项",
        r"读(?:一下|出来)",
        r"写了什么",
        r"配色方案",
        r"完成了什么",
        r"进展和具体任务",
        r"(?:写完|完成后).*确认.*(?:章节|结构|层级)",
        r"(?:金额|比例|收入|评分|nps).{0,10}(?:念|告诉|多少|是什么)",
        r"写的?多少",
    ),
    "read_metadata": (
        r"drive/get",
        r"docs/info",
        r"元数据",
        r"文件信息",
        r"文档信息",
        r"什么类型",
        r"多大",
        r"修改时间",
        r"什么时候修改",
        r"最后改于",
        r"文件\s*id.{0,30}(?:信息|元数据)",
        r"(?:分享)?链接.{0,4}(?:发|给|列|$)",
        r"(?:给我|获取).{0,15}链接",
    ),
    "read_comments": (
        r"comments/(?:list|get|search)",
        r"哪些评论",
        r"未解决的评论",
        r"评论.*内容",
        r"评论说了啥",
    ),
    "replace": (
        r"find-replace",
        r"find_replace",
        r"替换",
        r"改成",
        r"换成",
        r"填成",
    ),
    "edit": (
        r"docs/(?:edit|insert|delete|sed|update|write|clear)",
        r"追加",
        r"插入",
        r"删掉",
        r"删除",
        r"加一段",
        r"写入",
        r"清空",
        r"补一句",
        r"更新一下",
    ),
    "comment": (
        r"comments/(?:add|reply|resolve|delete)",
        r"回复.*评论",
        r"加一条评论",
        r"(?:加|添加|新建|回复|删除|清理).{0,12}评论",
        r"(?<!未)解决.{0,12}评论",
        r"评论里@",
        r"评论都清理",
    ),
    "create": (
        r"docs/create",
        r"drive/mkdir",
        r"新建",
        r"创建",
        r"生成一份",
        r"建一个.*文件夹",
    ),
    "copy": (r"docs/copy", r"drive/copy", r"复制"),
    "export": (
        r"docs/export",
        r"sheets/export",
        r"导出",
        r"导成",
        r"转换成",
    ),
    "download": (r"drive/download", r"download", r"下载", r"下下来"),
    "upload": (r"drive/upload", r"upload", r"上传", r"传上去", r"存到.*drive", r"存到.*云盘"),
    "move": (r"drive/move", r"移到", r"移走", r"放到.*文件夹"),
    "delete": (
        r"drive/delete",
        r"删除.*文件",
        r"文件删了",
        r"(?:删除|清理).{0,12}评论",
        r"评论.{0,12}(?:删除|清理)",
    ),
    "permissions": (
        r"drive/permissions",
        r"谁能访问",
        r"都有谁能访问",
        r"权限",
    ),
    "share": (
        r"drive/share",
        r"分享给",
        r"公开分享",
        r"公开到网上",
        r"加 owner",
    ),
    "aggregate": (
        r"对比",
        r"比较",
        r"哪个.*(?:最低|最高|最多|最少)",
        r"环比",
        r"按.*(?:分组|排序)",
        r"倒序|升序|降序",
        r"算一下",
        r"帮我总结|顺便总结|总结(?:一下|三点|要点|我们的|出)",
        r"核心优势和短板",
        r"提取.+按.+排序",
    ),
}


def _metadata_text(case: CaseRecord) -> str:
    description = str(case.metadata.get("description") or case.metadata.get("benchmark_description") or "")
    tags = case.metadata.get("tags") or case.metadata.get("benchmark_tags") or []
    return " ".join((description, " ".join(str(tag) for tag in tags))).lower()


def _ordered_operations(text: str) -> tuple[str, ...]:
    positions: list[tuple[int, str]] = []
    for operation, patterns in _OPERATION_PATTERNS.items():
        match_positions = [match.start() for pattern in patterns if (match := re.search(pattern, text, re.IGNORECASE))]
        if match_positions:
            positions.append((min(match_positions), operation))
    positions.sort()
    return tuple(operation for _, operation in positions)


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _is_boundary(case: CaseRecord, metadata_text: str) -> tuple[bool, list[str]]:
    query = case.query.lower()
    signals: list[str] = []
    description = str(case.metadata.get("description") or case.metadata.get("benchmark_description") or "").strip()
    negative_benchmark = bool(
        description.startswith("负-")
        or description.startswith("负_")
        or re.search(r"(?:^|[|；; ])负(?:向)?(?:$|[|；; ])", description)
    )
    seed_names = seed_resource_names(case.seed_summary)
    mentioned = quoted_mentions(case.query)
    if negative_benchmark and mentioned and seed_names and not match_seed_resources(case.query, seed_names):
        signals.append("negative benchmark named resource missing from seed")
    boundary_text = f"{description.lower()} {query}"
    for term in _BOUNDARY_TERMS:
        if term in boundary_text:
            signals.append(f"boundary term: {term}")
    for phrase in _VAGUE_QUERIES:
        if phrase.lower() in query:
            signals.append("underspecified mutation")
    if re.search(r"(?:\.ai\s*格式|epub\s*格式|图片文件转换成\s*word)", query, re.IGNORECASE):
        signals.append("unsupported conversion")
    if re.search(r"第\s*(?:999|100\s*到\s*300)\s*(?:号|个字符)", query):
        signals.append("out-of-range target")
    if "文件名带客户名" in query and not re.search(r"客户(?:名)?[：:]\s*[\w\u4e00-\u9fff]+", query):
        signals.append("missing required output name")
    if re.search(r"这份.+(?:存|上传).*(?:drive|云盘)", query, re.IGNORECASE):
        signals.append("missing upload source")
    if re.search(r"只读[^》』\"”']{0,20}[》』\"”']?.{0,24}(?:加|添加|修改|删除|回复)", query):
        signals.append("read-only target mutation")
    if "需admin权限" in metadata_text or "需要管理员权限" in metadata_text:
        signals.append("admin permission required")
    return bool(signals), signals


def _insert_before(operations: list[str], value: str, before: set[str]) -> None:
    if value in operations:
        return
    positions = [operations.index(operation) for operation in before if operation in operations]
    operations.insert(min(positions) if positions else len(operations), value)


def _append_unique(operations: list[str], value: str) -> None:
    if value not in operations:
        operations.append(value)


def _needs_existing_resource(operations: set[str]) -> bool:
    existing = {
        "read_content",
        "read_metadata",
        "read_comments",
        "replace",
        "edit",
        "comment",
        "copy",
        "export",
        "download",
        "move",
        "delete",
        "permissions",
        "share",
        "aggregate",
    }
    if operations <= {"create", "edit", "read_content"} and "create" in operations:
        return False
    if "upload" in operations and operations <= {"upload", "permissions", "share"}:
        return False
    return bool(operations & existing)


def _needs_edit_scope(query: str, operations: set[str]) -> bool:
    if "replace" in operations:
        return bool(
            re.search(r"(?:只|所有|全部|第|这一|整|段|节|行|标题|之间|后面|前面|保持|不要动)", query)
            or re.search(r"(?:里|中的?)\s*[^，。；;]{1,24}\s*(?:改成|换成|填成)", query)
        )
    return bool(re.search(r"(?:删除|删掉|第\s*\d+|负责人|段落|章节|标题|之间|后面|保持.*不变)", query))


def _direct_boundary(signals: list[str]) -> bool:
    direct_terms = (
        "pii",
        "敏感",
        "公开到网上",
        "全员通知",
        "owner",
        "unsupported",
        "underspecified",
        "missing required",
        "missing upload",
        "read-only target",
    )
    return any(any(term in signal.lower() for term in direct_terms) for signal in signals)


def _intent_operations(case: CaseRecord, *, boundary: bool, signals: list[str]) -> list[str]:
    query = case.query.lower()
    operations = list(_ordered_operations(query))
    operation_set = set(operations)
    metadata_text = _metadata_text(case)
    has_destination = (
        bool(destination_resource_names(case.query))
        or bool(re.search(r"(?:放到|移到|复制到).{0,30}文件夹", query))
        or bool("文件夹" in query and re.search(r"(?:新建|创建)", query))
    )

    if "copy" in operation_set and "create" in operation_set and "复制" in query and not has_destination:
        operations.remove("create")
        operation_set.remove("create")
    if (
        "replace" in operation_set
        and "edit" in operation_set
        and not re.search(r"(?:追加|插入|写入|补一句|加一段)", query)
    ):
        operations.remove("edit")
        operation_set.remove("edit")
    if "replace" in operation_set and re.search(r"(?:负责人|deadline\s*段落).{0,30}(?:改成|更新)", query):
        replace_index = operations.index("replace")
        operations[replace_index] = "edit"
        operation_set = set(operations)
    if "审阅备注" in query and "comments/add" in metadata_text:
        # The official benchmark contract treats this note as a Docs comment, not body text.
        operations = [operation for operation in operations if operation != "edit"]
        _append_unique(operations, "comment")
        operation_set = set(operations)
    if "comment" in operation_set and re.search(r"(?:已解决|删除|清理|回复|第\s*\d+\s*号)", query):
        _insert_before(operations, "read_comments", {"comment"})
    if "delete" in operation_set and "评论" in query:
        operations = [operation for operation in operations if operation not in {"edit", "comment"}]
        _insert_before(operations, "read_comments", {"delete"})
    if "share" in operation_set:
        _insert_before(operations, "permissions", {"share"})
    if case.domain == "drive" and "export" in operation_set:
        export_index = operations.index("export")
        if "download" not in operations:
            operations.insert(export_index + 1, "download")
    if (
        case.domain == "drive"
        and "download" in operation_set
        and re.search(r"(?:pdf|xlsx|excel|docx|txt|md|html)\s*(?:文件|格式)", query, re.IGNORECASE)
    ):
        _insert_before(operations, "export", {"download"})
    if "copy" in operation_set and "create" in operation_set and has_destination:
        operations = [operation for operation in operations if operation != "move"]
        operations.remove("create")
        operations.insert(operations.index("copy"), "create")
    operation_set = set(operations)

    if "read_comments" in operation_set and "read_content" in operation_set and not re.search(r"(?:原文|引用)", query):
        operations.remove("read_content")
        operation_set.remove("read_content")
    if "read_comments" in operation_set and re.search(r"(?:原文|引用)", query):
        _insert_before(operations, "read_content", {"read_comments"})
    if (
        "aggregate" not in operation_set
        and re.search(r"(?:对比计算|计算)", metadata_text)
        and re.search(r"(?:收入|金额|比例|率|总计|合计|完成率)", query)
    ):
        _append_unique(operations, "aggregate")
        operation_set.add("aggregate")
    analytical_summary = bool(
        re.search(
            r"(?:对比|比较|环比|计算|算一下|分组|排序|最低|最高|最多|最少|"
            r"核心优势和短板|提取.+按.+排序)",
            query,
        )
    )
    artifact_operations = {"create", "copy", "export", "download", "upload", "share"}
    explicit_sources = match_seed_resources(case.query, seed_resource_names(case.seed_summary))
    if (
        "aggregate" in operation_set
        and "总结" in query
        and not analytical_summary
        and not operation_set & artifact_operations
        and len(explicit_sources) <= 1
        and len(quoted_mentions(case.query)) <= 1
    ):
        # Summarizing one document is a read/answer task, not a multi-source aggregation.
        operations.remove("aggregate")
        operation_set.remove("aggregate")
    if "aggregate" in operation_set and not operation_set & {"read_content", "read_metadata"}:
        evidence_operation = (
            "read_metadata"
            if re.search(r"(?:mime|修改时间|文件信息|大小|类型)", query, re.IGNORECASE)
            else "read_content"
        )
        _insert_before(operations, evidence_operation, {"aggregate"})
    if "提取" in query and re.search(r"(?:里面|其中|从.+里).{0,40}提取", query):
        _insert_before(operations, "read_content", {"create", "edit", "aggregate"})
    if "aggregate" in operations:
        evidence = [operation for operation in operations if operation in {"read_content", "read_metadata"}]
        for operation in evidence:
            operations.remove(operation)
        aggregate_index = operations.index("aggregate")
        for offset, operation in enumerate(evidence):
            operations.insert(aggregate_index + offset, operation)
    operation_set = set(operations)
    if operation_set & {"replace", "edit"} and re.search(r"(?:执行摘要|根据.+(?:摘要|总结))", query):
        _insert_before(operations, "read_content", {"replace", "edit"})
    if (
        operation_set & {"replace", "edit"}
        and "create" not in operation_set
        and _needs_edit_scope(query, operation_set)
    ):
        _insert_before(operations, "read_content", {"replace", "edit"})
    if "create" not in operations and "read_content" in operations and set(operations) & {"replace", "edit"}:
        read_index = operations.index("read_content")
        action_index = min(operations.index(operation) for operation in ("replace", "edit") if operation in operations)
        if read_index > action_index:
            operations.pop(read_index)
            operations.insert(action_index, "read_content")
    if _needs_existing_resource(set(operations)):
        _insert_before(operations, "locate", set(operations))

    if not operations and not case.query.strip():
        metadata_operations = list(_ordered_operations(_metadata_text(case)))
        if metadata_operations:
            operations.append(metadata_operations[-1])

    if not boundary:
        return operations
    if "只读" in query and "permissions" not in operations:
        _insert_before(operations, "permissions", {"comment", "replace", "edit"})
    if "指向" in query and "read_content" not in operations:
        _insert_before(operations, "read_content", {"establish_boundary"})
        _insert_before(operations, "locate", {"read_content"})
    if _direct_boundary(signals):
        # A safety/permission boundary changes the outcome, not the requested
        # operation graph. Retain observable read and operation components so
        # safe partial progress can still be represented; authorization guards
        # prevent an unsafe mutation from receiving action credit.
        if "missing upload source" in signals:
            return ["establish_boundary"]
        _append_unique(operations, "establish_boundary")
        return operations
    if re.search(r"(?:根本)?不存在的(?:文档|文件)", query):
        return ["locate", "establish_boundary"]

    evidence = [
        operation
        for operation in operations
        if operation in {"locate", "read_content", "read_metadata", "read_comments", "permissions"}
    ]
    if not evidence and set(operations) & {"replace", "edit"}:
        evidence = ["locate", "read_content"]
    elif set(operations) & {"replace", "edit"} and "read_content" not in evidence:
        _insert_before(evidence, "read_content", {"establish_boundary"})
    if not evidence and "comment" in operations:
        evidence = ["locate", "read_comments"]
    elif "comment" in operations and not set(evidence) & {"read_comments", "permissions"}:
        _append_unique(evidence, "read_comments")
    if not evidence and set(operations) & {"delete", "move", "copy", "export", "download"}:
        evidence = ["locate"]
    _append_unique(evidence, "establish_boundary")
    return evidence


def _primary_operation(family: str, operations: list[str]) -> str:
    preferences = {
        "docs.read": ("read_comments", "read_metadata", "read_content", "locate"),
        "docs.replace_or_edit": ("replace", "edit"),
        "docs.comments": ("comment", "delete", "read_comments"),
        "docs.create_or_export": ("create", "copy", "export", "download", "upload"),
        "drive.retrieve": ("read_comments", "read_metadata", "read_content", "aggregate", "locate"),
        "drive.file_operation": ("create", "copy", "move", "upload", "export", "download", "delete"),
        "drive.transfer_or_share": ("share", "permissions"),
        "multi_source_aggregate": ("aggregate",),
        "boundary_or_negative": ("establish_boundary",),
    }
    for preferred in preferences[family]:
        if preferred in operations:
            return preferred
    return operations[-1]


def classify_case(case: CaseRecord) -> CaseClassification:
    """Classify one case using domain + operation + outcome + cardinality + complexity."""

    query = case.query.lower()
    metadata_text = _metadata_text(case)
    boundary, signals = _is_boundary(case, metadata_text)
    operations = _intent_operations(case, boundary=boundary, signals=signals)
    operation_set = set(operations)
    cardinality = infer_resource_cardinality(case)
    if _contains_any(query, _MULTIPLE_TERMS):
        cardinality = "multiple"

    mutating = {"replace", "edit", "comment", "create", "copy", "upload", "move", "delete", "share"}
    artifact = {"create", "copy", "export", "download", "upload"}
    reading = {"locate", "read_content", "read_metadata", "read_comments", "permissions", "aggregate"}
    if boundary:
        outcome = "clarify_or_refuse"
    elif operation_set & mutating and operation_set & reading:
        outcome = "mixed"
    elif operation_set & artifact:
        outcome = "artifact"
    elif operation_set & mutating:
        outcome = "mutation"
    else:
        outcome = "read"

    is_aggregate = "aggregate" in operation_set and bool(
        re.search(
            r"(?:环比|对比|比较|最低|最高|最多|最少|分组|排序|倒序|升序|降序|目标完成率|帮我总结|概述|归纳|提炼)",
            query,
        )
    )
    if boundary:
        family = "boundary_or_negative"
    elif operation_set & {"comment", "read_comments"} and not operation_set & {
        "create",
        "copy",
        "export",
        "download",
        "upload",
        "move",
        "share",
    }:
        family = "docs.comments"
    elif is_aggregate and not operation_set & {"create", "copy", "export", "download", "upload"}:
        family = "multi_source_aggregate"
    elif case.domain == "docs":
        if operation_set & {"create", "copy", "export", "download", "upload"}:
            family = "docs.create_or_export"
        elif operation_set & {"replace", "edit"}:
            family = "docs.replace_or_edit"
        else:
            family = "docs.read"
    elif "create" in operation_set and re.search(
        r"(?:新建|创建|生成).{0,30}(?:文档|docx?)", query, re.IGNORECASE
    ):
        family = "docs.create_or_export"
    else:
        if operation_set & {"share", "permissions"}:
            family = "drive.transfer_or_share"
        elif operation_set & {"create", "copy", "export", "download", "upload", "move", "delete"}:
            family = "drive.file_operation"
        else:
            family = "drive.retrieve"

    if not operations:
        operations = [
            {
                "docs.read": "read_content",
                "docs.replace_or_edit": "edit",
                "docs.comments": "read_comments",
                "docs.create_or_export": "export",
                "drive.retrieve": "locate",
                "drive.file_operation": "copy",
                "drive.transfer_or_share": "permissions",
                "multi_source_aggregate": "aggregate",
                "boundary_or_negative": "establish_boundary",
            }[family]
        ]
        signals.append("family default operation")

    compound = len(operations) > 1 or _contains_any(query, _COMPOUND_TERMS)
    if boundary:
        complexity = "boundary"
    elif compound:
        complexity = "compound"
    else:
        complexity = "atomic"

    signals.extend(
        (
            f"domain: {case.domain}",
            f"operations: {','.join(operations)}",
        )
    )
    return CaseClassification(
        family=family,
        domain=case.domain,
        operation=_primary_operation(family, operations),
        outcome=outcome,
        cardinality=cardinality,
        complexity=complexity,
        expected_operations=tuple(operations),
        signals=tuple(dict.fromkeys(signals)),
    )
