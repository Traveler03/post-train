"""Resolve concrete benchmark resources from queries and offline seed summaries."""

from __future__ import annotations

import re
from typing import Any

from .models import CaseClassification, CaseRecord, JsonObject

_QUOTED = (
    re.compile(r"《([^》]{1,120})》"),
    re.compile(r"『([^』]{1,120})』"),
    re.compile(r"[\"“]([^\"”]{1,120})[\"”]"),
    re.compile(r"'([^']{1,120})'"),
)
_TITLE_QUOTED = (
    re.compile(r"《([^》]{1,120})》"),
    re.compile(r"『([^』]{1,120})』"),
)
_FILE_ID = re.compile(r"(?:文件\s*ID|file\s*id)\s*[:：]?\s*([A-Za-z0-9_-]{5,})", re.IGNORECASE)
_FILE_NAME = re.compile(
    r"(?<![\w.-])([A-Za-z0-9_\-\u4e00-\u9fff][A-Za-z0-9_\-.\u4e00-\u9fff]{0,119}"
    r"\.(?:docx?|xlsx?|pptx?|pdf|txt|md|markdown|html?|csv))(?![\w.-])",
    re.IGNORECASE,
)
_COLLECTION = re.compile(
    r"(?:所有(?:的)?(?:文件|文档|资源)|全部(?:文件|文档|资源)|(?:有哪些|都有哪些|都有什么|列出|列一下|找出).{0,16}(?:文件|文档|资源)|"
    r"找一下所有|三个(?:文件|文档|资源|报价|合同)|三份(?:文件|文档|资源|报价|合同)|两个文件|这两个文件|"
    r"\d+\s*个[^，。；;]{0,20}文件|近\s*\d+\s*天.*文件|上周.*(?:文件|周报)|有没有.*文件|"
    r"相关的文件|(?:所有|全部)\s*(?:pdf|docx?|xlsx?|pptx?|txt|csv)(?:\s*(?:类型的)?文件)?|"
    r"(?:云盘|drive|根目录).{0,16}(?:都有什么|有哪些)|PDF\s*(?:类型的)?文件|多人)",
    re.IGNORECASE,
)
_DYNAMIC_COLLECTION = re.compile(
    r"(?:所有(?:的)?(?:文件|文档|资源)|全部(?:文件|文档|资源)|"
    r"(?:有哪些|都有哪些|都有什么|列出|列一下|找出).{0,16}(?:文件|文档|资源)|找一下所有|"
    r"近\s*\d+\s*天|上周|有没有.*文件|相关的文件|名字(?:里)?带|名称(?:里)?带|"
    r"(?:所有|全部)\s*(?:pdf|docx?|xlsx?|pptx?|txt|csv)(?:\s*(?:类型的)?文件)?|"
    r"(?:云盘|drive|根目录).{0,16}(?:都有什么|有哪些)|PDF\s*(?:类型的)?文件)",
    re.IGNORECASE,
)
_NON_RESOURCE = re.compile(
    r"^(?:已完成|待定|本公司|todo|收到，?已处理|reader|writer|review)$",
    re.IGNORECASE,
)
_OUTPUT_VERB = re.compile(
    r"(?:生成一份|新建(?:一个)?(?:文档)?|创建(?:一个)?(?:文档)?|建(?:一个)?|改名(?:为)?|叫|名为)"
)
_DESTINATION_VERB = re.compile(r"(?:复制到|移动到|移到|放到|放进|文件夹(?:下|里))")
_DOMAIN = re.compile(r"(?<![\w.-])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![\w.-])")
_GENERIC_FRAGMENTS = {"文档", "文件", "报告", "模板", "数据", "内容", "计划"}
_SEMANTIC_ALIASES = {
    "nda": ("nda", "保密协议", "non-disclosure", "nondisclosure"),
}
_FORMAT_ALIASES = {
    "excel": "xlsx",
    "htm": "html",
    "html": "html",
    "markdown": "md",
    "md": "md",
    "pdf": "pdf",
    "纯文本": "txt",
    "txt": "txt",
    "word": "docx",
    "doc": "docx",
    "docx": "docx",
    "xlsx": "xlsx",
}


def _resource_name(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("filename") or value.get("name") or value.get("title") or "").strip()


def seed_resource_names(seed_summary: JsonObject) -> list[str]:
    names: list[str] = []
    for key, value in seed_summary.items():
        if isinstance(value, list):
            for item in value:
                name = _resource_name(item)
                if name:
                    names.append(name)
        elif isinstance(value, dict):
            name = _resource_name(value)
            if name:
                names.append(name)
        if key.endswith("_files") and isinstance(value, dict):
            names.extend(str(item) for item in value if str(item).strip())
    return list(dict.fromkeys(names))[:100]


def normalize_resource(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"\.(?:docx?|xlsx?|pptx?|pdf|txt|md|html?)$", "", lowered)
    return re.sub(r"[\s_\-《》『』\"'“”()（）【】]+", "", lowered)


def _seed_aliases(name: str) -> list[str]:
    base = re.sub(r"\.(?:docx?|xlsx?|pptx?|pdf|txt|md|html?)$", "", name, flags=re.IGNORECASE)
    return [base] if len(normalize_resource(base)) >= 3 else []


def _distinctive_fragment_score(query: str, seed_name: str) -> int:
    normalized_query = normalize_resource(query)
    normalized_seed = normalize_resource(seed_name)
    for size in range(min(len(normalized_seed), 12), 2, -1):
        for start in range(0, len(normalized_seed) - size + 1):
            fragment = normalized_seed[start : start + size]
            if fragment in _GENERIC_FRAGMENTS or fragment.isdigit():
                continue
            if fragment in normalized_query:
                return size
    base = re.sub(r"\.(?:docx?|xlsx?|pptx?|pdf|txt|md|html?)$", "", seed_name, flags=re.IGNORECASE)
    tokens = [normalize_resource(token) for token in re.split(r"[_\-\s]+", base)]
    return max(
        (len(token) for token in tokens if re.search(r"\d", token) and len(token) >= 2 and token in normalized_query),
        default=0,
    )


def quoted_mentions(query: str) -> list[str]:
    mentions = [match.group(1).strip() for pattern in _QUOTED for match in pattern.finditer(query)]
    return [value for value in dict.fromkeys(mentions) if value and not _NON_RESOURCE.search(value)]


def explicit_file_names(query: str) -> list[str]:
    return list(dict.fromkeys(match.group(1).strip() for match in _FILE_NAME.finditer(query)))


def _semantic_alias_matches(query: str, seed_names: list[str]) -> list[str]:
    lowered = query.lower()
    aliases = {
        alias
        for trigger, values in _SEMANTIC_ALIASES.items()
        if trigger in lowered
        for alias in values
    }
    if not aliases:
        return []
    return [
        seed_name
        for seed_name in seed_names
        if any(normalize_resource(alias) in normalize_resource(seed_name) for alias in aliases)
    ]


def match_seed_resources(query: str, seed_names: list[str]) -> list[str]:
    normalized_query = normalize_resource(query)
    mentions = [normalize_resource(value) for value in quoted_mentions(query)]
    mention_matches: list[str] = []
    for seed_name in seed_names:
        normalized_seed = normalize_resource(seed_name)
        if any(mention and (mention in normalized_seed or normalized_seed in mention) for mention in mentions):
            mention_matches.append(seed_name)
    if mention_matches:
        return list(dict.fromkeys(mention_matches))

    semantic_matches = _semantic_alias_matches(query, seed_names)
    if semantic_matches:
        return list(dict.fromkeys(semantic_matches))

    alias_matches = [
        seed_name
        for seed_name in seed_names
        if any(normalize_resource(alias) in normalized_query for alias in _seed_aliases(seed_name))
    ]
    if alias_matches:
        return list(dict.fromkeys(alias_matches))

    scored: list[tuple[int, str]] = []
    for seed_name in seed_names:
        score = _distinctive_fragment_score(query, seed_name)
        if score:
            scored.append((score, seed_name))
    if not scored:
        return []
    if re.search(r"(?:对比|比较|分别|各自|和.+(?:报告|纪要|文档|文件))", query):
        return [seed_name for score, seed_name in scored if score >= 3]
    best = max(score for score, _ in scored)
    return [seed_name for score, seed_name in scored if score == best]


def infer_resource_cardinality(case: CaseRecord) -> str:
    seed_matches = match_seed_resources(case.query, seed_resource_names(case.seed_summary))
    if len(seed_matches) > 1:
        return "multiple"
    title_mentions = [match.group(1).strip() for pattern in _TITLE_QUOTED for match in pattern.finditer(case.query)]
    if len(set(title_mentions)) > 1:
        return "multiple"
    if len(set(title_mentions)) == 1 and re.search(r"(?:段落结构|章节|评论)", case.query):
        return "single"
    if re.search(
        r"(?:(?:共享云端硬盘|shared\s+drive).*(?:哪些|列|所有)|"
        r"(?:哪些|列|所有).*(?:共享云端硬盘|shared\s+drive))",
        case.query,
        re.IGNORECASE,
    ):
        return "multiple"
    return "multiple" if _COLLECTION.search(case.query) else "single"


def is_dynamic_resource_set(query: str) -> bool:
    return bool(_DYNAMIC_COLLECTION.search(query))


def requested_resources(case: CaseRecord, classification: CaseClassification) -> list[str]:
    seed_names = seed_resource_names(case.seed_summary)
    matched_seed_names = match_seed_resources(case.query, seed_names)
    identifiers = [match.group(1) for match in _FILE_ID.finditer(case.query)]
    mentions = quoted_mentions(case.query)

    if classification.cardinality == "multiple" and is_dynamic_resource_set(case.query):
        return ["<requested_resource_set>"]

    resources = matched_seed_names or identifiers or mentions
    resources = list(dict.fromkeys(resources))[:8]
    if classification.cardinality == "multiple" and len(resources) > 1:
        return resources
    if resources:
        return resources[:1]
    return ["<requested_resource_set>"]


def _mentions_after(query: str, pattern: re.Pattern[str]) -> list[str]:
    values: list[str] = []
    for match in pattern.finditer(query):
        values.extend(quoted_mentions(query[match.end() : match.end() + 120])[:1])
    return list(dict.fromkeys(values))


def output_resource_names(query: str) -> list[str]:
    resources = _mentions_after(query, _OUTPUT_VERB)
    if re.search(r"(?:上传|upload).{0,100}(?:文件|txt|docx?|xlsx?|pdf|csv)", query, re.IGNORECASE):
        resources.extend(explicit_file_names(query))
    return list(dict.fromkeys(resources))[:12]


def _named_folders(query: str) -> list[str]:
    folders: list[str] = []
    for pattern in _QUOTED:
        for match in pattern.finditer(query):
            before = query[max(0, match.start() - 30) : match.start()]
            after = query[match.end() : match.end() + 20]
            if re.search(r"(?:新建|创建|建一个|目标|放到|移到|复制到)[^，。；;]{0,20}$", before) or re.match(
                r"\s*(?:文件夹|目录)", after
            ):
                folders.append(match.group(1).strip())
    return list(dict.fromkeys(folders))


def destination_resource_names(query: str) -> list[str]:
    resources = _mentions_after(query, _DESTINATION_VERB)
    if re.search(r"(?:放进去|往里面|放到|移到|移动到|复制到|上传)", query):
        resources.extend(_named_folders(query))
    return list(dict.fromkeys(resources))[:8]


def share_targets(query: str) -> list[str]:
    if not re.search(r"(?:分享|共享|权限|访问者|谁能访问|公开)", query):
        return []
    targets = [match.group(0) for match in _DOMAIN.finditer(query)]
    match = re.search(r"(?:分享|共享)给\s*([^，。；;]+)", query)
    if match:
        targets.append(match.group(1).strip())
    return list(dict.fromkeys(targets))[:8]


def comment_targets(query: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(r"第\s*(\d+)\s*号评论", query, re.IGNORECASE)
        )
    )


def requested_output_format(query: str) -> str | None:
    if not re.search(r"(?:导出|导成|转换成|下载)", query, re.IGNORECASE):
        return None
    lowered = query.lower()
    for raw, normalized in _FORMAT_ALIASES.items():
        if re.search(rf"(?<![\w]){re.escape(raw)}(?![\w])", lowered, re.IGNORECASE):
            return normalized
    if "网页" in query:
        return "html"
    match = re.search(r"(?:导出|导成|转换成).{0,20}?(?:\.\s*)?([a-z][a-z0-9]{1,8})\s*格式", query, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def search_constraints(query: str) -> JsonObject:
    constraints: JsonObject = {}
    match = re.search(r"名字(?:里)?带\s*[\"“'『]?([^\"”'』，。；;]+)", query, re.IGNORECASE)
    if match:
        constraints["name_contains"] = match.group(1).strip()
    match = re.search(r"跟\s*[《『\"“']([^》』\"”']+)[》』\"”']\s*相关", query)
    if match:
        constraints["semantic_query"] = match.group(1).strip()
    if "semantic_query" not in constraints:
        match = re.search(r"(?:和|与)\s*([^，。；;？?]{1,40}?)\s*相关的?(?:文件|文档|资源)", query)
        if match:
            constraints["semantic_query"] = match.group(1).strip()
    if re.search(r"(?:找|搜|列|所有|哪些|查询)", query):
        match = re.search(r"(pdf|docx?|xlsx?|pptx?|txt|csv)\s*(?:类型的)?文件", query, re.IGNORECASE)
        if not match and re.search(r"(?:所有|全部|列出|列一下|找出)", query):
            match = re.search(r"(?<![A-Za-z0-9])(pdf|docx?|xlsx?|pptx?|txt|csv)(?![A-Za-z0-9])", query, re.IGNORECASE)
        if match:
            constraints["file_type"] = match.group(1).lower()
    match = re.search(r"(\d{1,2}/\d{1,2}\s*[~～至到-]\s*(?:\d{1,2}/)?\d{1,2})", query)
    if match:
        constraints["date_range"] = match.group(1).replace(" ", "")
    if "上周" in query:
        constraints["relative_time"] = "上周"
    if "周报" in query and ({"date_range", "relative_time"} & constraints.keys()):
        constraints.setdefault("name_contains", "周报")
    match = re.search(r"近\s*(\d+)\s*天(?:修改|改过|更新)", query)
    if match:
        constraints["modified_within_days"] = int(match.group(1))
    if "根目录" in query:
        constraints["parent_scope"] = "drive_root"
    if "最近" in query:
        constraints["sort_by"] = "modified_time"
        constraints["sort_order"] = "descending"
    if re.search(r"(?:共享云端硬盘|shared\s+drive)", query, re.IGNORECASE):
        constraints["resource_type"] = "shared_drive"
        if re.search(r"(?:我能访问|可访问)", query):
            constraints["access_scope"] = "current_user_accessible"
    return constraints


def requested_view_constraints(query: str) -> JsonObject:
    constraints: JsonObject = {}
    match = re.search(
        r"(?:里|中的?)\s*[\"“'『]([^\"”'』]{1,80})[\"”'』]\s*(?:这一|该)?(?:节|章节|段)",
        query,
    )
    if match:
        constraints["section"] = match.group(1).strip()
    if re.search(r"(?:最前面|开头)", query):
        constraints["position"] = "document_start"
    elif "文末" in query:
        constraints["position"] = "document_end"
    if re.search(r"(?:段落|章节)结构", query):
        constraints["view"] = "document_structure"
    if "用中文" in query:
        constraints["response_language"] = "zh-CN"
    if "执行摘要" in query:
        constraints.pop("position", None)
        constraints["scope"] = "full_document"
        constraints["purpose"] = "derive_summary"
    return constraints


_EDIT_LITERAL = re.compile(
    r"(\{\{[^{}]{1,80}\}\}|\[\[[^\[\]]{1,80}\]\]|「[^」]{1,160}」|"
    r"『[^』]{1,160}』|《[^》]{1,160}》|[\"“][^\"”]{1,160}[\"”]|'[^']{1,160}'|"
    r"<URL(?:_\d+)?>)",
    re.IGNORECASE,
)


def _unquote_edit_literal(value: str) -> str:
    pairs = (("「", "」"), ("『", "』"), ("《", "》"), ('"', '"'), ("“", "”"), ("'", "'"))
    for left, right in pairs:
        if value.startswith(left) and value.endswith(right):
            return value[len(left) : len(value) - len(right)]
    return value


def edit_constraints(query: str) -> JsonObject:
    constraints = requested_view_constraints(query)
    if re.search(r"(?:第一个|第一处)", query):
        constraints["occurrence"] = "first"
    if re.search(r"第二个.{0,30}(?:保留|不动)", query):
        constraints["preserve_occurrences"] = ["second"]
    match = re.search(r"后面\s*([两二三四五六七八九\d]+)\s*个.{0,20}(?:保留|不变)", query)
    if match:
        constraints["preserve_remaining_matches"] = match.group(1)
    if re.search(r"(?:正文|下面).{0,20}(?:不要动|不变)", query):
        constraints["preserve"] = ["body"]
    replacements: list[JsonObject] = []
    action_pattern = re.compile(r"(?:替换为|替换成|改成|换成|填成)")
    for action in action_pattern.finditer(query):
        before = query[max(0, action.start() - 100) : action.start()]
        after = query[action.end() : action.end() + 180].lstrip(" ：:")
        old_literals = list(_EDIT_LITERAL.finditer(before))
        new_literal = _EDIT_LITERAL.match(after)
        field = re.search(
            r"(?:里|中的?)\s*(?:的)?([A-Za-z_\u4e00-\u9fff]{2,30})(?:\s*段落)?\s*$",
            before,
        )
        explicit_case_target = re.search(r"大写的\s*([A-Za-z0-9_\-]+).{0,20}$", before)
        if explicit_case_target:
            old_value = explicit_case_target.group(1)
        elif field:
            old_value = field.group(1).strip()
        elif old_literals:
            old_value = _unquote_edit_literal(old_literals[-1].group(0))
        elif re.search(r"deadline", before, re.IGNORECASE):
            old_value = "deadline paragraph"
        else:
            old_value = ""
        if new_literal:
            new_value = _unquote_edit_literal(new_literal.group(0))
        elif re.search(r"^\s*[：:]", query[action.end() : action.end() + 3]) and re.search(
            r"。\s*其他(?:章节|内容)", after
        ):
            new_value = re.split(r"。\s*其他(?:章节|内容)", after, maxsplit=1)[0].strip()
        else:
            new_value = re.split(r"[，。；;、（）()]", after, maxsplit=1)[0].strip()
        if old_value and new_value:
            replacements.append({"from": old_value, "to": new_value})
    if replacements:
        constraints["replacements"] = replacements
    if re.search(r"负责人.{0,20}(?:改成|设为)我", query):
        constraints.pop("replacements", None)
        constraints["field_updates"] = [{"field": "负责人", "value": "current_user"}]
    deadline_update = re.search(r"deadline\s*段落.{0,30}改成\s*([^，。；;]+)", query, re.IGNORECASE)
    if deadline_update:
        constraints.pop("replacements", None)
        constraints["range_update"] = {
            "selector": {"paragraph_label": "deadline"},
            "new_content": deadline_update.group(1).strip(),
        }
    section_update = re.search(
        r"[「『\"“']([^」』\"”']+)[」』\"”']这一节的内容整段(?:换成|改成)[：:]\s*(.+?)。\s*其他章节",
        query,
    )
    if section_update:
        constraints.pop("replacements", None)
        constraints["section_update"] = {
            "section": section_update.group(1).strip(),
            "preserve_heading": True,
            "new_content": section_update.group(2).strip(),
        }
    for replacement in constraints.get("replacements") or []:
        if replacement.get("to") == "空" and re.search(r"(?:替换成空|换成空)", query):
            replacement["to"] = ""
    append_match = re.search(
        r"文末.{0,40}(?:追加|添加|加|补).{0,40}[：:]\s*[\"“'『]([^\"”'』]+)[\"”'』]",
        query,
    )
    if append_match:
        constraints["append_text"] = append_match.group(1).strip()
    match = re.search(r"(\d+)\s*字", query)
    if match:
        constraints["length_chars"] = int(match.group(1))
    match = re.search(r"第\s*(\d+)\s*到\s*(\d+)\s*个字符", query)
    if match:
        constraints["character_range"] = {"start": int(match.group(1)), "end": int(match.group(2))}
    if re.search(r"(?:所有|全部).{0,20}(?:替换|改成|换成)", query):
        constraints["replace_all"] = True
    if re.search(r"(?:其它|其他|正文|后面).{0,24}(?:不要动|不变|保持)", query):
        constraints.setdefault("preserve", []).append("all_unrequested_content")
    if re.search(r"大写", query) and re.search(r"小写.{0,20}(?:保持|不要动|不变)", query):
        constraints["case_sensitive"] = True
    if "markdown" in query.lower():
        constraints["content_format"] = "markdown"
    if re.search(r"小标题", query):
        constraints["required_heading_count"] = 1
    match = re.search(r"([一二两三四五六七八九十\d]+)\s*条要点", query)
    if match:
        chinese_counts = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        raw_count = match.group(1)
        constraints["required_bullet_count"] = chinese_counts.get(
            raw_count, int(raw_count) if raw_count.isdigit() else raw_count
        )
    anchors = [_unquote_edit_literal(match.group(0)) for match in _EDIT_LITERAL.finditer(query)]
    if "之间" in query and len(anchors) >= 2:
        constraints["between_anchors"] = anchors[-2:]
    return constraints


def content_constraints(query: str) -> JsonObject:
    constraints: JsonObject = {}
    match = re.search(r"(?:上传|创建|新建)[^，。；;]{0,80}?(\d+)\s*个[^，。；;]{0,20}文件", query)
    if match:
        constraints["file_count"] = int(match.group(1))
    match = re.search(r"内容(?:都)?是[^\"“'『]{0,20}[\"“'『]([^\"”'』]+)[\"”'』]", query)
    if match:
        constraints["literal_content"] = match.group(1)
    if "单行" in query:
        constraints["line_count"] = 1
    if re.search(r"每个.+(?:单独|作为).{0,10}章节", query):
        constraints["section_per_item"] = True
    match = re.search(r"\d+\s*个章节[：:]\s*([^；;。]+)", query)
    if match:
        constraints["required_sections"] = [
            item.strip() for item in re.split(r"[、,，]", match.group(1)) if item.strip()
        ]
    if re.search(r"(?:章节层级|层级结构).{0,12}(?:正确|确认|检查)", query):
        constraints["verify_section_hierarchy"] = True
    if re.search(r"(?:无分页|pageless)", query, re.IGNORECASE):
        constraints["document_mode"] = "pageless"
    match = re.search(r"[（(]([^（）()]{2,120})[）)].{0,30}每个.+章节", query)
    if match:
        constraints["required_items"] = [item.strip() for item in re.split(r"[、,，]", match.group(1)) if item.strip()]
    return constraints


def comment_constraints(query: str) -> JsonObject:
    constraints: JsonObject = {}
    if "未解决" in query:
        constraints["status_filter"] = "unresolved"
    elif "已解决" in query:
        constraints["status_filter"] = "resolved"
    if re.search(r"(?:删除|清理)", query):
        constraints["operation"] = "delete"
    elif re.search(r"回复", query):
        constraints["operation"] = "reply"
    elif re.search(r"(?:加|添加|新建).{0,12}评论", query):
        constraints["operation"] = "add"
    if constraints.get("operation") in {"add", "reply"}:
        match = re.search(r"(?:评论[：:]|评论说|回复.+?说)\s*([\"“'『])(.+?)[\"”'』]", query)
        if match:
            constraints["comment_text"] = match.group(2).strip()
        else:
            match = re.search(r"(?:评论[：:]|评论说|回复.+?说)\s*([^，。]+)", query)
            if match:
                constraints["comment_text"] = match.group(1).strip()
    if "文末" in query:
        constraints["position"] = "document_end"
    return constraints


def share_constraints(query: str) -> JsonObject:
    constraints: JsonObject = {}
    match = re.search(r"(?:留言|附言)[：:]\s*([^，。；;]+)", query)
    if match:
        constraints["message"] = match.group(1).strip()
    role_pairs = re.findall(r"([A-Za-z0-9_@.\-]+)\s*[（(](reader|writer|owner)[）)]", query, re.IGNORECASE)
    if role_pairs:
        constraints["recipient_roles"] = [
            {"recipient": recipient, "role": role.lower()} for recipient, role in role_pairs
        ]
    return constraints


def indirect_target_names(query: str) -> list[str]:
    match = re.search(r"指向的\s*[《『\"“']([^》』\"”']+)[》』\"”']", query)
    return [match.group(1).strip()] if match else []
