"""Tool evidence normalization, causal pruning, and experience-card extraction."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from typing import Any

from .authorization import EXTERNAL_MUTATION_TOOL_PATTERNS
from .models import CaseClassification, JsonObject, RolloutRecord, ToolEvent

_WRAPPED_TOOL = re.compile(r"^(?:execute_tool|proxy_tool)\s*\[(.+)]$", re.IGNORECASE)
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,}(?!\w)")
_URL = re.compile(r"(?:https?|migoo-file)://[^\s\"'<>]+")
_RESOURCE_ID = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])")

_META_TOOLS = {
    "agent",
    "bash",
    "browser_close",
    "browser_navigate",
    "browser_snapshot",
    "browser_wait_for",
    "connector_list",
    "data_agent",
    "device_permission_check",
    "edit",
    "file_read",
    "file_read_logger",
    "grep",
    "image_agent",
    "load_connector",
    "load_google_tools",
    "load_skill",
    "load_skill_resource",
    "memory_save",
    "memory_search",
    "merged_tools",
    "office_agent",
    "read",
    "research_agent",
    "roadrunner_activate",
    "sandbox_agent",
    "set_model_response",
    "skill",
    "temporal_resolve",
    "tool_search",
    "transfer_to_agent",
    "video_agent",
    "web_search",
    "webfetch",
    "write",
}

_ALIASES = {
    "gdrive_upload": "google_drive_upload",
    "google_docs_find-replace": "google_docs_find_replace",
    "merged tools": "merged_tools",
    "(merged tools)": "merged_tools",
}

_FAMILY_EVIDENCE: dict[str, tuple[str, ...]] = {
    "docs.read": (
        "google_drive_search",
        "google_drive_get",
        "google_docs_cat",
        "google_docs_info",
        "google_docs_structure",
    ),
    "docs.replace_or_edit": (
        "google_docs_edit",
        "google_docs_find_replace",
        "google_docs_insert",
        "google_docs_delete",
        "google_docs_sed",
        "google_docs_update",
        "google_docs_write",
        "google_docs_clear",
    ),
    "docs.comments": (
        "google_docs_comments_*",
        "google_docs_explicit_comments",
        "google_docs_search_comments",
    ),
    "docs.create_or_export": (
        "google_docs_create",
        "google_docs_export",
        "google_drive_copy",
        "google_drive_download",
        "google_drive_upload",
        "download_file",
        "upload_file",
        "create_pdf_from_docx",
    ),
    "drive.retrieve": (
        "google_drive_search",
        "google_drive_ls",
        "google_drive_get",
        "google_drive_url",
        "google_drive_drives",
        "google_docs_cat",
        "google_sheets_get",
        "google_sheets_metadata",
        "google_drive_download",
        "download_file",
    ),
    "drive.file_operation": (
        "google_drive_copy",
        "google_drive_delete",
        "google_drive_download",
        "google_drive_mkdir",
        "google_drive_move",
        "google_drive_upload",
        "google_docs_export",
        "google_sheets_export",
        "google_slides_export",
        "download_file",
        "upload_file",
    ),
    "drive.transfer_or_share": (
        "google_drive_permissions",
        "google_drive_share",
        "google_drive_url",
    ),
    "multi_source_aggregate": (
        "google_drive_search",
        "google_drive_get",
        "google_docs_cat",
        "google_sheets_get",
        "google_sheets_metadata",
        "google_docs_create",
        "google_docs_write",
        "google_drive_share",
    ),
    "boundary_or_negative": (),
}


def normalize_tool_name(raw_name: Any) -> str:
    name = str(raw_name or "").strip()
    match = _WRAPPED_TOOL.match(name)
    if match:
        name = match.group(1).strip()
    name = name.lstrip("@").rstrip(">")
    name = re.sub(r"^mcp__[^_]+__", "", name)
    name = re.sub(r"^mcp__proxy__", "", name)
    name = _ALIASES.get(name.lower(), name)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return _ALIASES.get(name, name)


def _has_transport_error(response: Any) -> bool:
    if isinstance(response, dict):
        for key, value in response.items():
            key_text = str(key).lower()
            if key_text in {"error", "exception", "traceback"} and value:
                return True
            if key_text in {"success", "ok"} and value is False:
                return True
        # Tool-discovery responses embed executable documentation. Error-handling
        # phrases inside a returned schema describe the discovered tool and are
        # not failures of the tool_search call itself.
        return any(
            _has_transport_error(value)
            for key, value in response.items()
            if str(key).lower() not in {"tool_schemas", "schemas"}
        )
    if isinstance(response, list):
        return any(_has_transport_error(value) for value in response)
    if isinstance(response, str):
        stripped = response.strip()
        if not stripped:
            return False
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            lowered = stripped.lower()
            if re.search(r"(?im)^\s*(?:error|failed|failure)\b", stripped):
                return True
            if re.search(r"(?i)\bexit(?:ed)?(?:\s+with)?(?:\s+code)?\s*[1-9]\d*\b", stripped):
                return True
            return any(
                marker in lowered
                for marker in ("traceback (most recent call last)", "toolerror:", "internal server error")
            )
        return _has_transport_error(parsed)
    return False


def extract_tool_events(actual_outcome: JsonObject) -> list[ToolEvent]:
    raw_events: list[tuple[str, JsonObject]] = []
    for event in actual_outcome.get("tool") or []:
        if isinstance(event, dict):
            raw_events.append(("actual_outcome.tool", event))
    for event in actual_outcome.get("sandbox_execution_chain") or []:
        if isinstance(event, dict) and event.get("type") == "tool":
            raw_events.append(("actual_outcome.sandbox_execution_chain", event))

    result: list[ToolEvent] = []
    seen: set[str] = set()
    for position, (source, event) in enumerate(raw_events, start=1):
        name = normalize_tool_name(event.get("tool_name") or event.get("name"))
        if not name:
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        response = event.get("response", event.get("output"))
        fingerprint = json.dumps([name, args, response], ensure_ascii=False, sort_keys=True, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(
            ToolEvent(
                index=int(event.get("index") or position),
                name=name,
                args=args,
                response=response,
                source=source,
                successful=response is not None and not _has_transport_error(response),
            )
        )
    return result


def is_core_event(event: ToolEvent) -> bool:
    if event.name in _META_TOOLS:
        return False
    return event.name.startswith(
        ("google_docs_", "google_drive_", "google_sheets_", "google_slides_")
    ) or event.name in {
        "create_pdf_from_docx",
        "download_file",
        "upload_file",
    }


def is_mutation_event(event: ToolEvent) -> bool:
    return any(fnmatch.fnmatchcase(event.name, pattern) for pattern in EXTERNAL_MUTATION_TOOL_PATTERNS)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def verify_extraction_candidate(
    rollout: RolloutRecord,
    classification: CaseClassification,
) -> tuple[bool, str, list[ToolEvent]]:
    """Independently gate benchmark PASS trajectories before skill extraction."""

    core = [event for event in rollout.events if is_core_event(event) and event.successful]
    if not rollout.benchmark_pass:
        return False, "benchmark overall result is not PASS", core
    if str(rollout.reward.get("benchmark_judge_error") or "").strip():
        return False, "benchmark judge recorded an error", core
    if not rollout.answer.strip():
        return False, "assistant response is empty", core

    if classification.family == "boundary_or_negative":
        mutations = [event for event in core if is_mutation_event(event)]
        if mutations:
            return False, "boundary case contains a successful mutation", core
        return True, "PASS response respected the boundary without mutation", core

    patterns = _FAMILY_EVIDENCE[classification.family]
    matched = [event for event in core if _matches_any(event.name, patterns)]
    if not matched:
        return False, f"no successful core evidence for {classification.family}", core
    if classification.family == "multi_source_aggregate":
        read_events = [
            event
            for event in matched
            if _matches_any(
                event.name,
                (
                    "google_drive_search",
                    "google_drive_get",
                    "google_docs_cat",
                    "google_sheets_get",
                    "google_sheets_metadata",
                ),
            )
        ]
        if not read_events:
            return False, "aggregate PASS lacks retrievable source evidence", core
    return True, f"PASS has successful {classification.family} core evidence", core


def _replace_distinct(value: str, pattern: re.Pattern[str], label: str) -> str:
    matches = list(dict.fromkeys(match.group(0) for match in pattern.finditer(value)))
    if not matches:
        return value
    replacements = {
        match: f"<{label}>" if len(matches) == 1 else f"<{label}_{index}>"
        for index, match in enumerate(matches, start=1)
    }
    return pattern.sub(lambda match: replacements[match.group(0)], value)


def _sanitize_scalar(value: str, *, key: str = "") -> str:
    if "email" in key.lower() or key.lower() in {"account", "user"}:
        return "<ACCOUNT_EMAIL>"
    value = _replace_distinct(value, _EMAIL, "EMAIL")
    value = _replace_distinct(value, _URL, "URL")
    value = _replace_distinct(value, _RESOURCE_ID, "RESOURCE_ID")
    if key.lower() in {"docid", "documentid", "fileid", "folderid", "driveid"}:
        return f"<{key.upper()}>"
    return value


def sanitize_value(value: Any, *, key: str = "", max_string: int = 600) -> Any:
    if isinstance(value, dict):
        return {
            str(child_key): sanitize_value(child_value, key=str(child_key), max_string=max_string)
            for child_key, child_value in list(value.items())[:24]
        }
    if isinstance(value, list):
        return [sanitize_value(item, key=key, max_string=max_string) for item in value[:12]]
    if isinstance(value, str):
        text = _sanitize_scalar(value, key=key)
        if len(text) > max_string:
            return text[:max_string] + "...<truncated>"
        return text
    return value


def prune_causal_events(events: list[ToolEvent], *, include_failed: bool = False) -> list[ToolEvent]:
    """Keep compact task-relevant events and discard routing/tool-discovery noise."""

    selected: list[ToolEvent] = []
    seen: set[str] = set()
    for event in events:
        if not is_core_event(event) or (not include_failed and not event.successful):
            continue
        signature = json.dumps([event.name, sanitize_value(event.args)], ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(event)
    if len(selected) > 12:
        selected = selected[:4] + selected[-8:]
    return selected


def _judge_failure_summary(reward: JsonObject) -> JsonObject:
    text = str(reward.get("benchmark_judge_result_json") or "")
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    failures: list[str] = []
    overall: JsonObject = {}
    for key, value in parsed.items() if isinstance(parsed, dict) else []:
        if not isinstance(value, dict):
            continue
        if str(value.get("状态") or value.get("status") or "").upper() == "FAIL":
            failures.append(str(key))
        if str(key).endswith("评估结果"):
            overall = value
    return {
        "failed_sections": failures,
        "severity": overall.get("severity"),
        "failure_type": overall.get("失败子类型"),
        "analysis": sanitize_value(str(overall.get("结果分析") or ""), max_string=400),
    }


def build_experience_card(
    rollout: RolloutRecord,
    classification: CaseClassification,
) -> JsonObject:
    verified, reason, _ = verify_extraction_candidate(rollout, classification)
    events = prune_causal_events(rollout.events, include_failed=not verified)
    action_sequence = [
        {
            "step": index,
            "tool": event.name,
            "args": sanitize_value(event.args, max_string=300),
            "result": sanitize_value(event.response, max_string=500),
            "successful": event.successful,
        }
        for index, event in enumerate(events, start=1)
    ]
    card_id = hashlib.sha256(f"{rollout.domain}:{rollout.case_id}:{rollout.request_id}".encode()).hexdigest()[:20]
    return {
        "schema_version": 1,
        "card_id": card_id,
        "label": "verified_success" if verified else "regression_negative",
        "selection_reason": reason,
        "family": classification.family,
        "case_id": rollout.case_id,
        "request_id": rollout.request_id,
        "trace_id": rollout.trace_id,
        "query": sanitize_value(rollout.query, max_string=800),
        "classification": classification.to_dict(),
        "action_sequence": action_sequence,
        "final_response": sanitize_value(rollout.answer, max_string=1000),
        "reward": {
            "score": rollout.score,
            "overall_pass": rollout.benchmark_pass,
            "d1": float(rollout.reward.get("benchmark_d1_pass") or 0.0),
            "d2": float(rollout.reward.get("benchmark_d2_pass") or 0.0),
            "d3": float(rollout.reward.get("benchmark_d3_pass") or 0.0),
            "d4": float(rollout.reward.get("benchmark_d4_pass") or 0.0),
            "d5": float(rollout.reward.get("benchmark_d5_pass") or 0.0),
        },
        "failure": {} if verified else _judge_failure_summary(rollout.reward),
    }


def select_prompt_cards(cards: list[JsonObject], *, positive_limit: int, negative_limit: int) -> list[JsonObject]:
    """Select short, diverse cards without using clustering."""

    positives = [card for card in cards if card["label"] == "verified_success"]
    negatives = [card for card in cards if card["label"] == "regression_negative"]
    positives.sort(
        key=lambda card: (
            -sum(float(card["reward"].get(f"d{i}") or 0.0) for i in range(1, 6)),
            len(card["action_sequence"]),
            card["case_id"],
        )
    )
    negatives.sort(key=lambda card: (-float(card["reward"]["score"]), card["case_id"]))

    selected_positive: list[JsonObject] = []
    signatures: set[tuple[str, ...]] = set()
    cases: set[str] = set()
    for card in positives:
        signature = tuple(step["tool"] for step in card["action_sequence"])
        if card["case_id"] in cases:
            continue
        if signature in signatures and len(selected_positive) >= max(2, positive_limit // 2):
            continue
        selected_positive.append(card)
        cases.add(card["case_id"])
        signatures.add(signature)
        if len(selected_positive) >= positive_limit:
            break
    if len(selected_positive) < positive_limit:
        for card in positives:
            if card not in selected_positive:
                selected_positive.append(card)
                if len(selected_positive) >= positive_limit:
                    break

    selected_negative: list[JsonObject] = []
    negative_cases: set[str] = set()
    for card in negatives:
        if card["case_id"] in negative_cases:
            continue
        selected_negative.append(card)
        negative_cases.add(card["case_id"])
        if len(selected_negative) >= negative_limit:
            break
    return selected_positive + selected_negative
