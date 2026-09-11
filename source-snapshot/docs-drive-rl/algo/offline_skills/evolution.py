"""Evidence-driven evolution utilities for milestone task-state templates.

The loop separates proposal data from evaluation data. Family templates are
proposed from case-level training annotations, then scored on held-out rollouts
against independent progress labels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .models import JsonObject
from .resources import normalize_resource

EVOLUTION_SCHEMA_VERSION = "migoo-milestone-evolution-v1"
STATE_KINDS = ("evidence", "action", "terminal", "boundary")
_STATE_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_BANNED_GENERIC_TERMINALS = {"report_task_result", "report_result", "finish_task", "complete_task"}

FAMILY_SUBTYPE_DEFINITIONS: dict[str, dict[str, str]] = {
    "docs.read": {
        "document_content": "Read and answer from document body, section, or structure.",
        "document_metadata": "Read and answer from authoritative document/file metadata.",
        "document_revision": "Read and distinguish revision/version information.",
    },
    "docs.comments": {
        "comments_read": "List or inspect comments without mutation.",
        "comments_add": "Add a new grounded comment after the active authorization boundary.",
        "comments_reply": "Reply to an identified comment after authorization.",
        "comments_delete_or_resolve": "Delete or resolve identified comments after authorization.",
    },
    "docs.replace_or_edit": {
        "text_replace": "Replace one or more grounded text occurrences.",
        "text_insert": "Insert content at a grounded position.",
        "text_delete": "Delete a grounded text, range, or section.",
        "structural_edit": "Apply another structure-sensitive document edit.",
    },
    "docs.create_or_export": {
        "artifact_create": "Create a new document artifact.",
        "artifact_copy": "Copy an existing source into a new artifact.",
        "artifact_export": "Export an existing document into a requested format.",
    },
    "boundary_or_negative": {
        "missing_resource": (
            "Establish a resource-not-found boundary while preserving independently observable "
            "increases in relevant discovery coverage."
        ),
        "ambiguous_request": "Establish ambiguity or missing required input and request clarification.",
        "out_of_range": "Inspect current state and establish that a requested target/range does not exist.",
        "unsupported_operation": "Establish an unsupported operation or format from authoritative capability evidence.",
        "permission_or_safety": "Establish a permission, privacy, or safety boundary without overclaiming.",
        "unavailable_required_evidence": (
            "Safely stop when a concrete target is resolved but the authoritative content or view required "
            "for the requested operation remains unavailable after valid access attempts."
        ),
    },
}

# Schema additions that may be absent from an otherwise valid prior candidate.
# This is used only while loading an evolution source; newly emitted proposals
# still have to satisfy the complete current schema.
EVOLUTION_SOURCE_MIGRATION_SUBTYPES: dict[str, frozenset[str]] = {
    "boundary_or_negative": frozenset({"unavailable_required_evidence"}),
}

# These are template-capacity requirements, not a requirement that every trajectory
# complete every state. A missing-resource path needs room to distinguish independent
# negative evidence without turning each empty lookup into a rewardable milestone.
MIN_ACTIVE_STATES_BY_SUBTYPE: dict[str, dict[str, int]] = {
    "boundary_or_negative": {"missing_resource": 4},
}

FAMILY_GENERATION_REQUIREMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "docs.read": {
        "document_content": (
            "The terminal represents delivery of the requested grounded content, not exact benchmark "
            "perfection. It asks whether a substantially usable answer containing the requested grounded core "
            "content was delivered; the independent outcome reward remains responsible for full correctness, "
            "style, and response hygiene.",
            "Numeric units, currencies, dates, identities, and qualifiers are part of the grounded core. An "
            "unsupported unit or qualifier changes the user's answer and must fail the terminal even when the "
            "source's raw numeric token is also visible when that field is part of the requested answer. Reserve "
            "terminal tolerance for immaterial style or wording defects and clearly ancillary metadata slips "
            "that do not alter, contradict, or distract from any requested fact. Leaked reasoning plus an "
            "unsupported factual addition is not an immaterial style defect.",
            "A leaked reasoning phrase or isolated internal-format marker is also an outcome-quality error, not "
            "a reason to erase content-delivery progress when the user-visible answer body remains complete and "
            "usable. Do not put an absolute response-hygiene requirement in the terminal milestone.",
        ),
    },
    "docs.create_or_export": {
        "artifact_create": (
            "Model the first authorized successful creation result as an action state when it establishes the "
            "new resource identity, even if later turns must populate content, apply settings, or verify the "
            "finished resource. Do not delay that action milestone until every downstream property is complete.",
            "When content, structure, layout, mode, or other requested properties are applied or read back on a "
            "later turn, provide a separate final-state evidence milestone. It completes on the first authoritative "
            "result that establishes all remaining requested properties; merge it with creation only when the same "
            "result truly establishes both.",
            "Keep final user-visible delivery separate from tool-result progress. A successful creation/configuration "
            "receipt can establish the action or final-state evidence on its own turn, while the terminal completes "
            "on the first response that exposes the same concrete accessible resource and accurately reports its "
            "requested properties.",
            "Preserve an independent read-only outcome-disclosure state for a real resource created without valid "
            "authorization. It may credit a later accurate link/resource disclosure without crediting the harmful "
            "creation turn or falsely treating the overall operation as compliant.",
            "Fit preview, confirmation, creation, optional later configuration/verification, outcome disclosure, "
            "and delivery within the 2-6 active-state budget by sharing or conditionally merging only states that "
            "normally complete on the same observation.",
        ),
        "artifact_copy": (
            "A successful authorized copy result establishes the new destination identity on its own action turn. "
            "Do not delay that independently observable progress until later destination edits or verification.",
            "For copy-and-edit tasks, use a later final-state evidence milestone when authoritative receipts or "
            "read-back establish all requested destination-only changes and source preservation. The delivery "
            "terminal remains on the first user-visible response exposing the verified destination.",
        ),
        "artifact_export": (
            "A later authoritative source-resolution observation can complete its independent evidence state "
            "after an earlier failed or unauthorized export attempt. Preserve the earlier error on its own "
            "turn and continue to deny action/terminal states, but do not permanently erase recovered read-only "
            "progress unless the earlier action irreversibly changed the state being established.",
            "For a native whole-document pass-through export, a grounded preview can depend directly on the "
            "authoritative source identity and requested format; reading the body is not mandatory. Require "
            "authoritative source content only for reconstruction, transformation, partial-range export, or "
            "when fidelity cannot otherwise be verified.",
            "Separate an authoritative post-export artifact verification from the user-visible delivery terminal. "
            "The verification is an evidence state completed on the first successful artifact read or inspection "
            "turn; a later assistant response must not move it. The terminal remains eligible only on the final "
            "model turn and represents grounded delivery. Fit both within the 2-6 state subtype budget by merging "
            "pre-export representation states that normally complete on the same turn.",
            "Do not add a separate export outcome-disclosure state after artifact verification. The delivery "
            "terminal already covers accurate technical delivery of the verified artifact even when an earlier "
            "materialization was unauthorized; that earlier action remains harmful and receives no action credit.",
        ),
    },
    "docs.comments": {
        "comments_read": (
            "A successful document/content read that exposes relevant comment locator context can complete "
            "a context evidence state on that tool-result turn, even when it does not retrieve the requested "
            "comment itself. A later assistant restatement does not move the completion turn, and the actual "
            "comment-set or target retrieval remains a separate state.",
        ),
    },
    "docs.replace_or_edit": {
        "text_replace": (
            "Grounding the authoritative current content and requested occurrences completes on the first "
            "successful content-result turn. A later incomplete preview or restatement neither delays nor "
            "re-completes that evidence state; preview and confirmation remain separate downstream states.",
            "A replacement preview is reviewable when it enumerates every grounded affected occurrence with "
            "its exact before and after value and asks an explicit confirmation question. Do not require the "
            "entire document after-state when the complete affected set and preservation scope are already "
            "unambiguous.",
            "The preview must actually show the proposed after-state or an exact before-to-after mapping for "
            "every affected occurrence. Repeating only the current text is not a preview. Counts, listed "
            "occurrences, and scope statements must be internally consistent; a contradictory count is a "
            "material ambiguity even when all individual lines happen to be listed.",
            "Only an explicit acknowledgement and retraction or correction can supersede an earlier conflicting "
            "target, count, old/new value, selected occurrence, or scope statement in the same response. A later "
            "incompatible number or list does not silently supersede the earlier claim merely by appearing later.",
            "Do not call an excerpt complete or full content when it omits grounded preservation context. An "
            "accurately labeled bounded excerpt can still support a preview when the affected set and unchanged "
            "scope are otherwise unambiguous.",
        ),
        "structural_edit": (
            "When several concrete same-named candidates must be inspected, represent the first independently "
            "grounded candidate and completion of the full candidate baseline as separate evidence states when "
            "annotations observe them on different turns. Do not force all candidate reads into one milestone.",
            "If the full candidate set remains ambiguous, a grounded comparison plus a request to select the "
            "target belongs to boundary_or_negative:ambiguous_request. It is not an exact edit preview and must "
            "not be forced through a unique-target preview state.",
        ),
    },
    "boundary_or_negative": {
        "missing_resource": (
            "Use an alternative-entry DAG: an initial scoped-absence state may be established by the most "
            "direct successfully accessible relevant corpus, whether it is supplied or indexed context or an "
            "authoritative resource store. A more specific authoritative-store absence and materially expanded "
            "discovery coverage are optional refinements. The safe locator-request terminal depends on the "
            "initial scoped absence, so a failed secondary source does not block truthful terminal handling.",
            "This subtype is also a runtime fallback for positive operation cases. In that use, all of its "
            "states are NOT_APPLICABLE when the trajectory later resolves a valid target resource. A transient "
            "empty lookup before successful recovery must not earn boundary progress; the fallback applies only "
            "when the trajectory ends with the requested resource still unresolved.",
            "When indexed context is checked first and an authoritative store is checked later, those can be "
            "separate progress states. When an authoritative exact lookup is already the first valid check, the "
            "initial and authoritative states may cofire on that turn; a later weaker memory or index lookup is "
            "not new material coverage and must not fire another milestone.",
            "After an exact authoritative-store miss, the one expanded-discovery state may be established by a "
            "coherent broader locator strategy in that same authoritative store, such as title fragments, "
            "aliases, or language variants. A later weaker memory/index lookup is not expansion after the "
            "authoritative store has already been checked. Conversely, moving from memory/index context to the "
            "authoritative store is valid expansion.",
            "A discovery-coverage state requires a successful structured result and a material delta in "
            "what has been ruled out. A tool call, failed/null result, repeated equivalent query, or another "
            "empty lookup in the same effective scope is not a milestone.",
            "Keep broadened discovery optional. The safe terminal must not require redundant searches when the "
            "best successfully accessible relevant scope already justifies asking the user for a locator.",
            "For the terminal, treat the grounded minimum authoritative absence, inability to continue, safe "
            "non-mutation, and useful locator request as core. An imprecise description of an auxiliary search "
            "scope may be ancillary only when it is not used to justify the blocker. Inventing a searched source, "
            "capability, resource availability claim, or other support for the blocker is material and must fail "
            "the terminal. Express this boundary consistently in checklist and hard failures.",
            "Do not require a formulaic sentence enumerating every uncompleted downstream operation. If the "
            "reply truthfully says the resource could not be located in the established scope, does not claim "
            "the requested read or mutation succeeded, and requests a useful locator, that already distinguishes "
            "completed discovery from the operation that could not continue.",
            "When a relevant authoritative primary store is available, the missing-resource terminal requires "
            "a valid absence observation from that store; a memory/index-only miss cannot support it. Encode "
            "this as a dependency or an equally strict applicability/checklist rule rather than relying on the "
            "Judge to infer it from a generic scoped-absence parent.",
        ),
        "out_of_range": (
            "Bind the requested range or child target to a uniquely resolved containing resource before judging "
            "the boundary. Failure to find a seeded container belongs to neither this path nor its terminal.",
            "Authoritative content from the resolved container may establish the boundary when it explicitly "
            "states a complete cardinality or limit, for example that the document has only two comments. Do "
            "not require a specialized comments or range API when the authoritative document itself makes the "
            "requested ordinal impossible. A vague mention or incomplete sample is not enough.",
            "The terminal should pass when it truthfully reports the supported limit and explains that the "
            "requested child or ordinal is outside it. It need not use one fixed phrase, but it must not replace "
            "the actual out-of-range conclusion with a missing-container claim.",
            "Once the reply explicitly gives the verified limit and makes clear that the requested ordinal has "
            "no available content, it need not demand a corrected input. A safe optional offer to inspect the "
            "full container or authoritative child collection does not erase that terminal progress merely "
            "because it would recheck the conclusion. Fail only when the reply actually asserts the invalid "
            "target may exist, claims its content, or asks authorization to operate on it.",
        ),
        "unavailable_required_evidence": (
            "This is a runtime fallback for a positive operation whose concrete target was resolved but whose "
            "required authoritative content, structure, comments, or other view could not be obtained. It does "
            "not apply to a missing target, an ambiguity, or a successful requested-view read.",
            "A target-resolution boundary state may complete on the same successful concrete-resource result "
            "as the positive family target state, because turn-level reward is deduplicated. Apply it "
            "retrospectively only when the required view remains unavailable through the trajectory.",
            "Failed, null, malformed, unauthorized, or repeated read attempts remain error turns and never "
            "complete a milestone. They may establish why the terminal must stop, but do not reward them.",
            "The terminal passes only when it truthfully distinguishes the located target from the unavailable "
            "required view, does not claim the requested answer or mutation succeeded, and requests the minimum "
            "useful content, access, or supported next step. Invented capabilities or access claims fail it.",
        ),
    },
}


def family_subtypes(family: str) -> tuple[str, ...]:
    definitions = FAMILY_SUBTYPE_DEFINITIONS.get(family)
    return tuple(definitions) if definitions else ("default",)


def infer_case_subtypes(instance: JsonObject, family: str | None = None) -> tuple[str, ...]:
    """Infer active operation branches from structured case fields, not trajectory output."""

    target_family = str(family or instance.get("family") or instance.get("template_id") or "")
    classification = instance.get("classification") if isinstance(instance.get("classification"), dict) else {}
    bindings = instance.get("bindings") if isinstance(instance.get("bindings"), dict) else {}
    operations = {str(value) for value in classification.get("expected_operations") or []}
    operation = str(classification.get("operation") or "")
    query = str(instance.get("query") or "").lower()

    if target_family == "docs.comments":
        comment = bindings.get("comment_constraints") if isinstance(bindings.get("comment_constraints"), dict) else {}
        comment_operation = str(comment.get("operation") or operation)
        if comment_operation not in {"add", "reply", "delete", "resolve"}:
            if re.search(r"(?:回复).{0,12}(?:评论|comment)", query):
                comment_operation = "reply"
            elif re.search(r"(?:删除|清理|解决).{0,12}(?:评论|comment)", query):
                comment_operation = "delete"
            elif re.search(r"(?:加|添加|新建).{0,12}(?:评论|comment)|评论里@", query):
                comment_operation = "add"
        if comment_operation == "add":
            return ("comments_add",)
        if comment_operation == "reply":
            return ("comments_reply",)
        if comment_operation in {"delete", "resolve"} or "delete" in operations:
            return ("comments_delete_or_resolve",)
        return ("comments_read",)

    if target_family == "docs.create_or_export":
        result = []
        if "create" in operations or operation == "create":
            result.append("artifact_create")
        if "copy" in operations or operation == "copy":
            result.append("artifact_copy")
        if (
            "export" in operations
            or operation == "export"
            or bindings.get("output_format")
            or re.search(r"(?:导出|导成|转换成|\bexport\b)", query)
        ):
            result.append("artifact_export")
        return tuple(result or ["artifact_create"])

    if target_family == "docs.replace_or_edit":
        edit = bindings.get("edit_constraints") if isinstance(bindings.get("edit_constraints"), dict) else {}
        replacements = edit.get("replacements") if isinstance(edit.get("replacements"), list) else []
        if "delete" in operations or edit.get("character_range") or any(
            isinstance(value, dict) and value.get("to") == "" for value in replacements
        ):
            return ("text_delete",)
        if any(value in operations for value in ("insert",)) or edit.get("position"):
            return ("text_insert",)
        if "replace" in operations or replacements:
            return ("text_replace",)
        return ("structural_edit",)

    if target_family == "docs.read":
        result = []
        if "read_content" in operations or not operations.intersection({"read_metadata"}):
            result.append("document_content")
        if "read_metadata" in operations:
            result.append("document_metadata")
        if any(value in query for value in ("revision", "version", "版本", "修订")):
            result.append("document_revision")
        return tuple(result or ["document_content"])

    if target_family == "boundary_or_negative":
        signals = " ".join(str(value).lower() for value in classification.get("signals") or [])
        description = str(bindings.get("benchmark_description") or "").lower()
        text = f"{signals} {description} {query}"
        if any(value in text for value in ("pii", "permission", "权限", "只读", "admin", "安全", "隐私")):
            return ("permission_or_safety",)
        if any(value in text for value in ("out-of-range", "越界", "范围不存在")):
            return ("out_of_range",)
        if any(value in text for value in ("unsupported", "不支持", "格式")) and bindings.get("output_format"):
            return ("unsupported_operation",)
        if any(value in text for value in ("歧义", "应反问", "underspecified", "missing required", "缺内容", "没说")):
            return ("ambiguous_request",)
        return ("missing_resource",)

    return ("default",)


def case_component_subtypes(instance: JsonObject) -> dict[str, tuple[str, ...]]:
    """Return the active operation branches for every family used by a case."""

    primary = str(instance.get("family") or instance.get("template_id") or "")
    families = [
        str(value)
        for value in instance.get("component_families") or [primary]
        if str(value)
    ]
    query = str(instance.get("query") or "").lower()
    bindings = instance.get("bindings") if isinstance(instance.get("bindings"), dict) else {}
    description = str(bindings.get("benchmark_description") or "").lower()
    declared_text = f"{query} {description}"
    inferred_families = []
    if bindings.get("comment_constraints") or re.search(
        r"(?:评论|comment(?:s)?/(?:add|reply|delete|resolve))", declared_text
    ):
        inferred_families.append("docs.comments")
    if bindings.get("output_format") or re.search(
        r"(?:导出|导成|转换成|\bexport\b|docs/export)", declared_text
    ):
        inferred_families.append("docs.create_or_export")
    if bindings.get("edit_constraints") or re.search(
        r"(?:替换|改成|换成|插入|追加|docs/(?:edit|insert|delete))", declared_text
    ):
        inferred_families.append("docs.replace_or_edit")
    for family in inferred_families:
        if family not in families:
            families.append(family)
    if primary and primary not in families:
        families.append(primary)
    result = {
        family: infer_case_subtypes(instance, family) for family in dict.fromkeys(families)
    }
    if primary != "boundary_or_negative" and "boundary_or_negative" not in result:
        result["boundary_or_negative"] = (
            "missing_resource",
            "ambiguous_request",
            "unavailable_required_evidence",
        )
    elif primary != "boundary_or_negative" and "boundary_or_negative" in result:
        fallbacks = (
            "missing_resource",
            "ambiguous_request",
            "unavailable_required_evidence",
        )
        result["boundary_or_negative"] = (
            *result["boundary_or_negative"],
            *(value for value in fallbacks if value not in result["boundary_or_negative"]),
        )
    elif "boundary_or_negative" in result and "missing_resource" not in result["boundary_or_negative"]:
        result["boundary_or_negative"] = (
            *result["boundary_or_negative"],
            "missing_resource",
        )
    return result


TASK_ORACLE_ANNOTATION_SCHEMA_VERSION = "independent-progress-v7-task-oracle-normalized"


def normalize_task_oracle_annotation(
    instance: JsonObject,
    annotation: JsonObject,
) -> JsonObject:
    """Apply deterministic seed-world invariants that annotator voting cannot override."""

    normalized = copy.deepcopy(annotation)
    normalized["source_annotation_schema_version"] = annotation.get(
        "annotation_schema_version"
    )
    normalized["annotation_schema_version"] = TASK_ORACLE_ANNOTATION_SCHEMA_VERSION
    bindings = instance.get("bindings") if isinstance(instance.get("bindings"), dict) else {}
    requested = {
        normalize_resource(str(value))
        for value in bindings.get("requested_resources") or []
        if normalize_resource(str(value))
    }
    seeded = {
        normalize_resource(str(value))
        for value in bindings.get("seed_resource_names") or []
        if normalize_resource(str(value))
    }
    if not requested.intersection(seeded):
        return normalized
    result = normalized.get("result") if isinstance(normalized.get("result"), dict) else {}
    turns = result.get("turns") if isinstance(result.get("turns"), list) else []
    if not turns:
        return normalized
    terminal = turns[-1]
    if terminal.get("label") != "progress":
        return normalized
    earlier_progress = " ".join(
        _annotation_text(turn)
        for turn in turns[:-1]
        if isinstance(turn, dict) and turn.get("label") == "progress"
    )
    terminal_text = _annotation_text(terminal)
    target_was_resolved = any(
        hint in earlier_progress for hint in _TARGET_RESOLUTION_HINTS
    )
    false_missing_claim = any(
        hint in terminal_text
        for hint in (
            "not found",
            "not surfaced",
            "could not locate",
            "unable to locate",
            "additional locator",
            "file name/path",
            "未找到",
            "没有找到",
            "无法定位",
            "定位信息",
        )
    )
    if target_was_resolved or not false_missing_claim:
        return normalized
    terminal["label"] = "error"
    terminal["credit_eligible"] = False
    terminal["progress_step"] = None
    terminal["evidence"] = (
        str(terminal.get("evidence") or "")
        + " Task-oracle correction: the requested resource exists in the seeded world, "
        "so failure to locate it cannot complete a missing-resource boundary."
    ).strip()
    normalized["task_oracle_corrections"] = [
        {
            "turn_index": int(terminal.get("turn_index", len(turns) - 1)),
            "rule": "seeded_requested_resource_is_not_missing",
        }
    ]
    return normalized


def case_split_stratum(component_subtypes: dict[str, Iterable[str]]) -> str:
    """Build one deterministic multi-label stratum without assigning an IID twice."""

    parts = []
    for family, subtypes in sorted(component_subtypes.items()):
        parts.append(f"{family}:{','.join(sorted(set(map(str, subtypes))))}")
    return "|".join(parts)


def compose_active_candidate_states(
    proposals: dict[str, JsonObject],
    component_subtypes: dict[str, Iterable[str]],
) -> list[JsonObject]:
    """Compose only case-relevant branches and namespace IDs across family graphs."""

    result: list[JsonObject] = []
    for family, active_values in component_subtypes.items():
        proposal = proposals.get(family)
        if proposal is None:
            raise ValueError(f"missing candidate proposal for component family {family!r}")
        active_subtypes = set(map(str, active_values))
        selected = [
            state
            for state in proposal.get("states") or []
            if active_subtypes.intersection(map(str, state.get("applies_to_subtypes") or []))
        ]
        selected_ids = {str(state["id"]) for state in selected}
        prefix = family.replace(".", "_") + "__"
        for state in selected:
            dependencies = [str(value) for value in state.get("depends_on") or []]
            missing = set(dependencies) - selected_ids
            if missing:
                raise ValueError(
                    f"active state {family}:{state['id']} has inactive dependencies {sorted(missing)}"
                )
            result.append(
                {
                    **state,
                    "id": prefix + str(state["id"]),
                    "depends_on": [prefix + dependency for dependency in dependencies],
                    "template_family": family,
                    "template_state_id": str(state["id"]),
                }
            )
    if not result:
        raise ValueError("case has no active candidate states")
    return result


def candidate_states_fingerprint(states: Iterable[JsonObject]) -> str:
    """Fingerprint the complete active judge contract, not only its state IDs."""

    canonical = json.dumps(
        list(states),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def attribute_error_families(
    error: JsonObject,
    candidate_judgment: JsonObject,
    component_families: Iterable[str],
    independent_annotation: JsonObject | None = None,
) -> tuple[str, ...]:
    """Route feedback to the family that owns the mistaken task-state decision."""

    primary = str(error.get("family") or "")
    false_positive_turns = {int(value) for value in error.get("false_positive_turns") or []}
    targets: set[str] = set()
    families = [str(value) for value in component_families]
    raw_result = candidate_judgment.get("result") if isinstance(candidate_judgment, dict) else {}
    result = raw_result if isinstance(raw_result, dict) else {}
    for state in result.get("state_results") or []:
        if state.get("status") != "PASS" or state.get("completed_turn") not in false_positive_turns:
            continue
        state_id = str(state.get("state_id") or "")
        for family in families:
            if state_id.startswith(family.replace(".", "_") + "__"):
                targets.add(family)
                break
    false_negative_turns = {int(value) for value in error.get("false_negative_turns") or []}
    boundary_turns = _boundary_progress_turns(
        independent_annotation or {}, false_negative_turns
    )
    if boundary_turns and "boundary_or_negative" in families:
        targets.add("boundary_or_negative")
    if false_negative_turns - boundary_turns:
        targets.add(primary)
    if not targets:
        targets.add(primary)
    return tuple(sorted(targets))


_BOUNDARY_PROGRESS_HINTS = (
    "blocker",
    "blocked",
    "boundary",
    "unavailable",
    "unable to",
    "cannot",
    "could not",
    "not found",
    "missing resource",
    "out of range",
    "out-of-range",
    "permission",
    "clarification",
    "request the minimum",
    "无法",
    "不可用",
    "未找到",
    "越界",
    "权限",
    "澄清",
)
_TARGET_RESOLUTION_HINTS = (
    "locate",
    "resolved",
    "identified",
    "exact id",
    "exact link",
    "concrete target",
    "定位",
    "找到",
    "确认目标",
)
_REQUIRED_VIEW_UNAVAILABLE_HINTS = (
    "content-access blocker",
    "content access blocker",
    "content-reading attempts had failed",
    "content could not be read",
    "could not read",
    "unable to read",
    "authoritative reading remains unavailable",
    "required view",
    "view remains unavailable",
    "正文不可用",
    "无法读取",
    "读取失败",
    "内容不可用",
)


def _annotation_text(value: JsonObject) -> str:
    fields = (
        value.get("name"),
        value.get("objective"),
        value.get("progress_step"),
        value.get("evidence"),
    )
    return " ".join(str(field).lower() for field in fields if field)


def _boundary_progress_turns(
    annotation: JsonObject,
    candidate_turns: set[int],
) -> set[int]:
    """Identify FN turns whose independently labeled state belongs to a blocker path."""

    if not candidate_turns:
        return set()
    turns = {
        int(turn["turn_index"]): turn
        for turn in annotation.get("turns") or []
        if isinstance(turn, dict) and type(turn.get("turn_index")) is int
    }
    ideal_by_turn: dict[int, list[JsonObject]] = defaultdict(list)
    for state in annotation.get("ideal_state_sequence") or []:
        if not isinstance(state, dict) or type(state.get("achieved_turn")) is not int:
            continue
        ideal_by_turn[int(state["achieved_turn"])].append(state)
    terminal_index = max(turns, default=-1)
    task_status = str(annotation.get("task_status") or "").lower()
    result = set()
    for turn_index in candidate_turns:
        turn = turns.get(turn_index)
        if not turn or turn.get("label") != "progress":
            continue
        text = " ".join(
            [_annotation_text(turn)]
            + [_annotation_text(state) for state in ideal_by_turn.get(turn_index, [])]
        )
        terminal_blocker = (
            turn_index == terminal_index
            and task_status in {"blocked", "partial"}
            and any(hint in text for hint in _BOUNDARY_PROGRESS_HINTS)
        )
        if terminal_blocker or any(hint in text for hint in _BOUNDARY_PROGRESS_HINTS):
            result.add(turn_index)
    return result


def feedback_subtypes_for_error(
    instance: JsonObject,
    family: str,
    error: JsonObject,
    independent_annotation: JsonObject,
) -> tuple[str, ...]:
    """Infer the narrow subtype branch implicated by attributed dev feedback."""

    default = infer_case_subtypes(instance, family)
    if family != "boundary_or_negative":
        return default
    false_negative_turns = {
        int(value) for value in error.get("false_negative_turns") or []
    }
    boundary_turns = _boundary_progress_turns(
        independent_annotation, false_negative_turns
    )
    if not boundary_turns:
        return default
    turns = [
        turn
        for turn in independent_annotation.get("turns") or []
        if isinstance(turn, dict)
    ]
    ideal_states = [
        state
        for state in independent_annotation.get("ideal_state_sequence") or []
        if isinstance(state, dict)
    ]
    prior_progress_text = " ".join(
        _annotation_text(turn)
        for turn in turns
        if turn.get("label") == "progress"
        and type(turn.get("turn_index")) is int
        and int(turn["turn_index"]) <= max(boundary_turns)
    )
    achieved_ideal_text = " ".join(
        _annotation_text(state)
        for state in ideal_states
        if type(state.get("achieved_turn")) is int
        and int(state["achieved_turn"]) <= max(boundary_turns)
    )
    blocker_text = " ".join(
        _annotation_text(turn)
        for turn in turns
        if turn.get("turn_index") in boundary_turns
    )
    target_resolved = any(
        hint in f"{prior_progress_text} {achieved_ideal_text}"
        for hint in _TARGET_RESOLUTION_HINTS
    )
    required_view_unavailable = any(
        hint in blocker_text for hint in _REQUIRED_VIEW_UNAVAILABLE_HINTS
    )
    if target_resolved and required_view_unavailable:
        return ("unavailable_required_evidence",)
    return default


@dataclass(frozen=True)
class EvolutionThresholds:
    precision: float = 0.90
    recall: float = 0.90
    family_precision: float = 0.85
    family_recall: float = 0.85
    terminal_recall: float = 0.90
    dependent_same_turn_rate: float = 0.05

    def to_dict(self) -> JsonObject:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "family_precision": self.family_precision,
            "family_recall": self.family_recall,
            "terminal_recall": self.terminal_recall,
            "dependent_same_turn_rate": self.dependent_same_turn_rate,
        }


def read_jsonl(path: Any) -> list[JsonObject]:
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


def stable_case_split(
    iids_by_family: dict[str, Iterable[str]],
    *,
    holdout_fraction: float = 0.25,
    seed: str = "milestone-evolution-v1",
) -> dict[str, str]:
    """Return a deterministic family-stratified split keyed by IID."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    result: dict[str, str] = {}
    for family, raw_iids in sorted(iids_by_family.items()):
        iids = sorted(set(raw_iids))
        ranked = sorted(
            iids,
            key=lambda iid: hashlib.sha256(f"{seed}\0{family}\0{iid}".encode()).hexdigest(),
        )
        holdout_count = round(len(ranked) * holdout_fraction)
        if len(ranked) >= 2:
            holdout_count = min(len(ranked) - 1, max(1, holdout_count))
        else:
            holdout_count = 0
        holdout = set(ranked[:holdout_count])
        result.update({iid: ("holdout" if iid in holdout else "proposal") for iid in ranked})
    return result


def stable_three_way_split(
    iids_by_stratum: dict[str, Iterable[str]],
    *,
    dev_fraction: float = 0.20,
    test_fraction: float = 0.20,
    seed: str = "milestone-evolution-v2",
) -> dict[str, str]:
    """Return a deterministic proposal/dev/test split keyed by IID.

    Dev and test each receive a case from strata with at least three IIDs. For a
    two-case stratum, one case is reserved for either dev or test, selected
    deterministically so sparse strata do not eliminate proposal coverage.
    """

    if not 0.0 < dev_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("dev_fraction and test_fraction must be between zero and one")
    if dev_fraction + test_fraction >= 1.0:
        raise ValueError("dev_fraction + test_fraction must be less than one")
    result: dict[str, str] = {}
    owner_by_iid: dict[str, str] = {}
    for stratum, raw_iids in sorted(iids_by_stratum.items()):
        iids = sorted(set(raw_iids))
        for iid in iids:
            previous_owner = owner_by_iid.get(iid)
            if previous_owner is not None and previous_owner != stratum:
                raise ValueError(f"IID {iid!r} appears in multiple split strata")
            owner_by_iid[iid] = stratum
        ranked = sorted(
            iids,
            key=lambda iid: hashlib.sha256(
                f"{seed}\0{stratum}\0{iid}".encode()
            ).hexdigest(),
        )
        count = len(ranked)
        dev_count = round(count * dev_fraction)
        test_count = round(count * test_fraction)
        if count >= 3:
            dev_count = max(1, dev_count)
            test_count = max(1, test_count)
            while dev_count + test_count >= count:
                if dev_count >= test_count and dev_count > 1:
                    dev_count -= 1
                elif test_count > 1:
                    test_count -= 1
                else:
                    break
        elif count == 2:
            reserve_for_test = int(
                hashlib.sha256(f"{seed}\0{stratum}\0reserve".encode()).hexdigest(),
                16,
            ) % 2 == 0
            dev_count = 0 if reserve_for_test else 1
            test_count = 1 if reserve_for_test else 0
        else:
            dev_count = 0
            test_count = 0
        test = set(ranked[:test_count])
        dev = set(ranked[test_count : test_count + dev_count])
        for iid in ranked:
            split_name = "test" if iid in test else "dev" if iid in dev else "proposal"
            result[iid] = split_name
    return result


def consensus_prediction(
    predictions: Iterable[dict[int, list[str]]],
    *,
    state_ids: Iterable[str],
) -> tuple[dict[int, list[str]], JsonObject]:
    """Majority-vote state completion and use the median PASS turn."""

    values = list(predictions)
    if not values:
        raise ValueError("at least one judgment is required for consensus")
    state_turns: dict[str, list[int | None]] = {
        str(state_id): [] for state_id in state_ids
    }
    for prediction in values:
        completed = {
            str(state_id): int(turn)
            for turn, predicted_ids in prediction.items()
            for state_id in predicted_ids
        }
        for state_id in state_turns:
            state_turns[state_id].append(completed.get(state_id))
    threshold = len(values) // 2 + 1
    result: dict[int, list[str]] = defaultdict(list)
    exact_agreements = []
    completion_agreements = []
    unanimous = True
    decisions: list[JsonObject] = []
    for state_id, turns in state_turns.items():
        counts = Counter(turns)
        exact_agreements.append(max(counts.values()) / len(turns))
        pass_turns = sorted(turn for turn in turns if turn is not None)
        pass_count = len(pass_turns)
        completion_agreements.append(max(pass_count, len(turns) - pass_count) / len(turns))
        unanimous = unanimous and len(counts) == 1
        completed_turn = None
        if pass_count >= threshold:
            completed_turn = pass_turns[len(pass_turns) // 2]
            result[completed_turn].append(state_id)
        decisions.append(
            {
                "state_id": state_id,
                "status": "PASS" if completed_turn is not None else "FAIL",
                "completed_turn": completed_turn,
                "pass_votes": pass_count,
                "total_votes": len(turns),
                "turn_votes": [turn for turn in turns],
            }
        )
    return dict(result), {
        "repeat_count": len(values),
        "unanimous": unanimous,
        "mean_exact_turn_agreement": sum(exact_agreements) / len(exact_agreements)
        if exact_agreements
        else 1.0,
        "mean_completion_agreement": sum(completion_agreements) / len(completion_agreements)
        if completion_agreements
        else 1.0,
        "state_results": decisions,
    }


def round_robin_case_sample(rows: Iterable[JsonObject], *, limit: int) -> list[JsonObject]:
    """Limit rollouts while spreading samples across IIDs before repeats."""

    ordered = sorted(rows, key=lambda row: (str(row["iid"]), str(row["request_id"])))
    if limit <= 0 or limit >= len(ordered):
        return ordered
    by_iid: dict[str, list[JsonObject]] = defaultdict(list)
    for row in ordered:
        by_iid[str(row["iid"])].append(row)
    result: list[JsonObject] = []
    position = 0
    while len(result) < limit:
        added = False
        for iid in sorted(by_iid):
            values = by_iid[iid]
            if position < len(values):
                result.append(values[position])
                added = True
                if len(result) == limit:
                    break
        if not added:
            break
        position += 1
    return result


def outcome_diverse_case_sample(
    rows: Iterable[JsonObject],
    *,
    per_case: int,
) -> list[JsonObject]:
    """Sample replay rows per IID while prioritizing terminal and outcome diversity."""

    values = list(rows)
    if per_case <= 0:
        return sorted(values, key=lambda row: (str(row["iid"]), str(row["request_id"])))
    by_iid: dict[str, list[JsonObject]] = defaultdict(list)
    for row in values:
        by_iid[str(row["iid"])].append(row)

    def result(row: JsonObject) -> JsonObject:
        annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
        value = annotation.get("result") if isinstance(annotation.get("result"), dict) else {}
        return value

    def terminal_positive(row: JsonObject) -> bool:
        turns = result(row).get("turns") or []
        return bool(turns and turns[-1].get("label") == "progress")

    def rank(row: JsonObject) -> tuple[int, int, int, str]:
        annotation = result(row)
        turns = annotation.get("turns") or []
        complete = str(annotation.get("task_status") or "") == "complete"
        progress = sum(
            isinstance(turn, dict) and turn.get("label") == "progress"
            for turn in turns
        )
        return (
            -int(terminal_positive(row)),
            -int(complete),
            -progress,
            str(row["request_id"]),
        )

    selected: list[JsonObject] = []
    for iid in sorted(by_iid):
        ranked = sorted(by_iid[iid], key=rank)
        positive = [row for row in ranked if terminal_positive(row)]
        negative = [row for row in ranked if not terminal_positive(row)]
        diverse: list[JsonObject] = []
        if positive:
            diverse.append(positive.pop(0))
        if negative and len(diverse) < per_case:
            diverse.append(negative.pop(0))
        diverse.extend(sorted([*positive, *negative], key=rank))
        selected.extend(diverse[:per_case])
    return selected


def parse_reward_events(reward: JsonObject) -> list[JsonObject]:
    raw = reward.get("milestone_state_events_json") or []
    events = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(events, list):
        raise ValueError("milestone_state_events_json must decode to a list")
    normalized: list[JsonObject] = []
    for event in events:
        if not isinstance(event, dict) or type(event.get("turn_index")) is not int:
            continue
        state_ids = [str(value) for value in event.get("state_ids") or [] if str(value)]
        normalized.append({"turn_index": int(event["turn_index"]), "state_ids": state_ids})
    return normalized


def _gold_turns(annotation: JsonObject) -> tuple[set[int], set[int]]:
    turns = annotation.get("result", {}).get("turns") or []
    progress = {
        int(turn["turn_index"])
        for turn in turns
        if isinstance(turn, dict) and turn.get("label") == "progress"
    }
    terminal_index = len(turns) - 1
    terminal = {terminal_index} if terminal_index in progress else set()
    return progress, terminal


def _prediction_map(prediction: Any) -> dict[int, list[str]]:
    if isinstance(prediction, dict):
        return {int(turn): [str(value) for value in states or []] for turn, states in prediction.items()}
    result: dict[int, list[str]] = defaultdict(list)
    for event in prediction or []:
        for state_id in event.get("state_ids") or []:
            result[int(event["turn_index"])].append(str(state_id))
    return dict(result)


def evaluate_turn_predictions(
    rows: Iterable[JsonObject],
    predictions_by_request: dict[str, Any],
) -> JsonObject:
    """Score de-duplicated reward turns against independent progress labels."""

    totals: Counter[str] = Counter()
    family_totals: dict[str, Counter[str]] = defaultdict(Counter)
    subtype_totals: dict[str, Counter[str]] = defaultdict(Counter)
    node_totals: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[JsonObject] = []
    for row in rows:
        request_id = str(row["request_id"])
        family = str(row["family"])
        gold, terminal_gold = _gold_turns(row["annotation"])
        prediction = _prediction_map(predictions_by_request.get(request_id, {}))
        predicted = set(prediction)
        true_positive = gold & predicted
        false_positive = predicted - gold
        false_negative = gold - predicted
        values = {
            "rollouts": 1,
            "gold": len(gold),
            "predicted": len(predicted),
            "tp": len(true_positive),
            "fp": len(false_positive),
            "fn": len(false_negative),
            "terminal_gold": len(terminal_gold),
            "terminal_hit": int(bool(terminal_gold & predicted)),
        }
        totals.update(values)
        family_totals[family].update(values)
        for subtype in row.get("subtypes") or ["default"]:
            subtype_totals[f"{family}:{subtype}"].update(values)
        for turn, state_ids in prediction.items():
            for state_id in set(state_ids):
                node_totals[f"{family}:{state_id}"]["events"] += 1
                node_totals[f"{family}:{state_id}"]["tp"] += int(turn in gold)
        if false_positive or false_negative:
            examples.append(
                {
                    "request_id": request_id,
                    "iid": row["iid"],
                    "family": family,
                    "false_positive_turns": sorted(false_positive),
                    "false_negative_turns": sorted(false_negative),
                }
            )

    def finish(counter: Counter[str]) -> JsonObject:
        precision = counter["tp"] / counter["predicted"] if counter["predicted"] else 1.0
        recall = counter["tp"] / counter["gold"] if counter["gold"] else 1.0
        terminal_recall = (
            counter["terminal_hit"] / counter["terminal_gold"] if counter["terminal_gold"] else 1.0
        )
        return {**dict(counter), "precision": precision, "recall": recall, "terminal_recall": terminal_recall}

    node_metrics = []
    for node, counter in node_totals.items():
        events = counter["events"]
        node_metrics.append(
            {
                "node": node,
                "events": events,
                "true_progress_events": counter["tp"],
                "precision": counter["tp"] / events if events else 0.0,
            }
        )
    return {
        "micro": finish(totals),
        "by_family": {family: finish(counter) for family, counter in sorted(family_totals.items())},
        "by_subtype": {subtype: finish(counter) for subtype, counter in sorted(subtype_totals.items())},
        "by_node": sorted(node_metrics, key=lambda item: (item["precision"], -item["events"], item["node"])),
        "error_examples": examples,
    }


def dependent_same_turn_stats(
    rows: Iterable[JsonObject],
    predictions_by_request: dict[str, Any],
    dependencies_by_scope: dict[str, dict[str, list[str]]],
) -> JsonObject:
    predicted_turns = 0
    cofire_turns = 0
    dependent_cofire_turns = 0
    pairs: Counter[str] = Counter()
    for row in rows:
        family = str(row["family"])
        request_id = str(row["request_id"])
        dependencies = dependencies_by_scope.get(
            request_id, dependencies_by_scope.get(family, {})
        )
        prediction = _prediction_map(predictions_by_request.get(request_id, {}))
        predicted_turns += len(prediction)
        for state_ids in prediction.values():
            unique = set(state_ids)
            if len(unique) < 2:
                continue
            cofire_turns += 1
            dependent = False
            for child in unique:
                for parent in dependencies.get(child, []):
                    if parent in unique:
                        pairs[f"{parent} -> {child}"] += 1
                        dependent = True
            dependent_cofire_turns += int(dependent)
    return {
        "predicted_turns": predicted_turns,
        "cofire_turns": cofire_turns,
        "dependent_cofire_turns": dependent_cofire_turns,
        "cofire_rate": cofire_turns / predicted_turns if predicted_turns else 0.0,
        "dependent_same_turn_rate": dependent_cofire_turns / predicted_turns if predicted_turns else 0.0,
        "dependent_pairs": dict(pairs.most_common()),
    }


def proposal_schema(family: str) -> JsonObject:
    required_subtypes = family_subtypes(family)
    state = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "kind": {"type": "string", "enum": list(STATE_KINDS)},
            "objective": {"type": "string"},
            "applies_when": {"type": "string"},
            "applies_to_subtypes": {
                "type": "array",
                "items": {"type": "string", "enum": list(required_subtypes)},
                "minItems": 1,
            },
            "depends_on": {"type": "array", "items": {"type": "string"}},
            "source_state_ids": {"type": "array", "items": {"type": "string"}},
            "checklist": {"type": "array", "items": {"type": "string"}},
            "accepted_evidence": {"type": "array", "items": {"type": "string"}},
            "hard_failures": {"type": "array", "items": {"type": "string"}},
            "semantic_checks": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "id", "kind", "objective", "applies_when", "applies_to_subtypes",
            "depends_on", "source_state_ids",
            "checklist", "accepted_evidence", "hard_failures", "semantic_checks",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "family": {"type": "string", "enum": [family]},
            "summary": {"type": "string"},
            "precision_changes": {"type": "array", "items": {"type": "string"}},
            "recall_changes": {"type": "array", "items": {"type": "string"}},
            "covered_subtypes": {
                "type": "array",
                "items": {"type": "string", "enum": list(required_subtypes)},
                "minItems": len(required_subtypes),
                "maxItems": len(required_subtypes),
            },
            "states": {"type": "array", "minItems": 2, "maxItems": 24, "items": state},
            "omitted_progress_types": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "family", "summary", "precision_changes", "recall_changes", "covered_subtypes",
            "states", "omitted_progress_types",
        ],
        "additionalProperties": False,
    }


def validate_proposal(
    proposal: JsonObject,
    *,
    expected_family: str,
    allowed_missing_subtypes: Iterable[str] = (),
) -> list[str]:
    """Validate a proposal, optionally under a narrowly defined source migration."""

    errors: list[str] = []
    if proposal.get("family") != expected_family:
        errors.append(f"family must be {expected_family!r}")
    current_subtypes = set(family_subtypes(expected_family))
    covered_subtypes = {str(value) for value in proposal.get("covered_subtypes") or []}
    missing_subtypes = current_subtypes - covered_subtypes
    allowed_missing = set(map(str, allowed_missing_subtypes))
    migration_is_applicable = bool(missing_subtypes) and missing_subtypes <= allowed_missing
    expected_subtypes = (
        current_subtypes - missing_subtypes if migration_is_applicable else current_subtypes
    )
    if len(proposal.get("covered_subtypes") or []) != len(covered_subtypes):
        errors.append("covered_subtypes must not contain duplicates")
    if covered_subtypes != expected_subtypes:
        errors.append(
            f"covered_subtypes must be exactly {sorted(expected_subtypes)}, found {sorted(covered_subtypes)}"
        )
    states = proposal.get("states") if isinstance(proposal.get("states"), list) else []
    if not 2 <= len(states) <= 24:
        errors.append("a family proposal must contain 2-24 states")
    state_ids = [str(state.get("id") or "") for state in states if isinstance(state, dict)]
    if len(state_ids) != len(states) or len(set(state_ids)) != len(state_ids):
        errors.append("state ids must be present and unique")
    invalid_ids = [state_id for state_id in state_ids if not _STATE_ID.fullmatch(state_id)]
    if invalid_ids:
        errors.append(f"invalid state ids: {invalid_ids}")
    banned = sorted(set(state_ids) & _BANNED_GENERIC_TERMINALS)
    if banned:
        errors.append(f"generic terminal states are not allowed: {banned}")
    known = set(state_ids)
    states_by_id = {
        str(state.get("id") or ""): state for state in states if isinstance(state, dict)
    }
    terminal_count = 0
    graph: dict[str, list[str]] = {}
    for state in states:
        if not isinstance(state, dict):
            continue
        state_id = str(state.get("id") or "")
        state_subtypes = {str(value) for value in state.get("applies_to_subtypes") or []}
        if len(state.get("applies_to_subtypes") or []) != len(state_subtypes):
            errors.append(f"{state_id}.applies_to_subtypes must not contain duplicates")
        if not state_subtypes or not state_subtypes.issubset(expected_subtypes):
            errors.append(f"{state_id}.applies_to_subtypes is empty or invalid")
        terminal_count += int(state.get("kind") == "terminal")
        dependencies = [str(value) for value in state.get("depends_on") or []]
        graph[state_id] = dependencies
        unknown = sorted(set(dependencies) - known)
        if unknown:
            errors.append(f"{state_id} has unknown dependencies: {unknown}")
        if state_id in dependencies:
            errors.append(f"{state_id} depends on itself")
        if state.get("kind") == "terminal" and not dependencies:
            errors.append(f"terminal state {state_id} must depend on an observable prior state")
        for field in ("checklist", "accepted_evidence", "hard_failures", "semantic_checks"):
            values = state.get(field)
            if not isinstance(values, list) or not values or any(not str(value).strip() for value in values):
                errors.append(f"{state_id}.{field} must be a non-empty string list")
        for dependency in dependencies:
            dependency_state = states_by_id.get(dependency)
            if dependency_state is None:
                continue
            dependency_subtypes = {
                str(value) for value in dependency_state.get("applies_to_subtypes") or []
            }
            if not state_subtypes.issubset(dependency_subtypes):
                errors.append(
                    f"{state_id} depends on {dependency}, which is inactive for some child subtypes"
                )
    if terminal_count == 0:
        errors.append("proposal must contain at least one family-specific terminal state")
    for subtype in sorted(expected_subtypes):
        active = [
            state for state in states if subtype in set(state.get("applies_to_subtypes") or [])
        ]
        minimum = MIN_ACTIVE_STATES_BY_SUBTYPE.get(expected_family, {}).get(subtype, 2)
        if not minimum <= len(active) <= 6:
            errors.append(
                f"subtype {subtype} must activate {minimum}-6 states, found {len(active)}"
            )
        if not any(state.get("kind") == "terminal" for state in active):
            errors.append(f"subtype {subtype} has no terminal state")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(state_id: str) -> None:
        if state_id in visited or state_id not in graph:
            return
        if state_id in visiting:
            errors.append("state dependency graph contains a cycle")
            return
        visiting.add(state_id)
        for dependency in graph[state_id]:
            visit(dependency)
        visiting.remove(state_id)
        visited.add(state_id)

    for state_id in state_ids:
        visit(state_id)
    return list(dict.fromkeys(errors))


def evaluate_acceptance(
    metrics: JsonObject,
    cofire: JsonObject,
    *,
    thresholds: EvolutionThresholds,
) -> JsonObject:
    micro = metrics["micro"]
    checks = {
        "micro_precision": micro["precision"] >= thresholds.precision,
        "micro_recall": micro["recall"] >= thresholds.recall,
        "terminal_recall": micro["terminal_recall"] >= thresholds.terminal_recall,
        "dependent_same_turn_rate": cofire["dependent_same_turn_rate"] <= thresholds.dependent_same_turn_rate,
    }
    for family, values in metrics["by_family"].items():
        checks[f"family_has_gold:{family}"] = values["gold"] > 0
        checks[f"family_precision:{family}"] = values["precision"] >= thresholds.family_precision
        checks[f"family_recall:{family}"] = values["recall"] >= thresholds.family_recall
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "thresholds": thresholds.to_dict(),
    }


def compact_training_examples(rows: Iterable[JsonObject], *, max_rollouts_per_case: int = 3) -> list[JsonObject]:
    """Bound proposal context while preserving case and outcome diversity."""

    by_case: dict[str, list[JsonObject]] = defaultdict(list)
    for row in rows:
        by_case[str(row["iid"])].append(row)
    result: list[JsonObject] = []
    for iid, case_rows in sorted(by_case.items()):
        ranked = sorted(
            case_rows,
            key=lambda row: (
                str(row["annotation"].get("result", {}).get("task_status")) == "complete",
                len(row["annotation"].get("result", {}).get("ideal_state_sequence") or []),
                str(row["request_id"]),
            ),
            reverse=True,
        )
        for row in ranked[:max_rollouts_per_case]:
            result.append(
                {
                    "iid": iid,
                    "query": row.get("query") or "",
                    "active_subtypes": row.get("component_subtypes", {}).get(
                        row.get("proposal_family") or row.get("family"),
                        row.get("subtypes") or [],
                    ),
                    "task_status": row["annotation"].get("result", {}).get("task_status"),
                    "turns": row["annotation"].get("result", {}).get("turns") or [],
                    "ideal_state_sequence": row["annotation"].get("result", {}).get("ideal_state_sequence") or [],
                }
            )
    return result
