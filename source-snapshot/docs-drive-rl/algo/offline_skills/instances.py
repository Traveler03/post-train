"""Instantiate category state templates for concrete benchmark cases."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .authorization import build_instance_authorization_policy
from .blueprints import get_blueprint
from .classifier import family_for_operation
from .evidence import sanitize_value
from .models import CaseClassification, CaseRecord, FamilyBlueprint, JsonObject, StateBlueprint
from .resources import (
    comment_constraints,
    comment_targets,
    content_constraints,
    destination_resource_names,
    edit_constraints,
    indirect_target_names,
    match_seed_resources,
    output_resource_names,
    quoted_mentions,
    requested_output_format,
    requested_resources,
    requested_view_constraints,
    search_constraints,
    seed_resource_names,
    share_constraints,
    share_targets,
)

_TERMINAL_STATE_IDS = {
    "answer_from_evidence",
    "report_edit_result",
    "report_comment_result",
    "report_artifact",
    "report_file_result",
    "report_permission_result",
    "explain_boundary",
}
_SECONDARY_RESOLUTION_STATES = {
    "resolve_document",
    "resolve_source",
    "locate_drive_resource",
    "resolve_sources",
    "resolve_resource",
    "identify_sources",
}
_MULTI_SOURCE_EVIDENCE_STATES = {"identify_sources", "collect_source_evidence"}

_SYNTHETIC_STATE_GUIDANCE: dict[str, JsonObject] = {
    "apply_comment_action": {
        "checklist": [
            "The mutation targets the document and comment context established by its dependencies.",
            "The executed operation is the requested add, reply, resolve, or delete action.",
            "Any requested comment or reply text is preserved exactly enough to satisfy the request.",
            "The tool result explicitly confirms success and identifies the affected comment when available.",
            "No duplicate or unrelated comment mutation is performed.",
        ],
        "accepted_evidence": [
            "A successful comment mutation result with document, comment, operation, content, or status fields.",
            "The resolved document or comment evidence from the dependency state, tied to the mutation target.",
        ],
        "hard_failures": [
            "The tool was merely called, failed, or returned an ambiguous result that does not establish the mutation.",
            "The operation affects the wrong document or comment, uses the wrong action, or changes the wrong text.",
            "An unrequested or duplicate comment mutation is performed.",
        ],
        "semantic_checks": [
            "Do the mutation arguments and result jointly match the requested operation, target, and content?",
            "Is the successful result tied to the comment context established by the dependency state?",
        ],
    },
    "resolve_added_comment": {
        "checklist": [
            "The target is the exact comment created by the dependency state.",
            "A resolve action is executed after the comment is created.",
            "The resolve result explicitly confirms success for that comment.",
            "No different comment is resolved as a substitute.",
        ],
        "accepted_evidence": [
            "The created comment identifier from the preceding add result.",
            "A successful resolve result for the same comment identifier and document.",
        ],
        "hard_failures": [
            "No successful resolve result is present.",
            "The resolved comment cannot be linked to the newly created comment.",
            "A different or additional comment is resolved without authorization.",
        ],
        "semantic_checks": [
            "Does comment identity remain consistent from creation through resolution?",
            "Does the result establish a completed resolve action rather than an attempted call?",
        ],
    },
    "verify_comment_resolution": {
        "checklist": [
            "A read occurs after the resolve action.",
            "The read targets the exact newly created and resolved comment.",
            "The returned comment state explicitly indicates that it is resolved.",
            "Verification does not rely only on the earlier mutation response or assistant claim.",
        ],
        "accepted_evidence": [
            "A post-action comment get, list, or search result containing the target comment and resolved status.",
            "Matching comment identifiers across the add, resolve, and verification results.",
        ],
        "hard_failures": [
            "There is no post-action read of the target comment.",
            "The read shows unresolved status or a different comment.",
            "The assistant claims verification without an observable verification result.",
        ],
        "semantic_checks": [
            "Is the verification chronologically after resolution and bound to the same comment?",
            "Does the returned status unambiguously establish the requested final state?",
        ],
    },
    "resolve_actor_identity": {
        "checklist": [
            "The deictic reference to the current user is recognized as an identity-dependent value.",
            "The selected value is grounded in the active account or authoritative current-user evidence.",
            "The resolved identity is passed unchanged into the downstream edit.",
            "No unrelated person's identity is substituted.",
        ],
        "accepted_evidence": [
            "The active account identity together with downstream edit arguments that use the matching identity value.",
            "An authoritative current-user lookup result tied to the active account.",
        ],
        "hard_failures": [
            "The value for the current user is guessed without account or tool evidence.",
            "The downstream edit uses a different person's identity or an unresolved placeholder.",
        ],
        "semantic_checks": [
            "Does the resolved value denote the active user in the seeded environment?",
            "Can the downstream edit value be traced to the resolved identity evidence?",
        ],
    },
    "derive_edit_content": {
        "checklist": [
            "The complete source evidence required by the request is available before derivation.",
            "The derived content covers the requested facts and excludes unsupported additions.",
            "Explicit length, structure, language, and formatting constraints are satisfied.",
            "The exact derived content is passed into the downstream edit.",
        ],
        "accepted_evidence": [
            "Retrieved source content plus assistant or tool output containing the derived edit text.",
            "Downstream edit arguments containing the same derived text and requested placement.",
        ],
        "hard_failures": [
            "The content is derived without the required source evidence.",
            "The derived content contradicts the source or violates an explicit content constraint.",
            "The downstream edit uses materially different content.",
        ],
        "semantic_checks": [
            "Is every material claim in the derived content grounded in retrieved source evidence?",
            "Does the content satisfy the user's semantic and exact-form constraints before it is written?",
        ],
    },
    "verify_edit_result": {
        "checklist": [
            "Verification occurs after the write action and targets the edited document.",
            "The final content contains the requested edit at the requested position.",
            "Exact length, structure, or formatting constraints are checked against observable output.",
            "Unrelated source content remains consistent with the authorized edit scope.",
        ],
        "accepted_evidence": [
            "A post-write document read showing the inserted or replaced content in context.",
            "A successful edit result with exact counts or positions when those fields prove the requested constraint.",
        ],
        "hard_failures": [
            "No post-action evidence establishes the requested final document state.",
            "The final content violates a required length, position, structure, or grounding constraint.",
            "Verification reads a different document or predates the edit.",
        ],
        "semantic_checks": [
            "Does the post-edit evidence establish the requested final state, not just a successful write call?",
            "Are exact constraints and preservation of unrelated content supported by the available evidence?",
        ],
    },
    "report_task_result": {
        "checklist": [
            "A non-empty user-visible final response is present.",
            "The response covers every requested result and identifies the relevant resources or outputs.",
            "Completion claims agree with observable tool results and completed dependency states.",
            "Failures, ambiguity, and partial completion are disclosed rather than reported as full success.",
            "The response contains no unsupported facts or unrequested action claims.",
        ],
        "accepted_evidence": [
            "The final assistant response together with the successful results supporting each reported outcome.",
            "Explicit failure or partial-completion results accurately reflected in the final response.",
        ],
        "hard_failures": [
            "The final response claims completion while a required dependency is missing or failed.",
            "The response invents facts, artifacts, links, mutations, or verification not present in the trajectory.",
            "A material requested result or known failure is omitted.",
        ],
        "semantic_checks": [
            "Can each material statement in the final response be traced to trajectory evidence?",
            "Does the response accurately distinguish complete, partial, blocked, and failed outcomes?",
        ],
    },
}


def _synthetic_state_guidance(state_id: str) -> JsonObject:
    try:
        guidance = _SYNTHETIC_STATE_GUIDANCE[state_id]
    except KeyError as exc:
        raise ValueError(f"missing judge guidance for synthetic state {state_id!r}") from exc
    return {key: list(values) for key, values in guidance.items()}


def _annotation_map(generated: JsonObject) -> dict[str, JsonObject]:
    guidance = generated.get("state_guidance") if isinstance(generated.get("state_guidance"), dict) else {}
    annotations = guidance.get("state_annotations") if isinstance(guidance.get("state_annotations"), list) else []
    return {
        str(annotation.get("state_id")): annotation
        for annotation in annotations
        if isinstance(annotation, dict) and annotation.get("state_id")
    }


def _is_required(state: StateBlueprint, classification: CaseClassification, case: CaseRecord) -> bool:
    if state.required == "always":
        return True
    operations = set(classification.expected_operations)
    if state.id == "inspect_edit_scope" and "read_content" in operations:
        ordered = list(classification.expected_operations)
        action_positions = [ordered.index(operation) for operation in ("replace", "edit") if operation in ordered]
        return bool(action_positions and ordered.index("read_content") < min(action_positions))
    if state.id == "establish_preconditions":
        return bool(operations - {"establish_boundary"})
    if state.id == "resolve_destination" and destination_resource_names(case.query):
        return True
    return bool(set(state.required_operations) & operations)


def required_skill_components(classification: CaseClassification) -> tuple[str, ...]:
    primary = classification.family
    if primary == "boundary_or_negative" and "admin permission required" in classification.signals:
        return (primary,)
    components: list[str] = []
    for operation in classification.expected_operations:
        if operation == "locate":
            family = "drive.retrieve" if classification.domain == "drive" and primary != "drive.retrieve" else None
        elif operation == "establish_boundary":
            family = "boundary_or_negative"
        elif operation == "delete" and primary == "docs.comments":
            family = primary
        elif operation in {"create", "copy", "export", "download", "upload"} and primary == "docs.create_or_export":
            family = primary
        else:
            family = family_for_operation(classification.domain, operation)
        if family and family not in components:
            components.append(family)
    if primary not in components:
        components.append(primary)
    return tuple(components)


def _node_binding(resources: list[str]) -> JsonObject:
    if len(resources) == 1:
        return {"requested_resource": resources[0]}
    return {"requested_resources": resources}


def _case_mutation_constraints(case: CaseRecord, sanitized_query: str) -> tuple[JsonObject, JsonObject]:
    edit = edit_constraints(sanitized_query)
    comments = comment_constraints(sanitized_query)
    benchmark_description = str(
        case.metadata.get("description") or case.metadata.get("benchmark_description") or ""
    ).lower()
    if "审阅备注" in sanitized_query and "comments/add" in benchmark_description:
        comments["operation"] = "add"
        appended_text = edit.pop("append_text", None)
        if appended_text:
            comments["comment_text"] = appended_text
        if edit.get("position") == "document_end":
            edit.pop("position")
        if "comments/resolve" in benchmark_description:
            comments["resolve_after_add"] = True
    return edit, comments


def _state_binding(
    *,
    family: str,
    state_id: str,
    resources: list[str],
    source_resources: list[str],
    case: CaseRecord,
    classification: CaseClassification,
    resource_from_state: str | None = None,
) -> JsonObject:
    destinations = destination_resource_names(case.query)
    outputs = output_resource_names(case.query)
    recipients = share_targets(case.query)
    comment_ids = comment_targets(case.query)
    output_format = requested_output_format(case.query)
    sanitized_query = str(sanitize_value(case.query, max_string=700))
    view = requested_view_constraints(sanitized_query)
    edit, comments = _case_mutation_constraints(case, sanitized_query)
    search = search_constraints(sanitized_query)
    content = content_constraints(sanitized_query)
    sharing = share_constraints(sanitized_query)
    indirect_targets = indirect_target_names(sanitized_query)

    if state_id == "resolve_destination" and destinations:
        binding = _node_binding(destinations)
    elif resource_from_state and family == "drive.transfer_or_share":
        binding = {
            "resource_from_state": resource_from_state,
            "selection": "resource selected by the upstream task state",
        }
    else:
        binding = _node_binding(resources)

    operation_bound_states = {
        "inspect_requested_view",
        "answer_from_evidence",
        "inspect_resource",
        "establish_preconditions",
        "resolve_comment_context",
        "apply_comment_action",
        "resolve_actor_identity",
        "derive_edit_content",
        "verify_edit_result",
        "report_comment_result",
        "inspect_edit_scope",
        "apply_scoped_edit",
        "report_edit_result",
        "materialize_artifact",
        "report_artifact",
        "execute_file_operation",
        "report_file_result",
        "inspect_or_apply_permissions",
        "report_permission_result",
        "derive_aggregate",
        "materialize_or_transfer",
        "preserve_world_state",
        "explain_boundary",
        "report_task_result",
    }
    if state_id in operation_bound_states:
        binding["operation_request"] = sanitized_query

    if (
        state_id
        in {
            "resolve_document",
            "resolve_source",
            "locate_drive_resource",
            "resolve_sources",
            "resolve_resource",
            "identify_sources",
            "establish_preconditions",
        }
        and search
    ):
        binding["search_constraints"] = search
    if state_id in {"inspect_requested_view", "inspect_resource", "collect_source_evidence"}:
        if view:
            binding["requested_view"] = view
        if content:
            binding["content_constraints"] = content
    if state_id in {"inspect_edit_scope", "apply_scoped_edit", "derive_edit_content", "verify_edit_result"} and edit:
        binding["edit_constraints"] = edit
    if state_id in {"apply_scoped_edit", "derive_edit_content", "verify_edit_result"} and content:
        binding["content_constraints"] = content
    if (state_id == "resolve_comment_context" or family == "boundary_or_negative") and comment_ids:
        binding["comment_ids"] = comment_ids
    if state_id == "resolve_comment_context" and "out-of-range target" in classification.signals:
        binding["expected_lookup_outcome"] = "not_found_or_out_of_range"
        binding["required_evidence"] = "authoritative_comment_lookup_result"
    if state_id in {"resolve_comment_context", "apply_comment_action", "report_comment_result"} and comments:
        binding["comment_constraints"] = comments
    if family == "boundary_or_negative":
        if "admin permission required" in classification.signals:
            binding["required_authorization"] = "administrator"
            binding["expected_boundary_evidence"] = "permission_denied_or_missing_admin_authorization"
        references = quoted_mentions(case.query)
        if references:
            binding["named_references"] = references
        if indirect_targets:
            binding["target_resources"] = indirect_targets
        if output_format:
            binding["requested_output_format"] = output_format
        if edit:
            binding["attempted_edit_constraints"] = edit
    if state_id in {"materialize_artifact", "execute_file_operation", "materialize_or_transfer"}:
        if outputs:
            binding["output_resources"] = outputs
        if destinations:
            binding["destination_resources"] = destinations
        if output_format:
            binding["output_format"] = output_format
        delegates_content_write = bool(
            state_id == "materialize_artifact"
            and content
            and set(classification.expected_operations) & {"replace", "edit"}
        )
        if content and not delegates_content_write:
            binding["content_constraints"] = content
        if delegates_content_write:
            binding["materialization_scope"] = "create_or_copy_target_only"
        if source_resources != resources:
            binding["source_resources"] = source_resources
        if "copy" in classification.expected_operations:
            file_outputs = [output for output in outputs if output not in destinations]
            if not file_outputs:
                binding["derived_outputs"] = [
                    {
                        "operation": "copy",
                        "source_resource": source,
                        "output_name": "same_as_source",
                    }
                    for source in source_resources
                ]
    if state_id in {"inspect_or_apply_permissions", "materialize_or_transfer"} and recipients:
        binding["share_targets"] = recipients
    if state_id in {"inspect_or_apply_permissions", "materialize_or_transfer"} and sharing:
        binding["share_constraints"] = sharing
    if state_id.startswith("report_"):
        if outputs:
            binding["output_resources"] = outputs
        if destinations:
            binding["destination_resources"] = destinations
        if output_format:
            binding["output_format"] = output_format
        if recipients:
            binding["share_targets"] = recipients
        if comment_ids:
            binding["comment_ids"] = comment_ids
        if content:
            binding["content_constraints"] = content
        if comments:
            binding["comment_constraints"] = comments
        if sharing:
            binding["share_constraints"] = sharing
    return binding


def _nearest_retained_dependencies(
    dependency: str,
    *,
    selected_ids: set[str],
    states_by_id: dict[str, StateBlueprint],
) -> list[str]:
    if dependency in selected_ids:
        return [dependency]
    state = states_by_id.get(dependency)
    if state is None:
        return []
    result: list[str] = []
    for parent in state.depends_on:
        result.extend(_nearest_retained_dependencies(parent, selected_ids=selected_ids, states_by_id=states_by_id))
    return list(dict.fromkeys(result))


def _component_nodes(
    *,
    family: str,
    classification: CaseClassification,
    case: CaseRecord,
    generated: JsonObject,
    resources: list[str],
    namespace: bool,
    secondary: bool,
    prior_has_evidence: bool,
    resource_from_state: str | None,
    suppress_materialization: bool,
    source_resources: list[str],
    known_missing_resource: bool,
) -> list[JsonObject]:
    blueprint = get_blueprint(family)
    annotations = _annotation_map(generated)
    states_by_id = {state.id: state for state in blueprint.states}
    selected = [
        state
        for state in blueprint.states
        if state.id not in _TERMINAL_STATE_IDS and _is_required(state, classification, case)
    ]
    sanitized_query = str(sanitize_value(case.query, max_string=700))
    _, case_comment_constraints = _case_mutation_constraints(case, sanitized_query)
    if family == "docs.comments" and case_comment_constraints.get("operation") == "add":
        selected = [state for state in selected if state.id != "resolve_comment_context"]
    if known_missing_resource and family in {"docs.read", "docs.comments", "drive.retrieve"}:
        selected = [
            state for state in selected if state.id in {"resolve_document", "locate_drive_resource", "resolve_resource"}
        ]
    if secondary:
        selected = [state for state in selected if state.id not in _SECONDARY_RESOLUTION_STATES]
    if secondary and prior_has_evidence and family == "multi_source_aggregate":
        selected = [state for state in selected if state.id not in _MULTI_SOURCE_EVIDENCE_STATES]
    if secondary and prior_has_evidence and family == "boundary_or_negative" and not indirect_target_names(case.query):
        selected = [state for state in selected if state.id != "establish_preconditions"]
    if family == "multi_source_aggregate" and suppress_materialization:
        selected = [state for state in selected if state.id != "materialize_or_transfer"]
    selected_ids = {state.id for state in selected}
    prefix = family.replace(".", "_") + "__" if namespace else ""
    template_to_instances: dict[str, list[str]] = {}
    nodes: list[JsonObject] = []

    for state in selected:
        expansion_values: list[str | None]
        if (
            state.expansion == "per_requested_resource" or state.id in {"inspect_requested_view", "inspect_resource"}
        ) and len(resources) > 1:
            expansion_values = resources
        else:
            expansion_values = [None]
        instance_ids: list[str] = []
        for index, binding in enumerate(expansion_values, start=1):
            suffix = "" if binding is None else f"__{index}"
            instance_id = f"{prefix}{state.id}{suffix}"
            instance_ids.append(instance_id)
            annotation = annotations.get(state.id, {})
            node_binding = _state_binding(
                family=family,
                state_id=state.id,
                resources=resources,
                source_resources=source_resources,
                case=case,
                classification=classification,
                resource_from_state=resource_from_state,
            )
            if binding is not None:
                node_binding.pop("requested_resources", None)
                node_binding["requested_resource"] = binding
            nodes.append(
                {
                    "id": instance_id,
                    "template_family": family,
                    "template_state_id": state.id,
                    "objective": str(annotation.get("objective") or state.applies_when),
                    "required": True,
                    "required_mode": "instantiated",
                    "applies_when": state.applies_when,
                    "binding": node_binding,
                    "depends_on": list(state.depends_on),
                    "deterministic_checks": [check.to_dict() for check in state.checks],
                    "checklist": list(annotation.get("checklist") or []),
                    "accepted_evidence": list(annotation.get("accepted_evidence") or []),
                    "hard_failures": list(annotation.get("hard_failures") or []),
                    "semantic_checks": list(annotation.get("semantic_checks") or []),
                }
            )
            if state.id == "materialize_artifact" and set(classification.expected_operations) & {
                "replace",
                "edit",
                "comment",
            }:
                nodes[-1]["objective"] = (
                    "Create or copy the target artifact exactly once, preserving source content when applicable; "
                    "delegate requested content edits and comments to the downstream mutation states."
                )
            if known_missing_resource and state.id in {"resolve_document", "locate_drive_resource"}:
                nodes[-1]["objective"] = (
                    "Attempt the requested read-only lookup and retain authoritative not-found evidence."
                )
            if (
                family == "docs.comments"
                and state.id == "resolve_comment_context"
                and "out-of-range target" in classification.signals
            ):
                nodes[-1]["objective"] = (
                    "Look up the requested comment ID and retain authoritative not-found or out-of-range "
                    "evidence without inventing comment content."
                )
            if (
                family == "boundary_or_negative"
                and state.id == "establish_preconditions"
                and "admin permission required" in classification.signals
            ):
                nodes[-1]["objective"] = (
                    "Check the current account's administrator authorization by attempting the restricted "
                    "Shared Drive enumeration and retain its permission result as boundary evidence."
                )
        template_to_instances[state.id] = instance_ids

    for node in nodes:
        expanded_dependencies: list[str] = []
        for dependency in node["depends_on"]:
            retained = _nearest_retained_dependencies(
                str(dependency), selected_ids=selected_ids, states_by_id=states_by_id
            )
            for state_id in retained:
                expanded_dependencies.extend(template_to_instances[state_id])
        node["depends_on"] = list(dict.fromkeys(expanded_dependencies))

    if family == "docs.comments" and set(classification.expected_operations) & {"comment", "delete"}:
        context_ids = [str(node["id"]) for node in nodes if node.get("template_state_id") == "resolve_comment_context"]
        document_ids = [str(node["id"]) for node in nodes if node.get("template_state_id") == "resolve_document"]
        comment_operation = str(case_comment_constraints.get("operation") or "mutate")
        action_patterns = {
            "add": ["google_docs_comments_add"],
            "reply": ["google_docs_comments_reply"],
            "delete": ["google_docs_comments_delete"],
            "resolve": ["google_docs_comments_resolve"],
        }.get(
            comment_operation,
            [
                "google_docs_comments_add",
                "google_docs_comments_reply",
                "google_docs_comments_resolve",
                "google_docs_comments_delete",
            ],
        )
        action_id = f"{prefix}apply_comment_action"
        nodes.append(
            {
                "id": action_id,
                "template_family": None,
                "template_state_id": None,
                "synthetic": True,
                "objective": f"Apply the requested comment {comment_operation} action exactly once.",
                "required": True,
                "required_mode": "instantiated",
                "applies_when": "the request adds, replies to, resolves, or deletes comments",
                "binding": _state_binding(
                    family=family,
                    state_id="apply_comment_action",
                    resources=resources,
                    source_resources=source_resources,
                    case=case,
                    classification=classification,
                ),
                "depends_on": context_ids or document_ids,
                "deterministic_checks": [
                    {
                        "kind": "tool_called",
                        "tool_patterns": action_patterns,
                        "min_count": 1,
                        "description": "The requested comment mutation succeeds.",
                    }
                ],
                **_synthetic_state_guidance("apply_comment_action"),
            }
        )
        if comment_operation == "add" and case_comment_constraints.get("resolve_after_add"):
            resolve_id = f"{prefix}resolve_added_comment"
            resolve_binding = _state_binding(
                family=family,
                state_id="apply_comment_action",
                resources=resources,
                source_resources=source_resources,
                case=case,
                classification=classification,
            )
            resolve_binding["comment_from_state"] = action_id
            resolve_binding["comment_constraints"] = {
                "operation": "resolve",
                "expected_status": "resolved",
            }
            nodes.append(
                {
                    "id": resolve_id,
                    "template_family": None,
                    "template_state_id": None,
                    "synthetic": True,
                    "objective": "Resolve the exact comment created by the preceding add action.",
                    "required": True,
                    "required_mode": "instantiated",
                    "applies_when": "the benchmark contract requires a newly added comment to be resolved",
                    "binding": resolve_binding,
                    "depends_on": [action_id],
                    "deterministic_checks": [
                        {
                            "kind": "tool_called",
                            "tool_patterns": ["google_docs_comments_resolve"],
                            "min_count": 1,
                            "description": "The newly added comment is resolved.",
                        }
                    ],
                    **_synthetic_state_guidance("resolve_added_comment"),
                }
            )
            nodes.append(
                {
                    "id": f"{prefix}verify_comment_resolution",
                    "template_family": None,
                    "template_state_id": None,
                    "synthetic": True,
                    "objective": "Verify that the newly added comment now has resolved status.",
                    "required": True,
                    "required_mode": "instantiated",
                    "applies_when": "after resolving a newly added comment",
                    "binding": {
                        **_node_binding(resources),
                        "comment_from_state": action_id,
                        "expected_status": "resolved",
                    },
                    "depends_on": [resolve_id],
                    "deterministic_checks": [
                        {
                            "kind": "tool_called",
                            "tool_patterns": [
                                "google_docs_comments_get",
                                "google_docs_comments_list",
                                "google_docs_explicit_comments",
                                "google_docs_search_comments",
                            ],
                            "min_count": 1,
                            "description": "A post-action comment read confirms resolved status.",
                        }
                    ],
                    **_synthetic_state_guidance("verify_comment_resolution"),
                }
            )

    resolve_actor = family == "docs.replace_or_edit" and re.search(
        r"(?:负责人|owner).{0,20}(?:改成|设为)我", case.query, re.IGNORECASE
    )
    if resolve_actor:
        apply_node = next(
            (node for node in nodes if node.get("template_state_id") == "apply_scoped_edit"),
            None,
        )
        if apply_node is not None:
            identity_id = f"{prefix}resolve_actor_identity"
            identity_dependencies = list(apply_node.get("depends_on") or [])
            nodes.insert(
                nodes.index(apply_node),
                {
                    "id": identity_id,
                    "template_family": None,
                    "template_state_id": None,
                    "synthetic": True,
                    "objective": "Resolve the deictic value '我' to the current user's writable identity value.",
                    "required": True,
                    "required_mode": "instantiated",
                    "applies_when": "an edit value refers to the current user",
                    "binding": {
                        "identity_reference": "current_user",
                        "operation_request": str(sanitize_value(case.query, max_string=700)),
                    },
                    "depends_on": identity_dependencies,
                    "deterministic_checks": [],
                    **_synthetic_state_guidance("resolve_actor_identity"),
                },
            )
            apply_node["depends_on"] = [identity_id]
            for update in apply_node["binding"].get("edit_constraints", {}).get("field_updates") or []:
                if update.get("value") == "current_user":
                    update.pop("value", None)
                    update["value_from_state"] = identity_id

    if family == "docs.replace_or_edit" and "执行摘要" in case.query:
        apply_node = next(
            (node for node in nodes if node.get("template_state_id") == "apply_scoped_edit"),
            None,
        )
        if apply_node is not None:
            derive_id = f"{prefix}derive_edit_content"
            derive_dependencies = list(apply_node.get("depends_on") or [])
            derive_node = {
                "id": derive_id,
                "template_family": None,
                "template_state_id": None,
                "synthetic": True,
                "objective": "Derive the requested execution summary from the full retrieved document content.",
                "required": True,
                "required_mode": "instantiated",
                "applies_when": "the requested edit content must be derived from source evidence",
                "binding": _state_binding(
                    family=family,
                    state_id="derive_edit_content",
                    resources=resources,
                    source_resources=source_resources,
                    case=case,
                    classification=classification,
                ),
                "depends_on": derive_dependencies,
                "deterministic_checks": [],
                **_synthetic_state_guidance("derive_edit_content"),
            }
            nodes.insert(nodes.index(apply_node), derive_node)
            apply_node["depends_on"] = [derive_id]
            nodes.append(
                {
                    "id": f"{prefix}verify_edit_result",
                    "template_family": None,
                    "template_state_id": None,
                    "synthetic": True,
                    "objective": (
                        "Verify the inserted summary's character count, position, and grounding after the edit."
                    ),
                    "required": True,
                    "required_mode": "instantiated",
                    "applies_when": "the generated edit has an exact length or placement constraint",
                    "binding": _state_binding(
                        family=family,
                        state_id="verify_edit_result",
                        resources=resources,
                        source_resources=source_resources,
                        case=case,
                        classification=classification,
                    ),
                    "depends_on": [str(apply_node["id"])],
                    "deterministic_checks": [],
                    **_synthetic_state_guidance("verify_edit_result"),
                }
            )
    return nodes


def _sink_ids(nodes: list[JsonObject]) -> list[str]:
    dependencies = {str(dependency) for node in nodes for dependency in node.get("depends_on") or []}
    return [str(node["id"]) for node in nodes if str(node["id"]) not in dependencies]


def _terminal_node(
    *,
    primary_blueprint: FamilyBlueprint,
    generated: JsonObject,
    resources: list[str],
    dependencies: list[str],
    composite: bool,
    case: CaseRecord,
    classification: CaseClassification,
    source_resources: list[str],
    resource_from_state: str | None,
) -> JsonObject:
    terminal = next((state for state in primary_blueprint.states if state.id in _TERMINAL_STATE_IDS), None)
    annotation = _annotation_map(generated).get(terminal.id, {}) if terminal else {}
    guidance = annotation if terminal else _synthetic_state_guidance("report_task_result")
    checks = (
        [check.to_dict() for check in terminal.checks]
        if terminal
        else [
            {
                "kind": "response_nonempty",
                "tool_patterns": [],
                "min_count": 1,
                "description": "A user-visible result for the complete task is returned.",
            }
        ]
    )
    return {
        "id": "report_task_result" if composite or terminal is None else terminal.id,
        "template_family": primary_blueprint.family if terminal else None,
        "template_state_id": terminal.id if terminal else None,
        "synthetic": terminal is None,
        "objective": str(
            annotation.get("objective")
            or (terminal.applies_when if terminal else "Report every requested result without unsupported claims.")
        ),
        "required": True,
        "required_mode": "instantiated",
        "applies_when": "after every required task state is complete",
        "binding": _state_binding(
            family=primary_blueprint.family,
            state_id="report_task_result" if composite or terminal is None else terminal.id,
            resources=resources,
            source_resources=source_resources,
            case=case,
            classification=classification,
            resource_from_state=resource_from_state,
        ),
        "depends_on": dependencies,
        "deterministic_checks": checks,
        "checklist": list(guidance.get("checklist") or []),
        "accepted_evidence": list(guidance.get("accepted_evidence") or []),
        "hard_failures": list(guidance.get("hard_failures") or []),
        "semantic_checks": list(guidance.get("semantic_checks") or []),
    }


def instantiate_case(
    case: CaseRecord,
    classification: CaseClassification,
    blueprint: FamilyBlueprint,
    generated: JsonObject,
    *,
    verifier_version: str,
    template_version: str,
    generated_packages: Mapping[str, JsonObject] | None = None,
) -> JsonObject:
    resources = requested_resources(case, classification)
    output_resources = output_resource_names(case.query)
    sanitized_query = str(sanitize_value(case.query, max_string=700))
    instance_edit_constraints, instance_comment_constraints = _case_mutation_constraints(case, sanitized_query)
    packages = dict(generated_packages or {})
    packages.setdefault(blueprint.family, generated)
    components = required_skill_components(classification)
    state_nodes: list[JsonObject] = []
    prior_sinks: list[str] = []
    prior_has_evidence = False
    seed_names = seed_resource_names(case.seed_summary)
    known_missing_resource = bool(
        classification.family == "boundary_or_negative"
        and seed_names
        and resources != ["<requested_resource_set>"]
        and not match_seed_resources(case.query, seed_names)
    )
    for index, family in enumerate(components):
        component_resources = output_resources if index > 0 and output_resources else resources
        resource_from_state = None
        if family == "drive.transfer_or_share" and index > 0 and components[index - 1] == "multi_source_aggregate":
            resource_from_state = prior_sinks[-1] if prior_sinks else None
        later_components = set(components[index + 1 :])
        component_nodes = _component_nodes(
            family=family,
            classification=classification,
            case=case,
            generated=packages.get(family, {}),
            resources=component_resources,
            namespace=len(components) > 1,
            secondary=index > 0,
            prior_has_evidence=prior_has_evidence,
            resource_from_state=resource_from_state,
            suppress_materialization=bool(
                family == "multi_source_aggregate"
                and later_components & {"docs.create_or_export", "drive.file_operation", "drive.transfer_or_share"}
            ),
            source_resources=resources,
            known_missing_resource=known_missing_resource,
        )
        roots = [node for node in component_nodes if not node["depends_on"]]
        for node in roots:
            node["depends_on"] = list(prior_sinks)
        state_nodes.extend(component_nodes)
        prior_sinks = _sink_ids(component_nodes) or prior_sinks
        prior_has_evidence = prior_has_evidence or family in {
            "docs.read",
            "docs.replace_or_edit",
            "docs.comments",
            "drive.retrieve",
            "multi_source_aggregate",
        }

    terminal = _terminal_node(
        primary_blueprint=blueprint,
        generated=generated,
        resources=output_resources or resources,
        dependencies=prior_sinks,
        composite=len(components) > 1,
        case=case,
        classification=classification,
        source_resources=resources,
        resource_from_state=(
            prior_sinks[-1]
            if len(components) > 1
            and components[-1] == "drive.transfer_or_share"
            and "multi_source_aggregate" in components
            else None
        ),
    )
    state_nodes.append(terminal)
    authorization_policy = build_instance_authorization_policy(classification, state_nodes)
    if authorization_policy is not None:
        execution_ids = set(authorization_policy["execution_state_ids"])
        for node in state_nodes:
            if str(node["id"]) in execution_ids:
                node["authorization_guard"] = "confirmed_or_preauthorized"
    node_by_id = {str(node["id"]): node for node in state_nodes}

    _assert_acyclic(node_by_id)
    return {
        "schema": "migoo-task-state-instance-v1",
        "case_id": case.case_id,
        "iid": case.iid,
        "domain": case.domain,
        "query": case.query,
        "family": classification.family,
        "template_id": classification.family,
        "component_families": list(components),
        "template_version": template_version,
        "verifier_version": verifier_version,
        "classification": classification.to_dict(),
        "bindings": {
            "requested_resources": resources,
            "mentioned_resources": quoted_mentions(case.query),
            "output_resources": output_resources,
            "destination_resources": destination_resource_names(case.query),
            "share_targets": share_targets(case.query),
            "comment_targets": comment_targets(case.query),
            "output_format": requested_output_format(case.query),
            "search_constraints": search_constraints(sanitized_query),
            "requested_view": requested_view_constraints(sanitized_query),
            "edit_constraints": instance_edit_constraints,
            "content_constraints": content_constraints(sanitized_query),
            "comment_constraints": instance_comment_constraints,
            "share_constraints": share_constraints(sanitized_query),
            "seed_resource_names": seed_resource_names(case.seed_summary),
            "seed_data_file": case.metadata.get("seed_data_file"),
            "benchmark_description": case.metadata.get("description") or case.metadata.get("benchmark_description"),
        },
        "state_graph": state_nodes,
        "authorization_policy": authorization_policy,
        "completion_rule": "all required state nodes pass deterministic checks and semantic judge criteria",
    }


def _assert_acyclic(nodes: dict[str, JsonObject]) -> None:
    unknown = {
        dependency for node in nodes.values() for dependency in node.get("depends_on") or [] if dependency not in nodes
    }
    if unknown:
        raise ValueError(f"state graph has unknown dependencies: {sorted(unknown)}")
    indegree = {node_id: 0 for node_id in nodes}
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node_id, node in nodes.items():
        for dependency in node.get("depends_on") or []:
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
    if visited != len(nodes):
        raise ValueError("state graph contains a cycle")


def compact_case_catalog(case: CaseRecord, classification: CaseClassification) -> JsonObject:
    return {
        "domain": case.domain,
        "case_id": case.case_id,
        "query": sanitize_value(case.query, max_string=700),
        "description": sanitize_value(
            str(case.metadata.get("description") or case.metadata.get("benchmark_description") or ""),
            max_string=300,
        ),
        "difficulty": case.metadata.get("difficulty") or case.metadata.get("benchmark_difficulty"),
        "tags": case.metadata.get("tags") or case.metadata.get("benchmark_tags") or [],
        "classification": classification.to_dict(),
        "requested_resources": requested_resources(case, classification),
        "seed_resources": seed_resource_names(case.seed_summary)[:20],
        "rollout_count": len(case.rollouts),
        "benchmark_pass_count": sum(rollout.benchmark_pass for rollout in case.rollouts),
    }


def instance_fingerprint(instance: JsonObject) -> str:
    return json.dumps(instance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
