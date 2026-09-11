"""Versioned deterministic state skeletons for each task family."""

from __future__ import annotations

from .models import FamilyBlueprint, StateBlueprint, StateCheck

TEMPLATE_VERSION = "1.4.0"
VERIFIER_VERSION = "task-state-verifier-v5"


def _tool(*patterns: str, min_count: int = 1, description: str) -> StateCheck:
    return StateCheck(
        kind="tool_called",
        tool_patterns=patterns,
        min_count=min_count,
        description=description,
    )


def _response(description: str) -> StateCheck:
    return StateCheck(kind="response_nonempty", min_count=1, description=description)


def _no_mutation(description: str) -> StateCheck:
    return StateCheck(
        kind="forbidden_tool_absent",
        tool_patterns=(
            "google_docs_create",
            "google_docs_edit",
            "google_docs_find_replace",
            "google_docs_insert",
            "google_docs_delete",
            "google_docs_sed",
            "google_docs_update",
            "google_docs_write",
            "google_docs_clear",
            "google_docs_comments_add",
            "google_docs_comments_delete",
            "google_docs_comments_resolve",
            "google_drive_copy",
            "google_drive_delete",
            "google_drive_mkdir",
            "google_drive_move",
            "google_drive_share",
            "google_drive_upload",
            "google_sheets_create",
            "google_sheets_update",
            "upload_file",
        ),
        min_count=0,
        description=description,
    )


BLUEPRINTS: dict[str, FamilyBlueprint] = {
    "docs.read": FamilyBlueprint(
        family="docs.read",
        purpose="Locate a Google Doc, inspect the requested view, and answer only from retrieved evidence.",
        states=(
            StateBlueprint(
                id="resolve_document",
                required="always",
                applies_when="all cases",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool(
                        "google_drive_search",
                        "google_drive_get",
                        "google_docs_cat",
                        "google_docs_info",
                        "google_docs_structure",
                        description="A concrete document or requested view is resolved.",
                    ),
                ),
            ),
            StateBlueprint(
                id="inspect_requested_view",
                required="conditional",
                required_operations=("read_content", "read_metadata", "read_comments"),
                applies_when="the request asks for content, structure, metadata, or comments beyond search results",
                depends_on=("resolve_document",),
                expansion="none",
                checks=(
                    _tool(
                        "google_docs_cat",
                        "google_docs_info",
                        "google_docs_structure",
                        "google_docs_comments_*",
                        "google_drive_get",
                        description="The exact requested document view is read.",
                    ),
                ),
            ),
            StateBlueprint(
                id="answer_from_evidence",
                required="always",
                applies_when="all cases",
                depends_on=("inspect_requested_view",),
                expansion="none",
                checks=(_response("A user-visible evidence-grounded answer is returned."),),
            ),
        ),
    ),
    "docs.replace_or_edit": FamilyBlueprint(
        family="docs.replace_or_edit",
        purpose="Apply a precisely scoped document edit without changing unrelated content.",
        states=(
            StateBlueprint(
                id="resolve_document",
                required="conditional",
                required_operations=("locate",),
                applies_when="the document id is not already available",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool("google_drive_search", "google_drive_get", description="The target document is resolved."),
                ),
            ),
            StateBlueprint(
                id="inspect_edit_scope",
                required="conditional",
                required_operations=("read_content",),
                applies_when="the edit target depends on current content, position, or occurrence count",
                depends_on=("resolve_document",),
                expansion="none",
                checks=(
                    _tool(
                        "google_docs_cat",
                        "google_docs_structure",
                        description="Current content needed to delimit the edit is inspected.",
                    ),
                ),
            ),
            StateBlueprint(
                id="apply_scoped_edit",
                required="always",
                applies_when="all executable edit cases",
                depends_on=("inspect_edit_scope",),
                expansion="none",
                checks=(
                    _tool(
                        "google_docs_edit",
                        "google_docs_find_replace",
                        "google_docs_insert",
                        "google_docs_delete",
                        "google_docs_sed",
                        "google_docs_update",
                        "google_docs_write",
                        "google_docs_clear",
                        description="A document mutation tool applies the requested scoped change.",
                    ),
                ),
            ),
            StateBlueprint(
                id="report_edit_result",
                required="always",
                applies_when="all cases",
                depends_on=("apply_scoped_edit",),
                expansion="none",
                checks=(_response("The response accurately reports the applied edit and its scope."),),
            ),
        ),
    ),
    "docs.comments": FamilyBlueprint(
        family="docs.comments",
        purpose="Read or mutate document comments while preserving comment identity and permission boundaries.",
        states=(
            StateBlueprint(
                id="resolve_document",
                required="conditional",
                required_operations=("locate",),
                applies_when="the document id is not already available",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool("google_drive_search", "google_drive_get", description="The target document is resolved."),
                ),
            ),
            StateBlueprint(
                id="resolve_comment_context",
                required="always",
                applies_when="all comment read and mutation cases",
                depends_on=("resolve_document",),
                expansion="none",
                checks=(
                    _tool(
                        "google_docs_comments_*",
                        "google_docs_explicit_comments",
                        "google_docs_search_comments",
                        description="The requested comment is read or the requested comment mutation succeeds.",
                    ),
                ),
            ),
            StateBlueprint(
                id="report_comment_result",
                required="always",
                applies_when="all cases",
                depends_on=("resolve_comment_context",),
                expansion="none",
                checks=(_response("The comment content or mutation result is reported without invention."),),
            ),
        ),
    ),
    "docs.create_or_export": FamilyBlueprint(
        family="docs.create_or_export",
        purpose="Create, copy, or export document artifacts in the requested format and destination.",
        states=(
            StateBlueprint(
                id="resolve_source",
                required="conditional",
                required_operations=("locate", "copy", "export", "download"),
                applies_when="the operation uses an existing source document",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool(
                        "google_drive_search",
                        "google_drive_get",
                        "google_docs_cat",
                        description="The source document is unambiguously resolved.",
                    ),
                ),
            ),
            StateBlueprint(
                id="materialize_artifact",
                required="always",
                applies_when="all executable create, copy, and export cases",
                depends_on=("resolve_source",),
                expansion="none",
                checks=(
                    _tool(
                        "google_docs_create",
                        "google_docs_export",
                        "google_drive_copy",
                        "google_drive_download",
                        "google_drive_upload",
                        "download_file",
                        "upload_file",
                        "create_pdf_from_docx",
                        description="The requested document artifact is created or exported.",
                    ),
                ),
            ),
            StateBlueprint(
                id="report_artifact",
                required="always",
                applies_when="all cases",
                depends_on=("materialize_artifact",),
                expansion="none",
                checks=(_response("The response identifies the created artifact and does not overclaim its format."),),
            ),
        ),
    ),
    "drive.retrieve": FamilyBlueprint(
        family="drive.retrieve",
        purpose="Resolve Drive files or folders and return requested metadata or content with evidence.",
        states=(
            StateBlueprint(
                id="locate_drive_resource",
                required="always",
                applies_when="all cases",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool(
                        "google_drive_search",
                        "google_drive_ls",
                        "google_drive_get",
                        "google_drive_drives",
                        "google_drive_url",
                        "google_docs_cat",
                        "google_sheets_get",
                        description="A Drive resource or requested listing is resolved.",
                    ),
                ),
            ),
            StateBlueprint(
                id="inspect_resource",
                required="conditional",
                required_operations=("read_content", "read_metadata", "read_comments"),
                applies_when="the answer requires file contents, metadata, sheet cells, or comments",
                depends_on=("locate_drive_resource",),
                expansion="none",
                checks=(
                    _tool(
                        "google_drive_get",
                        "google_drive_url",
                        "google_docs_cat",
                        "google_docs_comments_*",
                        "google_sheets_get",
                        "google_sheets_metadata",
                        description="The requested resource details are read.",
                    ),
                ),
            ),
            StateBlueprint(
                id="answer_from_evidence",
                required="always",
                applies_when="all cases",
                depends_on=("inspect_resource",),
                expansion="none",
                checks=(_response("The final answer is grounded in retrieved Drive evidence."),),
            ),
        ),
    ),
    "drive.file_operation": FamilyBlueprint(
        family="drive.file_operation",
        purpose="Create, copy, move, upload, download, delete, or export Drive artifacts with exact targets.",
        states=(
            StateBlueprint(
                id="resolve_sources",
                required="conditional",
                required_operations=("locate", "copy", "move", "delete", "download", "export"),
                applies_when="the operation acts on existing resources",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool("google_drive_search", "google_drive_get", description="Source resources are resolved."),
                ),
            ),
            StateBlueprint(
                id="resolve_destination",
                required="conditional",
                required_operations=("move", "upload"),
                applies_when="the operation names or creates a destination",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool(
                        "google_drive_search",
                        "google_drive_get",
                        "google_drive_mkdir",
                        description="The destination is resolved or created.",
                    ),
                ),
            ),
            StateBlueprint(
                id="execute_file_operation",
                required="always",
                applies_when="all executable file-operation cases",
                depends_on=("resolve_sources", "resolve_destination"),
                expansion="none",
                checks=(
                    _tool(
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
                        description="The requested Drive operation succeeds.",
                    ),
                ),
            ),
            StateBlueprint(
                id="report_file_result",
                required="always",
                applies_when="all cases",
                depends_on=("execute_file_operation",),
                expansion="none",
                checks=(_response("The response reports the actual file operation result."),),
            ),
        ),
    ),
    "drive.transfer_or_share": FamilyBlueprint(
        family="drive.transfer_or_share",
        purpose="Inspect or change Drive sharing permissions with exact recipients and roles.",
        states=(
            StateBlueprint(
                id="resolve_resource",
                required="conditional",
                required_operations=("locate",),
                applies_when="the resource id is not already available",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool("google_drive_search", "google_drive_get", description="The shared resource is resolved."),
                ),
            ),
            StateBlueprint(
                id="inspect_or_apply_permissions",
                required="always",
                applies_when="all permission read and mutation cases",
                depends_on=("resolve_resource",),
                expansion="none",
                checks=(
                    _tool(
                        "google_drive_permissions",
                        "google_drive_share",
                        "google_drive_url",
                        description="Permissions are inspected or applied with the requested role.",
                    ),
                ),
            ),
            StateBlueprint(
                id="report_permission_result",
                required="always",
                applies_when="all cases",
                depends_on=("inspect_or_apply_permissions",),
                expansion="none",
                checks=(_response("The response accurately reports permission state or sharing result."),),
            ),
        ),
    ),
    "multi_source_aggregate": FamilyBlueprint(
        family="multi_source_aggregate",
        purpose=(
            "Collect evidence from multiple requested sources, derive a comparison or synthesis, "
            "and optionally materialize it."
        ),
        states=(
            StateBlueprint(
                id="identify_sources",
                required="always",
                applies_when="all cases",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool(
                        "google_drive_search",
                        "google_drive_get",
                        "google_docs_cat",
                        "google_sheets_get",
                        min_count=1,
                        description="The requested source set is identified.",
                    ),
                ),
            ),
            StateBlueprint(
                id="collect_source_evidence",
                required="always",
                applies_when="once per explicitly requested source; otherwise once for the source set",
                depends_on=("identify_sources",),
                expansion="per_requested_resource",
                checks=(
                    _tool(
                        "google_drive_search",
                        "google_drive_get",
                        "google_docs_cat",
                        "google_sheets_get",
                        "google_sheets_metadata",
                        min_count=1,
                        description="Evidence required from each source is retrieved.",
                    ),
                ),
            ),
            StateBlueprint(
                id="derive_aggregate",
                required="always",
                applies_when="all cases",
                depends_on=("collect_source_evidence",),
                expansion="none",
                checks=(_response("The requested comparison, calculation, or synthesis is returned."),),
            ),
            StateBlueprint(
                id="materialize_or_transfer",
                required="conditional",
                required_operations=("create", "copy", "export", "upload", "share"),
                applies_when="the query also asks to create, export, upload, or share the derived result",
                depends_on=("derive_aggregate",),
                expansion="none",
                checks=(
                    _tool(
                        "google_docs_create",
                        "google_docs_write",
                        "google_drive_upload",
                        "google_drive_share",
                        "google_docs_export",
                        "upload_file",
                        description="The derived artifact or transfer is completed.",
                    ),
                ),
            ),
        ),
    ),
    "boundary_or_negative": FamilyBlueprint(
        family="boundary_or_negative",
        purpose=(
            "Detect missing, ambiguous, unsupported, unsafe, or unauthorized preconditions and stop "
            "without harmful mutation."
        ),
        states=(
            StateBlueprint(
                id="establish_preconditions",
                required="conditional",
                required_operations=("locate", "read_content", "read_metadata", "read_comments", "permissions"),
                applies_when="a read-only lookup can establish the boundary",
                depends_on=(),
                expansion="none",
                checks=(
                    _tool(
                        "google_drive_search",
                        "google_drive_get",
                        "google_docs_cat",
                        "google_docs_info",
                        "google_docs_comments_*",
                        "google_drive_permissions",
                        "google_drive_drives",
                        description="Available read-only evidence is used to establish the blocker.",
                    ),
                ),
            ),
            StateBlueprint(
                id="preserve_world_state",
                required="always",
                applies_when="all boundary cases",
                depends_on=("establish_preconditions",),
                expansion="none",
                checks=(_no_mutation("No state-changing tool succeeds while required preconditions are missing."),),
            ),
            StateBlueprint(
                id="explain_boundary",
                required="always",
                applies_when="all boundary cases",
                depends_on=("preserve_world_state",),
                expansion="none",
                checks=(
                    _response(
                        "The response states the blocker and requests only the information or authorization needed."
                    ),
                ),
            ),
        ),
    ),
}


def get_blueprint(family: str) -> FamilyBlueprint:
    try:
        return BLUEPRINTS[family]
    except KeyError as exc:
        raise ValueError(f"unsupported task family: {family}") from exc
