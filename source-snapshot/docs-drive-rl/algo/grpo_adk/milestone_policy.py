"""Resolve confirmation state and select the active milestone graph branch."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from algo.offline_skills.authorization import (
    AUTHORIZATION_POLICY_VERSION,
    infer_instance_authorization_policy,
    infer_requested_mutation_operations,
    materialize_authorization_graph,
)
from algo.offline_skills.models import JsonObject

_MSG_TIME = re.compile(r"<msg_time>.*?</msg_time>\s*", re.IGNORECASE | re.DOTALL)
_INJECTED_CONTEXT_PREFIXES = (
    "<connectors",
    "<system-reminder",
    "<user-context",
    "<query-context",
    "<temporal_context",
)
_CONFIRMATION_REQUEST = re.compile(
    r"(?:\bconfirm(?:ation)?\b|\bwrite this\b|\bsend this\b|\bproceed\b|"
    r"确认|是否(?:执行|继续|写入|修改|删除|分享|创建)|(?:可以|要我).{0,12}(?:执行|写入|修改|删除|分享|创建))",
    re.IGNORECASE,
)
_EXPLICIT_CONFIRMATION = re.compile(
    r"^(?:"
    r"yes|y|ok(?:ay)?|confirm(?:ed)?|send|go(?: ahead)?|proceed|do it|continue|approved|"
    r"是|对|确认|好的?|可以|执行|继续|同意|批准|就这样|就这么做|"
    r"确认[，,、\s].*|好的?[，,、\s].*|对[，,、\s].*|可以[，,、\s].*|"
    r"已授权[，,、\s].*|按.{0,80}(?:执行|处理|修改|写入)|.*(?:全撤|都收回)"
    r")[。.!！?？\s]*$",
    re.IGNORECASE | re.DOTALL,
)
_EXPLICIT_REJECTION = re.compile(
    r"^(?:no|cancel|stop|don't|do not|不用了?|不要|取消|停止|别(?:执行|写|改|删|分享)|不确认)"
    r"[。.!！?？\s]*$",
    re.IGNORECASE,
)
_OVERRIDE_STATES = {
    "confirmation_required",
    "ambiguous",
    "confirmed",
    "preauthorized",
    "rejected",
}


@dataclass(frozen=True)
class AuthorizationContext:
    authorization_state: str
    source: str
    policy_version: str
    prior_confirmation_request: bool
    current_user_message: str
    mutation_operations: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        return asdict(self)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "message", "query"):
            if isinstance(value.get(key), str):
                return str(value[key])
    return ""


def _strip_message_time(value: str) -> str:
    return _MSG_TIME.sub("", value).strip()


def conversation_history(extra_info: JsonObject) -> list[JsonObject]:
    """Return actual user/assistant messages without injected runtime context blocks."""

    result: list[JsonObject] = []
    for raw_message in extra_info.get("adk_history_context") or []:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role") or "").strip().lower()
        content = _text(raw_message.get("content"))
        if role not in {"user", "assistant"} or not content.strip():
            continue
        if content.lstrip().lower().startswith(_INJECTED_CONTEXT_PREFIXES):
            continue
        result.append({"role": role, "content": content})
    return result


def current_user_message(extra_info: JsonObject) -> str:
    value = _text(extra_info.get("adk_user_query_with_msg_time"))
    if not value:
        value = _text(extra_info.get("question"))
    return _strip_message_time(value)


def _prior_confirmation_request(history: list[JsonObject]) -> bool:
    assistant_messages = [str(message["content"]) for message in history if message.get("role") == "assistant"]
    if not assistant_messages:
        return False
    return bool(_CONFIRMATION_REQUEST.search(assistant_messages[-1]))


def resolve_authorization_context(instance: JsonObject, extra_info: JsonObject) -> AuthorizationContext:
    policy = infer_instance_authorization_policy(instance)
    mutation_operations = tuple(sorted(infer_requested_mutation_operations(instance)))
    current_message = current_user_message(extra_info)
    if policy is None:
        if mutation_operations:
            return AuthorizationContext(
                authorization_state="confirmation_required",
                source="boundary_case_has_external_mutation_intent",
                policy_version=AUTHORIZATION_POLICY_VERSION,
                prior_confirmation_request=False,
                current_user_message=current_message,
                mutation_operations=mutation_operations,
            )
        return AuthorizationContext(
            authorization_state="not_required",
            source="task_has_no_external_mutation",
            policy_version=AUTHORIZATION_POLICY_VERSION,
            prior_confirmation_request=False,
            current_user_message=current_message,
            mutation_operations=(),
        )

    history = conversation_history(extra_info)
    prior_request = _prior_confirmation_request(history)
    override = str(
        extra_info.get("adk_authorization_state") or extra_info.get("authorization_state") or ""
    ).strip().lower()
    if override:
        if override not in _OVERRIDE_STATES:
            raise ValueError(f"unsupported authorization-state override: {override!r}")
        return AuthorizationContext(
            authorization_state=override,
            source="trusted_runtime_override",
            policy_version=str(policy.get("version") or AUTHORIZATION_POLICY_VERSION),
            prior_confirmation_request=prior_request,
            current_user_message=current_message,
            mutation_operations=mutation_operations,
        )

    if prior_request and _EXPLICIT_CONFIRMATION.fullmatch(current_message):
        state = "confirmed"
        source = "later_user_confirmation"
    elif prior_request and _EXPLICIT_REJECTION.fullmatch(current_message):
        state = "rejected"
        source = "later_user_rejection"
    elif prior_request:
        state = "ambiguous"
        source = "later_user_message_not_explicit"
    else:
        state = str(policy.get("default_authorization_state") or "confirmation_required")
        source = "initial_request_is_intent"
    return AuthorizationContext(
        authorization_state=state,
        source=source,
        policy_version=str(policy.get("version") or AUTHORIZATION_POLICY_VERSION),
        prior_confirmation_request=prior_request,
        current_user_message=current_message,
        mutation_operations=mutation_operations,
    )


def prepare_policy_aware_instance(
    instance: JsonObject,
    extra_info: JsonObject,
) -> tuple[JsonObject, AuthorizationContext]:
    context = resolve_authorization_context(instance, extra_info)
    active = materialize_authorization_graph(instance, context.authorization_state)
    active["authorization_context"] = context.to_dict()
    return active, context
