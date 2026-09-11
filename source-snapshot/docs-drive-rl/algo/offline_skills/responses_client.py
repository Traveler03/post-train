"""Direct Responses API client using endpoint credentials from a TOML config."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib
from openai import OpenAI

from .models import JsonObject


@dataclass(frozen=True)
class ResponsesProviderConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    reasoning_effort: str | None = None
    service_tier: str | None = None
    provider_name: str = ""
    wire_api: str = "responses"

    @classmethod
    def from_toml(
        cls,
        path: Path,
        *,
        model_override: str | None = None,
        reasoning_effort_override: str | None = None,
    ) -> ResponsesProviderConfig:
        raw = tomllib.loads(path.expanduser().read_text(encoding="utf-8"))
        provider_name = str(raw.get("model_provider") or "").strip()
        providers = raw.get("model_providers") if isinstance(raw.get("model_providers"), dict) else {}
        provider = providers.get(provider_name) if isinstance(providers.get(provider_name), dict) else {}
        api_key = str(
            os.environ.get("CODEX_API_KEY")
            or provider.get("experimental_bearer_token")
            or raw.get("experimental_bearer_token")
            or ""
        ).strip()
        base_url = str(os.environ.get("CODEX_BASE_URL") or provider.get("base_url") or "").strip()
        model = str(model_override or os.environ.get("CODEX_MODEL") or raw.get("model") or "").strip()
        wire_api = str(provider.get("wire_api") or "responses").strip()
        if not api_key:
            raise ValueError(f"Responses API bearer token is missing from {path}")
        if not base_url:
            raise ValueError(f"Responses API base_url is missing from {path}")
        if not model:
            raise ValueError(f"Responses API model is missing from {path}")
        if wire_api != "responses":
            raise ValueError(f"offline skill generation requires wire_api=responses, found {wire_api!r}")
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            reasoning_effort=reasoning_effort_override or raw.get("model_reasoning_effort"),
            service_tier=raw.get("service_tier"),
            provider_name=provider_name,
            wire_api=wire_api,
        )

    def public_metadata(self) -> JsonObject:
        return {
            "provider": self.provider_name,
            "base_url": self.base_url,
            "model": self.model,
            "wire_api": self.wire_api,
            "reasoning_effort": self.reasoning_effort,
            "service_tier": self.service_tier,
            "credential_source": "config_toml_or_environment",
        }


@dataclass(frozen=True)
class GeneratedResponse:
    content: JsonObject
    response_id: str
    model: str
    status: str
    usage: JsonObject
    elapsed_s: float
    attempts: int


class DirectResponsesClient:
    def __init__(
        self,
        config: ResponsesProviderConfig,
        *,
        timeout_s: float = 300.0,
        max_attempts: int = 3,
    ) -> None:
        self.config = config
        self.max_attempts = max(1, max_attempts)
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=timeout_s,
            max_retries=0,
        )

    def generate_json(
        self,
        *,
        instructions: str,
        prompt: str,
        schema_name: str,
        schema: JsonObject,
        max_output_tokens: int,
    ) -> GeneratedResponse:
        last_error: BaseException | None = None
        started = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "instructions": instructions,
                    "input": prompt,
                    "max_output_tokens": max_output_tokens,
                    "store": False,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        }
                    },
                }
                if self.config.reasoning_effort:
                    kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
                if self.config.service_tier:
                    kwargs["service_tier"] = self.config.service_tier
                response = self._client.responses.create(**kwargs)
                if str(response.status) != "completed":
                    raise RuntimeError(f"Responses API status is {response.status!r}")
                content = json.loads(response.output_text)
                if not isinstance(content, dict):
                    raise ValueError("structured response is not a JSON object")
                usage = response.usage.model_dump(mode="json") if response.usage is not None else {}
                return GeneratedResponse(
                    content=content,
                    response_id=str(response.id),
                    model=str(response.model),
                    status=str(response.status),
                    usage=usage,
                    elapsed_s=time.monotonic() - started,
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(min(20.0, 2.0 ** (attempt - 1)))
        assert last_error is not None
        raise RuntimeError(
            f"direct Responses API generation failed after {self.max_attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error
