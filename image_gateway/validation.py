from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from .models import ArtifactResult, OptimizerResult


class StrictJSONError(ValueError):
    """Raised when a job output is not the exact small JSON contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_limited_json(
    raw: str | bytes, *, max_bytes: int = 4096, validator: Callable[[dict[str, Any]], Any]
) -> Any:
    encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(encoded) > max_bytes:
        raise StrictJSONError(f"JSON output exceeds {max_bytes} bytes")
    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError) as exc:
        raise StrictJSONError(f"invalid JSON output: {exc}") from exc
    if not isinstance(value, dict):
        raise StrictJSONError("JSON output must be an object")
    try:
        return validator(value)
    except (ValidationError, ValueError, TypeError) as exc:
        raise StrictJSONError(str(exc)) from exc


def parse_optimizer_output(raw: str | bytes) -> OptimizerResult:
    encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(encoded) > 4096:
        raise StrictJSONError("optimizer output exceeds 4096 bytes")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJSONError(f"invalid optimizer output encoding: {exc}") from exc
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise StrictJSONError("optimizer output did not contain a JSON object")
    value = parse_limited_json(
        text[start : end + 1], validator=OptimizerResult.model_validate
    )
    if set(value.model_dump()) != {"rewritten_prompt", "wh_ratio"}:
        raise StrictJSONError("optimizer output must contain only rewritten_prompt and wh_ratio")
    return value


def parse_artifact_output(raw: str | bytes) -> ArtifactResult:
    value = parse_limited_json(raw, validator=ArtifactResult.model_validate)
    if set(value.model_dump()) != {"key", "sha256", "content_type", "width", "height"}:
        raise StrictJSONError("artifact output has an unexpected field")
    return value
