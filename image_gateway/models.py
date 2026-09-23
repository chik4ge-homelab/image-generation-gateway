from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RATIO_RE = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")


def validate_ratio(value: str) -> str:
    if not RATIO_RE.fullmatch(value):
        raise ValueError("aspect ratio must be a positive width:height string")
    return value


class ImageJobCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    idempotency_key: str = Field(alias="idempotencyKey", min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=8192)
    aspect_ratio: str = Field(default="1:1", alias="aspectRatio")
    steps: int = Field(default=40, ge=1, le=100)
    seed: int = Field(default=42, ge=-(2**63), le=2**63 - 1)

    @field_validator("idempotency_key", "prompt")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("aspect_ratio")
    @classmethod
    def check_ratio(cls, value: str) -> str:
        return validate_ratio(value)


class OpenAIImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=8192)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=1)
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"
    quality: Literal["standard", "hd"] | None = None
    style: Literal["vivid", "natural"] | None = None
    user: str | None = None

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("size")
    @classmethod
    def check_size(cls, value: str) -> str:
        if value not in OPENAI_SIZE_RATIOS:
            raise ValueError("size is not supported by this image-generation endpoint")
        return value


OPENAI_SIZE_RATIOS = {
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1216x832": "3:2",
    "832x1216": "2:3",
    "1344x768": "16:9",
    "768x1344": "9:16",
}


class ImageJobSpec(ImageJobCreate):
    suspend: bool = True


class OptimizerStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    rewritten_prompt: str = Field(alias="rewrittenPrompt")
    wh_ratio: str = Field(alias="whRatio")


class ArtifactStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    key: str
    sha256: str
    content_type: str = Field(alias="contentType")
    width: int
    height: int


class FailureStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    reason: str
    message: str


class ImageJobResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    job_id: str
    name: str
    phase: Literal[
        "Pending",
        "StoppingText",
        "Optimizing",
        "Generating",
        "RestoringText",
        "Succeeded",
        "Failed",
    ]
    spec: ImageJobSpec
    optimizer: OptimizerStatus | None = None
    artifact: ArtifactStatus | None = None
    failure: FailureStatus | None = None
    conditions: list[dict[str, Any]] | None = None


class ArtifactResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    artifact: ArtifactStatus
    url: str | None = None


class ProbeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "ready"]


class OptimizerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_prompt: str = Field(min_length=1, alias="rewritten_prompt")
    wh_ratio: str = Field(alias="wh_ratio")

    @field_validator("rewritten_prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rewritten_prompt must not be blank")
        return value

    @field_validator("wh_ratio")
    @classmethod
    def ratio_is_valid(cls, value: str) -> str:
        return validate_ratio(value)


class ArtifactResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    content_type: str = Field(alias="content_type", min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @field_validator("key")
    @classmethod
    def key_is_relative(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("artifact key must be a relative object key")
        return value


TERMINAL_PHASES = frozenset({"Succeeded", "Failed"})
PHASES = frozenset(
    {"Pending", "StoppingText", "Optimizing", "Generating", "RestoringText", "Succeeded", "Failed"}
)
