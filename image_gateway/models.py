from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RATIO_RE = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")


def validate_ratio(value: str) -> str:
    if not RATIO_RE.fullmatch(value):
        raise ValueError("aspect ratio must be a positive width:height string")
    return value


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
