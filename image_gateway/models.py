from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ProbeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "ready"]


TERMINAL_PHASES = frozenset({"Succeeded", "Failed"})
