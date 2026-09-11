from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExecutionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    robot: Literal["ur5e"]
    route: Literal["A", "B"]
    task_dir: str
    scene: str
    scene_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    interaction_registry: str
    commands: str

    @field_validator("task_dir", "scene", "interaction_registry", "commands")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("execution bundle paths must be absolute")
        return value

