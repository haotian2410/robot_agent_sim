from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from .task_intent import TaskType


class SemanticSubtask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    description: str = Field(min_length=1, max_length=300)


class SkillStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(pattern=r"^step-[0-9]+$")
    semantic_subtask: SemanticSubtask
    skill_name: Literal["locate", "search", "move", "grasp", "release", "press"]
    target_object: str | None = None
    reference_object: str | None = None
    semantic_target: str | None = None
    depends_on: list[str] = Field(default_factory=list)


class SkillPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_types: list[TaskType]
    steps: list[SkillStep]
    model_call_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_steps(self):
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step_id must be unique")
        position = {step_id: index for index, step_id in enumerate(ids)}
        for step in self.steps:
            if step.step_id in step.depends_on:
                raise ValueError("step cannot depend on itself")
            if any(dep not in position or position[dep] >= position[step.step_id] for dep in step.depends_on):
                raise ValueError("depends_on must point to previous steps")
        return self
