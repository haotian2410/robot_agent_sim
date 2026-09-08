"""Minimal planning-model contract; enrichment stays deterministic."""
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts.grounded_task import GroundedTask
from ..contracts.skill_plan import SkillPlan, SkillStep


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannerOperation(StrictModel):
    id: str
    type: str
    source: str | None = None
    destination: str | None = None
    target: str | None = None
    reference: str | None = None


class PlannerContext(StrictModel):
    operations: list[PlannerOperation]
    goals: list[dict[str, str | None]] = Field(default_factory=list)


class LLMPlanStep(StrictModel):
    skill: str
    target: Literal["source", "destination", "target", "reference"] | None = None
    reference: Literal["source", "destination", "target", "reference"] | None = None
    region: str | None = None


class LLMOperationPlan(StrictModel):
    id: str
    steps: list[LLMPlanStep]


class SkillPlanLLMOutput(StrictModel):
    operations: list[LLMOperationPlan]

    @model_validator(mode="after")
    def unique_operations(self):
        ids = [operation.id for operation in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("planner operation ids must be unique")
        return self


class SkillPlanningRequest(StrictModel):
    context: PlannerContext
    skill_catalog: str


class SkillPlanningProvider(Protocol):
    def plan(self, request: SkillPlanningRequest) -> SkillPlanLLMOutput: ...


def planner_context(task: GroundedTask) -> PlannerContext:
    objects = {entity.entity_id: entity.object_id for entity in task.entities}
    operations = [PlannerOperation(
        id=operation.operation_id, type=operation.task_type.value,
        source=objects.get(operation.source), destination=objects.get(operation.destination),
        target=objects.get(operation.target), reference=objects.get(operation.reference),
    ) for operation in task.operations]
    goals = [{
        "subject": objects.get(relation.subject), "relation": relation.relation.value,
        "reference": objects.get(relation.reference),
    } for relation in task.spatial_relations if relation.scope == "goal"]
    return PlannerContext(operations=operations, goals=goals)


def enrich_skill_plan(output: SkillPlanLLMOutput, task: GroundedTask) -> SkillPlan:
    context = planner_context(task)
    operations = {operation.id: operation for operation in context.operations}
    expected_ids = [operation.operation_id for operation in task.operations]
    if [operation.id for operation in output.operations] != expected_ids:
        raise ValueError("planner must preserve the exact operation order")
    steps: list[SkillStep] = []
    for operation_plan in output.operations:
        operation = operations[operation_plan.id]
        for raw in operation_plan.steps:
            target = getattr(operation, raw.target) if raw.target else None
            reference = getattr(operation, raw.reference) if raw.reference else None
            if raw.target and target is None:
                raise ValueError(f"operation {operation.id} has no {raw.target} object")
            if raw.reference and reference is None:
                raise ValueError(f"operation {operation.id} has no {raw.reference} object")
            step_id = f"step-{len(steps) + 1}"
            steps.append(SkillStep(
                step_id=step_id, operation_id=operation.id, skill_name=raw.skill,
                target_object=target, reference_object=reference, semantic_target=raw.region,
                depends_on=[steps[-1].step_id] if steps else [],
            ))
    return SkillPlan(task_types=task.task_types, steps=steps)
