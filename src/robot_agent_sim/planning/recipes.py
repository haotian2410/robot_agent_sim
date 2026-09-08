from __future__ import annotations

from ..skills.registry import REGISTRY


RECIPES: dict[str, tuple[str, ...]] = {
    "grasp": ("locate", "move", "grasp"),
    "press": ("locate", "move", "press"),
    "pick_and_place": ("locate", "move", "grasp", "locate", "move", "release"),
    "locate": ("locate",),
    "search": ("search",),
    "move": ("locate", "move"),
    "release": ("locate", "release"),
}


def recipe_prompt() -> str:
    return ";".join(f"{name}:{'>'.join(steps)}" for name, steps in RECIPES.items())


def validate_plan(plan, task) -> None:
    object_ids = {entity.object_id for entity in task.entities}
    operation_ids = [operation.operation_id for operation in task.operations]
    positions = {operation_id: index for index, operation_id in enumerate(operation_ids)}
    by_operation = {operation_id: [] for operation_id in operation_ids}
    last_operation_position = -1
    for step in plan.steps:
        definition = REGISTRY.require(step.skill_name)
        if step.operation_id not in by_operation:
            raise ValueError(f"skill references unknown operation: {step.operation_id}")
        if definition.requires_target and not step.target_object:
            raise ValueError(f"skill {step.skill_name} requires target_object")
        if step.target_object and step.target_object not in object_ids:
            raise ValueError(f"skill target is not grounded: {step.target_object}")
        if step.reference_object and step.reference_object not in object_ids:
            raise ValueError(f"skill reference is not grounded: {step.reference_object}")
        if step.semantic_target and definition.allowed_regions and step.semantic_target not in definition.allowed_regions:
            raise ValueError(f"unsupported region for {step.skill_name}: {step.semantic_target}")
        current_position = positions[step.operation_id]
        if current_position < last_operation_position:
            raise ValueError("skill steps must preserve operation order")
        last_operation_position = current_position
        by_operation[step.operation_id].append(step.skill_name)
    for operation in task.operations:
        actual = tuple(by_operation[operation.operation_id])
        expected = RECIPES[operation.task_type.value]
        if actual != expected:
            raise ValueError(f"invalid skill recipe for {operation.operation_id}: expected {expected}, got {actual}")
        for dependency in operation.depends_on:
            if positions[dependency] >= positions[operation.operation_id]:
                raise ValueError("operation dependency order violated")
            if not by_operation[dependency] or not by_operation[operation.operation_id]:
                raise ValueError("operation dependency has no steps")
