"""Deterministically compile semantic plans into control command documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts.grounded_task import GroundedTask
from ..contracts.skill_plan import SkillPlan
from .contracts import ExecutionBundle
from .interaction_registry_builder import build_authored_registry


REGION_TO_ANCHOR = {
    "grasp_region": "grasp",
    "container_interior": "interior",
    "button_surface": "button_surface",
}


def compile_execution_bundle(
    skill_plan: SkillPlan,
    grounded_task: GroundedTask,
    *,
    scene_path: str | Path,
    interaction_registry_path: str | Path,
    output_dir: str | Path,
    route: str,
    robot: str = "ur5e",
) -> ExecutionBundle:
    try:
        from robot_agent_control import CommandDocument, ExecutionOptions, SkillCommand
        from robot_agent_control.contracts import scene_sha256
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("robot_agent_control"):
            raise ValueError(
                "control dependency is not installed; run "
                "python -m pip install -e /home/cscvlab/lht/robot-agent-control"
            ) from exc
        raise

    if robot != "ur5e":
        raise ValueError("control execution currently supports ur5e only")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = Path(scene_path).expanduser().resolve()
    registry_output = build_authored_registry(
        scene, interaction_registry_path, output / "interaction_registry.json"
    )
    registry = json.loads(registry_output.read_text(encoding="utf-8"))
    objects = registry["objects"]
    operations = {item.operation_id: item for item in grounded_task.operations}

    commands: list[SkillCommand] = []

    def add(skill: str, target: str, source_step: str, **parameters: Any) -> None:
        commands.append(
            SkillCommand(
                command_id=f"command-{len(commands) + 1:03d}",
                source_skill_step_id=source_step,
                skill_name=skill,
                parameters={"target": target, **parameters},
            )
        )

    for step in skill_plan.steps:
        operation = operations[step.operation_id]
        target = step.target_object
        if step.skill_name == "search":
            raise ValueError("UNRESOLVED_PERCEPTION_STEP: search must finish before execution")
        if not target:
            raise ValueError(f"skill {step.step_id} has no grounded target")
        if target not in objects and target != "home":
            raise ValueError(f"TARGET_NOT_FOUND: {target}")

        if step.skill_name == "locate":
            add("locate", target, step.step_id)
        elif step.skill_name == "move":
            spatial = objects[target].get("spatial", {})
            desired = REGION_TO_ANCHOR.get(step.semantic_target or "")
            anchors = spatial.get("anchors", {})
            anchor = desired if desired in anchors else spatial.get("default_anchor")
            parameters = {"planning_method": "auto"}
            if anchor:
                parameters["anchor"] = anchor
            add("move", target, step.step_id, **parameters)
        elif step.skill_name in {"grasp", "release", "press", "pull", "push"}:
            parameters = {}
            if step.reference_object:
                parameters["reference"] = step.reference_object
            add(step.skill_name, target, step.step_id, **parameters)
        else:
            raise ValueError(f"SKILL_UNSUPPORTED: {step.skill_name}")

        # Execution micro-steps are deterministic profile macros. Numeric
        # distances never come from the planner or model.
        if step.skill_name == "grasp" and operation.task_type.value == "pick_and_place":
            add(
                "move",
                "end_effector",
                step.step_id,
                relation="above",
                distance_m=0.125,
                planning_method="linear",
            )
            add("move", "home", step.step_id)
        if step.skill_name == "release":
            if operation.task_type.value == "pick_and_place":
                add(
                    "move",
                    "end_effector",
                    step.step_id,
                    relation="behind",
                    distance_m=0.23,
                    planning_method="linear",
                )
                add("move", "home", step.step_id)
            elif operation.task_type.value in {"open", "close"}:
                add("move", "home", step.step_id)

    fingerprint = scene_sha256(scene)
    command_document = CommandDocument(
        robot="ur5e",
        scene=str(scene),
        registry=str(registry_output),
        scene_fingerprint=fingerprint,
        runtime=ExecutionOptions(),
        request_defaults=registry.get("move_defaults", {}),
        commands=commands,
    )
    commands_path = output / "commands.json"
    commands_path.write_text(command_document.model_dump_json(indent=2), encoding="utf-8")
    bundle = ExecutionBundle(
        robot="ur5e",
        route=route,
        task_dir=str(output),
        scene=str(scene),
        scene_fingerprint=fingerprint,
        interaction_registry=str(registry_output),
        commands=str(commands_path),
    )
    (output / "execution_bundle.json").write_text(
        bundle.model_dump_json(indent=2), encoding="utf-8"
    )
    return bundle


def compile_directory(
    task_dir: str | Path,
    *,
    scene_path: str | Path | None = None,
    interaction_registry_path: str | Path | None = None,
) -> ExecutionBundle:
    directory = Path(task_dir).expanduser().resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    scene = Path(scene_path).resolve() if scene_path else Path(summary.get("source_scene") or directory / "scene.xml").resolve()
    authored = interaction_registry_path or summary.get("interaction_registry")
    if authored is None:
        raise ValueError("execution_metadata_missing: provide an authored interaction registry")
    return compile_execution_bundle(
        SkillPlan.model_validate_json((directory / "skill_plan.json").read_text(encoding="utf-8")),
        GroundedTask.model_validate_json((directory / "grounded_task.json").read_text(encoding="utf-8")),
        scene_path=scene,
        interaction_registry_path=authored,
        output_dir=directory,
        route=str(summary["route"]),
        robot=str(json.loads((directory / "scene_registry.json").read_text(encoding="utf-8"))["robot"]),
    )
