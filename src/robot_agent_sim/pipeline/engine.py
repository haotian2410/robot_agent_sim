from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ..assets.registry import AssetRegistry
from ..assets.resolver import AssetResolver
from ..backends.mujoco.backend import MujocoSceneBackend
from ..contracts.grounded_task import GroundedEntity, GroundedTask
from ..contracts.task_intent import TaskStatus, TaskType
from ..grounding.iou import match_detections
from ..models.fake import FakeSkillPlanningProvider, FakeTaskUnderstandingProvider, FakeVisionGroundingProvider
from ..models.skill_planning import SkillPlanningRequest
from ..models.task_understanding import TaskUnderstandingRequest
from ..models.vision_grounding import VisionGroundingRequest
from ..scene.composer import SceneComposer
from ..skills.registry import REGISTRY


class PipelineResult(BaseModel):
    task_intent: dict
    scene_registry: dict = Field(default_factory=dict)
    grounded_task: dict | None = None
    visual_grounding: dict | None = None
    skill_plan: dict | None = None
    model_call_count: int = 0
    status: str = "accepted"
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class PipelineEngine:
    def __init__(self, understanding=None, vision=None, planner=None, asset_registry=None):
        self.understanding = understanding or FakeTaskUnderstandingProvider()
        self.vision = vision or FakeVisionGroundingProvider()
        self.planner = planner or FakeSkillPlanningProvider()
        self.assets = asset_registry or AssetRegistry()
        self.backend = MujocoSceneBackend()

    def plan(self, instruction: str, robot: str = "panda", scene: Path | None = None, seed: int = 0, output_dir: Path | str | None = None) -> PipelineResult:
        calls = 0
        out = Path(output_dir or "var")
        out.mkdir(parents=True, exist_ok=True)
        intent = None
        registry = None
        visual_grounding = None
        try:
            calls += 1
            intent = self.understanding.understand(TaskUnderstandingRequest(
                instruction=instruction,
                supported_task_types=[item.value for item in TaskType if item != TaskType.MIXED],
                supported_directions=["left", "right", "front", "back", "up", "down"],
                asset_catalog=self.assets.catalog() if scene is None else [],
            ))
            if intent.status != TaskStatus.ACCEPTED:
                return self._write_result(PipelineResult(task_intent=intent.model_dump(mode="json"), status=intent.status.value, model_call_count=calls), out)

            if scene is None:
                assets = AssetResolver(self.assets).resolve_entities(intent.entities)
                registry = SceneComposer().compose(intent, assets, robot, seed)
                xml_path = self.backend.compose(registry, assets, out)
                observation = self.backend.renderer.render(xml_path, registry, out)
                grounded = []
                for entity in intent.entities:
                    object_id = registry.bindings.get(entity.entity_id)
                    if object_id is None: raise ValueError(f"no selected object for entity {entity.entity_id}")
                    obj = registry.by_object_id(object_id)
                    instance = next(item for item in observation.instances if item.object_id == object_id)
                    grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=object_id, body_name=obj.body_name, role=entity.role, model_id=obj.model_id, model_name=obj.model_name, grounding_method="asset_scene_binding", instance_bbox=instance.bbox))
            else:
                registry = self.backend.load_uploaded(Path(scene), robot)
                observation = self.backend.renderer.render(Path(scene), registry, out)
                calls += 1
                detection = self.vision.detect(VisionGroundingRequest(instruction=instruction, entities=[entity.model_dump(mode="json") for entity in intent.entities], rgb_path=str(observation.rgb_path)))
                truth = [{"object_id": item.object_id, "bbox": item.bbox} for item in observation.instances if item.bbox is not None]
                expected_entities = {entity.entity_id for entity in intent.entities}
                relevant_detections = [item for item in detection.detections if item.entity_id in expected_entities]
                ignored_detections = [item for item in detection.detections if item.entity_id not in expected_entities]
                matches, unmatched, ambiguous = match_detections(relevant_detections, truth)
                visual_grounding = {
                    "status": detection.status,
                    "detections": [item.model_dump(mode="json") for item in detection.detections],
                    "ignored_detections": [item.model_dump(mode="json") for item in ignored_detections],
                    "truth": truth,
                    "matches": [
                        {"entity_id": entity_id, "object_id": object_id, "iou": score}
                        for entity_id, object_id, score in matches
                    ],
                    "unmatched": unmatched,
                    "ambiguous": ambiguous,
                    "minimum_iou": 0.20,
                    "ambiguity_margin": 0.05,
                    "explanation": detection.explanation,
                }
                if detection.status != "accepted" or unmatched or ambiguous:
                    status = "grounding_ambiguous" if ambiguous else "grounding_failed"
                    result = PipelineResult(task_intent=intent.model_dump(mode="json"), scene_registry=registry.model_dump(mode="json"), visual_grounding=visual_grounding, status=status, model_call_count=calls, error=f"vision_status={detection.status}; unmatched={unmatched}; ambiguous={ambiguous}")
                    self._add_observation_artifacts(result.artifacts, observation)
                    return self._write_result(result, out)
                by_entity = {entity_id: (object_id, score) for entity_id, object_id, score in matches}
                missing_entities = {entity.entity_id for entity in intent.entities} - set(by_entity)
                if missing_entities:
                    result = PipelineResult(
                        task_intent=intent.model_dump(mode="json"),
                        scene_registry=registry.model_dump(mode="json"),
                        visual_grounding=visual_grounding,
                        status="grounding_failed",
                        model_call_count=calls,
                        error=f"missing required detections: {sorted(missing_entities)}",
                    )
                    self._add_observation_artifacts(result.artifacts, observation)
                    return self._write_result(result, out)
                grounded = []
                for entity in intent.entities:
                    object_id, score = by_entity[entity.entity_id]
                    instance = next(item for item in observation.instances if item.object_id == object_id)
                    detected = next(item for item in relevant_detections if item.entity_id == entity.entity_id)
                    grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=object_id, body_name=instance.body_name, role=entity.role, grounding_method="vlm_iou", detection_bbox=tuple(detected.bbox), instance_bbox=instance.bbox, bbox_iou=score))

            task = GroundedTask(instruction=intent.instruction, task_types=intent.task_types, entities=grounded, operations=intent.operations, spatial_relations=intent.spatial_relations, scene_id=registry.scene_id)
            calls += 1
            skill = self.planner.plan(SkillPlanningRequest(task=task, skill_catalog=[item.model_dump(mode="json") for item in REGISTRY.list()]))
            skill.model_call_count = calls
            self._validate_skill_targets(skill, task)
            result = PipelineResult(task_intent=intent.model_dump(mode="json"), scene_registry=registry.model_dump(mode="json"), grounded_task=task.model_dump(mode="json"), visual_grounding=visual_grounding, skill_plan=skill.model_dump(mode="json"), model_call_count=calls, status="accepted")
            self._add_observation_artifacts(result.artifacts, observation)
            if scene is None:
                result.artifacts["scene.xml"] = str(xml_path)
            return self._write_result(result, out)
        except (OSError, ValueError, KeyError, RuntimeError, ValidationError) as exc:
            text = str(exc)
            status = "asset_missing" if "asset_missing" in text else "planning_failed"
            return self._write_result(PipelineResult(task_intent=intent.model_dump(mode="json") if intent else {"instruction": instruction}, scene_registry=registry.model_dump(mode="json") if registry else {}, status=status, model_call_count=calls, error=text), out)

    @staticmethod
    def _validate_skill_targets(skill, task):
        object_ids = {entity.object_id for entity in task.entities}
        names = [step.skill_name for step in skill.steps]
        for step in skill.steps:
            REGISTRY.require(step.skill_name)
            if step.target_object and step.target_object not in object_ids: raise ValueError(f"skill target is not grounded: {step.target_object}")
            if step.reference_object and step.reference_object not in object_ids: raise ValueError(f"skill reference is not grounded: {step.reference_object}")
            if step.skill_name in {"locate", "search", "move", "grasp", "release", "press"} and not step.target_object:
                raise ValueError(f"skill {step.skill_name} requires target_object")

        # The planner is free to choose wording and semantic_target values,
        # but the canonical high-level order is deterministic and checked at
        # this boundary.  This catches malformed model responses before they
        # are written as a seemingly valid plan.
        task_type_values = {
            item.value if hasattr(item, "value") else str(item)
            for item in task.task_types
        }
        for task_type, pattern in (
            ("pick_and_place", ("locate", "move", "grasp", "locate", "move", "release")),
            ("press", ("locate", "move", "press")),
            ("grasp", ("locate", "move", "grasp")),
            ("locate", ("locate",)),
            ("move", ("locate", "move")),
            ("release", ("locate", "release")),
            ("search", ("search",)),
        ):
            if task_type in task_type_values and not _contains_subsequence(names, pattern):
                raise ValueError(f"invalid skill order for task type {task_type}: expected {pattern}")

    @staticmethod
    def _write_result(result, out):
        payloads = {
            "task_intent.json": result.task_intent,
            "scene_registry.json": result.scene_registry,
            "grounded_task.json": result.grounded_task,
            "visual_grounding.json": result.visual_grounding,
            "skill_plan.json": result.skill_plan,
            "summary.json": {"status": result.status, "model_call_count": result.model_call_count, "error": result.error},
        }
        # Do not leave a previous successful plan looking current after an
        # unsupported or grounding-failed run in the same output directory.
        for name in (*payloads, "visual_grounding.json"):
            path = out / name
            if payloads.get(name) is None and path.exists():
                path.unlink()
        for name, payload in payloads.items():
            if payload is not None:
                path = out / name
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                result.artifacts[name] = str(path)
        return result

    @staticmethod
    def _add_observation_artifacts(artifacts, observation):
        """Expose renderer outputs in the pipeline result without coupling schemas to paths."""
        for name, path in {
            "rgb.png": observation.rgb_path,
            "segmentation.npy": observation.segmentation_path,
            "segmentation.png": observation.segmentation_visualization_path,
            "instances.json": observation.instance_index_path,
        }.items():
            artifacts[name] = str(path)


def _contains_subsequence(values, pattern):
    """Return whether *pattern* occurs in order (possibly with extra steps)."""
    cursor = 0
    for value in values:
        if cursor < len(pattern) and value == pattern[cursor]:
            cursor += 1
    return cursor == len(pattern)
