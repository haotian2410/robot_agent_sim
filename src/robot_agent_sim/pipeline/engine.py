from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ..assets.registry import AssetRegistry
from ..assets.resolver import AssetResolver
from ..backends.mujoco.backend import MujocoSceneBackend
from ..contracts.grounded_task import GroundedEntity, GroundedTask
from ..contracts.task_intent import TaskStatus
from ..grounding.iou import match_detections
from ..grounding.interaction_registry import ground_with_interaction_registry
from ..grounding.world_relation import WorldRelationResolver
from ..execution.interaction_registry_builder import build_generated_registry
from ..models.budget import ModelCallBudget, ModelCallBudgetExceeded
from ..models.fake import FakeSkillPlanningProvider, FakeTaskUnderstandingProvider, FakeVisionGroundingProvider
from ..models.skill_planning import SkillPlanningRequest, enrich_skill_plan, planner_context
from ..models.task_understanding import TaskUnderstandingRequest, enrich_task
from ..models.vision_grounding import VisionGroundingRequest, VisionQuery
from ..planning.recipes import validate_plan
from ..planning.recipe_planner import RecipePlanner
from ..scene.composer import SceneComposer
from ..skills.registry import REGISTRY


class PipelineResult(BaseModel):
    task_intent: dict
    scene_registry: dict = Field(default_factory=dict)
    grounded_task: dict | None = None
    visual_grounding: dict | None = None
    skill_plan: dict | None = None
    model_call_count: int = 0
    model_usage: dict = Field(default_factory=dict)
    planner: str = "recipe"
    route: str = "A"
    status: str = "accepted"
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    source_scene: str | None = None
    interaction_registry: str | None = None


class PipelineEngine:
    def __init__(self, understanding=None, vision=None, planner=None, asset_registry=None):
        self.understanding = understanding or FakeTaskUnderstandingProvider()
        self.vision = vision or FakeVisionGroundingProvider()
        self.planner = planner or FakeSkillPlanningProvider()
        self.assets = asset_registry or AssetRegistry()
        self.backend = MujocoSceneBackend()

    def plan(self, instruction: str, robot: str = "panda", scene: Path | None = None, seed: int = 0, output_dir: Path | str | None = None, planner: str = "recipe", interaction_registry: Path | None = None) -> PipelineResult:
        out = Path(output_dir or "var"); out.mkdir(parents=True, exist_ok=True)
        intent = None; registry = None; observation = None; visual_grounding = None
        if planner not in {"recipe", "qwen", "auto"}:
            raise ValueError(f"unsupported planner: {planner}")
        budget = ModelCallBudget.for_route(scene is not None, planner)
        route = "B" if scene is not None else "A"
        planner_used = planner
        try:
            budget.consume("task_understanding")
            parsed = self.understanding.understand(TaskUnderstandingRequest(instruction=instruction))
            self._capture(budget, "task_understanding", self.understanding)
            intent = enrich_task(parsed, instruction)
            if intent.status != TaskStatus.ACCEPTED:
                return self._write_result(PipelineResult(task_intent=intent.model_dump(mode="json"), status=intent.status.value, model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route), out)

            if scene is None:
                assets = AssetResolver(self.assets).resolve_entities(intent.entities)
                registry = SceneComposer().compose(intent, assets, robot, seed)
                xml_path = self.backend.compose(registry, assets, out)
                generated_interactions = build_generated_registry(
                    xml_path, registry, out / "interaction_registry.json"
                )
                observation = self.backend.renderer.render(xml_path, registry, out)
                grounded = []
                for entity in intent.entities:
                    object_id = registry.bindings.get(entity.entity_id)
                    if object_id is None: raise ValueError(f"no selected object for entity {entity.entity_id}")
                    item = registry.by_object_id(object_id)
                    instance = next(instance for instance in observation.instances if instance.object_id == object_id)
                    grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=object_id, body_name=item.body_name, model_id=item.model_id, model_name=item.model_name, grounding_method="asset_scene_binding", instance_bbox=instance.bbox))
                asset_bindings = {entity.entity_id: {"model_id": assets[entity.entity_id].model_id, "model_name": assets[entity.entity_id].model_name} for entity in intent.entities}
            else:
                registry = self.backend.load_uploaded(Path(scene), robot)
                observation = self.backend.renderer.render(Path(scene), registry, out)
                if interaction_registry is not None:
                    grounded = ground_with_interaction_registry(
                        intent.entities, interaction_registry, scene
                    )
                    visual_grounding = {
                        "method": "interaction_registry",
                        "registry": str(Path(interaction_registry).resolve()),
                    }
                    asset_bindings = {}
                else:
                    selection_entities = {
                    value for relation in intent.spatial_relations if relation.scope == "selection"
                    for value in (relation.subject, relation.reference) if value
                    }
                    queries = [VisionQuery(id=entity.entity_id, name=entity.semantic_name, category=entity.category, color=entity.color, all=entity.entity_id in selection_entities) for entity in intent.entities]
                    budget.consume("vision_grounding")
                    detection = self.vision.detect(VisionGroundingRequest(entities=queries, rgb_path=str(observation.rgb_path)))
                    self._capture(budget, "vision_grounding", self.vision)
                    truth = [{"object_id": item.object_id, "bbox": item.bbox} for item in observation.instances if item.bbox is not None]
                    expected = {entity.entity_id for entity in intent.entities}
                    relevant = [item for item in detection.detections if item.entity_id in expected]
                    for index, item in enumerate(relevant, 1):
                        if item.detection_id is None:
                            item.detection_id = f"d-{index}"
                    matches, unmatched, ambiguous = match_detections(relevant, truth)
                    visual_grounding = {"detections": [item.model_dump(mode="json") for item in detection.detections], "truth": truth, "matches": [{"detection_id": detection_id, "object_id": object_id, "iou": score} for detection_id, object_id, score in matches], "unmatched": unmatched, "ambiguous": ambiguous, "minimum_iou": 0.2, "ambiguity_margin": 0.05}
                    if unmatched or ambiguous:
                        result = PipelineResult(task_intent=intent.model_dump(mode="json"), scene_registry=registry.model_dump(mode="json"), visual_grounding=visual_grounding, status="grounding_ambiguous" if ambiguous else "grounding_failed", model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, error=f"unmatched={unmatched}; ambiguous={ambiguous}", source_scene=str(Path(scene).resolve()))
                        self._add_observation_artifacts(result.artifacts, observation); return self._write_result(result, out)
                    instance_by_id = {item.object_id: item for item in observation.instances}
                    positions = {item.object_id: item.world_position for item in observation.instances}
                    candidate_map = {entity.entity_id: [] for entity in intent.entities}
                    for detection_id, object_id, score in matches:
                        candidate = next(item for item in relevant if item.detection_id == detection_id)
                        candidate_map[candidate.entity_id].append({"object_id": object_id, "detection_bbox": tuple(candidate.bbox), "instance_bbox": instance_by_id[object_id].bbox, "bbox_iou": score})
                    selected = WorldRelationResolver().resolve(intent, candidate_map, positions)
                    grounded = []
                    for entity in intent.entities:
                        choice = selected[entity.entity_id]; instance = instance_by_id[choice["object_id"]]
                        grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=choice["object_id"], body_name=instance.body_name, grounding_method="vlm_iou", detection_bbox=choice["detection_bbox"], instance_bbox=choice["instance_bbox"], bbox_iou=choice["bbox_iou"]))
                    asset_bindings = {}

            task = GroundedTask(instruction=intent.instruction, task_types=intent.task_types, entities=grounded, operations=intent.operations, spatial_relations=intent.spatial_relations, scene_id=registry.scene_id)
            recipe_supported = RecipePlanner.supports(task)
            if planner == "recipe" or (planner == "auto" and recipe_supported):
                if not recipe_supported:
                    raise ValueError("unsupported_recipe")
                skill = RecipePlanner().plan(task)
                planner_used = "recipe"
            else:
                planner_used = "qwen"
                budget.consume("skill_planning")
                raw_plan = self.planner.plan(SkillPlanningRequest(
                    context=planner_context(task),
                    skill_catalog=REGISTRY.prompt_catalog(),
                ))
                self._capture(budget, "skill_planning", self.planner)
                skill = enrich_skill_plan(raw_plan, task)
                validate_plan(skill, task)
            result = PipelineResult(task_intent=intent.model_dump(mode="json"), scene_registry=registry.model_dump(mode="json"), grounded_task=task.model_dump(mode="json"), visual_grounding=visual_grounding, skill_plan=skill.model_dump(mode="json"), model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, source_scene=str((xml_path if scene is None else Path(scene)).resolve()), interaction_registry=str(generated_interactions) if scene is None else (str(Path(interaction_registry).resolve()) if interaction_registry else None))
            result.artifacts["asset_bindings.json"] = str(out / "asset_bindings.json")
            Path(result.artifacts["asset_bindings.json"]).write_text(json.dumps(asset_bindings, ensure_ascii=False, indent=2), encoding="utf-8")
            self._add_observation_artifacts(result.artifacts, observation)
            if scene is None:
                result.artifacts["scene.xml"] = str(xml_path)
                result.artifacts["interaction_registry.json"] = str(generated_interactions)
            return self._write_result(result, out)
        except (OSError, ValueError, KeyError, RuntimeError, ValidationError, ModelCallBudgetExceeded) as exc:
            status = "model_call_budget_exceeded" if isinstance(exc, ModelCallBudgetExceeded) else ("asset_missing" if "asset_missing" in str(exc) else ("unsupported_recipe" if "unsupported_recipe" in str(exc) else ("grounding_ambiguous" if "grounding_ambiguous" in str(exc) else ("relation_not_satisfied" if "relation_not_satisfied" in str(exc) else "planning_failed"))))
            result = PipelineResult(task_intent=intent.model_dump(mode="json") if intent else {"instruction": instruction}, scene_registry=registry.model_dump(mode="json") if registry else {}, status=status, model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, error=str(exc))
            if observation is not None: self._add_observation_artifacts(result.artifacts, observation)
            return self._write_result(result, out)

    @staticmethod
    def _capture(budget, stage, provider):
        calls = getattr(provider, "calls", [])
        budget.update(stage, calls[-1] if calls and calls[-1].get("stage") == stage else None)

    @staticmethod
    def _write_result(result, out):
        result.model_usage.setdefault("route", result.route)
        result.model_usage.setdefault("planner", result.planner)
        payloads = {"task_intent.json": result.task_intent, "scene_registry.json": result.scene_registry, "grounded_task.json": result.grounded_task, "visual_grounding.json": result.visual_grounding, "skill_plan.json": result.skill_plan, "model_usage.json": result.model_usage, "summary.json": {"status": result.status, "route": result.route, "planner": result.planner, "model_call_count": result.model_call_count, "model_usage": result.model_usage, "error": result.error, "source_scene": result.source_scene, "interaction_registry": result.interaction_registry}}
        for name, payload in payloads.items():
            path = out / name
            if payload is None:
                if path.exists(): path.unlink()
            else:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); result.artifacts[name] = str(path)
        return result

    @staticmethod
    def _add_observation_artifacts(artifacts, observation):
        for name, path in {"rgb.png": observation.rgb_path, "segmentation.npy": observation.segmentation_path, "segmentation.png": observation.segmentation_visualization_path, "instances.json": observation.instance_index_path}.items(): artifacts[name] = str(path)
