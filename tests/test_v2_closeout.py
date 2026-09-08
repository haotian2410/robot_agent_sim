import pytest

from robot_agent_sim.contracts.task_intent import Operation, SpatialRelation, SpatialRelationType, TaskEntity, TaskIntent, TaskStatus, TaskType
from robot_agent_sim.grounding.world_relation import RelationAmbiguous, WorldRelationResolver
from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.models.budget import ModelCallBudget, ModelCallBudgetExceeded
from robot_agent_sim.models.task_understanding import ParseEntity, ParseOperation, TaskParseLLMOutput
from robot_agent_sim.models.fake import FakeVisionGroundingProvider
from robot_agent_sim.models.vision_grounding import VisionDetection
from pathlib import Path

SCENE_003 = Path(__file__).parents[1] / "assets/robots/ur5e/scenes/scene_003.xml"


def test_default_recipe_call_budget_route_a():
    result = PipelineEngine().plan("抓取红方块", output_dir="/tmp/robot-agent-sim-closeout")
    assert result.status == "accepted"
    assert result.planner == "recipe"
    assert result.model_call_count == 1


def test_qwen_planner_kept_as_explicit_mode():
    result = PipelineEngine().plan("抓取红方块", planner="qwen", output_dir="/tmp/robot-agent-sim-closeout")
    assert result.status == "accepted"
    assert result.planner == "qwen"
    assert result.model_call_count == 2


def test_binary_world_relation_does_not_choose_global_extreme():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED,
        instruction="抓取左侧红色物体",
        task_types=[TaskType.GRASP],
        entities=[
            TaskEntity(entity_id="red", semantic_name="red object", category="cube"),
            TaskEntity(entity_id="yellow", semantic_name="yellow object", category="cube"),
        ],
        operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="red")],
        spatial_relations=[SpatialRelation(subject="red", relation=SpatialRelationType.LEFT_OF, reference="yellow", scope="selection")],
    )
    candidates = {"red": [{"object_id": "r1"}, {"object_id": "r2"}], "yellow": [{"object_id": "y"}]}
    positions = {"r1": (-1.0, 0.0, 0.0), "r2": (-0.2, 0.0, 0.0), "y": (0.0, 0.0, 0.0)}
    try:
        WorldRelationResolver().resolve(intent, candidates, positions)
    except RelationAmbiguous:
        pass
    else:
        raise AssertionError("multiple binary-relation candidates must be ambiguous")


def test_budget_limits_stage_not_only_total_calls():
    budget = ModelCallBudget.for_route(False, "recipe")
    budget.consume("task_understanding")
    try:
        budget.consume("task_understanding")
    except ModelCallBudgetExceeded:
        pass
    else:
        raise AssertionError("task parser stage may only be called once")


@pytest.mark.parametrize(
    ("route_scene", "planner", "expected_calls"),
    [(False, "recipe", 1), (True, "recipe", 2), (False, "qwen", 2), (True, "qwen", 3)],
)
def test_fixed_planner_route_call_budgets(tmp_path, route_scene, planner, expected_calls):
    vision = FakeVisionGroundingProvider([VisionDetection(entity_id="button_01", bbox=[710, 412, 867, 525])]) if route_scene else None
    result = PipelineEngine(vision=vision).plan(
        "按按钮", robot="ur5e", scene=SCENE_003 if route_scene else None,
        planner=planner, output_dir=tmp_path / f"{planner}-{route_scene}",
    )
    assert result.model_call_count == expected_calls
    assert result.route == ("B" if route_scene else "A")
    assert result.planner == planner
    assert result.model_usage["calls"] == expected_calls
    assert len(result.model_usage["stages"]) == expected_calls


class UnsupportedRecipeProvider:
    def understand(self, request):
        return TaskParseLLMOutput(
            status="accepted",
            entities=[ParseEntity(id="button_01", name="button", category="button")],
            operations=[ParseOperation(type="pick_and_place")],
        )


@pytest.mark.parametrize("route_scene, expected_calls", [(False, 2), (True, 3)])
def test_auto_unsupported_recipe_falls_back_to_qwen_budget(tmp_path, route_scene, expected_calls):
    vision = FakeVisionGroundingProvider([VisionDetection(entity_id="button_01", bbox=[710, 412, 867, 525])]) if route_scene else None
    result = PipelineEngine(understanding=UnsupportedRecipeProvider(), vision=vision).plan(
        "按按钮", robot="ur5e", scene=SCENE_003 if route_scene else None,
        planner="auto", output_dir=tmp_path / f"auto-{route_scene}",
    )
    assert result.model_call_count == expected_calls
    assert result.route == ("B" if route_scene else "A")
    assert result.planner == "qwen"
    assert result.model_usage["calls"] == expected_calls
