from types import SimpleNamespace

from robot_agent_sim.contracts.grounded_task import GroundedEntity, GroundedTask
from robot_agent_sim.contracts.task_intent import Operation, SpatialRelation, SpatialRelationType, TaskEntity, TaskType
from robot_agent_sim.grounding.world_relation import RelationAmbiguous, WorldRelationResolver
from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.models.budget import ModelCallBudget, ModelCallBudgetExceeded


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
    intent = SimpleNamespace(entities=[SimpleNamespace(entity_id="red"), SimpleNamespace(entity_id="yellow")], spatial_relations=[SpatialRelation(subject="red", relation=SpatialRelationType.LEFT_OF, reference="yellow", scope="selection")])
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
