import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.contracts.task_intent import TaskStatus
from robot_agent_sim.grounding.iou import iou,match_detections
from robot_agent_sim.models.vision_grounding import VisionDetection
def test_pick_and_place_route_a():
    r=PipelineEngine().plan("把左边的红色方块放进右边蓝色盒子",seed=7); assert r.status=="accepted" and r.model_call_count==1; assert r.planner == "recipe"; assert [x["skill_name"] for x in r.skill_plan["steps"]]==["locate","move","grasp","locate","move","release"]
def test_seed_is_reproducible():
    assert PipelineEngine().plan("抓取红方块",seed=4).scene_registry==PipelineEngine().plan("抓取红方块",seed=4).scene_registry
def test_unsupported_direction(): assert PipelineEngine().plan("把东北角红方块抓起来").status==TaskStatus.DIRECTION_CLARIFICATION_REQUIRED
def test_unsupported_task(): assert PipelineEngine().plan("拧紧螺丝").status==TaskStatus.UNSUPPORTED_TASK
def test_iou_and_ambiguity():
    assert iou((0,0,100,100),(0,0,100,100))==1; d=VisionDetection(entity_id="x",bbox=[0,0,100,100]); m,u,a=match_detections([d],[{"object_id":"a","bbox":[0,0,100,100]},{"object_id":"b","bbox":[0,0,100,100]}]); assert not m and a==["x"]


def test_nearest_selection_binds_the_actual_nearest_candidate():
    import math

    result = PipelineEngine().plan("把离黄色方块最近的红色方块放进盒子", seed=2)
    objects = {item["object_id"]: item for item in result.scene_registry["objects"]}
    yellow = objects["yellow_cube_01"]["position"]
    candidates = [item for item in result.scene_registry["objects"] if item["candidate_for"] == "red_cube_01"]
    selected_id = result.scene_registry["bindings"]["red_cube_01"]
    selected_distance = math.dist(yellow[:2], objects[selected_id]["position"][:2])
    assert selected_distance == min(math.dist(yellow[:2], item["position"][:2]) for item in candidates)
