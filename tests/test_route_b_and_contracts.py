import json
from pathlib import Path

import httpx
import pytest

from robot_agent_sim.backends.mujoco.backend import MujocoSceneBackend
from robot_agent_sim.models.fake import FakeVisionGroundingProvider
from robot_agent_sim.models.qwen_http import QwenHTTPProvider
from robot_agent_sim.models.task_understanding import TaskUnderstandingRequest
from robot_agent_sim.models.vision_grounding import VisionDetection
from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.contracts.grounded_task import GroundedEntity, GroundedTask
from robot_agent_sim.contracts.task_intent import Operation, TaskType


SCENE_003 = Path(__file__).parents[1] / "assets/robots/ur5e/scenes/scene_003.xml"


def test_mujoco_headless_rgb_segmentation_smoke(tmp_path):
    registry = MujocoSceneBackend().load_uploaded(SCENE_003, "ur5e")
    observation = MujocoSceneBackend().renderer.render(SCENE_003, registry, tmp_path)
    assert observation.rgb_path.is_file()
    assert observation.segmentation_path.is_file()
    assert observation.segmentation_visualization_path.is_file()


def test_route_b_auto_discovers_task_body_and_grounding_artifact(tmp_path):
    registry = MujocoSceneBackend().load_uploaded(SCENE_003, "ur5e")
    assert [item.body_name for item in registry.objects] == ["push_button_base"]

    provider = FakeVisionGroundingProvider(
        [VisionDetection(entity_id="button_01", bbox=[710, 412, 867, 525])]
    )
    result = PipelineEngine(vision=provider).plan(
        "按按钮", robot="ur5e", scene=SCENE_003, output_dir=tmp_path
    )
    assert result.status == "accepted"
    assert result.model_call_count == 2
    assert Path(result.artifacts["visual_grounding.json"]).is_file()
    assert result.grounded_task["entities"][0]["object_id"] == "scene_object_001"
    assert [step["skill_name"] for step in result.skill_plan["steps"]] == [
        "locate", "move", "press"
    ]


def test_route_b_sidecar_takes_precedence(tmp_path):
    scene = tmp_path / "custom.xml"
    scene.write_text(SCENE_003.read_text(encoding="utf-8"), encoding="utf-8")
    sidecar = scene.with_name("custom.scene_registry.json")
    sidecar.write_text(json.dumps({
        "scene_id": "declared-scene",
        "robot": "ur5e",
        "objects": [{
            "object_id": "button-semantic",
            "body_name": "push_button_base",
            "semantic_name": "red button",
            "role": "target",
            "model_id": "button-model",
            "model_name": "button",
        }],
    }), encoding="utf-8")
    registry = MujocoSceneBackend().load_uploaded(scene, "panda")
    assert registry.scene_id == "declared-scene"
    assert registry.robot == "ur5e"
    assert registry.objects[0].object_id == "button-semantic"
    assert registry.objects[0].semantic_name == "red button"


def test_route_b_default_fake_detection_fails_without_forcing_binding(tmp_path):
    result = PipelineEngine().plan("按按钮", robot="ur5e", scene=SCENE_003, output_dir=tmp_path)
    assert result.status == "grounding_failed"
    assert result.grounded_task is None
    assert result.visual_grounding["unmatched"] == ["button_01"]
    assert "rgb.png" in result.artifacts


def test_qwen_provider_sends_fixed_stage_and_extracts_json(monkeypatch, tmp_path):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": '{"status":"unsupported_task","raw_task":"x"}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = QwenHTTPProvider("http://localhost:8000/v1", "qwen-test")
    intent = provider.understand(TaskUnderstandingRequest(instruction="x"))
    assert intent.status == "unsupported_task"
    assert len(calls) == 1
    assert calls[0][0].endswith("/chat/completions")
    body = calls[0][1]["json"]
    assert body["temperature"] == 0
    assert body["messages"][1]["content"] == '{"instruction":"x"}'
    assert provider.calls[0]["stage"] == "task_understanding"
    assert body["response_format"]["type"] == "json_schema"
    assert "json_schema" in body["response_format"]


@pytest.mark.parametrize(
    ("mode", "has_response_format"),
    [("json_schema", True), ("json_object", True), ("off", False)],
)
def test_qwen_structured_output_modes(monkeypatch, mode, has_response_format):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"status":"unsupported_task","raw_task":"x"}'}}]}

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = QwenHTTPProvider("http://localhost:8000/v1", "qwen-test", use_structured_output=mode)
    provider.understand(TaskUnderstandingRequest(instruction="x"))
    payload = calls[0]
    assert ("response_format" in payload) is has_response_format
    if mode == "json_schema":
        assert payload["response_format"]["type"] == "json_schema"
        assert "schema" in payload["response_format"]["json_schema"]
        assert "只输出一个 JSON 对象" not in payload["messages"][0]["content"]
    elif mode == "json_object":
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][0]["content"].endswith("只输出一个 JSON 对象，不要 markdown 或额外文字。")
    else:
        assert "只输出一个 JSON 对象，不要 markdown 或额外文字。" in payload["messages"][0]["content"]


def test_qwen_http_pipeline_level4_shape_with_local_transport(monkeypatch, tmp_path):
    """Exercise task -> vision -> skill planning through the real HTTP adapter.

    The transport is local and deterministic; this verifies the same request
    and response contracts used by a real Qwen-compatible server without
    pretending that a real model endpoint is available in CI.
    """
    responses = {
        "task_understanding": {
            "status": "accepted",
            "entities": [{"id": "button_01", "name": "red button", "category": "button", "color": "red"}],
            "operations": [{"type": "press", "target": "button_01"}],
            "relations": [],
        },
        "vision_grounding": {
            "detections": [{"entity": "button_01", "bbox": [710, 412, 867, 525]}],
        },
        "skill_planning": {
            "operations": [{"id": "op-1", "steps": [
                {"skill": "locate", "target": "target"},
                {"skill": "move", "target": "target", "region": "button_surface"},
                {"skill": "press", "target": "target"},
            ]}],
        },
    }

    class Response:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(self.body)}}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}}

    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        stage = kwargs["json"]["response_format"]["json_schema"]["name"]
        return Response(responses[stage])

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    provider = QwenHTTPProvider("http://local-mock/v1", "qwen-test")
    result = PipelineEngine(understanding=provider, vision=provider, planner=provider).plan(
        "按红色按钮", robot="ur5e", scene=SCENE_003,
        planner="qwen", output_dir=tmp_path,
    )
    assert result.status == "accepted"
    assert result.route == "B"
    assert result.planner == "qwen"
    assert result.model_call_count == 3
    assert [step["skill_name"] for step in result.skill_plan["steps"]] == ["locate", "move", "press"]
    assert [item["stage"] for item in provider.calls] == ["task_understanding", "vision_grounding", "skill_planning"]
    assert len(calls) == 3


def test_skill_plan_rejects_unknown_target_and_wrong_order():
    from robot_agent_sim.contracts.skill_plan import SkillPlan, SkillStep
    from robot_agent_sim.planning.recipes import validate_plan
    task = GroundedTask(
        instruction="按按钮",
        task_types=[TaskType.PRESS],
        entities=[GroundedEntity(entity_id="button", semantic_name="button", object_id="button-1", grounding_method="asset_scene_binding")],
        operations=[Operation(operation_id="op-1", task_type=TaskType.PRESS, target="button")],
        spatial_relations=[],
        scene_id="scene",
    )
    with pytest.raises(ValueError):
        validate_plan(SkillPlan(task_types=[TaskType.PRESS], steps=[SkillStep(step_id="step-1", operation_id="op-1", skill_name="press", target_object="missing")]), task)
