import json
from pathlib import Path

import httpx
import pytest

from robot_agent_sim.backends.mujoco.backend import MujocoSceneBackend
from robot_agent_sim.models.fake import FakeVisionGroundingProvider
from robot_agent_sim.models.qwen_http import QwenHTTPProvider
from robot_agent_sim.models.vision_grounding import VisionDetection
from robot_agent_sim.pipeline.engine import PipelineEngine


SCENE_003 = Path(__file__).parents[1] / "assets/robots/ur5e/scenes/scene_003.xml"


def test_route_b_auto_discovers_task_body_and_grounding_artifact(tmp_path):
    registry = MujocoSceneBackend().load_uploaded(SCENE_003, "ur5e")
    assert [item.body_name for item in registry.objects] == ["push_button_base"]

    provider = FakeVisionGroundingProvider(
        [VisionDetection(entity_id="button_01", bbox=[710, 412, 867, 525], confidence=1.0)]
    )
    result = PipelineEngine(vision=provider).plan(
        "按按钮", robot="ur5e", scene=SCENE_003, output_dir=tmp_path
    )
    assert result.status == "accepted"
    assert result.model_call_count == 3
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
                "choices": [{"message": {"content": '{"status":"unsupported_task","instruction":"x"}'}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = QwenHTTPProvider("http://localhost:8000/v1", "qwen-test")
    intent = provider.understand(
        type("Request", (), {
            "instruction": "x",
            "supported_task_types": ["locate"],
            "supported_directions": ["left"],
            "asset_catalog": [],
        })()
    )
    assert intent.status == "unsupported_task"
    assert len(calls) == 1
    assert calls[0][0].endswith("/chat/completions")
    body = calls[0][1]["json"]
    assert body["temperature"] == 0
    assert "supported_directions" in body["messages"][1]["content"]
    assert provider.calls[0]["stage"] == "task_understanding"


def test_skill_plan_rejects_unknown_target_and_wrong_order():
    from robot_agent_sim.contracts.skill_plan import SemanticSubtask, SkillPlan, SkillStep

    with pytest.raises(ValueError):
        PipelineEngine._validate_skill_targets(
            SkillPlan(
                task_types=["press"],
                steps=[SkillStep(
                    step_id="step-1",
                    semantic_subtask=SemanticSubtask(action="press", description="bad"),
                    skill_name="press",
                    target_object="missing",
                )],
                model_call_count=1,
            ),
            type("Task", (), {"entities": [], "task_types": ["press"]})(),
        )
