from pathlib import Path

import pytest
from typer.testing import CliRunner

from robot_agent_sim import cli
from robot_agent_sim.pipeline.engine import PipelineResult

runner = CliRunner()


@pytest.fixture
def calls(monkeypatch):
    received = []

    def plan(self, instruction, **kwargs):
        received.append((instruction, kwargs))
        return PipelineResult(task_intent={"status": "accepted"})

    monkeypatch.setattr(cli.PipelineEngine, "plan", plan)
    return received


def test_plan_subcommand_preserves_instruction_and_options(calls, tmp_path):
    instruction = "把左边的红色方块放进右边蓝色盒子"
    result = runner.invoke(cli.app, [
        "plan", instruction, "--robot", "panda", "--seed", "7",
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls == [(instruction, {
        "robot": "panda", "scene": None, "seed": 7, "output_dir": tmp_path, "planner": "recipe",
    })]


def test_uploaded_scene_options(calls, tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>")
    result = runner.invoke(cli.app, ["plan", "按按钮", "--robot", "ur5e", "--scene", str(scene)])
    assert result.exit_code == 0, result.output
    assert calls[0][1] == {
        "robot": "ur5e", "scene": scene, "seed": 0, "output_dir": Path("var"), "planner": "recipe",
    }


@pytest.mark.parametrize("args", [
    ["plan", "任务", "--robot", "invalid"],
    ["plan", "任务", "--seed", "seven"],
    ["plan", "任务", "--scene", "missing-scene.xml"],
    ["plan", "   "],
])
def test_invalid_arguments_do_not_start_pipeline(calls, args):
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    assert not calls


def test_unsupported_scene_extension(calls, tmp_path):
    path = tmp_path / "scene.txt"
    path.write_text("<mujoco/>")
    result = runner.invoke(cli.app, ["plan", "任务", "--scene", str(path)])
    assert result.exit_code == 2
    assert not calls


@pytest.mark.parametrize("args", [["--help"], ["plan", "--help"]])
def test_help_does_not_start_pipeline(calls, args):
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    assert "plan" in result.output
    assert not calls


def test_pipeline_error_is_readable(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("scene XML contains forbidden include/plugin")

    monkeypatch.setattr(cli.PipelineEngine, "plan", fail)
    result = runner.invoke(cli.app, ["plan", "任务"])
    assert result.exit_code == 1
    assert "规划失败" in result.output
    assert "Traceback" not in result.output
