from __future__ import annotations

import json
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated
from xml.etree.ElementTree import ParseError

import typer
from pydantic import ValidationError

from .execution.compiler import compile_directory
from .execution.contracts import ExecutionBundle
from .skills.registry import REGISTRY


app = typer.Typer(help="机器人任务规划、编译与 MuJoCo 执行。")


class Robot(str, Enum):
    PANDA = "panda"
    UR5E = "ur5e"


class Provider(str, Enum):
    FAKE = "fake"
    QWEN = "qwen"


class PlannerMode(str, Enum):
    RECIPE = "recipe"
    QWEN = "qwen"
    AUTO = "auto"


class StructuredOutputMode(str, Enum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    OFF = "off"


class ViewerMode(str, Enum):
    AUTO = "auto"
    STEP = "step"
    HEADLESS = "headless"


def _configure_planning_gl() -> None:
    """Select EGL before importing any MuJoCo planning/rendering module."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _new_engine(
    provider: Provider,
    structured_output: StructuredOutputMode,
    qwen_base_url: str | None,
    qwen_model: str | None,
):
    _configure_planning_gl()
    from .pipeline.engine import PipelineEngine

    if provider == Provider.FAKE:
        return PipelineEngine()
    base_url = qwen_base_url or os.environ.get("QWEN_BASE_URL")
    model = qwen_model or os.environ.get("QWEN_MODEL")
    if not base_url or not model:
        raise typer.BadParameter(
            "--provider qwen 需要 --qwen-base-url 和 --qwen-model（或同名环境变量）。",
            param_hint="--provider",
        )
    from .models.qwen_http import QwenHTTPProvider

    qwen = QwenHTTPProvider(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("QWEN_API_KEY", ""),
        use_structured_output=structured_output.value,
    )
    return PipelineEngine(understanding=qwen, vision=qwen, planner=qwen)


def _validate_plan_inputs(instruction: str, scene: Path | None) -> None:
    if not instruction.strip():
        raise typer.BadParameter("任务指令不能为空。", param_hint="instruction")
    if scene is not None and scene.suffix.lower() not in {".xml", ".mjcf"}:
        raise typer.BadParameter("场景必须是 .xml 或 .mjcf 文件。", param_hint="--scene")


def _plan(
    instruction: str,
    *,
    robot: Robot,
    scene: Path | None,
    interaction_registry: Path | None,
    seed: int,
    output_dir: Path,
    provider: Provider,
    planner: PlannerMode,
    structured_output: StructuredOutputMode,
    qwen_base_url: str | None,
    qwen_model: str | None,
):
    _validate_plan_inputs(instruction, scene)
    # An authored sidecar next to an uploaded XML is the preferred Route B
    # source.  Callers can still override it explicitly for shared fixtures.
    if scene is not None and interaction_registry is None:
        candidate = scene.with_name(f"{scene.stem}.interactions.json")
        if candidate.is_file():
            interaction_registry = candidate
    engine = _new_engine(provider, structured_output, qwen_base_url, qwen_model)
    return engine.plan(
        instruction,
        robot=robot.value,
        scene=scene,
        seed=seed,
        output_dir=output_dir,
        planner=planner.value,
        interaction_registry=interaction_registry,
    )


@app.command()
def plan(
    instruction: Annotated[str, typer.Argument(help="自然语言任务；包含空格时使用引号。")],
    robot: Annotated[Robot, typer.Option(help="机械臂类型。")]=Robot.PANDA,
    scene: Annotated[Path | None, typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True, help="已有 .xml/.mjcf 场景；省略则走 Route A 自动建场景。")]=None,
    interaction_registry: Annotated[Path | None, typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True, help="精确场景的交互元数据；提供后 Route B 可编译执行。")]=None,
    seed: Annotated[int, typer.Option(help="随机布局种子。")]=0,
    output_dir: Annotated[Path, typer.Option(file_okay=False, dir_okay=True, help="规划产物目录。")]=Path("var"),
    provider: Annotated[Provider, typer.Option(help="fake 离线可测；qwen 使用兼容 HTTP API。")]=Provider.FAKE,
    planner: Annotated[PlannerMode, typer.Option(help="recipe、qwen 或 auto。")]=PlannerMode.RECIPE,
    structured_output: Annotated[StructuredOutputMode, typer.Option(help="Qwen 结构化输出模式。")]=StructuredOutputMode.JSON_SCHEMA,
    qwen_base_url: Annotated[str | None, typer.Option(help="Qwen API 根地址；也可用 QWEN_BASE_URL。")]=None,
    qwen_model: Annotated[str | None, typer.Option(help="Qwen 模型名；也可用 QWEN_MODEL。")]=None,
):
    """只规划并保存 JSON/场景，不执行动作。"""
    try:
        result = _plan(
            instruction, robot=robot, scene=scene,
            interaction_registry=interaction_registry, seed=seed,
            output_dir=output_dir, provider=provider, planner=planner,
            structured_output=structured_output, qwen_base_url=qwen_base_url,
            qwen_model=qwen_model,
        )
    except (OSError, ValueError, KeyError, ParseError, ValidationError) as exc:
        typer.echo(f"规划失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_summary(result)
    typer.echo("\n完整 JSON：")
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command("compile")
def compile_command(
    task_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True, help="plan 的产物目录。")],
    scene: Annotated[Path | None, typer.Option(exists=True, file_okay=True, readable=True, help="覆盖规划记录的精确场景。")]=None,
    interaction_registry: Annotated[Path | None, typer.Option(exists=True, file_okay=True, readable=True, help="场景交互元数据。")]=None,
):
    """把语义技能计划确定性编译为控制命令。"""
    try:
        bundle = compile_directory(task_dir, scene_path=scene, interaction_registry_path=interaction_registry)
    except (OSError, ValueError, KeyError, ValidationError) as exc:
        typer.echo(f"编译失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(bundle.model_dump_json(indent=2))


def _bundle_path(task_or_bundle: Path) -> Path:
    path = task_or_bundle.expanduser().resolve()
    return path / "execution_bundle.json" if path.is_dir() else path


def _execute_subprocess(bundle_path: Path, viewer_mode: ViewerMode) -> None:
    env = os.environ.copy()
    if viewer_mode == ViewerMode.HEADLESS:
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
    else:
        env.pop("MUJOCO_GL", None)
        env.pop("PYOPENGL_PLATFORM", None)
    completed = subprocess.run(
        [sys.executable, "-m", "robot_agent_sim.execution.runner", "--bundle", str(bundle_path), "--viewer-mode", viewer_mode.value],
        env=env,
        check=False,
    )
    if completed.returncode:
        raise typer.Exit(code=completed.returncode)


@app.command()
def execute(
    task_or_bundle: Annotated[Path, typer.Argument(exists=True, readable=True, help="任务目录或 execution_bundle.json。")],
    viewer_mode: Annotated[ViewerMode, typer.Option(help="auto、step 或 headless。")]=ViewerMode.AUTO,
):
    """在独立进程中执行已编译命令，隔离 EGL 与 GLFW。"""
    bundle_path = _bundle_path(task_or_bundle)
    if not bundle_path.is_file():
        typer.echo(f"执行失败：找不到 {bundle_path}", err=True)
        raise typer.Exit(code=1)
    _execute_subprocess(bundle_path, viewer_mode)


@app.command()
def run(
    instruction: Annotated[str, typer.Argument(help="自然语言任务。")],
    robot: Annotated[Robot, typer.Option(help="执行阶段目前只支持 ur5e。")]=Robot.UR5E,
    scene: Annotated[Path | None, typer.Option(exists=True, file_okay=True, readable=True)]=None,
    interaction_registry: Annotated[Path | None, typer.Option(exists=True, file_okay=True, readable=True)]=None,
    seed: Annotated[int, typer.Option()]=0,
    output_dir: Annotated[Path, typer.Option(file_okay=False, dir_okay=True)]=Path("var"),
    provider: Annotated[Provider, typer.Option()]=Provider.FAKE,
    planner: Annotated[PlannerMode, typer.Option()]=PlannerMode.RECIPE,
    structured_output: Annotated[StructuredOutputMode, typer.Option()]=StructuredOutputMode.JSON_SCHEMA,
    qwen_base_url: Annotated[str | None, typer.Option()]=None,
    qwen_model: Annotated[str | None, typer.Option()]=None,
    viewer_mode: Annotated[ViewerMode, typer.Option()]=ViewerMode.AUTO,
):
    """一次完成自然语言规划、确定性编译和 MuJoCo 执行。"""
    try:
        result = _plan(
            instruction, robot=robot, scene=scene,
            interaction_registry=interaction_registry, seed=seed,
            output_dir=output_dir, provider=provider, planner=planner,
            structured_output=structured_output, qwen_base_url=qwen_base_url,
            qwen_model=qwen_model,
        )
        if result.status != "accepted":
            raise ValueError(f"planning status is {result.status}: {result.error or '-'}")
        bundle = compile_directory(output_dir)
    except (OSError, ValueError, KeyError, ParseError, ValidationError) as exc:
        typer.echo(f"运行准备失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_summary(result)
    _execute_subprocess(Path(bundle.task_dir) / "execution_bundle.json", viewer_mode)


def _print_summary(result) -> None:
    intent = getattr(result, "task_intent", None) or {}
    grounded = getattr(result, "grounded_task", None) or {}
    plan_data = getattr(result, "skill_plan", None) or {}
    typer.echo(f"状态：{getattr(result, 'status', 'accepted')}")
    typer.echo(f"任务类型：{', '.join(intent.get('task_types', [])) or '-'}")
    typer.echo("对象绑定结果：")
    for entity in grounded.get("entities", []):
        typer.echo(f"  {entity['entity_id']} -> {entity['object_id']} ({entity['grounding_method']})")
    if not grounded.get("entities"):
        typer.echo("  -")
    typer.echo("语义子任务顺序：")
    for step in plan_data.get("steps", []):
        typer.echo(f"  {step['step_id']}: {REGISTRY.describe(step['skill_name'], step.get('target_object'), step.get('reference_object'), step.get('semantic_target'))}")
    if not plan_data.get("steps"):
        typer.echo("  -")
    skills = " -> ".join(step["skill_name"] for step in plan_data.get("steps", [])) or "-"
    typer.echo(f"Atomic Skill 调用顺序：{skills}")
    typer.echo(f"模型调用次数：{getattr(result, 'model_call_count', 0)}")
    typer.echo(f"规划器：{getattr(result, 'planner', 'recipe')} / Route {getattr(result, 'route', '-')}")


if __name__ == "__main__":
    app()
