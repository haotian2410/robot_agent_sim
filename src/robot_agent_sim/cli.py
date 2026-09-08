import json
import os
from enum import Enum
from pathlib import Path
from typing import Annotated
from xml.etree.ElementTree import ParseError
from pydantic import ValidationError

import typer

from .pipeline.engine import PipelineEngine
from .models.qwen_http import QwenHTTPProvider
from .skills.registry import REGISTRY

app = typer.Typer(help="Planning-only 机器人技能规划。默认使用 Fake Providers，可选 Qwen HTTP。")


class Robot(str, Enum):
    PANDA = "panda"
    UR5E = "ur5e"


class Provider(str, Enum):
    FAKE = "fake"
    QWEN = "qwen"


@app.callback()
def main():
    """保留显式 plan 子命令，即使当前只有一个子命令。"""


@app.command()
def plan(
    instruction: Annotated[str, typer.Argument(help="自然语言任务；包含空格时使用引号。")],
    robot: Annotated[Robot, typer.Option(help="机械臂类型；上传 XML 时不会替换其中的机器人。")]
    = Robot.PANDA,
    scene: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True,
                     help="已有 .xml/.mjcf 场景文件路径；省略则自动生成场景。"),
    ] = None,
    seed: Annotated[int, typer.Option(help="随机布局种子，任意整数；7 只是一个示例值。")]
    = 0,
    output_dir: Annotated[
        Path,
        typer.Option(file_okay=False, dir_okay=True,
                     help="输出目录，相对当前工作目录；同名文件会覆盖。"),
    ] = Path("var"),
    provider: Annotated[
        Provider,
        typer.Option(help="模型 Provider；fake 离线可测，qwen 使用 OpenAI-compatible HTTP。"),
    ] = Provider.FAKE,
    qwen_base_url: Annotated[
        str | None,
        typer.Option(help="Qwen API 根地址，例如 http://localhost:8000/v1；也可用 QWEN_BASE_URL。"),
    ] = None,
    qwen_model: Annotated[
        str | None,
        typer.Option(help="Qwen 服务模型名；也可用 QWEN_MODEL。"),
    ] = None,
):
    """生成规划 JSON 和场景文件，不执行机器人动作。"""
    if not instruction.strip():
        raise typer.BadParameter("任务指令不能为空。", param_hint="instruction")
    if scene is not None and scene.suffix.lower() not in {".xml", ".mjcf"}:
        raise typer.BadParameter("场景必须是 .xml 或 .mjcf 文件。", param_hint="--scene")
    engine = PipelineEngine()
    if provider == Provider.QWEN:
        base_url = qwen_base_url or os.environ.get("QWEN_BASE_URL")
        model = qwen_model or os.environ.get("QWEN_MODEL")
        if not base_url or not model:
            raise typer.BadParameter(
                "--provider qwen 需要 --qwen-base-url 和 --qwen-model（或同名环境变量）。",
                param_hint="--provider",
            )
        qwen = QwenHTTPProvider(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("QWEN_API_KEY", ""),
        )
        engine = PipelineEngine(understanding=qwen, vision=qwen, planner=qwen)
    try:
        result = engine.plan(
            instruction, robot=robot.value, scene=scene, seed=seed, output_dir=output_dir
        )
    except (OSError, ValueError, KeyError, ParseError, ValidationError) as exc:
        typer.echo(f"规划失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_summary(result)
    typer.echo("\n完整 JSON：")
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _print_summary(result):
    intent = getattr(result, "task_intent", None) or {}
    grounded = getattr(result, "grounded_task", None) or {}
    plan = getattr(result, "skill_plan", None) or {}
    task_types = ", ".join(intent.get("task_types", [])) or "-"
    typer.echo(f"状态：{getattr(result, 'status', 'accepted')}")
    typer.echo(f"任务类型：{task_types}")
    typer.echo("涉及对象：")
    for entity in intent.get("entities", []):
        typer.echo(
            f"  {entity['entity_id']}: {entity['semantic_name']} "
            f"(category={entity['category']})"
        )
    if not intent.get("entities"):
        typer.echo("  -")
    typer.echo("对象绑定结果：")
    for entity in grounded.get("entities", []):
        typer.echo(
            f"  {entity['entity_id']} -> {entity['object_id']} "
            f"({entity['grounding_method']})"
        )
    if not grounded.get("entities"):
        typer.echo("  -")
    typer.echo("语义子任务顺序：")
    for step in plan.get("steps", []):
        typer.echo(f"  {step['step_id']}: {REGISTRY.describe(step['skill_name'], step.get('target_object'), step.get('reference_object'), step.get('semantic_target'))}")
    if not plan.get("steps"):
        typer.echo("  -")
    skills = " -> ".join(step["skill_name"] for step in plan.get("steps", [])) or "-"
    typer.echo(f"Atomic Skill 调用顺序：{skills}")
    typer.echo(f"模型调用次数：{getattr(result, 'model_call_count', 0)}")
    usage = getattr(result, "model_usage", {}) or {}
    typer.echo(
        f"模型 Tokens：输入 {usage.get('prompt_tokens', 0)} / "
        f"输出 {usage.get('completion_tokens', 0)} / 总计 {usage.get('total_tokens', 0)}"
    )


if __name__ == "__main__":
    app()
