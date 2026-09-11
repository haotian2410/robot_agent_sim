from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SkillDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    label: str
    description: str
    prompt_signature: str
    requires_target: bool = True
    allowed_regions: tuple[str, ...] = ()


class AtomicSkillRegistry:
    def __init__(self, definitions: tuple[SkillDefinition, ...]):
        self._items = {item.name: item for item in definitions}
        if len(self._items) != len(definitions):
            raise ValueError("skill names must be unique")

    def list(self): return list(self._items.values())

    def require(self, name):
        if name not in self._items:
            raise ValueError(f"unknown skill_name: {name}")
        return self._items[name]

    def prompt_catalog(self) -> str:
        return "\n".join(
            f"{item.prompt_signature}:{item.label}"
            for item in self._items.values()
        )

    def describe(self, name, target=None, reference=None, region=None) -> str:
        self.require(name)
        suffix = target or "目标"
        if reference: suffix += f"（参考 {reference}）"
        if region: suffix += f" / {region}"
        return f"{self.require(name).label}{suffix}"


REGISTRY = AtomicSkillRegistry((
    SkillDefinition(name="locate", label="定位", description="定位已知目标", prompt_signature="locate(target)"),
    SkillDefinition(name="search", label="搜索", description="搜索任务目标", prompt_signature="search(target)"),
    SkillDefinition(name="move", label="移动到", description="移动到语义区域", prompt_signature="move(target,reference?,region?)", allowed_regions=("grasp_region", "container_interior", "button_surface", "relative_region", "semantic_region")),
    SkillDefinition(name="grasp", label="抓取", description="抓取已定位目标", prompt_signature="grasp(target)"),
    SkillDefinition(name="release", label="释放", description="释放已抓取目标", prompt_signature="release(target,reference?,region?)", allowed_regions=("container_interior", "relative_region", "semantic_region")),
    SkillDefinition(name="press", label="按压", description="按压已定位按钮", prompt_signature="press(target)"),
    SkillDefinition(name="pull", label="拉动", description="拉动已抓住的关节机构", prompt_signature="pull(target)"),
    SkillDefinition(name="push", label="推动", description="推动已抓住的关节机构", prompt_signature="push(target)"),
))
