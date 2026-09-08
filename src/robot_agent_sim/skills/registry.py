from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SkillDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
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
        signatures = {
            "locate": "target", "search": "target",
            "move": "target,reference?,region?", "grasp": "target",
            "release": "target,reference?,region?", "press": "target",
        }
        labels = {"locate": "定位", "search": "搜索", "move": "移动",
                  "grasp": "抓取", "release": "释放", "press": "按压"}
        return "\n".join(
            f"{item.name}({signatures.get(item.name, 'target')}):{labels.get(item.name, item.description)}"
            for item in self._items.values()
        )

    def describe(self, name, target=None, reference=None, region=None) -> str:
        self.require(name)
        labels = {"locate": "定位", "search": "搜索", "move": "移动到", "grasp": "抓取", "release": "释放", "press": "按压"}
        suffix = target or "目标"
        if reference: suffix += f"（参考 {reference}）"
        if region: suffix += f" / {region}"
        return f"{labels[name]}{suffix}"


REGISTRY = AtomicSkillRegistry((
    SkillDefinition(name="locate", description="定位已知目标"),
    SkillDefinition(name="search", description="搜索任务目标"),
    SkillDefinition(name="move", description="移动到语义区域", allowed_regions=("grasp_region", "container_interior", "button_surface", "relative_region", "semantic_region")),
    SkillDefinition(name="grasp", description="抓取已定位目标"),
    SkillDefinition(name="release", description="释放已抓取目标", allowed_regions=("container_interior", "relative_region", "semantic_region")),
    SkillDefinition(name="press", description="按压已定位按钮"),
))
