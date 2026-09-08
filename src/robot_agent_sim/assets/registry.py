from __future__ import annotations

import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

from ..paths import MODEL_ROOT


class AssetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    model_name: str
    category: str
    aliases: list[str] = Field(default_factory=list)
    source: str = "mesh"
    root: Path | None = None
    visual_obj: Path | None = None
    texture: Path | None = None
    collision_stls: list[Path] = Field(default_factory=list)
    dimensions_m: tuple[float, float, float] | None = None
    bbox_min_m: tuple[float, float, float] | None = None
    bbox_max_m: tuple[float, float, float] | None = None
    bottom_offset_m: float = 0.0
    metadata: dict = Field(default_factory=dict)

    def exists(self) -> bool:
        return self.source == "primitive" or (self.visual_obj is not None and self.visual_obj.is_file())


class AssetRegistry:
    """Central catalog for user-provided meshes and controlled primitives."""

    def __init__(self, root: Path = MODEL_ROOT):
        self.root = root
        self.records = self._load()

    def _load(self) -> list[AssetRecord]:
        records: list[AssetRecord] = []
        if self.root.is_dir():
            for directory in sorted(item for item in self.root.iterdir() if item.is_dir()):
                records.append(self._from_directory(directory))
        records.extend([
            AssetRecord(model_id="primitive_cube", model_name="cube_basic", category="cube", aliases=["cube", "方块", "立方体"], source="primitive"),
            AssetRecord(model_id="primitive_open_box", model_name="open_box", category="container", aliases=["box", "open box", "盒子", "容器", "container"], source="primitive"),
            AssetRecord(model_id="primitive_button", model_name="button_basic", category="button", aliases=["button", "按钮"], source="primitive"),
        ])
        return records

    def _from_directory(self, directory: Path) -> AssetRecord:
        name = directory.name
        metadata_path = directory / f"{name}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        object_name = metadata.get("object_meta", {}).get("object_name", name)
        bounds = _obj_bounds(directory / "textured.obj")
        dimensions = tuple(round(bounds[1][i] - bounds[0][i], 6) for i in range(3)) if bounds else None
        return AssetRecord(
            model_id=f"mesh_{name}", model_name=name, category=_category(name),
            aliases=sorted(set([name, object_name, *_aliases(name)])), source="mesh", root=directory,
            visual_obj=directory / "textured.obj", texture=directory / "texture_map.png",
            collision_stls=sorted(directory.glob("textured_coacd_*.stl")),
            dimensions_m=dimensions, bottom_offset_m=dimensions[2] / 2 if dimensions else 0.0,
            bbox_min_m=bounds[0] if bounds else None,
            bbox_max_m=bounds[1] if bounds else None,
            metadata=metadata,
        )

    def catalog(self) -> list[dict]:
        return [record.model_dump(mode="json") for record in self.records]

    def resolve(self, category: str, name: str = "", aliases: list[str] | None = None) -> AssetRecord:
        query = " ".join([category, name, *(aliases or [])]).casefold()
        candidates = [r for r in self.records if r.exists() and (r.category.casefold() in query or any(a.casefold() in query for a in r.aliases))]
        if not candidates:
            raise KeyError(f"asset_missing: {category or name}")
        def score(record):
            exact = record.model_name.casefold() in query or any(alias.casefold() in query for alias in record.aliases)
            return (0 if exact else 1, 0 if record.source == "mesh" else 1, record.model_id)
        candidates.sort(key=score)
        if len(candidates) > 1 and score(candidates[0])[:2] == score(candidates[1])[:2]:
            raise ValueError(f"asset_ambiguous: {[r.model_id for r in candidates]}")
        return candidates[0]


def _category(name: str) -> str:
    return {"apple": "fruit", "banana": "fruit", "baseball": "ball", "rubiks_cube": "cube", "sponge": "sponge", "spoon": "utensil", "sugar_box": "package"}.get(name, "object")


def _aliases(name: str) -> list[str]:
    return {"apple": ["苹果"], "banana": ["香蕉"], "baseball": ["棒球"], "rubiks_cube": ["rubiks cube", "魔方"], "sponge": ["海绵"], "spoon": ["勺子", "汤匙"], "sugar_box": ["sugar box", "糖盒", "糖果盒"]}.get(name, [])


def _obj_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not path.is_file(): return None
    low = [float("inf")] * 3; high = [float("-inf")] * 3; count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "): continue
        values = [float(value) for value in line.split()[1:4]]
        low = [min(a, b) for a, b in zip(low, values)]; high = [max(a, b) for a, b in zip(high, values)]; count += 1
    if not count:
        return None
    return (
        tuple(round(value, 6) for value in low),
        tuple(round(value, 6) for value in high),
    )


def _obj_dimensions(path: Path) -> tuple[float, float, float] | None:
    bounds = _obj_bounds(path)
    return tuple(round(bounds[1][i] - bounds[0][i], 6) for i in range(3)) if bounds else None
