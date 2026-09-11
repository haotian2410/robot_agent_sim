"""Deterministic semantic grounding for authored interaction registries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mujoco

from ..contracts.grounded_task import GroundedEntity


def ground_with_interaction_registry(
    entities,
    registry_path: str | Path,
    scene_path: str | Path,
) -> list[GroundedEntity]:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    objects = registry.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("interaction registry must contain an objects mapping")
    model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))

    result: list[GroundedEntity] = []
    used: set[str] = set()
    for entity in entities:
        query_values = {
            _normalize(entity.semantic_name),
            *(_normalize(value) for value in entity.aliases),
        }
        exact: list[tuple[str, dict[str, Any]]] = []
        fuzzy: list[tuple[str, dict[str, Any]]] = []
        for object_id, item in objects.items():
            names = {
                _normalize(object_id),
                _normalize(str(item.get("object_id", object_id))),
                *(_normalize(str(value)) for value in item.get("aliases", [])),
            }
            if query_values & names:
                exact.append((object_id, item))
            elif any(
                query and name and (query in name or name in query)
                for query in query_values
                for name in names
            ):
                fuzzy.append((object_id, item))
        candidates = exact or fuzzy
        candidates = [candidate for candidate in candidates if candidate[0] not in used]
        if len(candidates) != 1:
            names = [candidate[0] for candidate in candidates]
            raise ValueError(
                f"interaction registry grounding for {entity.semantic_name!r} "
                f"is {'missing' if not candidates else 'ambiguous'}: {names}"
            )
        object_id, item = candidates[0]
        used.add(object_id)
        result.append(
            GroundedEntity(
                entity_id=entity.entity_id,
                semantic_name=entity.semantic_name,
                object_id=object_id,
                body_name=_source_body_name(model, item),
                grounding_method="interaction_registry",
            )
        )
    return result


def _normalize(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def _source_body_name(model: mujoco.MjModel, item: dict[str, Any]) -> str | None:
    source = item.get("spatial", {}).get("source", {})
    name = source.get("name")
    source_type = source.get("type")
    if not isinstance(name, str):
        return None
    if source_type == "body":
        return name
    kind = {
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
    }.get(source_type)
    if kind is None:
        return None
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"interaction registry source not found in scene: {source_type} {name}")
    body_id = int(model.site_bodyid[object_id]) if source_type == "site" else int(model.geom_bodyid[object_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
