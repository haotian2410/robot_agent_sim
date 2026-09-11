from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .task_intent import Operation, SpatialRelation, TaskType


class GroundedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_id: str
    semantic_name: str = Field(min_length=1, max_length=100)
    object_id: str
    body_name: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    grounding_method: Literal[
        "asset_scene_binding", "vlm_iou", "detector_iou", "interaction_registry"
    ]
    detection_bbox: tuple[int, int, int, int] | None = None
    instance_bbox: tuple[int, int, int, int] | None = None
    bbox_iou: float | None = Field(default=None, ge=0, le=1)


class GroundedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str
    task_types: list[TaskType]
    entities: list[GroundedEntity]
    operations: list[Operation]
    spatial_relations: list[SpatialRelation]
    scene_id: str

    @model_validator(mode="after")
    def refs(self):
        ids = {entity.entity_id for entity in self.entities}
        objects = {entity.object_id for entity in self.entities}
        if len(ids) != len(self.entities) or len(objects) != len(self.entities):
            raise ValueError("grounded entity and object IDs must be unique")
        for op in self.operations:
            if any(value and value not in ids for value in [op.source, op.destination, op.target, op.reference]):
                raise ValueError("grounded operation references unknown entity")
        return self
