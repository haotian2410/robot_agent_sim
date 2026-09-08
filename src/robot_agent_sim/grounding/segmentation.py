from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

class InstanceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: str
    body_name: str
    bbox: tuple[int, int, int, int] | None = None
    visible_pixel_count: int = Field(ge=0)

class SceneObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_id: str
    camera_id: str
    image_width_px: int
    image_height_px: int
    rgb_path: Path
    segmentation_path: Path
    segmentation_visualization_path: Path
    instance_index_path: Path
    instances: list[InstanceObservation]
