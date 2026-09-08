from typing import Literal, Protocol
from pydantic import BaseModel, Field, field_validator, model_validator
class VisionDetection(BaseModel):
    entity_id:str
    bbox:list[int]=Field(min_length=4,max_length=4)
    confidence:float=Field(ge=0,le=1)
    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls,v):
        if any(x<0 or x>1000 for x in v) or v[0]>=v[2] or v[1]>=v[3]: raise ValueError("bbox must be normalized yxyx 0..1000")
        return v
class VisionDetectionResult(BaseModel):
    status: Literal["accepted", "rejected"] = "accepted"
    detections: list[VisionDetection]
    explanation: str = ""

    @model_validator(mode="after")
    def unique_entities(self):
        ids = [item.entity_id for item in self.detections]
        if len(ids) != len(set(ids)):
            raise ValueError("vision detections must have unique entity_id")
        return self
class VisionGroundingRequest(BaseModel):
    instruction: str
    entities: list[dict]
    rgb_path: str
class VisionGroundingProvider(Protocol):
    def detect(self,request:VisionGroundingRequest)->VisionDetectionResult: ...
