from typing import Protocol
from pydantic import BaseModel, Field
from ..contracts.task_intent import TaskIntent
class TaskUnderstandingRequest(BaseModel):
    instruction: str
    supported_task_types: list[str]
    supported_directions: list[str]
    asset_catalog: list[dict] = Field(default_factory=list)
class TaskUnderstandingProvider(Protocol):
    def understand(self,request:TaskUnderstandingRequest)->TaskIntent: ...
