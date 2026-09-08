from typing import Protocol
from pydantic import BaseModel
from ..contracts.grounded_task import GroundedTask
from ..contracts.skill_plan import SkillPlan
class SkillPlanningRequest(BaseModel):
    task: GroundedTask
    skill_catalog: list[dict]
class SkillPlanningProvider(Protocol):
    def plan(self,request:SkillPlanningRequest)->SkillPlan: ...
