from pydantic import BaseModel
class SkillDefinition(BaseModel): skill_name:str; description:str
class AtomicSkillRegistry:
    names=("locate","search","move","grasp","release","press")
    def __init__(self): self._items={n:SkillDefinition(skill_name=n,description=f"Planning-only {n} skill") for n in self.names}
    def list(self): return list(self._items.values())
    def require(self,name):
        if name not in self._items: raise ValueError(f"unknown skill_name: {name}")
        return self._items[name]
REGISTRY=AtomicSkillRegistry()
