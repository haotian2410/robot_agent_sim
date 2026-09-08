from ..contracts.task_intent import Direction
ALIASES={"左":Direction.LEFT,"left":Direction.LEFT,"西":Direction.LEFT,"右":Direction.RIGHT,"right":Direction.RIGHT,"东":Direction.RIGHT,"前":Direction.FRONT,"front":Direction.FRONT,"南":Direction.FRONT,"后":Direction.BACK,"back":Direction.BACK,"北":Direction.BACK,"上":Direction.UP,"up":Direction.UP,"下":Direction.DOWN,"down":Direction.DOWN}
UNSUPPORTED={"东北","东南","西北","西南","左前方","右前方","斜上方","diagonal","northeast","northwest","southeast","southwest"}
def normalize_direction(value:str)->Direction:
    key=value.strip().lower()
    if key in UNSUPPORTED or any(x in key for x in UNSUPPORTED): raise ValueError("direction_clarification_required")
    try:return ALIASES[key]
    except KeyError as e: raise ValueError("direction_clarification_required") from e
