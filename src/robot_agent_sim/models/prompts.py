from __future__ import annotations

import json


TASK_UNDERSTANDING_PROMPT = """解析机器人任务，只输出规定 JSON。
支持 locate/search/move/grasp/release/pick_and_place/press/open/close。
open/close 的 target 是门或抽屉，reference 是对应把手；不要把 open/close 当成底层控制指令。
方向仅 left/right/front/back/up/down；东=right、西=left、南=back、北=front。
东北、左前方、斜上方等返回 direction_clarification_required；不支持的任务返回 unsupported_task。
禁止 explanation、operation_id、XYZ、object_id、模型信息、技能步骤和 task_types。"""


VISION_GROUNDING_PROMPT = """按实体语义返回 RGB 中所有相关候选 bbox。
bbox=[ymin,xmin,ymax,xmax]，整数范围 0..1000；同一 entity 可以有多个候选。
只输出 entity 和 bbox；不要输出 detection id、object_id、世界坐标、动作、置信度或解释。"""


SKILL_PLANNING_PROMPT = """根据 operations 与 Atomic Skill Catalog 生成技能顺序。
保持 operation 顺序，只使用注册技能；target/reference 只能使用 operation 角色。
不要输出 object_id、step id、depends_on、description、XYZ、轨迹或解释。"""


def prompt_payload(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
