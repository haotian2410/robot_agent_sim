from __future__ import annotations

import re

from ..contracts.task_intent import SpatialRelationType
from .skill_planning import LLMOperationPlan, LLMPlanStep, SkillPlanLLMOutput
from .task_understanding import ParseEntity, ParseOperation, ParseRelation, TaskParseLLMOutput
from .vision_grounding import VisionCandidate, VisionLLMOutput
from ..planning.recipes import RECIPE_DEFINITIONS


class FakeTaskUnderstandingProvider:
    """Offline parser used by tests and the CLI default; no model loop/retry."""

    def understand(self, request):
        text = request.instruction.strip()
        low = text.casefold()
        diagonals = ("东北", "东南", "西北", "西南", "左前方", "右前方", "斜上方", "northeast", "northwest", "southeast", "southwest", "diagonal")
        if any(token in low for token in diagonals):
            raw = next((token for token in diagonals if token in text or token in low), "diagonal")
            return TaskParseLLMOutput(status="direction_clarification_required", raw_direction=raw)
        if any(token in low for token in ("拧紧螺丝", "旋紧螺丝", "tighten screw")):
            return TaskParseLLMOutput(status="unsupported_task", raw_task=text)

        entities: list[ParseEntity] = []

        def add(eid, name, category, color=None):
            if not any(entity.id == eid for entity in entities):
                entities.append(ParseEntity(id=eid, name=name, category=category, color=color))

        if "红" in text or "red" in low: add("red_cube_01", "red cube", "cube", "red")
        if "蓝" in text or "blue" in low: add("blue_box_01", "blue box", "container", "blue")
        if "黄" in text or "yellow" in low: add("yellow_cube_01", "yellow cube", "cube", "yellow")
        if ("盒" in text or "box" in low) and not any(entity.category == "container" for entity in entities):
            add("open_box_01", "open box", "container", None)
        if "按钮" in text or "button" in low: add("button_01", "button", "button")
        for token, name, category in (("苹果", "apple", "fruit"), ("香蕉", "banana", "fruit"), ("棒球", "baseball", "ball"), ("魔方", "rubiks cube", "cube"), ("海绵", "sponge", "sponge"), ("勺子", "spoon", "utensil"), ("糖盒", "sugar box", "package")):
            if token in text or token in low:
                add(f"{name.replace(' ', '_')}_01", name, category)
        if "螺丝" in text or "screw" in low:
            add("screw_01", "screw", "screw")
        for label in ("a", "b", "c"):
            # Chinese characters are adjacent to the Latin labels, so ``\b``
            # does not create a boundary here.  Restrict the lookaround to
            # ASCII letters instead and still avoid matching words.
            if re.search(rf"(?<![a-z]){label}(?![a-z])", low):
                add(f"{label}_01", label, "object")
        if not entities: add("target_01", "target object", "cube")

        relations: list[ParseRelation] = []
        put = any(token in text for token in ("放进", "放入", "放到", "放在")) or "put" in low
        has_left = any(token in low for token in ("左", "西", "left", "west"))
        has_right = any(token in low for token in ("右", "东", "right", "east"))
        if len(entities) > 1 and has_left:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.LEFT_OF, reference=entities[1].id))
        elif len(entities) > 1 and has_right:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.RIGHT_OF, reference=entities[1].id))
        elif has_left:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.LEFT))
        elif has_right:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.RIGHT))
        if "最近" in text or "nearest" in low:
            red = next((entity for entity in entities if entity.color == "red"), entities[0])
            yellow = next((entity for entity in entities if entity.color == "yellow"), entities[-1])
            relations.append(ParseRelation(scope="selection", subject=red.id, relation=SpatialRelationType.NEAREST, reference=yellow.id))
        if "最远" in text or "farthest" in low:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.FARTHEST, reference=entities[-1].id))
        for tokens, relation in ((("前", "north", "front"), SpatialRelationType.FRONT), (("后", "south", "back"), SpatialRelationType.BACK), (("上", "above", "up"), SpatialRelationType.UP), (("下", "below", "down"), SpatialRelationType.DOWN)):
            if any(token in text or token in low for token in tokens) and not (relation == SpatialRelationType.BACK and "然后" in text):
                relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=relation))

        operations: list[ParseOperation] = []
        # Preserve entity reuse in simple chained pick-and-place language such
        # as “把 A 放进 B，再把 B 放进 C”.  Roles are local to each operation.
        letters = [entity for entity in entities if entity.category == "object" and entity.id.endswith("_01")]
        chained = len(letters) >= 3 and text.count("放") >= 2
        if chained:
            operations.extend([
                ParseOperation(type="pick_and_place", source=letters[0].id, destination=letters[1].id),
                ParseOperation(type="pick_and_place", source=letters[1].id, destination=letters[2].id),
            ])
            relations.extend([
                ParseRelation(scope="goal", subject=letters[0].id, relation=SpatialRelationType.INSIDE, reference=letters[1].id),
                ParseRelation(scope="goal", subject=letters[1].id, relation=SpatialRelationType.INSIDE, reference=letters[2].id),
            ])
        if chained and any(token in text or token in low for token in ("按", "press")):
            operations.append(ParseOperation(type="press", target=letters[2].id))
        if chained:
            return TaskParseLLMOutput(status="accepted", entities=entities, operations=operations, relations=relations)
        if put:
            source = next((entity for entity in entities if entity.color == "red"), entities[0])
            destination = next((entity for entity in entities if entity.category == "container" and entity.id != source.id), None)
            if destination is None and len(entities) > 1: destination = entities[1]
            if destination is not None:
                if destination.category != "container" and has_right:
                    operations.append(ParseOperation(type="move", target=source.id, reference=destination.id))
                else:
                    operations.append(ParseOperation(type="pick_and_place", source=source.id, destination=destination.id))
                    relations.append(ParseRelation(scope="goal", subject=source.id, relation=SpatialRelationType.INSIDE, reference=destination.id))
        if any(token in text or token in low for token in ("按", "press")):
            button = next((entity for entity in entities if entity.category == "button"), entities[-1])
            operations.append(ParseOperation(type="press", target=button.id))
        elif not operations and any(token in text or token in low for token in ("移动", "移到", "move")):
            operations.append(ParseOperation(type="move", target=entities[0].id, reference=entities[1].id if len(entities) > 1 else None))
        elif not operations and any(token in text or token in low for token in ("搜索", "寻找", "查找", "search")):
            operations.append(ParseOperation(type="search", target=entities[0].id))
        elif not operations and any(token in text or token in low for token in ("定位", "找到", "locate")):
            operations.append(ParseOperation(type="locate", target=entities[0].id))
        elif not operations and any(token in text or token in low for token in ("抓", "拿", "拾", "捡", "grasp", "pick")):
            operations.append(ParseOperation(type="grasp", target=entities[0].id))
        elif not operations and any(token in text or token in low for token in ("释放", "放开", "release")):
            operations.append(ParseOperation(type="release", target=entities[0].id))
        if not operations:
            return TaskParseLLMOutput(status="unsupported_task", raw_task=text)
        return TaskParseLLMOutput(status="accepted", entities=entities, operations=operations, relations=relations)


class FakeVisionGroundingProvider:
    def __init__(self, detections=None): self.detections = detections

    def detect(self, request):
        if self.detections is not None:
            return VisionLLMOutput(detections=self.detections)
        seen = {}
        values = []
        for index, entity in enumerate(request.entities):
            seen[entity.id] = seen.get(entity.id, 0) + 1
            detection_id = entity.id if seen[entity.id] == 1 else f"{entity.id}-{seen[entity.id]}"
            values.append(VisionCandidate(detection_id=detection_id, entity_id=entity.id, bbox=[100 + index * 100, 100 + index * 80, 280 + index * 100, 300 + index * 80]))
        return VisionLLMOutput(detections=values)


class FakeSkillPlanningProvider:
    def plan(self, request):
        plans = []
        for operation in request.context.operations:
            proxy = type("Operation", (), {
                "target": "target" if operation.has_target else None,
                "source": "source" if operation.has_source else None,
                "destination": "destination" if operation.has_destination else None,
                "reference": "reference" if operation.has_reference else None,
            })()
            steps = [LLMPlanStep(skill=skill, target=target, reference=reference, region=region) for skill, target, reference, region in RECIPE_DEFINITIONS[operation.type].build(proxy)]
            plans.append(LLMOperationPlan(id=operation.id, steps=steps))
        return SkillPlanLLMOutput(operations=plans)
