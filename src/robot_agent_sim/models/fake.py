from __future__ import annotations

import re

from ..contracts.skill_plan import SemanticSubtask, SkillPlan, SkillStep
from ..contracts.task_intent import (
    EntityRole, Operation, SpatialRelation, SpatialRelationType, TaskEntity,
    TaskIntent, TaskStatus, TaskType,
)
from .vision_grounding import VisionDetection, VisionDetectionResult


class FakeTaskUnderstandingProvider:
    def understand(self, request):
        text = request.instruction.strip(); low = text.casefold()
        if any(token in text.casefold() for token in ("东北", "东南", "西北", "西南", "左前方", "右前方", "斜上方", "northeast", "northwest", "southeast", "southwest", "diagonal")):
            return TaskIntent(status=TaskStatus.DIRECTION_CLARIFICATION_REQUIRED, instruction=text)
        if any(token in text for token in ("拧紧螺丝", "旋紧螺丝", "tighten screw")):
            return TaskIntent(status=TaskStatus.UNSUPPORTED_TASK, instruction=text)

        entities = []
        if "红" in text or "red" in low:
            entities.append(TaskEntity(entity_id="red_cube_01", semantic_name="red cube", category="cube", color="red", role=EntityRole.SOURCE))
        if "蓝" in text or "blue" in low:
            entities.append(TaskEntity(entity_id="blue_box_01", semantic_name="blue box", category="container", color="blue", role=EntityRole.DESTINATION))
        if "黄" in text or "yellow" in low:
            entities.append(TaskEntity(entity_id="yellow_cube_01", semantic_name="yellow cube", category="cube", color="yellow", role=EntityRole.REFERENCE))
        for token, name, category in (("苹果", "apple", "fruit"), ("香蕉", "banana", "fruit"), ("棒球", "baseball", "ball"), ("魔方", "rubiks cube", "cube"), ("海绵", "sponge", "sponge"), ("勺子", "spoon", "utensil"), ("糖盒", "sugar box", "package")):
            if token in text or token in low:
                entities.append(TaskEntity(entity_id=f"{name.replace(' ', '_')}_01", semantic_name=name, category=category, role=EntityRole.TARGET))
        if "按钮" in text or "button" in low:
            entities.append(TaskEntity(entity_id="button_01", semantic_name="button", category="button", role=EntityRole.TARGET))
        if not entities:
            entities = [TaskEntity(entity_id="target_01", semantic_name="target object", category="cube")]

        relations = []
        relative_placement = any(token in text for token in ("放到", "放进", "放入")) and len(entities) > 1
        if any(token in text or token in low for token in ("左", "西", "left", "west")):
            subject = entities[0] if relative_placement else entities[0]
            reference = entities[1] if len(entities) > 1 else None
            relations.append(SpatialRelation(subject=subject.entity_id, relation=SpatialRelationType.LEFT_OF, reference=reference.entity_id if reference else None, direction="left"))
        # In “left A ... right B” the left relation already fully specifies
        # the pair.  Emitting both inverse relations would create a dependency
        # cycle in deterministic scene placement.
        if (any(token in text or token in low for token in ("右", "东", "right", "east")) and not (relative_placement and any(token in text or token in low for token in ("左", "西", "left", "west")))):
            subject = entities[0] if relative_placement else (entities[1] if len(entities) > 1 else entities[0])
            reference = entities[1] if relative_placement else (entities[0] if len(entities) > 1 else None)
            relations.append(SpatialRelation(subject=subject.entity_id, relation=SpatialRelationType.RIGHT_OF, reference=reference.entity_id if reference else None, direction="right"))
        if "最近" in text or "nearest" in low:
            source = next((e for e in entities if e.role == EntityRole.SOURCE), entities[0]); ref = next((e for e in entities if e.role == EntityRole.REFERENCE), entities[-1])
            relations.append(SpatialRelation(subject=source.entity_id, relation=SpatialRelationType.NEAREST, reference=ref.entity_id))

        # Unary direction phrases (for example, “抓取前面的红方块”) still
        # constrain generated placement, but do not invent a second entity.
        direction_tokens = (
            (("前", "北", "front", "north"), SpatialRelationType.FRONT_OF, "front"),
            (("后", "南", "back", "south"), SpatialRelationType.BEHIND, "back"),
            (("上", "above", "up"), SpatialRelationType.ABOVE, "up"),
            (("下", "below", "down"), SpatialRelationType.BELOW, "down"),
        )
        for tokens, relation_type, direction in direction_tokens:
            if any(token in text or token in low for token in tokens) and not any(r.direction == direction for r in relations):
                relations.append(SpatialRelation(subject=entities[0].entity_id, relation=relation_type, direction=direction))

        operations = []
        put = any(token in text for token in ("放进", "放入", "放到")) or "put" in low
        move = any(token in text for token in ("移动", "移到", "搬到")) or "move" in low
        search = any(token in text for token in ("搜索", "寻找", "查找")) or "search" in low
        locate = any(token in text for token in ("定位", "找到")) or "locate" in low
        if put:
            source = next((e for e in entities if e.role == EntityRole.SOURCE), entities[0])
            destination = next((e for e in entities if e.role == EntityRole.DESTINATION), None)
            if destination is None and any(token in text for token in ("盒", "容器")):
                destination = TaskEntity(entity_id="open_box_01", semantic_name="open box", category="container", role=EntityRole.DESTINATION); entities.append(destination)
            if destination is None and len(entities) > 1:
                destination = entities[1]; destination.role = EntityRole.REFERENCE
            if destination is None:
                return TaskIntent(status=TaskStatus.INVALID, instruction=text, explanation="放置任务缺少 destination 实体")
            source.role = EntityRole.SOURCE
            if destination.role == EntityRole.DESTINATION:
                operations.append(Operation(operation_id="op-1", task_type=TaskType.PICK_AND_PLACE, source=source.entity_id, destination=destination.entity_id, description="pick and place"))
            else:
                # “放到香蕉右边” is a semantic relative move, not a
                # container placement.  Keep the reference explicit and do
                # not invent grasp/release steps.
                operations.append(Operation(operation_id="op-1", task_type=TaskType.MOVE, target=source.entity_id, reference=destination.entity_id, description="move relative to reference"))
        if "按" in text or "press" in low:
            target = next((e for e in entities if e.category == "button"), entities[-1])
            operations.append(Operation(operation_id=f"op-{len(operations)+1}", task_type=TaskType.PRESS, target=target.entity_id, description="press target", depends_on=[operations[-1].operation_id] if operations else []))
        elif move and not put:
            target = entities[0]
            operations.append(Operation(operation_id="op-1", task_type=TaskType.MOVE, target=target.entity_id, reference=entities[1].entity_id if len(entities) > 1 else None, description="move to semantic target"))
        elif search:
            operations.append(Operation(operation_id="op-1", task_type=TaskType.SEARCH, target=entities[0].entity_id, description="search for target"))
        elif locate:
            operations.append(Operation(operation_id="op-1", task_type=TaskType.LOCATE, target=entities[0].entity_id, description="locate target"))
        elif any(token in text for token in ("释放", "放开", "release")):
            operations.append(Operation(operation_id="op-1", task_type=TaskType.RELEASE, target=entities[0].entity_id, description="release target"))
        elif any(token in text for token in ("抓", "拿", "拾", "捡")) or "grasp" in low or "pick" in low:
            operations.append(Operation(operation_id="op-1", task_type=TaskType.GRASP, target=entities[0].entity_id, description="grasp target"))
        if not operations:
            return TaskIntent(status=TaskStatus.UNSUPPORTED_TASK, instruction=text, explanation="未识别到支持的任务动作")
        types = list(dict.fromkeys(op.task_type for op in operations))
        if len(types) > 1: types.append(TaskType.MIXED)
        return TaskIntent(status=TaskStatus.ACCEPTED, instruction=text, task_types=types, entities=entities, operations=operations, spatial_relations=relations)


class FakeVisionGroundingProvider:
    def __init__(self, detections=None): self.detections = detections
    def detect(self, request):
        values = self.detections if self.detections is not None else [VisionDetection(entity_id=e["entity_id"], bbox=[100+i*100, 100+i*80, 280+i*100, 300+i*80], confidence=0.95) for i, e in enumerate(request.entities)]
        return VisionDetectionResult(detections=values)


class FakeSkillPlanningProvider:
    def plan(self, request):
        steps = []
        object_by_entity = {entity.entity_id: entity.object_id for entity in request.task.entities}
        resolve = lambda value: object_by_entity.get(value) if value else None
        def add(skill, description, target=None, reference=None, semantic_target=None):
            step_id = f"step-{len(steps)+1}"
            steps.append(SkillStep(step_id=step_id, semantic_subtask=SemanticSubtask(action=skill, description=description), skill_name=skill, target_object=target, reference_object=reference, semantic_target=semantic_target, depends_on=[steps[-1].step_id] if steps else []))
        for op in request.task.operations:
            if op.task_type == TaskType.PICK_AND_PLACE:
                source = resolve(op.source); destination = resolve(op.destination or op.reference)
                semantic_target = "container_interior" if any(entity.entity_id == (op.destination or op.reference) and entity.role == EntityRole.DESTINATION for entity in request.task.entities) else "relative_region"
                add("locate", "定位源物体", source); add("move", "移动到源物体抓取区域", source, semantic_target="grasp_region"); add("grasp", "抓取源物体", source); add("locate", "定位目标或参考物体", destination); add("move", "携带物体移动到目标语义区域", destination, source, semantic_target); add("release", "在目标语义区域释放物体", source, destination, semantic_target)
            elif op.task_type == TaskType.PRESS:
                target = resolve(op.target); add("locate", "定位按钮", target); add("move", "移动到按钮表面", target, semantic_target="button_surface"); add("press", "按下按钮", target)
            elif op.task_type == TaskType.SEARCH:
                target = resolve(op.target); add("search", "搜索任务目标", target)
            elif op.task_type == TaskType.LOCATE:
                target = resolve(op.target); add("locate", "定位目标", target)
            elif op.task_type == TaskType.MOVE:
                target = resolve(op.target); reference = resolve(op.reference)
                add("locate", "定位移动目标", target)
                add("move", "移动到语义目标区域", target, reference, "relative_region" if reference else "semantic_region")
            elif op.task_type == TaskType.RELEASE:
                target = resolve(op.target or op.source); reference = resolve(op.reference or op.destination)
                add("locate", "定位释放目标", target)
                add("release", "在语义目标区域释放物体", target, reference, "semantic_region")
            elif op.task_type == TaskType.GRASP:
                target = resolve(op.target); add("locate", "定位目标", target); add("move", "移动到目标抓取区域", target, semantic_target="grasp_region"); add("grasp", "抓取目标", target)
        return SkillPlan(task_types=request.task.task_types, steps=steps, model_call_count=0)
