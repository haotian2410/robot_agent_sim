from __future__ import annotations

import math

from ..contracts.task_intent import SpatialRelationType


class WorldRelationError(ValueError):
    pass


class RelationNotSatisfied(WorldRelationError):
    pass


class RelationAmbiguous(WorldRelationError):
    pass


class WorldRelationResolver:
    """Select task instances using MuJoCo world coordinates, never image axes."""

    def resolve(self, intent, candidates, positions):
        selected = {}
        resolving = set()

        def choose(entity_id):
            if entity_id in selected: return selected[entity_id]
            if entity_id in resolving: raise WorldRelationError("cyclic selection relations")
            resolving.add(entity_id)
            values = candidates.get(entity_id, [])
            if not values: raise WorldRelationError(f"missing visual candidates for {entity_id}")
            relation = next((item for item in intent.spatial_relations if item.scope == "selection" and item.subject == entity_id), None)
            if relation is None:
                if len(values) != 1: raise RelationAmbiguous(f"grounding_ambiguous: multiple candidates without selection relation: {entity_id}")
                result = values[0]
            else:
                relation_type = relation.relation
                if relation_type in {SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST}:
                    reference = choose(relation.reference)
                    reference_position = positions[reference["object_id"]]
                    result = min(values, key=lambda value: math.dist(positions[value["object_id"]][:2], reference_position[:2])) if relation_type == SpatialRelationType.NEAREST else max(values, key=lambda value: math.dist(positions[value["object_id"]][:2], reference_position[:2]))
                else:
                    axis_sign = {
                        SpatialRelationType.LEFT: (0, 1), SpatialRelationType.LEFT_OF: (0, 1),
                        SpatialRelationType.RIGHT: (0, -1), SpatialRelationType.RIGHT_OF: (0, -1),
                        SpatialRelationType.FRONT: (1, -1), SpatialRelationType.FRONT_OF: (1, -1),
                        SpatialRelationType.BACK: (1, 1), SpatialRelationType.BEHIND: (1, 1),
                        SpatialRelationType.UP: (2, -1), SpatialRelationType.ABOVE: (2, -1),
                        SpatialRelationType.DOWN: (2, 1), SpatialRelationType.BELOW: (2, 1),
                    }.get(relation_type)
                    if axis_sign is None: raise WorldRelationError(f"unsupported selection relation: {relation_type}")
                    axis, sign = axis_sign
                    if relation.reference:
                        reference = choose(relation.reference)
                        reference_value = positions[reference["object_id"]][axis]
                        satisfying = [value for value in values if (positions[value["object_id"]][axis] - reference_value) * sign < 0]
                        if not satisfying:
                            raise RelationNotSatisfied(f"relation_not_satisfied: {entity_id} {relation_type} {relation.reference}")
                        if len(satisfying) > 1:
                            raise RelationAmbiguous(f"grounding_ambiguous: {entity_id} {relation_type} {relation.reference}")
                        result = satisfying[0]
                    else:
                        result = min(values, key=lambda value: sign * positions[value["object_id"]][axis])
            resolving.remove(entity_id); selected[entity_id] = result; return result

        for entity in intent.entities: choose(entity.entity_id)
        return selected
