from __future__ import annotations

import math
import random
import re

from ..contracts.task_intent import Direction, SpatialRelationType
from .registry import SceneObject, SceneRegistry

WORKSPACE_X = (-0.31, 0.31)
WORKSPACE_Y = (-0.66, 0.66)
MIN_GAP_M = 0.04


class SceneComposer:
    def compose(self, intent, assets, robot: str = "panda", seed: int = 0) -> SceneRegistry:
        rng = random.Random(seed)
        selection_subjects = {r.subject for r in intent.spatial_relations if r.relation in {SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST}}
        objects: list[SceneObject] = []
        bindings: dict[str, str] = {}

        by_entity = {entity.entity_id: entity for entity in intent.entities}
        preferred_by_entity: dict[str, tuple[float, float, float]] = {}
        for relation in intent.spatial_relations:
            if not relation.reference or relation.relation not in {
                SpatialRelationType.LEFT_OF,
                SpatialRelationType.RIGHT_OF,
                SpatialRelationType.FRONT_OF,
                SpatialRelationType.BEHIND,
                SpatialRelationType.ABOVE,
                SpatialRelationType.BELOW,
            }:
                continue
            subject_dimensions = assets[relation.subject].dimensions_m or _primitive_dimensions(assets[relation.subject].model_name)
            reference_dimensions = assets[relation.reference].dimensions_m or _primitive_dimensions(assets[relation.reference].model_name)
            subject_position, reference_position = self._relation_pair_positions(
                relation.relation, subject_dimensions, reference_dimensions
            )
            preferred_by_entity.setdefault(relation.subject, subject_position)
            preferred_by_entity.setdefault(relation.reference, reference_position)
        placed: set[str] = set()
        placing: set[str] = set()

        def place(entity_id: str) -> None:
            if entity_id in placed or entity_id in selection_subjects:
                return
            entity = by_entity[entity_id]
            if entity_id in placing:
                raise ValueError("cyclic spatial relations are not supported")
            placing.add(entity_id)
            relation = next(
                (
                    relation for relation in intent.spatial_relations
                    if relation.subject == entity_id
                    and relation.reference
                    and relation.relation in {
                        SpatialRelationType.LEFT_OF,
                        SpatialRelationType.RIGHT_OF,
                        SpatialRelationType.FRONT_OF,
                        SpatialRelationType.BEHIND,
                        SpatialRelationType.ABOVE,
                        SpatialRelationType.BELOW,
                    }
                ),
                None,
            )
            preferred = preferred_by_entity.get(entity_id)
            if relation is not None:
                place(relation.reference)
            item = self._make_object(entity, assets[entity_id], 1, objects, rng, intent, preferred=preferred)
            objects.append(item)
            bindings[entity.entity_id] = item.object_id
            placed.add(entity_id)
            placing.discard(entity_id)

        for entity in intent.entities:
            place(entity.entity_id)

        for entity in [e for e in intent.entities if e.entity_id in selection_subjects]:
            relation = next(r for r in intent.spatial_relations if r.subject == entity.entity_id and r.relation in {SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST})
            reference = next(item for item in objects if item.object_id == bindings[relation.reference])
            # Put the selected (near) candidate at a deterministic offset
            # that is guaranteed to be closer than the distractor.
            # Build both candidates against the already-placed objects.  The
            # near candidate's preferred offset is tried first; if clipping
            # or workspace bounds prevent it, the seeded sampler still finds
            # a valid location and ranking below preserves the selection.
            near = self._make_object(entity, assets[entity.entity_id], 1, objects, rng, intent, preferred=(reference.position[0] + 0.12, reference.position[1]))
            objects.append(near)
            far = self._make_object(entity, assets[entity.entity_id], 2, objects, rng, intent, preferred=(reference.position[0] - 0.26, reference.position[1]))
            objects.append(far)
            ranked = sorted([near, far], key=lambda item: _distance(item.position, reference.position), reverse=relation.relation == SpatialRelationType.FARTHEST)
            # Preserve the natural-language selection in the registry.
            bindings[entity.entity_id] = ranked[0].object_id

        return SceneRegistry(scene_id=f"{robot}_generated_{seed}", robot=robot, objects=objects, bindings=bindings)

    @staticmethod
    def _relation_pair_positions(relation, subject_dimensions, reference_dimensions):
        gap = MIN_GAP_M + 0.01
        subject = [0.0, 0.0, 0.0]
        reference = [0.0, 0.0, 0.0]
        if relation == SpatialRelationType.LEFT_OF:
            separation = reference_dimensions[0] / 2 + subject_dimensions[0] / 2 + gap
            subject[0], reference[0] = -separation / 2, separation / 2
        elif relation == SpatialRelationType.RIGHT_OF:
            separation = reference_dimensions[0] / 2 + subject_dimensions[0] / 2 + gap
            subject[0], reference[0] = separation / 2, -separation / 2
        elif relation == SpatialRelationType.FRONT_OF:
            separation = reference_dimensions[1] / 2 + subject_dimensions[1] / 2 + gap
            subject[1], reference[1] = separation / 2, -separation / 2
        elif relation == SpatialRelationType.BEHIND:
            separation = reference_dimensions[1] / 2 + subject_dimensions[1] / 2 + gap
            subject[1], reference[1] = -separation / 2, separation / 2
        elif relation == SpatialRelationType.ABOVE:
            subject[2] = reference_dimensions[2] + gap
        elif relation == SpatialRelationType.BELOW:
            reference[2] = subject_dimensions[2] + gap
        return tuple(subject), tuple(reference)

    def _make_object(self, entity, asset, index, existing, rng, intent, preferred=None):
        dimensions = asset.dimensions_m or _primitive_dimensions(asset.model_name)
        position = self._position(entity.entity_id, dimensions, existing, rng, intent, preferred)
        base = _slug(entity.semantic_name or asset.model_name)
        object_id = f"{base}_{index:02d}"
        return SceneObject(object_id=object_id, body_name=object_id, role=entity.role.value,
                           semantic_name=entity.semantic_name, position=position, dimensions_m=dimensions,
                           entity_id=entity.entity_id if index == 1 else None, candidate_for=entity.entity_id,
                           model_id=asset.model_id, model_name=asset.model_name)

    def _position(self, entity_id, dimensions, existing, rng, intent, preferred):
        unary = next((r.direction for r in intent.spatial_relations if r.subject == entity_id and r.direction), None)
        presets = {Direction.LEFT: (-0.22, 0.0, 0.0), Direction.RIGHT: (0.22, 0.0, 0.0),
                   Direction.FRONT: (0.0, 0.38, 0.0), Direction.BACK: (0.0, -0.38, 0.0),
                   Direction.UP: (0.0, 0.0, 0.20), Direction.DOWN: (0.0, 0.0, 0.02)}
        candidates = []
        if preferred is not None: candidates.append((preferred[0], preferred[1], preferred[2] if len(preferred) == 3 else 0.0))
        if unary in presets: candidates.append(presets[unary])
        candidates.extend((rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y), 0.0) for _ in range(200))
        for x, y, z in candidates:
            x = max(WORKSPACE_X[0] + dimensions[0] / 2, min(WORKSPACE_X[1] - dimensions[0] / 2, x))
            y = max(WORKSPACE_Y[0] + dimensions[1] / 2, min(WORKSPACE_Y[1] - dimensions[1] / 2, y))
            candidate = (round(x, 6), round(y, 6), round(z, 6))
            if all(not _overlap(candidate, dimensions, item.position, item.dimensions_m or (0, 0, 0)) for item in existing):
                return candidate
        raise ValueError("unable to find a valid non-overlapping placement")


def _primitive_dimensions(name):
    return {"cube_basic": (0.06, 0.06, 0.06), "open_box": (0.18, 0.18, 0.10), "button_basic": (0.08, 0.08, 0.04)}[name]


def _overlap(a, ad, b, bd):
    xy_overlap = (
        abs(a[0] - b[0]) < (ad[0] + bd[0]) / 2 + MIN_GAP_M
        and abs(a[1] - b[1]) < (ad[1] + bd[1]) / 2 + MIN_GAP_M
    )
    z_separated = a[2] >= b[2] + bd[2] + MIN_GAP_M or b[2] >= a[2] + ad[2] + MIN_GAP_M
    return xy_overlap and not z_separated


def _distance(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])


def _slug(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "scene_object"
