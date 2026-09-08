"""Global one-to-one matching for canonical normalized yxyx boxes."""
from __future__ import annotations


def iou(a, b):
    ay1, ax1, ay2, ax2 = a; by1, bx1, by2, bx2 = b
    intersection = max(0, min(ay2, by2) - max(ay1, by1)) * max(0, min(ax2, bx2) - max(ax1, bx1))
    union = (ay2 - ay1) * (ax2 - ax1) + (by2 - by1) * (bx2 - bx1) - intersection
    return intersection / union if union else 0.0


def match_detections(detections, truth, minimum_iou=0.20, ambiguity_margin=0.05):
    """Return detection_id/object_id bindings from a maximum-weight assignment."""
    if not detections:
        return [], [], []
    matrix = [[iou(detection.bbox, item["bbox"]) for item in truth] for detection in detections]
    assignment = _maximize_assignment(matrix)
    matches, unmatched, ambiguous = [], [], []
    for row, detection in enumerate(detections):
        detection_id = getattr(detection, "detection_id", getattr(detection, "entity_id", str(row)))
        column = assignment[row]
        if column is None or matrix[row][column] < minimum_iou:
            unmatched.append(detection_id)
            continue
        ranked = sorted(matrix[row], reverse=True)
        if len(ranked) > 1 and ranked[0] - ranked[1] < ambiguity_margin:
            ambiguous.append(detection_id)
            continue
        matches.append((detection_id, truth[column]["object_id"], matrix[row][column]))
    return matches, unmatched, ambiguous


def _maximize_assignment(weights):
    """Hungarian assignment for a rectangular non-negative weight matrix."""
    rows = len(weights); columns = len(weights[0]) if rows else 0
    size = max(rows, columns)
    maximum = max((value for row in weights for value in row), default=0.0)
    cost = [[maximum - (weights[i][j] if i < rows and j < columns else 0.0) for j in range(size)] for i in range(size)]
    u = [0.0] * (size + 1); v = [0.0] * (size + 1); p = [0] * (size + 1); way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i; j0 = 0; minimum = [float("inf")] * (size + 1); used = [False] * (size + 1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = float("inf"); j1 = 0
            for j in range(1, size + 1):
                if used[j]: continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]: minimum[j] = current; way[j] = j0
                if minimum[j] < delta: delta = minimum[j]; j1 = j
            for j in range(size + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minimum[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    result = [None] * rows
    for j in range(1, size + 1):
        if 1 <= p[j] <= rows and j <= columns: result[p[j] - 1] = j - 1
    return result
