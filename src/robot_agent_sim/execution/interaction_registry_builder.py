"""Build a scene-bound interaction registry for execution."""

from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

import mujoco

from ..scene.registry import SceneRegistry


DIRECTIONS = {
    "above": [0.0, 0.0, 1.0],
    "below": [0.0, 0.0, -1.0],
    "front": [1.0, 0.0, 0.0],
    "behind": [-1.0, 0.0, 0.0],
    "left": [0.0, 1.0, 0.0],
    "right": [0.0, -1.0, 0.0],
    "at": [0.0, 0.0, 0.0],
}

DEFAULT_DISTANCES = {
    "above": 0.12,
    "below": 0.08,
    "front": 0.10,
    "behind": 0.10,
    "left": 0.10,
    "right": 0.10,
    "at": 0.0,
}

TOOL_ORIENTATION = {
    "representation": "rpy",
    "roll": 180.0,
    "pitch": 0.0,
    "yaw": 0.0,
}


def _scene_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_generated_registry(
    scene_path: str | Path,
    scene_registry: SceneRegistry,
    output_path: str | Path,
) -> Path:
    """Generate execution metadata for simple Route A objects.

    Poses are read back from the final compiled MuJoCo model.  This builder
    intentionally supports only ordinary graspables, generated open boxes and
    generated buttons; articulated doors/drawers require authored metadata.
    """
    scene = Path(scene_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    objects: dict[str, dict] = {}

    for item in scene_registry.objects:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, item.body_name
        )
        if body_id < 0:
            raise ValueError(f"generated registry body not found: {item.body_name}")
        position = [float(value) for value in data.xpos[body_id]]
        quaternion = [float(value) for value in data.xquat[body_id]]
        dimensions = item.dimensions_m or (0.06, 0.06, 0.06)
        aliases = list(dict.fromkeys(filter(None, (
            item.semantic_name,
            item.object_id,
            item.model_name,
            item.entity_id,
        ))))
        common = {
            "object_id": item.object_id,
            "body_name": item.body_name,
            "aliases": aliases,
            "spatial": {
                "source": {"type": "body", "name": item.body_name},
                "reference_pose": {
                    "position": position,
                    "quaternion_wxyz": quaternion,
                },
                "directions_local": DIRECTIONS,
                "default_distances": DEFAULT_DISTANCES,
                "tool_orientation": TOOL_ORIENTATION,
            },
            "default_interactions": {},
            "action_requests": {},
        }

        if item.model_name == "open_box":
            common["interactable"] = False
            common["spatial"].update({
                "default_anchor": "interior",
                "anchors": {
                    "interior": {
                        "target_id": f"{item.object_id}_interior",
                        "aliases": ["interior", "inside", "内部", "里面"],
                        "local_position": [0.0, 0.0, max(0.02, dimensions[2] * 0.55)],
                    }
                },
            })
        elif item.model_name == "button_basic":
            common["interactable"] = True
            common["spatial"].update({
                "default_anchor": "button_surface",
                "anchors": {
                    "button_surface": {
                        "target_id": f"{item.object_id}_surface",
                        "aliases": ["button surface", "按钮表面", "按压点"],
                        # Stop above the physical cap.  The press controller
                        # owns the final displacement; placing the move pose
                        # directly in the button geom is rejected as a local
                        # collision during preflight.
                        "local_position": [0.0, 0.0, 0.10],
                    }
                },
            })
            common["default_interactions"] = {"press": "button_surface"}
            common["action_requests"] = {
                "press": {
                    "target": {
                        "type": "button",
                        "object_id": item.object_id,
                        "surface_normal": [0.0, 0.0, -1.0],
                    },
                    "press": {
                        "strategy": "displacement_press",
                        "control_mode": "displacement",
                        "operation": "press",
                        "direction_source": "surface_normal",
                        "retract_after_press": True,
                        "verification_mode": "position",
                    },
                    "constraints": {
                        "maximum_force": 100.0,
                        "maximum_travel": 0.10,
                        "contact_threshold": 0.2,
                        "press_speed": 0.01,
                        "retract_speed": 0.03,
                        "hold_time": 0.1,
                        "verify_press": True,
                        "retry_count": 0,
                        "timeout": 10.0,
                        "require_contact": False,
                        "emergency_retract": True,
                    },
                    "strategy_params": {
                        "press_depth": 0.06,
                        "stop_on_contact": False,
                    },
                }
            }
        else:
            common["interactable"] = True
            common["spatial"].update({
                "default_anchor": "grasp",
                "anchors": {
                    "target": {
                        "target_id": f"{item.object_id}_target",
                        "aliases": ["target", "目标处"],
                        "local_position": [0.0, 0.0, dimensions[2] * 0.5],
                    },
                    "grasp": {
                        "target_id": f"{item.object_id}_grasp",
                        "aliases": ["grasp", "抓取位姿"],
                        "local_position": [0.0, 0.0, dimensions[2] * 0.5],
                    },
                },
            })
            expected_width = min(float(dimensions[0]), float(dimensions[1]))
            common["default_interactions"] = {"grasp": "grasp"}
            common["action_requests"] = {
                "grasp": {
                    "target": {
                        "type": "object",
                        "object_id": item.object_id,
                        "expected_width": expected_width,
                    },
                    "grasp": {
                        "strategy": "auto",
                        "operation": "regrasp",
                        "verification_mode": "auto",
                    },
                    "constraints": {
                        "verify_grasp": True,
                        "retry_count": 1,
                        "hold_on_success": True,
                        "timeout": 5.0,
                        "maximum_force": 40.0,
                        "contact_required": False,
                        "slip_check": False,
                    },
                    "strategy_params": {
                        "open_width": 0.08,
                        "close_width": 0.0,
                        "close_speed": 0.5,
                        "grasp_force": 20.0,
                        "hold_force": 10.0,
                        "position_tolerance": 0.01,
                        "settle_time": 0.0,
                    },
                },
                "release": {
                    "target": {"type": "object", "object_id": item.object_id},
                    "constraints": {
                        "open_width": 0.08,
                        "timeout": 5.0,
                        "verify_release": False,
                    },
                },
            }
        objects[item.object_id] = common

    result = {
        "version": 2,
        "registry_version": 1,
        "coordinate_frame": "mujoco_world",
        "scene": str(scene),
        "scene_fingerprint": _scene_sha256(scene),
        "move_defaults": {
            "planning": {
                "mode": "auto",
                "allow_replan": True,
                "planning_time": 8.0,
                "max_attempts": 3,
            },
            "constraints": {
                "avoid_collision": True,
                "velocity_scale": 0.8,
                "acceleration_scale": 0.8,
                "position_tolerance": 0.005,
                "orientation_tolerance": 2.0,
                "timeout": 30.0,
            },
            # The control package's portable numerical IK is selected by its
            # public ``auto`` profile; backend-specific names stay internal.
            "strategy_params": {"ik_method": "auto", "cartesian_step": 0.01},
        },
        "named_targets": {
            "home": {
                "aliases": ["home", "初始位"],
                "request": {
                    "target": {
                        "type": "joint",
                        "joint_positions": [
                            0.0, -2.094395, 1.570796,
                            -1.5707963, -1.5707963, 0.0,
                        ],
                    }
                },
            }
        },
        "objects": objects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def build_authored_registry(
    scene_path: str | Path,
    authored_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Bind authored mechanism metadata to the exact scene it was made for."""
    from robot_agent_control.contracts import scene_sha256

    scene = Path(scene_path).expanduser().resolve()
    authored = Path(authored_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    data = json.loads(authored.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("objects"), dict):
        raise ValueError("authored interaction registry is invalid")
    actual = scene_sha256(scene)
    expected = data.get("scene_fingerprint")
    if expected is not None and expected != actual:
        raise ValueError(
            f"authored interaction registry does not match scene: expected {expected}, got {actual}"
        )
    result = deepcopy(data)
    result["registry_version"] = 1
    result["coordinate_frame"] = "mujoco_world"
    result["scene"] = str(scene)
    result["scene_fingerprint"] = actual
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
