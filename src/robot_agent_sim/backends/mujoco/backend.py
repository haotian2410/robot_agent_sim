from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from ...assets.registry import AssetRecord
from ...scene.registry import SceneObject, SceneRegistry, UploadedSceneRegistry
from .panda import panda_xml
from .renderer import MujocoRenderer
from .ur5e import ur5e_xml

class MujocoSceneBackend:
    def __init__(self): self.renderer = MujocoRenderer()
    def load_uploaded(self, path: Path, robot: str) -> SceneRegistry:
        root = ET.parse(path).getroot(); self._validate(root); sidecar = path.with_name(f"{path.stem}.scene_registry.json")
        if sidecar.is_file():
            declared = UploadedSceneRegistry.model_validate_json(sidecar.read_text(encoding="utf-8"))
            objects = [SceneObject(object_id=x.object_id, body_name=x.body_name, role=x.role, semantic_name=x.semantic_name or x.body_name, model_id=x.model_id, model_name=x.model_name, position=x.position, dimensions_m=x.dimensions_m, source="uploaded", expected_visible=x.expected_visible) for x in declared.objects]
            return SceneRegistry(scene_id=declared.scene_id or path.stem, robot=declared.robot or robot, objects=objects)
        names = self._discover_objects(root)
        objects = [SceneObject(object_id=f"scene_object_{i:03d}", body_name=name, role="target", semantic_name=name, source="uploaded") for i, name in enumerate(names, 1)]
        if not objects: raise ValueError("no discoverable task objects; add a .scene_registry.json sidecar")
        return SceneRegistry(scene_id=path.stem, robot=robot, objects=objects)
    def compose(self, registry: SceneRegistry, assets: dict[str, AssetRecord], output_dir: Path) -> Path:
        source = panda_xml() if registry.robot == "panda" else ur5e_xml(); root = ET.parse(source).getroot(); self._validate(root); self._paths(root, source, registry.robot); self._normalize(root, registry.robot)
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        world = root.find("worldbody")
        if world is None: raise ValueError("base scene has no worldbody")
        by_model = {record.model_id: record for record in assets.values()}
        for record in by_model.values(): self._add_asset(asset, record)
        for item in registry.objects: self._add_instance(world, item, by_model[item.model_id])
        output_dir.mkdir(parents=True, exist_ok=True); target = output_dir / "scene.xml"; ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True); return target
    @staticmethod
    def _validate(root):
        if root.tag != "mujoco" or root.find(".//include") is not None or root.find(".//plugin") is not None: raise ValueError("scene XML must be standalone without include/plugin")
    @staticmethod
    def _paths(root, source, robot):
        compiler = root.find("compiler")
        if compiler is not None: compiler.set("meshdir", str((source.parent / "assets").resolve()) if robot == "panda" else str((source.parent / "../assets/ur5e").resolve()))
    @staticmethod
    def _normalize(root, robot):
        world = root.find("worldbody"); assert world is not None
        if robot == "ur5e":
            for child in list(world):
                if child.get("pos"):
                    # The source UR5e scene places the tabletop upper surface
                    # at z=0.6 and its center at x=0.525.  Recenter that
                    # surface at z=0 so SceneObject.position denotes a
                    # contact point on the tabletop.
                    values = [float(v) for v in child.get("pos").split()]
                    child.set("pos", " ".join(str(v) for v in (values[0]-0.525, values[1], values[2]-0.6)))
            return
        link0 = world.find("body[@name='link0']")
        if link0 is not None: link0.set("pos", "-0.525 0 -0.1")
        ET.SubElement(world, "geom", name="ground", type="plane", size="3 3 0.1", pos="0 0 -0.6", rgba="0.82 0.84 0.88 1")
        # The tabletop upper surface is z=0 in the public tabletop-centered
        # frame; task-object body positions are the contact locations.
        table = ET.SubElement(world, "body", name="work_table", pos="0 0 -0.6"); ET.SubElement(table, "geom", name="work_table_top", type="box", pos="0 0 0.585", size="0.375 0.75 0.015", rgba="0.48 0.28 0.12 1")
        ET.SubElement(world, "camera", name="scene_camera", pos="0 0 1.5", quat="1 0 0 0", fovy="45")
    @staticmethod
    def _add_asset(asset, record):
        if record.source != "mesh": return
        ET.SubElement(asset, "mesh", name=f"visual_{record.model_id}", file=str(record.visual_obj.resolve()))
        if record.texture and record.texture.is_file():
            ET.SubElement(asset, "texture", name=f"texture_{record.model_id}", type="2d", file=str(record.texture.resolve())); ET.SubElement(asset, "material", name=f"material_{record.model_id}", texture=f"texture_{record.model_id}")
        for i, path in enumerate(record.collision_stls): ET.SubElement(asset, "mesh", name=f"collision_{record.model_id}_{i}", file=str(path.resolve()))
    @staticmethod
    def _add_instance(world, item, record):
        body = ET.SubElement(world, "body", name=item.body_name, pos=" ".join(str(v) for v in item.position)); dimensions = item.dimensions_m or record.dimensions_m or (0.06, 0.06, 0.06); z = dimensions[2] / 2
        # Generated ordinary objects are movable payloads.  A free joint is
        # required by the control gripper's persistent attachment model;
        # containers and buttons remain static scene geometry.
        if record.model_name not in {"open_box", "button_basic"}:
            ET.SubElement(body, "freejoint", name=f"{item.object_id}_free")
        if record.source == "mesh":
            minimum = record.bbox_min_m or (0.0, 0.0, 0.0)
            maximum = record.bbox_max_m or dimensions
            # SceneObject.position is the mesh footprint centre at the
            # tabletop.  Translate the imported mesh so its XY bbox is
            # centred on the body and its lowest vertex touches z=0.
            local_pos = (
                -(minimum[0] + maximum[0]) / 2,
                -(minimum[1] + maximum[1]) / 2,
                -minimum[2],
            )
            attrs = {"name": f"{item.object_id}_visual", "type": "mesh", "mesh": f"visual_{record.model_id}", "pos": " ".join(str(value) for value in local_pos), "contype": "0", "conaffinity": "0", "group": "2"}
            if record.texture and record.texture.is_file(): attrs["material"] = f"material_{record.model_id}"
            ET.SubElement(body, "geom", attrs)
            for i, _ in enumerate(record.collision_stls): ET.SubElement(body, "geom", name=f"{item.object_id}_collision_{i}", type="mesh", mesh=f"collision_{record.model_id}_{i}", pos=" ".join(str(value) for value in local_pos), rgba="0 0 0 0", contype="1", conaffinity="1", group="3")
        elif record.model_name == "cube_basic": ET.SubElement(body, "geom", name=f"{item.object_id}_geom", type="box", size=" ".join(str(v/2) for v in dimensions), pos=f"0 0 {z}", rgba=_rgba(item.semantic_name))
        elif record.model_name == "open_box":
            x, y, h = dimensions; t = 0.008
            for name, pos, size in (("floor", (0,0,t/2), (x/2,y/2,t/2)), ("left", (-x/2+t/2,0,h/2), (t/2,y/2,h/2)), ("right", (x/2-t/2,0,h/2), (t/2,y/2,h/2)), ("front", (0,y/2-t/2,h/2), (x/2,t/2,h/2)), ("back", (0,-y/2+t/2,h/2), (x/2,t/2,h/2))): ET.SubElement(body, "geom", name=f"{item.object_id}_{name}", type="box", pos=" ".join(str(v) for v in pos), size=" ".join(str(v) for v in size), rgba=_rgba(item.semantic_name))
        else: ET.SubElement(body, "geom", name=f"{item.object_id}_button", type="cylinder", size="0.025 0.012", pos="0 0 0.032", rgba="0.8 0.1 0.1 1")
    @staticmethod
    def _discover_objects(root):
        """Discover task objects without treating robot internals as objects.

        The XMLs used by different robot backends do not share a fixed list of
        top-level bodies, so filtering only ``worldbody.findall('body')`` is
        both incomplete and prone to returning robot links.  We exclude whole
        infrastructure subtrees and keep the remaining top-level task bodies;
        nested task bodies are included only when their parent is itself a task
        body (e.g. a button key).
        """
        world = root.find("worldbody")
        if world is None:
            return []
        infrastructure = {"base", "link0", "robot", "robot_pedestal", "work_table",
                          "table", "overhead_camera_mount", "wrist_camera_mount",
                          "camera", "scene_camera", "floor", "ground", "fixture",
                          "support", "lighting"}
        robot_tokens = ("link", "joint", "gripper", "robotiq", "camera", "mount",
                        "pedestal", "shoulder", "upper_arm", "forearm", "wrist")
        names: list[str] = []

        def walk(body, excluded=False, task_parent=False):
            name = body.get("name", "")
            low = name.casefold()
            subtree_excluded = (
                excluded or low in infrastructure
                or any(token in low for token in robot_tokens)
                or any(token in low for token in ("table", "floor", "ground", "light"))
            )
            # A direct body with a geom is a useful task object.  A nested body
            # under an infrastructure subtree is never task data.
            if not subtree_excluded and not task_parent and body.find("geom") is not None:
                names.append(name)
                task_parent = True
            for child in body.findall("body"):
                walk(child, subtree_excluded, task_parent)

        for body in world.findall("body"):
            walk(body)
        return names

def _rgba(name):
    low = name.casefold(); return "0.8 0.08 0.08 1" if "red" in low else "0.08 0.25 0.8 1" if "blue" in low else "0.9 0.72 0.08 1" if "yellow" in low else "0.65 0.65 0.65 1"
