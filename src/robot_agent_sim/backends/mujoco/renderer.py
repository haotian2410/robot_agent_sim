from __future__ import annotations
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from ...grounding.segmentation import InstanceObservation, SceneObservation

class MujocoRenderer:
    def render(self, xml_path: Path, registry, output_dir: Path, width: int = 640, height: int = 480) -> SceneObservation:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = mujoco.MjModel.from_xml_path(str(xml_path)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = "scene_camera" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_camera") >= 0 else -1
        renderer.update_scene(data, camera=camera); rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering(); renderer.update_scene(data, camera=camera); segmentation = renderer.render().copy(); renderer.close()
        rgb_path = output_dir / "rgb.png"; segmentation_path = output_dir / "segmentation.npy"; visual_path = output_dir / "segmentation.png"
        imageio.imwrite(rgb_path, rgb); np.save(segmentation_path, segmentation)
        visual = np.zeros((height, width, 3), dtype=np.uint8); instances = []
        for index, item in enumerate(registry.objects, start=1):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.body_name)
            if body_id < 0: raise ValueError(f"registered body not found: {item.body_name}")
            body_ids = _descendants(model, body_id)
            geom_ids = [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in body_ids]
            mask = np.isin(segmentation[:, :, 0], geom_ids); ys, xs = np.nonzero(mask); bbox = None
            if len(xs):
                bbox = (round(float(ys.min()) * 1000 / height), round(float(xs.min()) * 1000 / width), round(float(ys.max()+1) * 1000 / height), round(float(xs.max()+1) * 1000 / width))
                visual[mask] = _color(index)
            world_position = tuple(float(value) for value in data.xpos[body_id])
            instances.append(InstanceObservation(object_id=item.object_id, body_name=item.body_name, bbox=bbox, visible_pixel_count=int(mask.sum()), world_position=world_position))
        imageio.imwrite(visual_path, visual)
        # Keep both names: ``instances.json`` is the public artifact name,
        # while ``instance_index.json`` remains a backwards-compatible alias
        # for callers that used the initial prototype.
        instance_payload = json.dumps(
            [item.model_dump(mode="json") for item in instances],
            ensure_ascii=False,
            indent=2,
        )
        index_path = output_dir / "instances.json"
        index_path.write_text(instance_payload, encoding="utf-8")
        (output_dir / "instance_index.json").write_text(instance_payload, encoding="utf-8")
        return SceneObservation(scene_id=registry.scene_id, camera_id=str(camera), image_width_px=width, image_height_px=height, rgb_path=rgb_path, segmentation_path=segmentation_path, segmentation_visualization_path=visual_path, instance_index_path=index_path, instances=instances)

def _descendants(model, root_body):
    result = {root_body}
    for body_id in range(1, model.nbody):
        current = body_id
        while current > 0:
            if current == root_body: result.add(body_id); break
            current = int(model.body_parentid[current])
    return result

def _color(index): return np.array(((53*index)%205+50, (97*index)%205+50, (149*index)%205+50), dtype=np.uint8)
