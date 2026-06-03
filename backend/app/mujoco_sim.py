"""
MuJoCo Physics Simulation Engine

Provides:
- Dynamic scene construction (add objects to base MJCF)
- Multi-view rendering
- Trajectory simulation with collision detection
- Physics-grounded verification of spatial reasoning answers
"""
import io
import base64
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

SCENES_DIR = Path(__file__).parent.parent / "scenes"

# ─── Object primitives ──────────────────────────────────────────────

OBJECT_PRESETS = {
    "cup": {"type": "cylinder", "size": "0.035 0.05", "mass": "0.15"},
    "bowl": {"type": "ellipsoid", "size": "0.06 0.06 0.03", "mass": "0.2"},
    "box": {"type": "box", "size": "0.04 0.04 0.04", "mass": "0.3"},
    "bottle": {"type": "cylinder", "size": "0.03 0.1", "mass": "0.25"},
    "can": {"type": "cylinder", "size": "0.033 0.06", "mass": "0.35"},
    "ball": {"type": "sphere", "size": "0.04", "mass": "0.1"},
    "plate": {"type": "cylinder", "size": "0.1 0.01", "mass": "0.3"},
    "book": {"type": "box", "size": "0.1 0.07 0.015", "mass": "0.4"},
}

MATERIALS = ["obj_red", "obj_green", "obj_blue", "obj_yellow"]


@dataclass
class SceneObject:
    name: str
    preset: str  # key in OBJECT_PRESETS
    pos: list[float]  # [x, y, z]
    material: Optional[str] = None
    size: Optional[str] = None  # override preset size


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    t: float  # time in seconds


@dataclass
class SimResult:
    success: bool
    collisions: list[dict]
    trajectory_actual: list[dict]
    physics_plausible: bool
    reason: str
    frames: list[str] = field(default_factory=list)  # base64 PNGs


# ─── Scene Builder ───────────────────────────────────────────────────

def build_scene_xml(base_scene: str, objects: list[SceneObject]) -> str:
    """Add objects to a base MJCF scene XML and return the modified XML string."""
    xml_path = SCENES_DIR / f"{base_scene}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Base scene not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    for i, obj in enumerate(objects):
        preset = OBJECT_PRESETS.get(obj.preset)
        if not preset:
            raise ValueError(f"Unknown object preset: {obj.preset}")

        body = ET.SubElement(worldbody, "body")
        body.set("name", obj.name)
        body.set("pos", f"{obj.pos[0]} {obj.pos[1]} {obj.pos[2]}")

        # Free joint so object can fall / be pushed
        ET.SubElement(body, "freejoint", name=f"{obj.name}_jnt")

        geom_attrs = {
            "name": f"{obj.name}_geom",
            "type": preset["type"],
            "size": obj.size or preset["size"],
            "mass": preset["mass"],
            "material": obj.material or MATERIALS[i % len(MATERIALS)],
        }
        ET.SubElement(body, "geom", **geom_attrs)

    return ET.tostring(root, encoding="unicode")


# ─── Rendering ───────────────────────────────────────────────────────

def render_scene(model: mujoco.MjModel, data: mujoco.MjData,
                 camera: str = "front", width: int = 640, height: int = 480) -> str:
    """Render a single frame and return as base64 PNG data URI."""
    from PIL import Image

    # Use low-level rendering API (works on macOS CGL + Linux)
    gl_ctx = mujoco.GLContext(width, height)
    gl_ctx.make_current()

    scene = mujoco.MjvScene(model, maxgeom=1000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # Set camera to named camera in the model
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id >= 0:
        cam.fixedcamid = cam_id
    else:
        # Fallback: free camera looking at origin
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, 0.4]
        cam.distance = 2.0
        cam.elevation = -30
        cam.azimuth = 90

    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(viewport, scene, ctx)

    # Read pixels
    img_arr = np.empty((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(img_arr, None, viewport, ctx)
    img_arr = np.flipud(img_arr)  # OpenGL is bottom-up

    ctx.free()
    gl_ctx.free()

    pil_img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def render_multi_view(model: mujoco.MjModel, data: mujoco.MjData,
                      cameras: list[str] = None, width: int = 480, height: int = 360) -> list[str]:
    """Render from multiple cameras, return list of base64 PNGs."""
    if cameras is None:
        cameras = ["front", "top", "side"]
    return [render_scene(model, data, cam, width, height) for cam in cameras]


# ─── Trajectory Simulation ──────────────────────────────────────────

def simulate_trajectory(
    scene_xml: str,
    target_object: str,
    waypoints: list[Waypoint],
    record_interval: float = 0.1,
    record_cameras: list[str] = None,
) -> SimResult:
    """
    Simulate moving an object along waypoints and detect collisions.

    Instead of actuators, we directly set the object's position along
    the trajectory (kinematic mode) while letting other objects respond
    physically. This is simpler and sufficient for verification.
    """
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)

    # Find the target body and its joint
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_object)
    if target_body_id < 0:
        return SimResult(
            success=False, collisions=[], trajectory_actual=[],
            physics_plausible=False, reason=f"Object '{target_object}' not found in scene"
        )

    # Find the freejoint for this body
    jnt_name = f"{target_object}_jnt"
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    if jnt_id < 0:
        return SimResult(
            success=False, collisions=[], trajectory_actual=[],
            physics_plausible=False, reason=f"Joint '{jnt_name}' not found"
        )

    qpos_adr = model.jnt_qposadr[jnt_id]

    # Let scene settle first
    mujoco.mj_resetData(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)

    collisions: list[dict] = []
    trajectory_actual: list[dict] = []
    frames: list[str] = []
    last_record_time = -record_interval

    if not waypoints:
        return SimResult(
            success=False, collisions=[], trajectory_actual=[],
            physics_plausible=False, reason="No waypoints provided"
        )

    total_time = waypoints[-1].t
    dt = model.opt.timestep

    # Interpolate position at time t
    def interp_pos(t: float) -> np.ndarray:
        if t <= waypoints[0].t:
            wp = waypoints[0]
            return np.array([wp.x, wp.y, wp.z])
        if t >= waypoints[-1].t:
            wp = waypoints[-1]
            return np.array([wp.x, wp.y, wp.z])
        for i in range(len(waypoints) - 1):
            if waypoints[i].t <= t <= waypoints[i + 1].t:
                alpha = (t - waypoints[i].t) / (waypoints[i + 1].t - waypoints[i].t)
                p0 = np.array([waypoints[i].x, waypoints[i].y, waypoints[i].z])
                p1 = np.array([waypoints[i + 1].x, waypoints[i + 1].y, waypoints[i + 1].z])
                return p0 + alpha * (p1 - p0)
        wp = waypoints[-1]
        return np.array([wp.x, wp.y, wp.z])

    # Run simulation
    sim_time = 0.0
    while sim_time <= total_time + 0.5:  # extra 0.5s for settling
        # Set target object position (kinematic override)
        if sim_time <= total_time:
            pos = interp_pos(sim_time)
            data.qpos[qpos_adr:qpos_adr + 3] = pos
            data.qvel[model.jnt_dofadr[jnt_id]:model.jnt_dofadr[jnt_id] + 3] = 0

        mujoco.mj_step(model, data)
        sim_time += dt

        # Check contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            # Skip floor/table self-contacts
            if geom1 and geom2 and f"{target_object}_geom" in (geom1, geom2):
                other = geom2 if geom1 == f"{target_object}_geom" else geom1
                if other not in ("floor", "tabletop"):
                    collisions.append({
                        "time": round(sim_time, 4),
                        "object": target_object,
                        "other": other.replace("_geom", ""),
                        "position": [round(float(x), 4) for x in contact.pos],
                        "force": round(float(np.linalg.norm(contact.frame[:3])), 4),
                    })

        # Record snapshots
        if sim_time - last_record_time >= record_interval:
            actual_pos = data.qpos[qpos_adr:qpos_adr + 3].copy()
            trajectory_actual.append({
                "t": round(sim_time, 3),
                "x": round(float(actual_pos[0]), 4),
                "y": round(float(actual_pos[1]), 4),
                "z": round(float(actual_pos[2]), 4),
            })

            # Record frames (limit to avoid huge payloads)
            if record_cameras and len(frames) < 30:
                try:
                    frame = render_scene(model, data, record_cameras[0], 320, 240)
                    frames.append(frame)
                except Exception:
                    pass

            last_record_time = sim_time

    # Deduplicate collisions (same pair within 0.05s)
    unique_collisions = []
    for c in collisions:
        is_dup = any(
            uc["other"] == c["other"] and abs(uc["time"] - c["time"]) < 0.05
            for uc in unique_collisions
        )
        if not is_dup:
            unique_collisions.append(c)

    # Determine physics plausibility
    has_collision = len(unique_collisions) > 0
    reason_parts = []
    if has_collision:
        objects_hit = set(c["other"] for c in unique_collisions)
        reason_parts.append(f"Collisions detected with: {', '.join(objects_hit)}")
    else:
        reason_parts.append("No collisions — path is clear")

    return SimResult(
        success=True,
        collisions=unique_collisions,
        trajectory_actual=trajectory_actual,
        physics_plausible=True,
        reason="; ".join(reason_parts),
        frames=frames,
    )


# ─── Convenience: Quick scene from description ──────────────────────

def create_quick_scene(objects: list[dict], base_scene: str = "tabletop") -> tuple[str, list[SceneObject]]:
    """
    Create a scene from a simple list of dicts like:
      [{"name": "cup1", "type": "cup", "pos": [0.1, 0, 0.4]}]
    Returns (xml_string, scene_objects)
    """
    scene_objs = [
        SceneObject(
            name=o["name"],
            preset=o["type"],
            pos=o["pos"],
            material=o.get("material"),
            size=o.get("size"),
        )
        for o in objects
    ]
    xml = build_scene_xml(base_scene, scene_objs)
    return xml, scene_objs
