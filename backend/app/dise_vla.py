"""
DISE VLA Evaluation Environment

Evaluates Vision-Language-Action models via closed-loop MuJoCo simulation
with the SO-101 robot arm.

Pipeline:
  1. Build scene: SO-101 arm + table + objects
  2. Render observation: multi-view images + joint state
  3. Send to VLA model → get action (6 joint targets)
  4. Apply action → step MuJoCo simulation
  5. Check task success → record result
  6. Repeat until done or max steps

Robot: SO-101 (6-DOF: 5 arm joints + 1 gripper)
Action space: 6 joint position targets
Tasks: reach, pick, pick_place
"""

import io
import os
import base64
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image


# ─── Paths ────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent.parent / "models" / "so101"
URDF_PATH = MODEL_DIR / "so101.urdf"
ASSETS_DIR = MODEL_DIR / "assets"


# ─── Robot Config ─────────────────────────────────────────────────────

JOINT_NAMES = ["1", "2", "3", "4", "5", "6"]
NUM_JOINTS = 6

# Safe home pose: arm bent over table, gripper open
HOME_QPOS = [0.0, -0.8, 1.0, -0.4, 0.0, 0.5]


# ─── Colors & Objects ────────────────────────────────────────────────

COLORS = {
    "red":    [0.85, 0.15, 0.15, 1.0],
    "blue":   [0.15, 0.25, 0.85, 1.0],
    "green":  [0.15, 0.75, 0.25, 1.0],
    "yellow": [0.90, 0.80, 0.10, 1.0],
    "orange": [0.90, 0.45, 0.10, 1.0],
    "purple": [0.60, 0.15, 0.80, 1.0],
}

MANIP_OBJECTS = {
    "small_cube": {"geom": "box",      "size": [0.02, 0.02, 0.02], "mass": 0.05, "label": "cube"},
    "cylinder":   {"geom": "cylinder",  "size": [0.018, 0.03],      "mass": 0.04, "label": "cylinder"},
    "sphere":     {"geom": "sphere",    "size": [0.02],             "mass": 0.03, "label": "ball"},
}


# ─── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class ObjectSpec:
    """Object to place on the table."""
    obj_type: str               # key into MANIP_OBJECTS
    color: str                  # key into COLORS
    pos: list                   # [x, y, z]
    name: str = ""              # MuJoCo body name


@dataclass
class TaskSpec:
    """What the robot should do."""
    task_type: str              # "reach" | "pick" | "pick_place"
    instruction: str            # natural language for VLA
    target_obj: str = ""        # body name of target object
    place_target: list = None   # [x,y,z] for reach / pick_place target
    difficulty: str = "easy"


@dataclass
class VLAScenario:
    """A complete evaluation scenario."""
    scene_id: str
    objects: list               # list of ObjectSpec
    task: TaskSpec
    difficulty: str = "easy"


# ─── Asset Cache ──────────────────────────────────────────────────────

_assets: dict = {}
_urdf: str = ""


def _load_assets() -> dict:
    global _assets
    if _assets:
        return _assets
    for f in os.listdir(ASSETS_DIR):
        if f.endswith(".stl"):
            with open(ASSETS_DIR / f, "rb") as fh:
                _assets[f"assets/{f}"] = fh.read()
    return _assets


def _load_urdf() -> str:
    global _urdf
    if _urdf:
        return _urdf
    with open(URDF_PATH) as f:
        _urdf = f.read()
    return _urdf


# ─── Scene Builder ────────────────────────────────────────────────────

TABLE_POS = [0.0, -0.30, 0.0]
TABLE_HALF = [0.25, 0.20, 0.01]   # half-sizes
TABLE_SURFACE_Z = TABLE_POS[2] + TABLE_HALF[2]  # z of table top

_GEOM_MAP = {
    "box":      mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere":   mujoco.mjtGeom.mjGEOM_SPHERE,
}


def build_scene(scenario: VLAScenario):
    """
    Build MuJoCo model from scenario.

    Returns (model, data) with SO-101 + table + objects, arm at home pose.
    """
    assets = _load_assets()
    urdf = _load_urdf()

    spec = mujoco.MjSpec.from_string(urdf, assets)
    spec.option.gravity = [0, 0, -9.81]
    spec.option.timestep = 0.002

    # ── Actuators (position-controlled) ──
    for jname in JOINT_NAMES:
        act = spec.add_actuator()
        act.name = f"act_{jname}"
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm[0] = 100.0
        act.biasprm = [0.0, -100.0, -2.0, 0, 0, 0, 0, 0, 0, 0]
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    wb = spec.worldbody

    # ── Floor ──
    floor = wb.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [1, 1, 0.01]
    floor.rgba = [0.92, 0.90, 0.87, 1.0]

    # ── Lights ──
    l1 = wb.add_light()
    l1.pos = [0.1, -0.3, 0.8]
    l1.dir = [-0.05, 0, -1]
    l1.diffuse = [0.5, 0.5, 0.5]
    l1.specular = [0.1, 0.1, 0.1]
    l1.castshadow = True

    l2 = wb.add_light()
    l2.pos = [-0.3, 0.1, 0.6]
    l2.dir = [0.2, -0.3, -0.8]
    l2.diffuse = [0.35, 0.35, 0.38]

    # ── Table ──
    table = wb.add_body()
    table.name = "table"
    table.pos = TABLE_POS
    tg = table.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = TABLE_HALF
    tg.rgba = [0.82, 0.72, 0.58, 1.0]

    for lx, ly in [(0.22, 0.17), (-0.22, 0.17), (0.22, -0.17), (-0.22, -0.17)]:
        leg = table.add_geom()
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.size = [0.012, 0.15, 0]
        leg.pos = [lx, ly, -TABLE_HALF[2] - 0.15]
        leg.rgba = [0.6, 0.5, 0.4, 1.0]

    # ── Objects ──
    for i, obj in enumerate(scenario.objects):
        info = MANIP_OBJECTS[obj.obj_type]
        bname = obj.name or f"obj_{i}"

        body = wb.add_body()
        body.name = bname
        body.pos = obj.pos

        fj = body.add_freejoint()
        fj.name = f"{bname}_free"

        geom = body.add_geom()
        geom.type = _GEOM_MAP[info["geom"]]
        geom.rgba = COLORS[obj.color]
        geom.mass = info["mass"]
        geom.condim = 4
        geom.friction = [1.5, 0.05, 0.01]

        if info["geom"] == "box":
            geom.size = info["size"]
        elif info["geom"] == "cylinder":
            geom.size = [info["size"][0], info["size"][1], 0]
        elif info["geom"] == "sphere":
            geom.size = [info["size"][0], 0, 0]

    # ── Target marker (visual only — green disc on table) ──
    if scenario.task.place_target:
        mk = wb.add_body()
        mk.name = "target_marker"
        mk.pos = [scenario.task.place_target[0],
                   scenario.task.place_target[1],
                   TABLE_SURFACE_Z + 0.001]  # flat on table
        # Outer ring
        mg1 = mk.add_geom()
        mg1.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        mg1.size = [0.035, 0.001, 0]
        mg1.rgba = [0.1, 0.85, 0.2, 0.6]
        mg1.contype = 0
        mg1.conaffinity = 0
        # Center dot
        mg2 = mk.add_geom()
        mg2.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        mg2.size = [0.008, 0.002, 0]
        mg2.rgba = [0.1, 1.0, 0.1, 0.9]
        mg2.contype = 0
        mg2.conaffinity = 0

    # ── Compile & settle ──
    model = spec.compile()
    data = mujoco.MjData(model)

    data.ctrl[:NUM_JOINTS] = HOME_QPOS
    for _ in range(500):
        mujoco.mj_step(model, data)

    return model, data


# ─── Observation Rendering ────────────────────────────────────────────

# Camera views matching typical SmolVLA training setup
CAMERA_VIEWS = {
    "top":  {"lookat": [0.0, -0.25, 0.05], "dist": 0.65, "az": 180, "el": -89},
    "side": {"lookat": [0.0, -0.20, 0.15], "dist": 0.60, "az": 120, "el": -25},
}


def _render_view(model, data, view: dict, w=640, h=480) -> str:
    """Render one camera view → base64 JPEG string."""
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = view["lookat"]
    cam.distance = view["dist"]
    cam.azimuth = view["az"]
    cam.elevation = view["el"]
    renderer.update_scene(data, cam)
    px = renderer.render()
    renderer.close()

    buf = io.BytesIO()
    Image.fromarray(px).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def get_observation(model, data) -> dict:
    """
    Current observation for the VLA model.

    Returns::

        {
          "images": {"top": "data:image/...", "side": "data:image/..."},
          "state":  [j1, j2, j3, j4, j5, j6],   # joint positions
          "gripper_pos": [x, y, z],
        }
    """
    images = {name: _render_view(model, data, v) for name, v in CAMERA_VIEWS.items()}
    state = [float(data.qpos[i]) for i in range(NUM_JOINTS)]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    gpos = data.body(gid).xpos.tolist()
    return {"images": images, "state": state, "gripper_pos": gpos}


# ─── Action Execution ────────────────────────────────────────────────

SIM_STEPS_PER_ACTION = 50   # 50 × 0.002 s = 0.1 s real-time per action


def apply_action(model, data, action: list[float]):
    """
    Apply 6-DoF joint position targets and advance simulation.

    Args:
        action: [j1, j2, j3, j4, j5, j6] – target positions (radians / meters)
    """
    for i in range(NUM_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[i])
        lo, hi = model.jnt_range[jid]
        data.ctrl[i] = float(np.clip(action[i], lo, hi))

    for _ in range(SIM_STEPS_PER_ACTION):
        mujoco.mj_step(model, data)


# ─── Success Criteria ─────────────────────────────────────────────────

def check_success(model, data, task: TaskSpec) -> bool:
    """Check whether the task has been completed."""

    if task.task_type == "reach":
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        gpos = data.body(gid).xpos
        return float(np.linalg.norm(gpos - np.array(task.place_target))) < 0.03

    if task.task_type == "pick":
        oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task.target_obj)
        if oid < 0:
            return False
        return float(data.body(oid).xpos[2]) > TABLE_SURFACE_Z + 0.05

    if task.task_type == "pick_place":
        oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task.target_obj)
        if oid < 0:
            return False
        obj_pos = data.body(oid).xpos
        return float(np.linalg.norm(obj_pos - np.array(task.place_target))) < 0.04

    return False


# ─── Episode Runner ──────────────────────────────────────────────────

MAX_STEPS = 50
KEY_FRAME_INTERVAL = 10


def run_episode(
    scenario: VLAScenario,
    action_fn: Callable[[dict], list[float]],
    max_steps: int = MAX_STEPS,
) -> dict:
    """
    Closed-loop evaluation of one episode.

    Args:
        scenario:   defines task + objects
        action_fn:  observation → 6 joint targets  (calls the VLA model)
        max_steps:  budget

    Returns::

        {
          "success":     bool,
          "steps":       int,
          "trajectory":  [{step, state, gripper_pos, action}, ...],
          "key_frames":  [base64_img, ...],
          "scene_id":    str,
          "task_type":   str,
          "instruction": str,
        }
    """
    model, data = build_scene(scenario)

    trajectory: list[dict] = []
    key_frames: list[str] = []
    success = False

    # initial frame
    obs = get_observation(model, data)
    key_frames.append(obs["images"]["side"])

    for step in range(max_steps):
        obs = get_observation(model, data)
        action = action_fn(obs)

        trajectory.append({
            "step": step,
            "state": obs["state"],
            "gripper_pos": obs["gripper_pos"],
            "action": list(action),
        })

        apply_action(model, data, action)

        if check_success(model, data, scenario.task):
            success = True
            final = get_observation(model, data)
            key_frames.append(final["images"]["side"])
            break

        if step % KEY_FRAME_INTERVAL == 0 and step > 0:
            key_frames.append(get_observation(model, data)["images"]["side"])

    if not success:
        key_frames.append(get_observation(model, data)["images"]["side"])

    return {
        "success": success,
        "steps": len(trajectory),
        "trajectory": trajectory,
        "key_frames": key_frames,
        "scene_id": scenario.scene_id,
        "task_type": scenario.task.task_type,
        "instruction": scenario.task.instruction,
    }


# ─── Scenario Generation ─────────────────────────────────────────────

def generate_10_scenarios() -> list[VLAScenario]:
    """Ten curated evaluation scenarios: reach / pick / pick-place."""

    z = TABLE_SURFACE_Z   # shorthand for table surface height

    return [
        # ── Reach (easy) ──
        VLAScenario("vla_reach_1", [
            ObjectSpec("small_cube", "red", [0.05, -0.25, z + 0.02], "obj_0"),
        ], TaskSpec("reach", "Move to the red cube", "obj_0",
                    [0.05, -0.25, z + 0.08], "easy"), "easy"),

        VLAScenario("vla_reach_2", [
            ObjectSpec("sphere", "blue", [-0.05, -0.30, z + 0.02], "obj_0"),
        ], TaskSpec("reach", "Move above the blue ball", "obj_0",
                    [-0.05, -0.30, z + 0.08], "easy"), "easy"),

        # ── Pick (medium) ──
        VLAScenario("vla_pick_1", [
            ObjectSpec("small_cube", "red", [0.05, -0.25, z + 0.02], "obj_0"),
        ], TaskSpec("pick", "Pick up the red cube", "obj_0",
                    difficulty="medium"), "medium"),

        VLAScenario("vla_pick_2", [
            ObjectSpec("cylinder", "green", [0.0, -0.28, z + 0.03], "obj_0"),
        ], TaskSpec("pick", "Pick up the green cylinder", "obj_0",
                    difficulty="medium"), "medium"),

        VLAScenario("vla_pick_3", [
            ObjectSpec("small_cube", "yellow", [-0.10, -0.22, z + 0.02], "obj_0"),
            ObjectSpec("sphere", "blue", [0.10, -0.35, z + 0.02], "obj_1"),
        ], TaskSpec("pick", "Pick up the yellow cube", "obj_0",
                    difficulty="medium"), "medium"),

        # ── Pick with distractors (hard) ──
        VLAScenario("vla_pick_4", [
            ObjectSpec("small_cube", "red",   [0.06, -0.25, z + 0.02], "obj_0"),
            ObjectSpec("small_cube", "blue",  [-0.06, -0.28, z + 0.02], "obj_1"),
            ObjectSpec("cylinder",   "green", [0.0, -0.35, z + 0.03], "obj_2"),
        ], TaskSpec("pick", "Pick up the red cube", "obj_0",
                    difficulty="hard"), "hard"),

        # ── Pick & Place (medium) ──
        VLAScenario("vla_pp_1", [
            ObjectSpec("small_cube", "red", [0.08, -0.22, z + 0.02], "obj_0"),
        ], TaskSpec("pick_place", "Pick the red cube and place it on the green mark",
                    "obj_0", [-0.08, -0.32, z + 0.02], "medium"), "medium"),

        VLAScenario("vla_pp_2", [
            ObjectSpec("cylinder", "blue", [-0.05, -0.25, z + 0.03], "obj_0"),
        ], TaskSpec("pick_place", "Move the blue cylinder to the target",
                    "obj_0", [0.10, -0.35, z + 0.03], "medium"), "medium"),

        # ── Pick & Place with distractors (hard) ──
        VLAScenario("vla_pp_3", [
            ObjectSpec("small_cube", "orange", [0.06, -0.22, z + 0.02], "obj_0"),
            ObjectSpec("sphere", "purple", [-0.06, -0.30, z + 0.02], "obj_1"),
        ], TaskSpec("pick_place",
                    "Pick the orange cube and place it near the purple ball",
                    "obj_0", [-0.06, -0.30, z + 0.06], "hard"), "hard"),

        VLAScenario("vla_pp_4", [
            ObjectSpec("small_cube", "red",   [0.10, -0.25, z + 0.02], "obj_0"),
            ObjectSpec("small_cube", "blue",  [-0.08, -0.28, z + 0.02], "obj_1"),
            ObjectSpec("cylinder",   "green", [0.0, -0.35, z + 0.03], "obj_2"),
        ], TaskSpec("pick_place",
                    "Pick the red cube and place it on the green mark",
                    "obj_0", [0.0, -0.35, z + 0.06], "hard"), "hard"),
    ]


# ─── Preview Rendering (for datasets page) ───────────────────────────

def render_scenario(scenario: VLAScenario, width: int = 480) -> list[str]:
    """Render initial scene → [top_view, side_view] as base64 images."""
    model, data = build_scene(scenario)
    h = int(width * 0.75)
    return [
        _render_view(model, data, CAMERA_VIEWS["top"], w=width, h=h),
        _render_view(model, data, CAMERA_VIEWS["side"], w=width, h=h),
    ]


# ─── Review HTML ──────────────────────────────────────────────────────

_TASK_BADGE = {
    "reach":      ("bg-sky-100 text-sky-700",      "reach"),
    "pick":       ("bg-amber-100 text-amber-700",   "pick"),
    "pick_place": ("bg-violet-100 text-violet-700",  "pick & place"),
}


def generate_review_html(scenarios: list[VLAScenario] | None = None) -> str:
    """Generate HTML review page for all scenarios."""
    if scenarios is None:
        scenarios = generate_10_scenarios()

    cards = []
    for sc in scenarios:
        imgs = render_scenario(sc)
        badge_style, badge_text = _TASK_BADGE.get(sc.task.task_type, ("", sc.task.task_type))
        cards.append(f"""
<div style="background:white;border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.1)">
  <h3 style="margin:0 0 6px;color:#333">{sc.scene_id}</h3>
  <div style="font-size:13px;color:#888;margin-bottom:10px">
    <span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;{badge_style}">{badge_text}</span>
    <span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;background:#f3f4f6;color:#555">{sc.difficulty}</span>
    &nbsp;💬 "{sc.task.instruction}"
  </div>
  <div style="display:flex;gap:12px">
    <img src="{imgs[0]}" style="border-radius:8px;border:1px solid #e0e0e0;max-height:280px" alt="top">
    <img src="{imgs[1]}" style="border-radius:8px;border:1px solid #e0e0e0;max-height:280px" alt="side">
  </div>
</div>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VLA Scenarios</title></head>
<body style="font-family:-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f5f5f5">
<h1>🦾 VLA Evaluation Scenarios</h1>
<p style="color:#666">SO-101 arm · tabletop manipulation · SmolVLA evaluation</p>
{"".join(cards)}
</body></html>"""
