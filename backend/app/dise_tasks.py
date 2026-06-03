"""
DISE Quadrant Tasks — Intrinsic-Dynamic (I-D)

Task: Object Rotation Reasoning
  "If this object is rotated 90° clockwise around the Y-axis, which face will be on top?"

MuJoCo Setup: Single cube with 6 distinctly colored/labeled faces
Verification: Simulate rotation → read final quaternion → determine top face
"""
import io
import base64
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import mujoco

# ─── Face definitions ────────────────────────────────────────────────
# Each face has a label, color, and the outward normal in the object's local frame

FACE_NORMALS = {
    "top":    np.array([0, 0,  1]),   # +Z
    "bottom": np.array([0, 0, -1]),   # -Z
    "front":  np.array([0, -1, 0]),   # -Y  (facing camera)
    "back":   np.array([0,  1, 0]),   # +Y
    "right":  np.array([1,  0, 0]),   # +X
    "left":   np.array([-1, 0, 0]),   # -X
}

FACE_COLORS = {
    "top":    "0.9 0.2 0.2 1",   # Red
    "bottom": "0.2 0.8 0.3 1",   # Green
    "front":  "0.2 0.3 0.9 1",   # Blue
    "back":   "0.9 0.85 0.2 1",  # Yellow
    "right":  "0.9 0.5 0.1 1",   # Orange
    "left":   "0.7 0.2 0.9 1",   # Purple
}

FACE_LABELS = {
    "top": "1 (Red)",
    "bottom": "2 (Green)",
    "front": "3 (Blue)",
    "back": "4 (Yellow)",
    "right": "5 (Orange)",
    "left": "6 (Purple)",
}

# ─── Rotation axis definitions ──────────────────────────────────────

ROTATION_AXES = {
    "X": np.array([1, 0, 0]),
    "Y": np.array([0, 1, 0]),
    "Z": np.array([0, 0, 1]),
}


@dataclass
class RotationTask:
    """A single rotation reasoning task."""
    axis: str           # "X", "Y", or "Z"
    angle_deg: float    # e.g. 90, -90, 180
    direction: str      # "clockwise" or "counterclockwise" (when viewed from +axis)
    initial_top: str    # face label on top before rotation
    question: str       # natural language question
    ground_truth: str   # correct answer face label


@dataclass
class RotationResult:
    """Result of simulating a rotation task."""
    task: RotationTask
    predicted_top: str           # face on top after simulation
    correct: bool
    before_images: list[str]     # base64 renders before rotation
    after_images: list[str]      # base64 renders after rotation
    quaternion_before: list[float]
    quaternion_after: list[float]
    all_face_orientations: dict  # which direction each face points after rotation


# ─── MJCF scene builder ─────────────────────────────────────────────

def _build_labeled_cube_xml(cube_size: float = 0.08, table_height: float = 0.35) -> str:
    """
    Build MJCF XML with a cube whose 6 faces are individually colored thin geoms.
    The cube sits on a table and has a freejoint so it can rotate freely.
    """
    half = cube_size / 2
    face_thickness = 0.002  # thin colored panels on each face
    cube_z = table_height + 0.02 + half  # table top + margin

    xml = f"""<mujoco model="rotation_task">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <visual>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.7 0.7 0.7" width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.2"/>
    <material name="table_mat" rgba="0.55 0.35 0.2 1"/>
    <material name="cube_core" rgba="0.3 0.3 0.3 1"/>
    <material name="face_top" rgba="{FACE_COLORS['top']}"/>
    <material name="face_bottom" rgba="{FACE_COLORS['bottom']}"/>
    <material name="face_front" rgba="{FACE_COLORS['front']}"/>
    <material name="face_back" rgba="{FACE_COLORS['back']}"/>
    <material name="face_right" rgba="{FACE_COLORS['right']}"/>
    <material name="face_left" rgba="{FACE_COLORS['left']}"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid_mat"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1 -1 2" dir="-0.5 0.5 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Table -->
    <body name="table" pos="0 0 {table_height}">
      <geom name="tabletop" type="box" size="0.5 0.35 0.02" material="table_mat" mass="10"/>
      <geom name="leg1" type="cylinder" fromto="-0.45 -0.3 -{table_height}  -0.45 -0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg2" type="cylinder" fromto="0.45 -0.3 -{table_height}   0.45 -0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg3" type="cylinder" fromto="-0.45  0.3 -{table_height}  -0.45  0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg4" type="cylinder" fromto="0.45  0.3 -{table_height}   0.45  0.3 0" size="0.025" material="table_mat"/>
    </body>

    <!-- Labeled Cube -->
    <body name="cube" pos="0 0 {cube_z}">
      <freejoint name="cube_jnt"/>
      <!-- Core cube -->
      <geom name="cube_core" type="box" size="{half} {half} {half}" material="cube_core" mass="0.5"/>
      <!-- Face panels (thin boxes offset to each face) -->
      <geom name="face_top_geom"    type="box" size="{half-0.002} {half-0.002} {face_thickness}" pos="0 0 {half}" material="face_top" mass="0.01"/>
      <geom name="face_bottom_geom" type="box" size="{half-0.002} {half-0.002} {face_thickness}" pos="0 0 -{half}" material="face_bottom" mass="0.01"/>
      <geom name="face_front_geom"  type="box" size="{half-0.002} {face_thickness} {half-0.002}" pos="0 -{half} 0" material="face_front" mass="0.01"/>
      <geom name="face_back_geom"   type="box" size="{half-0.002} {face_thickness} {half-0.002}" pos="0 {half} 0" material="face_back" mass="0.01"/>
      <geom name="face_right_geom"  type="box" size="{face_thickness} {half-0.002} {half-0.002}" pos="{half} 0 0" material="face_right" mass="0.01"/>
      <geom name="face_left_geom"   type="box" size="{face_thickness} {half-0.002} {half-0.002}" pos="-{half} 0 0" material="face_left" mass="0.01"/>
    </body>

    <!-- Cameras -->
    <camera name="front" pos="0 -0.8 0.7" xyaxes="1 0 0 0 0.6 0.8"/>
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0"/>
    <camera name="side" pos="0.8 0 0.65" xyaxes="0 1 0 -0.5 0 0.85"/>
    <camera name="angle" pos="0.6 -0.6 0.8" xyaxes="0.7 0.7 0 -0.3 0.3 0.9"/>
  </worldbody>
</mujoco>"""
    return xml


# ─── Quaternion helpers ──────────────────────────────────────────────

def _axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Convert axis-angle to quaternion [w, x, y, z]."""
    axis = axis / np.linalg.norm(axis)
    w = math.cos(angle_rad / 2)
    xyz = axis * math.sin(angle_rad / 2)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q [w, x, y, z]."""
    # Using rotation matrix from quaternion for accuracy
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    return R @ v


def _determine_top_face(quat: np.ndarray) -> tuple[str, dict]:
    """
    Given the cube's quaternion, determine which face points upward (+Z in world).
    Returns (face_name, {face: world_direction}).
    """
    world_up = np.array([0, 0, 1])
    face_orientations = {}

    best_face = None
    best_dot = -2.0

    for face_name, local_normal in FACE_NORMALS.items():
        world_normal = _quat_rotate_vector(quat, local_normal)
        face_orientations[face_name] = {
            "world_normal": [round(float(x), 3) for x in world_normal],
            "up_alignment": round(float(np.dot(world_normal, world_up)), 3),
        }
        dot = np.dot(world_normal, world_up)
        if dot > best_dot:
            best_dot = dot
            best_face = face_name

    return best_face, face_orientations


# ─── Rendering ───────────────────────────────────────────────────────

def _render(model, data, camera: str, width: int = 640, height: int = 480) -> str:
    """Render and return base64 PNG data URI."""
    from PIL import Image

    gl_ctx = mujoco.GLContext(width, height)
    gl_ctx.make_current()

    scene = mujoco.MjvScene(model, maxgeom=1000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id >= 0:
        cam.fixedcamid = cam_id
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, 0.5]
        cam.distance = 1.5
        cam.elevation = -25
        cam.azimuth = 90

    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(viewport, scene, ctx)

    img_arr = np.empty((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(img_arr, None, viewport, ctx)
    img_arr = np.flipud(img_arr)

    ctx.free()
    gl_ctx.free()

    pil_img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ─── Core: Generate & Verify Rotation Task ──────────────────────────

def compute_rotation_ground_truth(axis: str, angle_deg: float) -> str:
    """
    Compute which face ends up on top after rotating the cube.
    Assumes cube starts with identity orientation (top=+Z, front=-Y, right=+X).
    """
    angle_rad = math.radians(angle_deg)
    axis_vec = ROTATION_AXES[axis]
    q = _axis_angle_to_quat(axis_vec, angle_rad)
    top_face, _ = _determine_top_face(q)
    return top_face


def generate_rotation_task(
    axis: str = "Y",
    angle_deg: float = 90,
    direction: str = "clockwise",
) -> RotationTask:
    """Generate a rotation reasoning task with ground truth."""
    # Clockwise when viewed from positive axis = negative rotation angle
    # (right-hand rule: positive = counterclockwise from +axis)
    effective_angle = -angle_deg if direction == "clockwise" else angle_deg

    ground_truth = compute_rotation_ground_truth(axis, effective_angle)

    question = (
        f"A cube with colored faces sits on a table. "
        f"The faces are: Red (top), Green (bottom), Blue (front), "
        f"Yellow (back), Orange (right), Purple (left). "
        f"If the cube is rotated {int(abs(angle_deg))}° {direction} "
        f"around the {axis}-axis (when viewed from the positive {axis} direction), "
        f"which face will be on top?"
    )

    return RotationTask(
        axis=axis,
        angle_deg=angle_deg,
        direction=direction,
        initial_top="top",
        question=question,
        ground_truth=ground_truth,
    )


def simulate_rotation(task: RotationTask, render_views: bool = True) -> RotationResult:
    """
    Simulate the rotation in MuJoCo and verify the result.
    """
    xml = _build_labeled_cube_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Step 1: Let cube settle on table
    mujoco.mj_resetData(model, data)
    for _ in range(1000):
        mujoco.mj_step(model, data)

    # Get cube joint
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_jnt")
    qpos_adr = model.jnt_qposadr[jnt_id]

    # Record before state
    quat_before = data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
    before_images = []
    if render_views:
        for cam in ["front", "angle", "top", "side"]:
            before_images.append(_render(model, data, cam))

    # Step 2: Apply rotation
    effective_angle = -task.angle_deg if task.direction == "clockwise" else task.angle_deg
    angle_rad = math.radians(effective_angle)
    axis_vec = ROTATION_AXES[task.axis]
    rot_quat = _axis_angle_to_quat(axis_vec, angle_rad)

    # Apply rotation to current quaternion
    new_quat = _quat_multiply(rot_quat, quat_before)
    new_quat = new_quat / np.linalg.norm(new_quat)

    # Set the new orientation, keep position, zero velocity
    pos = data.qpos[qpos_adr:qpos_adr + 3].copy()
    # Lift slightly so it doesn't clip through table during rotation
    data.qpos[qpos_adr + 2] = pos[2] + 0.02
    data.qpos[qpos_adr + 3:qpos_adr + 7] = new_quat
    # Zero out velocities
    dof_adr = model.jnt_dofadr[jnt_id]
    data.qvel[dof_adr:dof_adr + 6] = 0

    # Step 3: Let it settle after rotation
    for _ in range(2000):
        mujoco.mj_step(model, data)

    # Record after state
    quat_after = data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
    after_images = []
    if render_views:
        for cam in ["front", "angle", "top", "side"]:
            after_images.append(_render(model, data, cam))

    # Determine which face is on top
    predicted_top, face_orientations = _determine_top_face(quat_after)

    return RotationResult(
        task=task,
        predicted_top=predicted_top,
        correct=(predicted_top == task.ground_truth),
        before_images=before_images,
        after_images=after_images,
        quaternion_before=[round(float(x), 4) for x in quat_before],
        quaternion_after=[round(float(x), 4) for x in quat_after],
        all_face_orientations=face_orientations,
    )


# ─── Batch: Multiple rotation variations ────────────────────────────

def generate_rotation_battery() -> list[RotationTask]:
    """Generate a battery of rotation tasks covering different axes and angles."""
    tasks = []
    for axis in ["X", "Y", "Z"]:
        for angle in [90, 180]:
            for direction in ["clockwise", "counterclockwise"]:
                tasks.append(generate_rotation_task(axis, angle, direction))
    return tasks


# ─── HTML Demo Generator ────────────────────────────────────────────

def generate_rotation_demo_html(results: list[RotationResult]) -> str:
    """Generate interactive HTML showing rotation tasks and verification."""

    cards_html = ""
    for i, r in enumerate(results):
        t = r.task
        status = "✅ CORRECT" if r.correct else "❌ WRONG"
        status_class = "correct" if r.correct else "wrong"

        # Before/after image grids
        before_imgs = "".join(
            f'<div><img src="{img}" alt="before {j}"/><span>{["Front","Angle","Top","Side"][j]}</span></div>'
            for j, img in enumerate(r.before_images)
        )
        after_imgs = "".join(
            f'<div><img src="{img}" alt="after {j}"/><span>{["Front","Angle","Top","Side"][j]}</span></div>'
            for j, img in enumerate(r.after_images)
        )

        # Face orientation table
        face_rows = ""
        for face, info in r.all_face_orientations.items():
            align = info["up_alignment"]
            highlight = ' class="highlight"' if face == r.predicted_top else ""
            face_rows += f'<tr{highlight}><td>{face}</td><td>{FACE_LABELS[face]}</td><td>{info["world_normal"]}</td><td>{align:.3f}</td></tr>'

        cards_html += f"""
        <div class="task-card">
            <div class="task-header {status_class}">
                <h2>Task {i+1}: Rotate {t.angle_deg}° {t.direction} around {t.axis}-axis</h2>
                <span class="status">{status}</span>
            </div>
            <div class="question">{t.question}</div>
            <div class="answer-row">
                <div class="answer-box">
                    <strong>Ground Truth:</strong> {t.ground_truth} → {FACE_LABELS[t.ground_truth]}
                </div>
                <div class="answer-box">
                    <strong>MuJoCo Result:</strong> {r.predicted_top} → {FACE_LABELS[r.predicted_top]}
                </div>
            </div>
            <div class="views-section">
                <h3>Before Rotation</h3>
                <div class="image-grid">{before_imgs}</div>
            </div>
            <div class="views-section">
                <h3>After Rotation</h3>
                <div class="image-grid">{after_imgs}</div>
            </div>
            <details>
                <summary>Face Orientations After Rotation</summary>
                <table>
                    <tr><th>Face</th><th>Label</th><th>World Normal</th><th>Up Alignment</th></tr>
                    {face_rows}
                </table>
                <p>Quaternion before: {r.quaternion_before}</p>
                <p>Quaternion after: {r.quaternion_after}</p>
            </details>
        </div>
        """

    correct_count = sum(1 for r in results if r.correct)
    total = len(results)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>DISE I-D: Rotation Reasoning Tasks</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; margin: 20px 0; font-size: 1.8em; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
  .stats {{ display: flex; gap: 20px; justify-content: center; margin-bottom: 30px; }}
  .stat {{ background: #1a1d27; padding: 15px 25px; border-radius: 10px; text-align: center; }}
  .stat .num {{ font-size: 2em; font-weight: bold; }}
  .stat .label {{ color: #888; font-size: 0.9em; }}
  .task-card {{ background: #1a1d27; border-radius: 12px; padding: 20px; margin-bottom: 25px; border: 1px solid #2a2d37; }}
  .task-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .task-header h2 {{ font-size: 1.2em; }}
  .status {{ font-size: 1.1em; font-weight: bold; padding: 5px 15px; border-radius: 20px; }}
  .correct .status {{ background: #0d3320; color: #4ade80; }}
  .wrong .status {{ background: #3b1018; color: #f87171; }}
  .question {{ background: #12141c; padding: 12px 16px; border-radius: 8px; margin-bottom: 15px; font-style: italic; line-height: 1.5; border-left: 3px solid #6366f1; }}
  .answer-row {{ display: flex; gap: 15px; margin-bottom: 15px; }}
  .answer-box {{ flex: 1; background: #12141c; padding: 10px 15px; border-radius: 8px; }}
  .views-section {{ margin-bottom: 15px; }}
  .views-section h3 {{ margin-bottom: 8px; color: #aaa; font-size: 0.95em; }}
  .image-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
  .image-grid div {{ text-align: center; }}
  .image-grid img {{ width: 100%; border-radius: 6px; border: 1px solid #333; }}
  .image-grid span {{ font-size: 0.75em; color: #888; }}
  details {{ margin-top: 10px; }}
  summary {{ cursor: pointer; color: #6366f1; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ padding: 6px 10px; border: 1px solid #333; text-align: left; font-size: 0.85em; }}
  th {{ background: #12141c; }}
  .highlight {{ background: #1a3a1a !important; }}
  .legend {{ display: flex; gap: 15px; justify-content: center; margin-bottom: 20px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.9em; }}
  .color-swatch {{ width: 18px; height: 18px; border-radius: 3px; border: 1px solid #555; }}
</style>
</head><body>
<h1>🎲 DISE Intrinsic-Dynamic: Rotation Reasoning</h1>
<p class="subtitle">Can a VLM predict which face ends up on top after rotation? MuJoCo verifies.</p>

<div class="legend">
  <div class="legend-item"><div class="color-swatch" style="background:rgba(230,51,51,1)"></div> Top = Face 1 (Red)</div>
  <div class="legend-item"><div class="color-swatch" style="background:rgba(51,204,77,1)"></div> Bottom = Face 2 (Green)</div>
  <div class="legend-item"><div class="color-swatch" style="background:rgba(51,77,230,1)"></div> Front = Face 3 (Blue)</div>
  <div class="legend-item"><div class="color-swatch" style="background:rgba(230,217,51,1)"></div> Back = Face 4 (Yellow)</div>
  <div class="legend-item"><div class="color-swatch" style="background:rgba(230,128,26,1)"></div> Right = Face 5 (Orange)</div>
  <div class="legend-item"><div class="color-swatch" style="background:rgba(179,51,230,1)"></div> Left = Face 6 (Purple)</div>
</div>

<div class="stats">
  <div class="stat"><div class="num">{total}</div><div class="label">Tasks</div></div>
  <div class="stat"><div class="num" style="color:#4ade80">{correct_count}</div><div class="label">Verified ✅</div></div>
  <div class="stat"><div class="num" style="color:#f87171">{total - correct_count}</div><div class="label">Mismatch ❌</div></div>
</div>

{cards_html}

</body></html>"""
