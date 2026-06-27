"""
DISE Perspective Taking: 20 Hard Scenarios

Tests whether VLMs can reason about what a camera would see from
different positions. Designed to catch known failure modes:
  - Left/right reversal when camera faces viewer
  - Order reversal from non-cardinal angles
  - Occlusion from novel viewpoints
  - Depth/proximity reasoning without text hints

Design:
  - VLM always sees ONE fixed viewpoint (viewer at azimuth=90)
  - A camera on tripod is visible in the scene at various positions
  - Question asks what the camera would capture
  - NO position hints in questions — model must reason from the image

MuJoCo coordinate system (from viewer azimuth=90):
  - Viewer at -Y looking +Y
  - Viewer's left = -X, viewer's right = +X
  - azimuth=180 → camera at +X (right side)
  - azimuth=270 → camera at +Y (opposite/far side)
  - azimuth=0   → camera at -X (left side)
"""

import io
import base64
import os
import sys
from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import db


# ─── Camera math helpers ──────────────────────────────────────────────

def camera_right_vector(azimuth_deg: float) -> np.ndarray:
    """
    Given a MuJoCo azimuth, compute the camera's 'right' direction in world coords.
    This tells us which direction maps to the RIGHT side of the camera's image.
    """
    # MuJoCo convention (empirically confirmed):
    # az=90 → cam at -Y, az=180 → cam at +X, az=270 → cam at +Y, az=0 → cam at -X
    az_rad = np.radians(azimuth_deg)
    # Camera position direction from lookat (unit vector)
    # cam_pos_dir rotates clockwise: az=90→(0,-1), az=180→(1,0), az=270→(0,1), az=0→(-1,0)
    cam_x = np.sin(np.radians(azimuth_deg - 90))
    cam_y = -np.cos(np.radians(azimuth_deg - 90))
    # Forward = -cam_pos_dir (looking toward lookat)
    fwd = np.array([-cam_x, -cam_y, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    return right


def position_in_camera_image(obj_pos: np.ndarray, cam_azimuth: float,
                              cam_lookat: np.ndarray) -> str:
    """
    Determine if an object appears on 'left', 'center', or 'right' in
    a camera's image given the camera's azimuth and lookat point.
    """
    right = camera_right_vector(cam_azimuth)
    # Vector from lookat to object (in XY plane)
    delta = np.array(obj_pos[:2]) - np.array(cam_lookat[:2])
    # Project onto camera's right axis
    proj = np.dot(delta, right[:2])
    if proj > 0.08:
        return "right"
    elif proj < -0.08:
        return "left"
    return "center"


def distance_to_camera(obj_pos: np.ndarray, cam_azimuth: float,
                        cam_distance: float, cam_lookat: np.ndarray) -> float:
    """Compute distance from object to camera position."""
    az_rad = np.radians(cam_azimuth - 90)
    cam_x = cam_lookat[0] + cam_distance * np.sin(az_rad)
    cam_y = cam_lookat[1] - cam_distance * np.cos(az_rad)
    cam_pos = np.array([cam_x, cam_y])
    obj_xy = np.array(obj_pos[:2])
    return np.linalg.norm(obj_xy - cam_pos)


def order_left_to_right(objects: list, cam_azimuth: float,
                         cam_lookat: np.ndarray) -> list:
    """
    Order objects from LEFT to RIGHT as they appear in camera's image.
    """
    right = camera_right_vector(cam_azimuth)
    
    def proj(obj):
        delta = np.array(obj["pos"][:2]) - np.array(cam_lookat[:2])
        return np.dot(delta, right[:2])
    
    return sorted(objects, key=proj)


# ─── Scene definitions ─────────────────────────────────────────────────

OBJECTS_SCENE_A = [
    {"name": "red mug", "pos": [-0.25, 0.0, 0.425], "color": "0.9 0.1 0.1 1", "type": "mug"},
    {"name": "green book", "pos": [0.0, 0.0, 0.39], "color": "0.1 0.7 0.15 1", "type": "book"},
    {"name": "blue bottle", "pos": [0.25, 0.0, 0.45], "color": "0.1 0.25 0.9 1", "type": "bottle"},
]

OBJECTS_SCENE_B = [
    {"name": "red mug", "pos": [-0.20, 0.12, 0.425], "color": "0.9 0.1 0.1 1", "type": "mug"},
    {"name": "green book", "pos": [0.05, -0.10, 0.39], "color": "0.1 0.7 0.15 1", "type": "book"},
    {"name": "blue bottle", "pos": [0.22, 0.08, 0.45], "color": "0.1 0.25 0.9 1", "type": "bottle"},
    {"name": "yellow cup", "pos": [-0.10, -0.15, 0.41], "color": "0.9 0.8 0.1 1", "type": "cup"},
]

OBJECTS_SCENE_C = [
    {"name": "red mug", "pos": [-0.28, -0.08, 0.425], "color": "0.9 0.1 0.1 1", "type": "mug"},
    {"name": "green book", "pos": [0.0, 0.15, 0.39], "color": "0.1 0.7 0.15 1", "type": "book"},
    {"name": "blue bottle", "pos": [0.20, -0.12, 0.45], "color": "0.1 0.25 0.9 1", "type": "bottle"},
    {"name": "yellow cup", "pos": [-0.12, 0.18, 0.41], "color": "0.9 0.8 0.1 1", "type": "cup"},
    {"name": "white box", "pos": [0.28, 0.14, 0.40], "color": "0.95 0.95 0.95 1", "type": "box"},
]


def _build_object_xml(obj: dict) -> str:
    """Generate MuJoCo XML for a single object."""
    x, y, z = obj["pos"]
    rgba = obj["color"]
    name = obj["name"].replace(" ", "_")
    
    if obj["type"] == "mug":
        return f"""
    <body name="{name}" pos="{x} {y} {z}">
      <geom type="cylinder" size="0.04 0.05" rgba="{rgba}"/>
      <geom type="capsule" size="0.01" fromto="0.04 0 -0.03  0.065 0 0.03" rgba="{rgba}"/>
    </body>"""
    elif obj["type"] == "book":
        return f"""
    <body name="{name}" pos="{x} {y} {z}">
      <geom type="box" size="0.07 0.05 0.015" rgba="{rgba}"/>
    </body>"""
    elif obj["type"] == "bottle":
        return f"""
    <body name="{name}" pos="{x} {y} {z}">
      <geom type="cylinder" size="0.03 0.07" rgba="{rgba}"/>
      <geom type="cylinder" size="0.015 0.035" pos="0 0 0.105" rgba="{rgba}"/>
    </body>"""
    elif obj["type"] == "cup":
        return f"""
    <body name="{name}" pos="{x} {y} {z}">
      <geom type="cylinder" size="0.035 0.045" rgba="{rgba}"/>
    </body>"""
    elif obj["type"] == "box":
        return f"""
    <body name="{name}" pos="{x} {y} {z}">
      <geom type="box" size="0.045 0.045 0.035" rgba="{rgba}"/>
    </body>"""
    return ""


def _build_camera_xml(cam_pos: str, cam_azimuth: float) -> str:
    """Generate a camera on tripod at the given world position, pointing toward table center."""
    cx, cy, cz = [float(v) for v in cam_pos.split()]
    
    # Compute lens direction (pointing toward 0,0,table_height)
    target = np.array([0.0, 0.0, 0.4])
    pos = np.array([cx, cy, cz])
    direction = target - pos
    direction = direction / np.linalg.norm(direction)
    
    # Lens as a cylinder pointing in the direction
    # The lens points in local -Y. After euler="0 0 yaw" rotation:
    #   local -Y in world = (sin(yaw), -cos(yaw), 0)
    # We need this to equal direction_xy:
    #   sin(yaw) = direction[0]  →  yaw = atan2(direction[0], -direction[1])
    #   -cos(yaw) = direction[1]
    yaw = np.degrees(np.arctan2(direction[0], -direction[1]))
    
    return f"""
    <!-- Camera on Tripod -->
    <body name="camera_rig" pos="{cx} {cy} 0">
      <!-- Tripod legs -->
      <geom type="capsule" size="0.01" fromto="-0.08 -0.06 0  0 0 {cz-0.12}" rgba="0.2 0.2 0.2 1"/>
      <geom type="capsule" size="0.01" fromto="0.08 -0.06 0   0 0 {cz-0.12}" rgba="0.2 0.2 0.2 1"/>
      <geom type="capsule" size="0.01" fromto="0.0 0.08 0      0 0 {cz-0.12}" rgba="0.2 0.2 0.2 1"/>
      <!-- Pole -->
      <geom type="cylinder" size="0.01 0.06" pos="0 0 {cz-0.05}" rgba="0.2 0.2 0.2 1"/>
      <!-- Camera body -->
      <body name="cam_body" pos="0 0 {cz}" euler="0 0 {yaw}">
        <geom type="box" size="0.045 0.03 0.025" rgba="0.12 0.12 0.12 1"/>
        <!-- Lens pointing -Y in local frame (toward table after yaw rotation) -->
        <geom type="cylinder" size="0.018 0.022" pos="0 -0.052 0" euler="90 0 0" rgba="0.3 0.3 0.4 1"/>
        <geom type="cylinder" size="0.023 0.005" pos="0 -0.074 0" euler="90 0 0" rgba="0.25 0.25 0.35 1"/>
        <!-- Red LED -->
        <geom type="sphere" size="0.006" pos="0.035 0.0 0.025" rgba="1 0.15 0 1"/>
      </body>
    </body>"""


def build_full_scene_xml(objects: list, camera_world_pos: str, cam_azimuth: float) -> str:
    """Build complete MuJoCo scene XML."""
    obj_xml = "\n".join(_build_object_xml(o) for o in objects)
    cam_xml = _build_camera_xml(camera_world_pos, cam_azimuth)
    
    return f"""
<mujoco model="perspective_taking">
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1024" offheight="768"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.9" rgb2="0.78 0.78 0.75" width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="5 5" reflectance="0.05"/>
    <material name="table_mat" rgba="0.6 0.4 0.25 1"/>
  </asset>
  <worldbody>
    <light pos="0 -2 4" dir="0 0.5 -0.8" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <light pos="2 0 3" dir="-0.3 0 -0.7" diffuse="0.35 0.35 0.38"/>
    <light pos="-2 1 3" dir="0.3 -0.2 -0.6" diffuse="0.3 0.3 0.32"/>
    
    <geom type="plane" size="3 3 0.01" material="floor_mat"/>
    
    <!-- Table -->
    <body name="table" pos="0 0 0.35">
      <geom type="box" size="0.5 0.35 0.025" material="table_mat"/>
      <geom type="cylinder" size="0.025" fromto="-0.45 -0.30 -0.35  -0.45 -0.30 0" material="table_mat"/>
      <geom type="cylinder" size="0.025" fromto="0.45 -0.30 -0.35   0.45 -0.30 0" material="table_mat"/>
      <geom type="cylinder" size="0.025" fromto="-0.45 0.30 -0.35   -0.45 0.30 0" material="table_mat"/>
      <geom type="cylinder" size="0.025" fromto="0.45 0.30 -0.35    0.45 0.30 0" material="table_mat"/>
    </body>
    
    {obj_xml}
    {cam_xml}
  </worldbody>
</mujoco>
"""


# ─── Rendering ────────────────────────────────────────────────────────

def render_scene(xml: str) -> str:
    """Render from the fixed viewer angle (azimuth=90). Returns base64 JPEG."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 90
    cam.elevation = -25
    cam.distance = 2.0
    cam.lookat[:] = [0, 0, 0.4]
    renderer.update_scene(data, cam)
    pixels = renderer.render()
    renderer.close()

    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_ground_truth(xml: str, gt_azimuth: float, gt_dist: float = 0.7,
                         gt_elev: float = -10) -> str:
    """Render from the tripod camera's perspective for GT verification."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = gt_azimuth
    cam.elevation = gt_elev
    cam.distance = gt_dist
    cam.lookat[:] = [0, 0, 0.4]
    renderer.update_scene(data, cam)
    pixels = renderer.render()
    renderer.close()

    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ─── Scenario Definitions ─────────────────────────────────────────────

@dataclass
class PerspectiveScenario:
    scene_id: str
    objects: list          # which object set
    camera_world_pos: str  # "x y z"
    camera_azimuth: float  # for GT rendering
    question: str
    answer: str
    reasoning: str
    difficulty: str        # medium / hard
    question_type: str     # left_right, ordering, occlusion, closest, farthest


def compute_camera_world_pos(azimuth: float, distance: float = 0.75,
                              height: float = 0.40) -> str:
    """Convert camera azimuth + distance to world XYZ position."""
    az_rad = np.radians(azimuth - 90)
    x = distance * np.sin(az_rad)
    y = -distance * np.cos(az_rad)
    return f"{x:.3f} {y:.3f} {height:.3f}"


def generate_20_scenarios() -> list[PerspectiveScenario]:
    scenarios = []
    lookat = np.array([0, 0, 0.4])
    
    # ══════════════════════════════════════════════════════════════════
    # SCENE A: 3 objects in a row (red mug left, green book center, blue bottle right)
    # ══════════════════════════════════════════════════════════════════
    objs_a = OBJECTS_SCENE_A
    
    # ── Q1: Camera OPPOSITE (az=270). Red mug left or right? ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_01",
        objects=objs_a,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 3 objects on the table. "
            "When the photo is clicked through the camera, will the red mug appear on the left side or the right side? "
            "Answer with just: left or right"
        ),
        answer="right",
        reasoning="Camera at opposite side faces -Y. Its right = -X. Red mug at x=-0.25 is in -X direction = camera's RIGHT.",
        difficulty="hard",
        question_type="left_right",
    ))
    
    # ── Q2: Camera OPPOSITE (az=270). Blue bottle left or right? ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_02",
        objects=objs_a,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 3 objects on the table. "
            "When the photo is clicked through the camera, will the blue bottle appear on the left side or the right side? "
            "Answer with just: left or right"
        ),
        answer="left",
        reasoning="Camera opposite, right=-X. Blue bottle at x=+0.25 is in +X = camera's LEFT.",
        difficulty="hard",
        question_type="left_right",
    ))
    
    # ── Q3: Camera OPPOSITE (az=270). Full order reversal. ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_03",
        objects=objs_a,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 3 objects on the table. "
            "List all objects from left to right as they will appear in the photo clicked through the camera. "
            "Answer with just the names, comma-separated."
        ),
        answer="blue bottle, green book, red mug",
        reasoning="Opposite view reverses left/right. Your left-to-right (red,green,blue) becomes camera's right-to-left.",
        difficulty="hard",
        question_type="ordering",
    ))
    
    # ── Q4: Camera on RIGHT (az=180). Farthest object. ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_04",
        objects=objs_a,
        camera_world_pos=compute_camera_world_pos(180),
        camera_azimuth=180,
        question=(
            "There are 3 objects on the table. "
            "Which object will appear smallest in the photo clicked through the camera? "
            "Answer with just the object name."
        ),
        answer="red mug",
        reasoning="Camera at +X. Red mug at x=-0.25 is furthest from camera at x=+0.75.",
        difficulty="medium",
        question_type="farthest",
    ))
    
    # ── Q5: Camera on LEFT (az=0). Closest object. ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_05",
        objects=objs_a,
        camera_world_pos=compute_camera_world_pos(0),
        camera_azimuth=0,
        question=(
            "There are 3 objects on the table. "
            "Which object will appear closest in the photo clicked through the camera? "
            "Answer with just the object name."
        ),
        answer="red mug",
        reasoning="Camera at -X (left). Red mug at x=-0.25 is closest to camera at x=-0.75.",
        difficulty="medium",
        question_type="closest",
    ))
    
    # ── Q6: Camera on RIGHT (az=180). Occlusion. ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_06",
        objects=objs_a,
        camera_world_pos=compute_camera_world_pos(180),
        camera_azimuth=180,
        question=(
            "There are 3 objects on the table. "
            "Which object will be hidden (occluded) behind the blue bottle in the photo clicked through the camera? "
            "Answer with just the object name, or 'none' if all are visible."
        ),
        answer="green book",
        reasoning="From +X looking -X: blue(+0.25) is in front, green(0) directly behind it, red(-0.25) behind green.",
        difficulty="hard",
        question_type="occlusion",
    ))
    
    # ══════════════════════════════════════════════════════════════════
    # SCENE B: 4 objects scattered
    # red mug (-0.20, 0.12), green book (0.05, -0.10),
    # blue bottle (0.22, 0.08), yellow cup (-0.10, -0.15)
    # ══════════════════════════════════════════════════════════════════
    objs_b = OBJECTS_SCENE_B
    
    # ── Q7: Camera OPPOSITE (az=270). Far left? ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_07",
        objects=objs_b,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 4 objects on the table. "
            "Which object will appear on the far left when the photo is clicked through the camera? "
            "Answer with just the object name."
        ),
        answer="blue bottle",
        reasoning="Camera opposite, right=-X. Blue bottle at x=+0.22 has most negative projection on right axis = leftmost.",
        difficulty="hard",
        question_type="left_right",
    ))
    
    # ── Q8: Camera OPPOSITE (az=270). Far right? ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_08",
        objects=objs_b,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 4 objects on the table. "
            "Which object will appear on the far right when the photo is clicked through the camera? "
            "Answer with just the object name."
        ),
        answer="red mug",
        reasoning="Camera right=-X. Red mug at x=-0.20 has most positive projection on -X axis = rightmost.",
        difficulty="hard",
        question_type="left_right",
    ))
    
    # ── Q9: Camera at 45° RIGHT (az=135). Closest? ──
    cam_pos_9 = compute_camera_world_pos(135, distance=0.75)
    cx9, cy9 = float(cam_pos_9.split()[0]), float(cam_pos_9.split()[1])
    dists_9 = {o["name"]: np.sqrt((o["pos"][0]-cx9)**2 + (o["pos"][1]-cy9)**2) for o in objs_b}
    closest_9 = min(dists_9, key=dists_9.get)
    
    scenarios.append(PerspectiveScenario(
        scene_id="pt_09",
        objects=objs_b,
        camera_world_pos=cam_pos_9,
        camera_azimuth=135,
        question=(
            "There are 4 objects on the table. "
            "Which object will appear closest in the photo clicked through the camera? "
            "Answer with just the object name."
        ),
        answer=closest_9,
        reasoning=f"Camera at ({cx9:.2f},{cy9:.2f}). Distances: {', '.join(f'{k}={v:.2f}' for k,v in sorted(dists_9.items(), key=lambda x:x[1]))}",
        difficulty="hard",
        question_type="closest",
    ))
    
    # ── Q10: Camera at 45° LEFT (az=45). Closest? ──
    cam_pos_10 = compute_camera_world_pos(45, distance=0.75)
    cx10, cy10 = float(cam_pos_10.split()[0]), float(cam_pos_10.split()[1])
    dists_10 = {o["name"]: np.sqrt((o["pos"][0]-cx10)**2 + (o["pos"][1]-cy10)**2) for o in objs_b}
    closest_10 = min(dists_10, key=dists_10.get)
    
    scenarios.append(PerspectiveScenario(
        scene_id="pt_10",
        objects=objs_b,
        camera_world_pos=cam_pos_10,
        camera_azimuth=45,
        question=(
            "There are 4 objects on the table. "
            "Which object will appear closest in the photo clicked through the camera? "
            "Answer with just the object name."
        ),
        answer=closest_10,
        reasoning=f"Camera at ({cx10:.2f},{cy10:.2f}). Distances: {', '.join(f'{k}={v:.2f}' for k,v in sorted(dists_10.items(), key=lambda x:x[1]))}",
        difficulty="hard",
        question_type="closest",
    ))
    
    # ── Q11: Camera on RIGHT (az=180) of Scene B. Order closest to farthest. ──
    objs_b_by_x = sorted(objs_b, key=lambda o: -o["pos"][0])
    order_names = [o["name"] for o in objs_b_by_x]
    scenarios.append(PerspectiveScenario(
        scene_id="pt_11",
        objects=objs_b,
        camera_world_pos=compute_camera_world_pos(180),
        camera_azimuth=180,
        question=(
            "There are 4 objects on the table. "
            "List the objects from closest to farthest as they appear in the photo clicked through the camera. "
            "Answer with just the names, comma-separated."
        ),
        answer=", ".join(order_names),
        reasoning=f"Camera at +X. Objects sorted by x-coordinate (descending): {order_names}",
        difficulty="hard",
        question_type="ordering",
    ))
    
    # ── Q12: Camera OPPOSITE (az=270) of Scene B. Full L→R order. ──
    ordered_b_270 = order_left_to_right(objs_b, 270, lookat)
    order_names_12 = [o["name"] for o in ordered_b_270]
    scenarios.append(PerspectiveScenario(
        scene_id="pt_12",
        objects=objs_b,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 4 objects on the table. "
            "List all objects from left to right as they will appear in the photo clicked through the camera. "
            "Answer with just the names, comma-separated."
        ),
        answer=", ".join(order_names_12),
        reasoning=f"Camera opposite, right=-X. Order by projection on -X: {order_names_12}",
        difficulty="hard",
        question_type="ordering",
    ))
    
    # ══════════════════════════════════════════════════════════════════
    # SCENE C: 5 objects scattered widely
    # ══════════════════════════════════════════════════════════════════
    objs_c = OBJECTS_SCENE_C
    
    # ── Q13: Camera OPPOSITE (az=270). Leftmost of 5? ──
    ordered_c_270 = order_left_to_right(objs_c, 270, lookat)
    leftmost_c_270 = ordered_c_270[0]["name"]
    rightmost_c_270 = ordered_c_270[-1]["name"]
    
    scenarios.append(PerspectiveScenario(
        scene_id="pt_13",
        objects=objs_c,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 5 objects on the table. "
            "Which object will appear on the far left when the photo is clicked through the camera? "
            "Answer with just the object name."
        ),
        answer=leftmost_c_270,
        reasoning=f"Camera right=-X. Full L→R order: {[o['name'] for o in ordered_c_270]}",
        difficulty="hard",
        question_type="left_right",
    ))
    
    # ── Q14: Camera OPPOSITE (az=270). Rightmost of 5? ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_14",
        objects=objs_c,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 5 objects on the table. "
            "Which object will appear on the far right when the photo is clicked through the camera? "
            "Answer with just the object name."
        ),
        answer=rightmost_c_270,
        reasoning=f"Camera right=-X. Rightmost = most negative X = {rightmost_c_270}",
        difficulty="hard",
        question_type="left_right",
    ))
    
    # ── Q15: Camera on LEFT (az=0) of Scene C. Farthest? ──
    cam_pos_15 = compute_camera_world_pos(0)
    cx15, cy15 = float(cam_pos_15.split()[0]), float(cam_pos_15.split()[1])
    dists_15 = {o["name"]: np.sqrt((o["pos"][0]-cx15)**2 + (o["pos"][1]-cy15)**2) for o in objs_c}
    farthest_15 = max(dists_15, key=dists_15.get)
    
    scenarios.append(PerspectiveScenario(
        scene_id="pt_15",
        objects=objs_c,
        camera_world_pos=cam_pos_15,
        camera_azimuth=0,
        question=(
            "There are 5 objects on the table. "
            "Which object will appear smallest in the photo clicked through the camera? "
            "Answer with just the object name."
        ),
        answer=farthest_15,
        reasoning=f"Camera at ({cx15:.2f},{cy15:.2f}). Farthest: {farthest_15} ({dists_15[farthest_15]:.2f})",
        difficulty="hard",
        question_type="farthest",
    ))
    
    # ── Q16: Camera at 135° (az=225) of Scene C. Closest? ──
    cam_pos_16 = compute_camera_world_pos(225, distance=0.80)
    cx16, cy16 = float(cam_pos_16.split()[0]), float(cam_pos_16.split()[1])
    dists_16 = {o["name"]: np.sqrt((o["pos"][0]-cx16)**2 + (o["pos"][1]-cy16)**2) for o in objs_c}
    closest_16 = min(dists_16, key=dists_16.get)
    
    scenarios.append(PerspectiveScenario(
        scene_id="pt_16",
        objects=objs_c,
        camera_world_pos=cam_pos_16,
        camera_azimuth=225,
        question=(
            "There are 5 objects on the table. "
            "Which object will appear closest in the photo clicked through the camera? "
            "Answer with just the object name."
        ),
        answer=closest_16,
        reasoning=f"Camera at ({cx16:.2f},{cy16:.2f}). Closest: {closest_16} ({dists_16[closest_16]:.2f})",
        difficulty="hard",
        question_type="closest",
    ))
    
    # ── Q17: Camera on RIGHT (az=180) of Scene C. L→R ordering. ──
    ordered_c_180 = order_left_to_right(objs_c, 180, lookat)
    order_names_17 = [o["name"] for o in ordered_c_180]
    scenarios.append(PerspectiveScenario(
        scene_id="pt_17",
        objects=objs_c,
        camera_world_pos=compute_camera_world_pos(180),
        camera_azimuth=180,
        question=(
            "There are 5 objects on the table. "
            "List all objects from left to right as they will appear in the photo clicked through the camera. "
            "Answer with just the names, comma-separated."
        ),
        answer=", ".join(order_names_17),
        reasoning=f"Camera at +X, right=+Y. Objects sorted by Y-projection: {order_names_17}",
        difficulty="hard",
        question_type="ordering",
    ))
    
    # ── Q18: Camera OPPOSITE (az=270) Scene A. Between question. ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_18",
        objects=objs_a,
        camera_world_pos=compute_camera_world_pos(270),
        camera_azimuth=270,
        question=(
            "There are 3 objects on the table. "
            "In the photo clicked through the camera, which object appears between the blue bottle and the red mug? "
            "Answer with just the object name."
        ),
        answer="green book",
        reasoning="In opposite view L→R order is: blue, green, red. Green book is between them.",
        difficulty="medium",
        question_type="left_right",
    ))
    
    # ── Q19: Camera on LEFT (az=0) Scene C. Relative left/right. ──
    scenarios.append(PerspectiveScenario(
        scene_id="pt_19",
        objects=objs_c,
        camera_world_pos=compute_camera_world_pos(0),
        camera_azimuth=0,
        question=(
            "There are 5 objects on the table. "
            "In the photo clicked through the camera, does the green book appear to the left or right of the blue bottle? "
            "Answer with just: left or right"
        ),
        answer="left",
        reasoning="Camera at -X, right=-Y. Green(y=0.15) proj on -Y = -0.15 (LEFT). Blue(y=-0.12) proj = +0.12 (RIGHT). Green is LEFT of blue.",
        difficulty="hard",
        question_type="left_right",
    ))
    
    # ── Q20: Camera at 315° (between left and far) of Scene B. Farthest? ──
    cam_pos_20 = compute_camera_world_pos(315, distance=0.80)
    cx20, cy20 = float(cam_pos_20.split()[0]), float(cam_pos_20.split()[1])
    dists_20 = {o["name"]: np.sqrt((o["pos"][0]-cx20)**2 + (o["pos"][1]-cy20)**2) for o in objs_b}
    farthest_20 = max(dists_20, key=dists_20.get)
    
    scenarios.append(PerspectiveScenario(
        scene_id="pt_20",
        objects=objs_b,
        camera_world_pos=cam_pos_20,
        camera_azimuth=315,
        question=(
            "There are 4 objects on the table. "
            "Which object will appear furthest in the photo clicked through the camera? "
            "Answer with just the object name."
        ),
        answer=farthest_20,
        reasoning=f"Camera at ({cx20:.2f},{cy20:.2f}). Farthest: {farthest_20} ({dists_20[farthest_20]:.2f})",
        difficulty="hard",
        question_type="farthest",
    ))
    
    return scenarios


# ─── Publish to database ──────────────────────────────────────────────

def publish_dataset():
    scenarios = generate_20_scenarios()
    dataset_id = str(uuid4())
    
    print(f"{'='*60}")
    print(f"PERSPECTIVE TAKING — 20 Hard Scenarios")
    print(f"{'='*60}")
    
    all_records = []
    
    for idx, sc in enumerate(scenarios):
        print(f"\n[{idx+1}/20] {sc.scene_id} | {sc.question_type} | {sc.difficulty}")
        
        # Build and render scene
        xml = build_full_scene_xml(sc.objects, sc.camera_world_pos, sc.camera_azimuth)
        scene_image = render_scene(xml)
        gt_image = render_ground_truth(xml, sc.camera_azimuth)
        
        print(f"  Q: {sc.question[:80]}...")
        print(f"  A: {sc.answer}")
        
        record = {
            "id": str(uuid4()),
            "dataset_id": dataset_id,
            "pair_type": "ground_truth",
            "scene_id": sc.scene_id,
            "prompt": sc.question,
            "category": "perspective_taking",
            "difficulty": sc.difficulty,
            "ground_truth": {
                "answer": sc.answer,
                "reasoning": sc.reasoning,
                "question_type": sc.question_type,
                "camera_azimuth": sc.camera_azimuth,
            },
            "source": {
                "dataset": "mujoco:perspective_taking",
                "scene_id": sc.scene_id,
                "images": [scene_image],
                "gt_image": gt_image,
            },
            "status": "ready",
        }
        all_records.append(record)
    
    # Create dataset
    db.create_dataset({
        "id": dataset_id,
        "name": "perspective-taking-20",
        "task_type": "perspective_taking",
        "scenario_count": 20,
        "created_at": datetime.now().isoformat(),
        "config": {
            "environment": "perspective_taking",
            "mode": "curated",
            "answer_format": "free_form",
            "description": "20 hard perspective-taking questions: left/right reversal, ordering, occlusion, proximity",
        },
    })
    
    db.add_scenarios(all_records)
    
    # Summary
    by_type = {}
    for sc in scenarios:
        by_type.setdefault(sc.question_type, []).append(sc.scene_id)
    
    print(f"\n{'='*60}")
    print(f"✅ Published: perspective-taking-20")
    print(f"   Dataset ID: {dataset_id}")
    print(f"   Scenarios: 20")
    print(f"   Question types:")
    for qt, ids in by_type.items():
        print(f"     {qt}: {len(ids)} questions")
    print(f"{'='*60}")
    
    return dataset_id


if __name__ == "__main__":
    publish_dataset()
