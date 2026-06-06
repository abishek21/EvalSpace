"""
DISE Spatial Fitting: "Can object A fit through gap B?"

Ground truth is determined by comparing object dimensions against gap dimensions,
considering possible rotations/orientations. MuJoCo renders the scene for VLM evaluation.

Scenarios range from obvious (ball through large hole) to tricky (rotated box through slot).
"""
import io
import base64
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco
from PIL import Image


# ─── Object Primitives ──────────────────────────────────────────────

FITTING_OBJECTS = {
    "small_ball":     {"type": "sphere",   "dims": [0.03],               "label": "small ball",     "desc": "a small ball (3cm radius)",           "vlm_shape": "ball"},
    "large_ball":     {"type": "sphere",   "dims": [0.06],               "label": "large ball",     "desc": "a large ball (6cm radius)",           "vlm_shape": "ball"},
    "cube":           {"type": "box",      "dims": [0.04, 0.04, 0.04],   "label": "cube",           "desc": "a cube (4cm sides)",                  "vlm_shape": "cube"},
    "tall_box":       {"type": "box",      "dims": [0.03, 0.03, 0.08],   "label": "tall box",       "desc": "a tall box (3×3×8cm)",                "vlm_shape": "box"},
    "flat_plank":     {"type": "box",      "dims": [0.10, 0.04, 0.015],  "label": "flat plank",     "desc": "a flat plank (10×4×1.5cm)",           "vlm_shape": "plank"},
    "cylinder":       {"type": "cylinder", "dims": [0.03, 0.06],         "label": "cylinder",       "desc": "a cylinder (3cm radius, 6cm tall)",   "vlm_shape": "cylinder"},
    "thin_rod":       {"type": "cylinder", "dims": [0.01, 0.10],         "label": "thin rod",       "desc": "a thin rod (1cm radius, 10cm long)",  "vlm_shape": "rod"},
    "wide_disc":      {"type": "cylinder", "dims": [0.06, 0.015],        "label": "wide disc",      "desc": "a wide disc (6cm radius, 1.5cm thick)","vlm_shape": "disc"},
    "long_box":       {"type": "box",      "dims": [0.12, 0.03, 0.03],   "label": "long box",       "desc": "a long box (12×3×3cm)",               "vlm_shape": "box"},
    "small_cube":     {"type": "box",      "dims": [0.025, 0.025, 0.025],"label": "small cube",     "desc": "a small cube (2.5cm sides)",           "vlm_shape": "cube"},
}

# Gap types — defined by the opening shape and size
GAP_TYPES = {
    "square_small":    {"shape": "square",    "width": 0.05, "height": 0.05, "label": "small square opening (5×5cm)",          "vlm_label": "square opening"},
    "square_large":    {"shape": "square",    "width": 0.10, "height": 0.10, "label": "large square opening (10×10cm)",         "vlm_label": "square opening"},
    "rect_wide":       {"shape": "rectangle", "width": 0.12, "height": 0.04, "label": "wide rectangular slot (12×4cm)",         "vlm_label": "rectangular opening"},
    "rect_tall":       {"shape": "rectangle", "width": 0.04, "height": 0.12, "label": "tall rectangular slot (4×12cm)",         "vlm_label": "rectangular opening"},
    "circle_small":    {"shape": "circle",    "radius": 0.035, "label": "small circular hole (3.5cm radius)",                   "vlm_label": "circular opening"},
    "circle_large":    {"shape": "circle",    "radius": 0.07,  "label": "large circular hole (7cm radius)",                     "vlm_label": "circular opening"},
    "narrow_slit":     {"shape": "rectangle", "width": 0.10, "height": 0.02, "label": "narrow horizontal slit (10×2cm)",        "vlm_label": "opening"},
    "rect_medium":     {"shape": "rectangle", "width": 0.07, "height": 0.07, "label": "medium square opening (7×7cm)",          "vlm_label": "square opening"},
}

COLORS = {
    "red":    "0.85 0.15 0.15 1",
    "blue":   "0.15 0.25 0.85 1",
    "green":  "0.15 0.75 0.25 1",
    "yellow": "0.9 0.8 0.1 1",
    "orange": "0.9 0.45 0.1 1",
    "purple": "0.6 0.15 0.8 1",
    "cyan":   "0.1 0.75 0.8 1",
    "white":  "0.9 0.9 0.9 1",
}


@dataclass
class FittingScenario:
    """A spatial fitting scenario."""
    name: str
    object_type: str          # key in FITTING_OBJECTS
    object_color: str         # key in COLORS
    gap_type: str             # key in GAP_TYPES
    fits: bool                # ground truth: can the object fit through?
    best_orientation: str     # description of how it fits (or why it doesn't)
    question: str             # natural language question
    difficulty: str           # easy, medium, hard
    reasoning: str            # explanation for ground truth


@dataclass
class FittingResult:
    """Result with rendered views."""
    scenario: FittingScenario
    images: list[str]         # rendered views [front, angle, top, side]


# ─── Ground Truth Logic ─────────────────────────────────────────────

def _can_fit(obj_key: str, gap_key: str) -> tuple[bool, str]:
    """
    Determine if object can fit through gap in ANY valid orientation.
    Returns (fits: bool, explanation: str).
    """
    obj = FITTING_OBJECTS[obj_key]
    gap = GAP_TYPES[gap_key]
    obj_type = obj["type"]
    obj_dims = obj["dims"]

    if gap["shape"] == "circle":
        r = gap["radius"]
        if obj_type == "sphere":
            fits = obj_dims[0] < r
            return fits, f"Ball radius {obj_dims[0]*100:.1f}cm vs hole radius {r*100:.1f}cm"
        elif obj_type == "cylinder":
            cyl_r, cyl_h = obj_dims
            # Can go through end-first (need circle to fit cylinder's circular cross-section)
            # Or sideways (need circle to fit rectangle cyl_r*2 x cyl_h)
            end_first = cyl_r < r
            sideways = math.sqrt(cyl_r**2 + (cyl_h/2)**2) < r  # diagonal of side profile
            if end_first:
                return True, f"Cylinder can pass end-first: radius {cyl_r*100:.1f}cm < hole {r*100:.1f}cm"
            elif sideways:
                return True, f"Cylinder can fit diagonally through the circular hole"
            else:
                return False, f"Cylinder radius {cyl_r*100:.1f}cm and height {cyl_h*100:.1f}cm too large for hole radius {r*100:.1f}cm"
        elif obj_type == "box":
            # Try all 3 orientations: face XY, XZ, YZ going through
            dims = sorted(obj_dims)  # smallest to largest
            # Best case: smallest two dimensions form the cross-section
            min_cross = math.sqrt(dims[0]**2 + dims[1]**2)  # diagonal of smallest face
            if min_cross < r:
                return True, f"Box can fit through diagonally: smallest face diagonal {min_cross*100:.1f}cm < hole diameter {r*2*100:.1f}cm"
            # Check if smallest face fits within circle
            if dims[0] < r and dims[1] < r:
                return True, f"Box smallest face ({dims[0]*100:.1f}×{dims[1]*100:.1f}cm) fits within hole radius {r*100:.1f}cm"
            return False, f"Box too large: smallest face {dims[0]*100:.1f}×{dims[1]*100:.1f}cm won't fit in hole radius {r*100:.1f}cm"

    else:  # square or rectangle
        gw = gap["width"]
        gh = gap["height"]

        if obj_type == "sphere":
            d = obj_dims[0] * 2  # diameter
            fits = d < min(gw, gh)
            return fits, f"Ball diameter {d*100:.1f}cm vs opening {gw*100:.1f}×{gh*100:.1f}cm (min side: {min(gw,gh)*100:.1f}cm)"
        elif obj_type == "cylinder":
            cyl_r, cyl_h = obj_dims
            d = cyl_r * 2
            # End-first: need d < min(gw, gh)
            end_first = d < min(gw, gh)
            # Sideways: need d < one dim AND cyl_h < other dim
            sideways = (d < gw and cyl_h < gh) or (d < gh and cyl_h < gw)
            if end_first:
                return True, f"Cylinder can pass end-first: diameter {d*100:.1f}cm fits in {gw*100:.1f}×{gh*100:.1f}cm opening"
            elif sideways:
                return True, f"Cylinder can pass sideways: {d*100:.1f}cm × {cyl_h*100:.1f}cm fits in {gw*100:.1f}×{gh*100:.1f}cm opening"
            return False, f"Cylinder (diameter {d*100:.1f}cm, height {cyl_h*100:.1f}cm) won't fit in {gw*100:.1f}×{gh*100:.1f}cm opening"
        elif obj_type == "box":
            dims = obj_dims  # [x, y, z]
            # Try all 6 orientations (3 axes × 2 for which dim goes through)
            all_faces = [
                (dims[0]*2, dims[1]*2),  # XY face → Z goes through
                (dims[0]*2, dims[2]*2),  # XZ face → Y goes through
                (dims[1]*2, dims[2]*2),  # YZ face → X goes through
            ]
            for fw, fh in all_faces:
                # Can rotate face within the gap
                if (fw < gw and fh < gh) or (fh < gw and fw < gh):
                    return True, f"Box face {fw*100:.1f}×{fh*100:.1f}cm fits through {gw*100:.1f}×{gh*100:.1f}cm opening"
            # Diagonal fitting
            smallest_face = min(all_faces, key=lambda f: f[0]*f[1])
            return False, f"Box smallest face {smallest_face[0]*100:.1f}×{smallest_face[1]*100:.1f}cm too large for {gw*100:.1f}×{gh*100:.1f}cm opening"

    return False, "Cannot determine"


# ─── MJCF Scene Builder ─────────────────────────────────────────────

def _build_fitting_xml(scenario: FittingScenario, table_height: float = 0.35) -> str:
    """Build MJCF XML showing object next to a wall with a gap/opening."""

    obj = FITTING_OBJECTS[scenario.object_type]
    gap = GAP_TYPES[scenario.gap_type]
    color = scenario.object_color

    # Material definitions
    mat_defs = ""
    for cn, rgba in COLORS.items():
        mat_defs += f'    <material name="mat_{cn}" rgba="{rgba}" shininess="0.6" specular="0.5" reflectance="0.1"/>\n'

    # Object body — placed on left side of table
    obj_x = -0.15
    obj_y = 0.0
    geom_type = obj["type"]
    dims = obj["dims"]

    if geom_type == "box":
        obj_z = table_height + 0.02 + dims[2]
        size_str = f"{dims[0]} {dims[1]} {dims[2]}"
    elif geom_type == "cylinder":
        obj_z = table_height + 0.02 + dims[1]
        size_str = f"{dims[0]} {dims[1]}"
    elif geom_type == "sphere":
        obj_z = table_height + 0.02 + dims[0]
        size_str = f"{dims[0]}"
    else:
        obj_z = table_height + 0.1
        size_str = " ".join(str(d) for d in dims)

    object_body = f"""
    <body name="test_object" pos="{obj_x} {obj_y} {obj_z}">
      <geom name="object_geom" type="{geom_type}" size="{size_str}" mass="0.5" material="mat_{color}"/>
    </body>"""

    # Wall with gap — positioned on right side
    wall_x = 0.12
    wall_thickness = 0.008
    wall_height = 0.25
    wall_width = 0.30
    wall_z = table_height + 0.02 + wall_height / 2

    # Build wall as multiple box geoms with a hole in the middle
    wall_geoms = _build_wall_with_gap(gap, wall_x, wall_z, wall_thickness, wall_height, wall_width, table_height)

    xml = f"""<mujoco model="fitting_task">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.001"/>

  <visual>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.01" zfar="50"/>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.75 0.9" rgb2="0.25 0.35 0.55" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.92" rgb2="0.82 0.82 0.82" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.15" shininess="0.1" specular="0.2"/>
    <material name="table_mat" rgba="0.45 0.3 0.18 1" shininess="0.3" specular="0.3" reflectance="0.05"/>
    <material name="wall_mat" rgba="0.7 0.72 0.75 1" shininess="0.2" specular="0.3" reflectance="0.05"/>
{mat_defs}
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid_mat"/>
    <light pos="0.5 -0.8 2.5" dir="-0.15 0.3 -1" diffuse="0.65 0.6 0.55" specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="-0.7 0.3 2.0" dir="0.25 -0.1 -1" diffuse="0.35 0.38 0.45" specular="0.1 0.1 0.1" castshadow="false"/>
    <light pos="0 0.8 1.8" dir="0 -0.4 -1" diffuse="0.2 0.2 0.25" specular="0.05 0.05 0.05" castshadow="false"/>

    <body name="table" pos="0 0 {table_height}">
      <geom name="tabletop" type="box" size="0.5 0.35 0.02" material="table_mat" mass="10"/>
      <geom name="leg1" type="cylinder" fromto="-0.45 -0.3 -{table_height}  -0.45 -0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg2" type="cylinder" fromto="0.45 -0.3 -{table_height}   0.45 -0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg3" type="cylinder" fromto="-0.45  0.3 -{table_height}  -0.45  0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg4" type="cylinder" fromto="0.45  0.3 -{table_height}   0.45  0.3 0" size="0.025" material="table_mat"/>
    </body>

    <!-- Wall with gap -->
{wall_geoms}

    <!-- Object -->
{object_body}

    <!-- Cameras -->
    <camera name="front" pos="0 -0.85 0.55" xyaxes="1 0 0 0 0.4 0.92"/>
    <camera name="top" pos="0 0 1.4" xyaxes="1 0 0 0 1 0"/>
    <camera name="side" pos="0.9 0 0.7" xyaxes="0 1 0 -0.45 0 0.9"/>
    <camera name="angle" pos="0.55 -0.55 0.8" xyaxes="0.7 0.7 0 -0.3 0.3 0.9"/>
  </worldbody>
</mujoco>"""
    return xml


def _build_wall_with_gap(gap: dict, wall_x: float, wall_center_z: float,
                          thickness: float, wall_height: float, wall_width: float,
                          table_height: float) -> str:
    """Build wall geoms with a gap/opening. Returns MJCF body XML."""
    gap_shape = gap["shape"]
    geoms = ""

    base_z = table_height + 0.02  # top of table

    if gap_shape in ("square", "rectangle"):
        gw = gap["width"] / 2   # half-width
        gh = gap["height"] / 2  # half-height
        gap_center_z = base_z + wall_height * 0.45  # gap center slightly below wall center

        # Left panel
        left_w = (wall_width / 2 - gw) / 2
        left_x_off = -(gw + left_w)
        geoms += f'    <geom name="wall_left" type="box" pos="{wall_x} {left_x_off} {base_z + wall_height/2}" size="{thickness} {left_w} {wall_height/2}" material="wall_mat"/>\n'

        # Right panel
        right_x_off = gw + left_w
        geoms += f'    <geom name="wall_right" type="box" pos="{wall_x} {right_x_off} {base_z + wall_height/2}" size="{thickness} {left_w} {wall_height/2}" material="wall_mat"/>\n'

        # Top panel (above gap)
        top_h = (base_z + wall_height) - (gap_center_z + gh)
        if top_h > 0:
            top_z = gap_center_z + gh + top_h / 2
            geoms += f'    <geom name="wall_top" type="box" pos="{wall_x} 0 {top_z}" size="{thickness} {gw} {top_h/2}" material="wall_mat"/>\n'

        # Bottom panel (below gap)
        bottom_h = (gap_center_z - gh) - base_z
        if bottom_h > 0:
            bottom_z = base_z + bottom_h / 2
            geoms += f'    <geom name="wall_bottom" type="box" pos="{wall_x} 0 {bottom_z}" size="{thickness} {gw} {bottom_h/2}" material="wall_mat"/>\n'

    elif gap_shape == "circle":
        # Approximate circle with rectangular panels around it
        r = gap["radius"]
        gap_center_z = base_z + wall_height * 0.45

        # Left panel
        side_w = (wall_width / 2 - r) / 2
        geoms += f'    <geom name="wall_left" type="box" pos="{wall_x} {-(r + side_w)} {base_z + wall_height/2}" size="{thickness} {side_w} {wall_height/2}" material="wall_mat"/>\n'
        geoms += f'    <geom name="wall_right" type="box" pos="{wall_x} {r + side_w} {base_z + wall_height/2}" size="{thickness} {side_w} {wall_height/2}" material="wall_mat"/>\n'

        # Top
        top_h = (base_z + wall_height) - (gap_center_z + r)
        if top_h > 0:
            geoms += f'    <geom name="wall_top" type="box" pos="{wall_x} 0 {gap_center_z + r + top_h/2}" size="{thickness} {r} {top_h/2}" material="wall_mat"/>\n'

        # Bottom
        bottom_h = (gap_center_z - r) - base_z
        if bottom_h > 0:
            geoms += f'    <geom name="wall_bottom" type="box" pos="{wall_x} 0 {base_z + bottom_h/2}" size="{thickness} {r} {bottom_h/2}" material="wall_mat"/>\n'

        # Corner pieces to make it look more circular
        corner_size = r * 0.28
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                cy = dy * (r - corner_size * 0.5)
                cz = gap_center_z + dz * (r - corner_size * 0.5)
                geoms += f'    <geom name="wall_corner_{1 if dy>0 else 0}_{1 if dz>0 else 0}" type="box" pos="{wall_x} {cy} {cz}" size="{thickness+0.001} {corner_size} {corner_size}" material="wall_mat"/>\n'

    return geoms


# ─── Rendering ───────────────────────────────────────────────────────

def _render(model, data, camera: str, width: int = 1024, height: int = 768) -> str:
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


def _render_views(model, data) -> list[str]:
    return [_render(model, data, c) for c in ["front", "angle"]]


def render_scenario(scenario: FittingScenario) -> FittingResult:
    """Render a fitting scenario and return the result."""
    xml = _build_fitting_xml(scenario)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    images = _render_views(model, data)
    return FittingResult(scenario=scenario, images=images)


# ─── 10 Curated Scenarios ───────────────────────────────────────────

def generate_10_scenarios() -> list[FittingScenario]:
    """10 spatial fitting scenarios for review."""
    scenarios = []

    combos = [
        # (name, object, color, gap, difficulty)
        # --- EASY: Obvious fits / obvious no-fits ---
        ("1. Small ball → large circular hole",
         "small_ball", "red", "circle_large", "easy"),

        ("2. Large ball → small circular hole",
         "large_ball", "blue", "circle_small", "easy"),

        ("3. Small cube → large square opening",
         "small_cube", "green", "square_large", "easy"),

        # --- MEDIUM: Requires thinking about dimensions ---
        ("4. Cylinder → wide rectangular slot",
         "cylinder", "orange", "rect_wide", "medium"),

        ("5. Flat plank → narrow slit",
         "flat_plank", "purple", "narrow_slit", "medium"),

        ("6. Tall box → medium square opening",
         "tall_box", "cyan", "rect_medium", "medium"),

        # --- HARD: Requires considering rotation/orientation ---
        ("7. Long box → small square opening",
         "long_box", "yellow", "square_small", "hard"),

        ("8. Wide disc → tall rectangular slot",
         "wide_disc", "red", "rect_tall", "hard"),

        ("9. Thin rod → small circular hole",
         "thin_rod", "green", "circle_small", "hard"),

        ("10. Cube → narrow slit",
         "cube", "blue", "narrow_slit", "hard"),
    ]

    for name, obj_key, color, gap_key, diff in combos:
        fits, reasoning = _can_fit(obj_key, gap_key)
        obj = FITTING_OBJECTS[obj_key]
        gap = GAP_TYPES[gap_key]

        # VLM question: only color + shape, no sizes, no "large"/"small"
        question = f"Can the {color} {obj['vlm_shape']} fit through the {gap['vlm_label']} in the wall?"

        scenarios.append(FittingScenario(
            name=name,
            object_type=obj_key,
            object_color=color,
            gap_type=gap_key,
            fits=fits,
            best_orientation=reasoning,
            question=question,
            difficulty=diff,
            reasoning=reasoning,
        ))

    return scenarios


# ─── HTML Review Page ────────────────────────────────────────────────

def generate_review_html(results: list[FittingResult]) -> str:
    """Generate HTML review page with comment areas for each scenario."""

    cards = ""
    for i, r in enumerate(results):
        s = r.scenario
        gt = "FITS ✅" if s.fits else "DOESN'T FIT ❌"
        gt_class = "fits" if s.fits else "nofits"
        obj = FITTING_OBJECTS[s.object_type]
        gap = GAP_TYPES[s.gap_type]

        imgs = "".join(
            f'<div><img src="{img}"/><span>{["Front","Angle"][j]}</span></div>'
            for j, img in enumerate(r.images)
        )

        cards += f"""
        <div class="card" id="scenario-{i+1}">
            <div class="card-header {gt_class}">
                <div>
                    <h2>{s.name}</h2>
                    <span class="diff-badge {s.difficulty}">{s.difficulty}</span>
                </div>
                <div class="gt-badge">{gt}</div>
            </div>

            <div class="details-row">
                <div class="detail">
                    <span class="detail-label">Object:</span>
                    <span class="detail-value" style="border-color:{_css_color(s.object_color)}">{obj['desc']}</span>
                </div>
                <div class="detail">
                    <span class="detail-label">Gap:</span>
                    <span class="detail-value">{gap['label']}</span>
                </div>
            </div>

            <div class="question">💬 {s.question}</div>

            <div class="gt-answer">
                <strong>Ground Truth:</strong> {s.reasoning}
            </div>

            <div class="img-grid">{imgs}</div>

            <div class="comment-section">
                <label>📝 Review Comments:</label>
                <textarea placeholder="Your feedback on this scenario... (visual quality, question clarity, difficulty, etc.)" rows="3"></textarea>
                <div class="review-btns">
                    <button class="btn-approve" onclick="this.classList.toggle('active')">✅ Approve</button>
                    <button class="btn-reject" onclick="this.classList.toggle('active')">❌ Needs Changes</button>
                </div>
            </div>
        </div>
        """

    n_fits = sum(1 for r in results if r.scenario.fits)
    n_nofits = len(results) - n_fits

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>Spatial Fitting — Scenario Review</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#0f1117; color:#e0e0e0; padding:20px; max-width:1400px; margin:0 auto; }}
  h1 {{ text-align:center; margin:20px 0 5px; font-size:1.8em; }}
  .sub {{ text-align:center; color:#888; margin-bottom:25px; }}
  .stats {{ display:flex; gap:15px; justify-content:center; margin-bottom:30px; }}
  .stat {{ background:#1a1d27; padding:14px 22px; border-radius:10px; text-align:center; }}
  .stat .n {{ font-size:2em; font-weight:bold; }}
  .stat .l {{ color:#888; font-size:0.85em; }}
  .card {{ background:#1a1d27; border-radius:12px; padding:20px; margin-bottom:22px; border:1px solid #2a2d37; }}
  .card-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
  .card-header h2 {{ font-size:1.15em; }}
  .gt-badge {{ font-size:1.05em; font-weight:bold; padding:5px 16px; border-radius:20px; }}
  .fits .gt-badge {{ background:#0d3320; color:#4ade80; }}
  .nofits .gt-badge {{ background:#3b1018; color:#f87171; }}
  .diff-badge {{ font-size:0.7em; padding:2px 8px; border-radius:10px; margin-left:8px; vertical-align:middle; }}
  .diff-badge.easy {{ background:#0d3320; color:#4ade80; }}
  .diff-badge.medium {{ background:#3b2e08; color:#fbbf24; }}
  .diff-badge.hard {{ background:#3b1018; color:#f87171; }}
  .details-row {{ display:flex; gap:20px; margin-bottom:10px; flex-wrap:wrap; }}
  .detail {{ display:flex; align-items:center; gap:6px; }}
  .detail-label {{ color:#888; font-size:0.85em; }}
  .detail-value {{ background:#12141c; padding:4px 12px; border-radius:6px; font-size:0.9em; border-left:3px solid #555; }}
  .question {{ background:#12141c; padding:10px 14px; border-radius:8px; margin-bottom:10px; border-left:3px solid #6366f1; font-style:italic; line-height:1.5; }}
  .gt-answer {{ background:#12141c; padding:10px 14px; border-radius:8px; margin-bottom:15px; border-left:3px solid #f59e0b; line-height:1.5; }}
  .img-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:5px; margin-bottom:15px; }}
  .img-grid div {{ text-align:center; }}
  .img-grid img {{ width:100%; border-radius:5px; border:1px solid #333; }}
  .img-grid span {{ font-size:0.7em; color:#777; }}
  .comment-section {{ background:#12141c; padding:15px; border-radius:8px; border:1px dashed #333; }}
  .comment-section label {{ display:block; margin-bottom:8px; color:#aaa; font-size:0.9em; }}
  .comment-section textarea {{ width:100%; background:#1a1d27; color:#e0e0e0; border:1px solid #333; border-radius:6px; padding:10px; font-family:inherit; font-size:0.9em; resize:vertical; }}
  .comment-section textarea:focus {{ outline:none; border-color:#6366f1; }}
  .review-btns {{ display:flex; gap:10px; margin-top:10px; }}
  .btn-approve, .btn-reject {{ padding:6px 16px; border-radius:8px; border:1px solid #333; background:#1a1d27; color:#e0e0e0; cursor:pointer; font-size:0.9em; transition:all 0.2s; }}
  .btn-approve:hover, .btn-approve.active {{ background:#0d3320; border-color:#4ade80; color:#4ade80; }}
  .btn-reject:hover, .btn-reject.active {{ background:#3b1018; border-color:#f87171; color:#f87171; }}
  .export-btn {{ display:block; margin:30px auto; padding:12px 30px; background:#6366f1; color:white; border:none; border-radius:10px; font-size:1em; cursor:pointer; }}
  .export-btn:hover {{ background:#4f46e5; }}
</style>
</head><body>

<h1>🔲 Spatial Fitting — Scenario Review</h1>
<p class="sub">"Can object A fit through gap B?" — Review each scenario and leave comments.</p>

<div class="stats">
  <div class="stat"><div class="n">{len(results)}</div><div class="l">Scenarios</div></div>
  <div class="stat"><div class="n" style="color:#4ade80">{n_fits}</div><div class="l">Fits ✅</div></div>
  <div class="stat"><div class="n" style="color:#f87171">{n_nofits}</div><div class="l">Doesn't Fit ❌</div></div>
</div>

{cards}

<button class="export-btn" onclick="exportReview()">📋 Export Review Summary</button>

<script>
function exportReview() {{
    const cards = document.querySelectorAll('.card');
    let summary = 'Spatial Fitting Review Summary\\n' + '='.repeat(50) + '\\n\\n';
    cards.forEach((card, i) => {{
        const name = card.querySelector('h2').textContent;
        const textarea = card.querySelector('textarea');
        const approved = card.querySelector('.btn-approve').classList.contains('active');
        const rejected = card.querySelector('.btn-reject').classList.contains('active');
        const status = approved ? '✅ APPROVED' : rejected ? '❌ NEEDS CHANGES' : '⏳ NOT REVIEWED';
        summary += `${{name}}\\nStatus: ${{status}}\\nComments: ${{textarea.value || '(none)'}}\\n\\n`;
    }});
    navigator.clipboard.writeText(summary).then(() => alert('Review copied to clipboard!'));
}}
</script>
</body></html>"""


def _css_color(color_name: str) -> str:
    rgba = COLORS.get(color_name, "0.5 0.5 0.5 1")
    parts = [float(x) for x in rgba.split()]
    return f"rgb({int(parts[0]*255)},{int(parts[1]*255)},{int(parts[2]*255)})"
