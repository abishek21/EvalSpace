"""
DISE Extrinsic-Dynamic: Stacking Stability Tasks

Ground truth is determined ENTIRELY by MuJoCo physics simulation.
We build a stack → simulate → measure if objects toppled → that IS the answer.

No math shortcuts. No lookup tables. Pure physics.
"""
import io
import base64
import math
import random
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import mujoco


# ─── Stacking Primitives ────────────────────────────────────────────

STACKABLE_OBJECTS = {
    "large_box":    {"type": "box",      "size": [0.08, 0.08, 0.04], "mass": 1.0,  "label": "large box"},
    "small_box":    {"type": "box",      "size": [0.04, 0.04, 0.04], "mass": 0.3,  "label": "small box"},
    "flat_plate":   {"type": "cylinder", "size": [0.08, 0.008],      "mass": 0.4,  "label": "flat plate"},
    "tall_cylinder":{"type": "cylinder", "size": [0.025, 0.07],      "mass": 0.3,  "label": "tall cylinder"},
    "sphere":       {"type": "sphere",   "size": [0.04],             "mass": 0.2,  "label": "ball"},
    "small_sphere": {"type": "sphere",   "size": [0.025],            "mass": 0.1,  "label": "small ball"},
    "wide_box":     {"type": "box",      "size": [0.12, 0.06, 0.03], "mass": 0.8,  "label": "wide box"},
    "tiny_box":     {"type": "box",      "size": [0.02, 0.02, 0.02], "mass": 0.1,  "label": "tiny box"},
    "bowl":         {"type": "ellipsoid","size": [0.06, 0.06, 0.025],"mass": 0.3,  "label": "bowl"},
    "book":         {"type": "box",      "size": [0.10, 0.07, 0.012],"mass": 0.5,  "label": "book"},
}

COLORS = {
    "red":    "0.85 0.15 0.15 1",
    "blue":   "0.15 0.25 0.85 1",
    "green":  "0.15 0.75 0.25 1",
    "yellow": "0.9 0.8 0.1 1",
    "orange": "0.9 0.45 0.1 1",
    "purple": "0.6 0.15 0.8 1",
    "cyan":   "0.1 0.75 0.8 1",
    "pink":   "0.9 0.4 0.6 1",
}


@dataclass
class StackObject:
    """One object in a stack."""
    obj_type: str        # key in STACKABLE_OBJECTS
    color: str           # key in COLORS
    offset_x: float = 0  # lateral offset from center (makes it unstable)
    offset_y: float = 0


@dataclass
class StackingScenario:
    """A complete stacking scenario to test."""
    name: str
    objects: list[StackObject]  # bottom to top order
    question: str
    difficulty: str  # easy, medium, hard


@dataclass
class StackingResult:
    """Result after MuJoCo simulation determines ground truth."""
    scenario: StackingScenario
    stable: bool                    # THE ground truth — did it stay stacked?
    initial_positions: dict         # {name: [x,y,z]} before sim
    final_positions: dict           # {name: [x,y,z]} after sim
    max_displacement: float         # largest lateral movement of any object
    fell_objects: list[str]         # which objects fell off
    before_images: list[str]       # renders of initial setup
    after_images: list[str]        # renders after simulation
    settle_time: float             # how long we simulated


# ─── MJCF Builder ────────────────────────────────────────────────────

def _build_stacking_xml(objects: list[StackObject], table_height: float = 0.35) -> str:
    """Build MJCF XML with objects stacked on a table."""
    
    # Compute material definitions
    mat_defs = ""
    for color_name, rgba in COLORS.items():
        mat_defs += f'    <material name="mat_{color_name}" rgba="{rgba}" shininess="0.6" specular="0.5" reflectance="0.1"/>\n'

    # Compute object bodies — stack them bottom to top
    body_defs = ""
    current_z = table_height + 0.02  # start just above tabletop

    for i, obj in enumerate(objects):
        preset = STACKABLE_OBJECTS[obj.obj_type]
        geom_type = preset["type"]
        size = preset["size"]
        mass = preset["mass"]
        label = preset["label"]

        # Compute the height this object adds
        if geom_type == "box":
            half_h = size[2]
            obj_z = current_z + half_h
            size_str = f"{size[0]} {size[1]} {size[2]}"
        elif geom_type == "cylinder":
            half_h = size[1]
            obj_z = current_z + half_h
            size_str = f"{size[0]} {size[1]}"
        elif geom_type == "sphere":
            half_h = size[0]
            obj_z = current_z + half_h
            size_str = f"{size[0]}"
        elif geom_type == "ellipsoid":
            half_h = size[2]
            obj_z = current_z + half_h
            size_str = f"{size[0]} {size[1]} {size[2]}"
        else:
            half_h = 0.04
            obj_z = current_z + half_h
            size_str = " ".join(str(s) for s in size)

        obj_x = obj.offset_x
        obj_y = obj.offset_y
        name = f"obj_{i}_{obj.obj_type}"

        body_defs += f"""
    <body name="{name}" pos="{obj_x} {obj_y} {obj_z}">
      <freejoint name="{name}_jnt"/>
      <geom name="{name}_geom" type="{geom_type}" size="{size_str}" mass="{mass}" material="mat_{obj.color}"
            friction="0.4 0.005 0.001" condim="4" solref="-10000 -200"/>
    </body>"""

        # Next object starts on top of this one
        current_z = obj_z + half_h

    xml = f"""<mujoco model="stacking_task">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.001" integrator="implicit">
    <flag warmstart="disable"/>
  </option>

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
{mat_defs}
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid_mat"/>
    <!-- Key light (warm, from front-right) -->
    <light pos="0.5 -0.8 2.5" dir="-0.15 0.3 -1" diffuse="0.65 0.6 0.55" specular="0.3 0.3 0.3" castshadow="true"/>
    <!-- Fill light (cool, from left) -->
    <light pos="-0.7 0.3 2.0" dir="0.25 -0.1 -1" diffuse="0.35 0.38 0.45" specular="0.1 0.1 0.1" castshadow="false"/>
    <!-- Rim/back light (subtle, for depth) -->
    <light pos="0 0.8 1.8" dir="0 -0.4 -1" diffuse="0.2 0.2 0.25" specular="0.05 0.05 0.05" castshadow="false"/>

    <body name="table" pos="0 0 {table_height}">
      <geom name="tabletop" type="box" size="0.5 0.35 0.02" material="table_mat" mass="10"
            friction="0.4 0.005 0.001"/>
      <geom name="leg1" type="cylinder" fromto="-0.45 -0.3 -{table_height}  -0.45 -0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg2" type="cylinder" fromto="0.45 -0.3 -{table_height}   0.45 -0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg3" type="cylinder" fromto="-0.45  0.3 -{table_height}  -0.45  0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg4" type="cylinder" fromto="0.45  0.3 -{table_height}   0.45  0.3 0" size="0.025" material="table_mat"/>
    </body>
{body_defs}

    <!-- Cameras -->
    <camera name="front" pos="0 -0.85 0.55" xyaxes="1 0 0 0 0.4 0.92"/>
    <camera name="top" pos="0 0 1.4" xyaxes="1 0 0 0 1 0"/>
    <camera name="side" pos="0.9 0 0.7" xyaxes="0 1 0 -0.45 0 0.9"/>
    <camera name="angle" pos="0.55 -0.55 0.8" xyaxes="0.7 0.7 0 -0.3 0.3 0.9"/>
  </worldbody>
</mujoco>"""
    return xml


# ─── Rendering ───────────────────────────────────────────────────────

def _render(model, data, camera: str, width: int = 1024, height: int = 768) -> str:
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
    return [_render(model, data, c) for c in ["front", "angle", "top", "side"]]


# ─── Core: Simulate and Determine Ground Truth ──────────────────────

def _get_object_positions(model, data) -> dict:
    """Read position of every stacked object from MuJoCo state."""
    positions = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("obj_"):
            pos = data.xpos[i].copy()
            positions[name] = [round(float(x), 4) for x in pos]
    return positions


# Threshold: if an object moves more than this laterally, it "fell"
FALL_THRESHOLD_LATERAL = 0.05   # 5cm lateral displacement
FALL_THRESHOLD_Z_DROP  = 0.04   # 4cm drop from initial height


def simulate_stacking(scenario: StackingScenario, settle_seconds: float = 3.0) -> StackingResult:
    """
    THE KEY FUNCTION: MuJoCo decides the ground truth.

    1. Build scene with objects in stacked position
    2. Render "before" views (what VLM would see)
    3. Step physics for settle_seconds
    4. Check if objects stayed put or toppled
    5. Render "after" views
    6. Return ground truth: stable or unstable
    """
    xml = _build_stacking_xml(scenario.objects)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Initialize — just forward kinematics, no stepping yet
    mujoco.mj_forward(model, data)

    # Record initial positions
    initial_positions = _get_object_positions(model, data)

    # Render BEFORE — this is what the VLM will see and judge
    before_images = _render_views(model, data)

    # ═══════════════════════════════════════════════════════════
    # THIS IS WHERE MUJOCO DECIDES GROUND TRUTH
    # We step the physics engine and see what happens.
    # No math. No heuristics. Pure simulation.
    # ═══════════════════════════════════════════════════════════
    n_steps = int(settle_seconds / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    # Read final positions
    final_positions = _get_object_positions(model, data)

    # Render AFTER — shows what actually happened
    after_images = _render_views(model, data)

    # Determine which objects fell
    fell_objects = []
    max_displacement = 0.0

    for name in initial_positions:
        if name not in final_positions:
            continue
        init = np.array(initial_positions[name])
        final = np.array(final_positions[name])

        lateral_disp = np.sqrt((final[0] - init[0])**2 + (final[1] - init[1])**2)
        z_drop = init[2] - final[2]  # positive = dropped

        max_displacement = max(max_displacement, lateral_disp)

        if lateral_disp > FALL_THRESHOLD_LATERAL or z_drop > FALL_THRESHOLD_Z_DROP:
            fell_objects.append(name)

    # GROUND TRUTH: stable if nothing fell
    stable = len(fell_objects) == 0

    return StackingResult(
        scenario=scenario,
        stable=stable,
        initial_positions=initial_positions,
        final_positions=final_positions,
        max_displacement=round(max_displacement, 4),
        fell_objects=fell_objects,
        before_images=before_images,
        after_images=after_images,
        settle_time=settle_seconds,
    )


# ─── 10 Curated Scenarios ───────────────────────────────────────────

def generate_10_scenarios() -> list[StackingScenario]:
    """10 stacking scenarios ranging from obviously stable to clearly unstable."""
    return [
        # --- EASY STABLE ---
        StackingScenario(
            name="1. Large box on table",
            objects=[StackObject("large_box", "blue")],
            question="A blue box sits flat on the table. Is it stable?",
            difficulty="easy",
        ),

        StackingScenario(
            name="2. Small box on large box (centered)",
            objects=[
                StackObject("large_box", "blue"),
                StackObject("small_box", "red"),
            ],
            question="A red small box is placed centered on top of a blue large box. Will the stack remain stable?",
            difficulty="easy",
        ),

        StackingScenario(
            name="3. Book + small box (centered)",
            objects=[
                StackObject("book", "green"),
                StackObject("small_box", "yellow"),
            ],
            question="A yellow small box sits centered on a green book. Is this stack stable?",
            difficulty="easy",
        ),

        # --- MEDIUM ---
        StackingScenario(
            name="4. Three-high centered stack",
            objects=[
                StackObject("large_box", "blue"),
                StackObject("wide_box", "green"),
                StackObject("small_box", "red"),
            ],
            question="Three objects are stacked: blue large box (bottom), green wide box (middle), red small box (top). All centered. Will this stack stay upright?",
            difficulty="medium",
        ),

        StackingScenario(
            name="5. Box on cylinder (centered)",
            objects=[
                StackObject("tall_cylinder", "purple"),
                StackObject("small_box", "orange"),
            ],
            question="An orange small box is placed on top of a purple tall cylinder. Will it stay balanced?",
            difficulty="medium",
        ),

        StackingScenario(
            name="6. Slightly offset stack",
            objects=[
                StackObject("large_box", "blue"),
                StackObject("small_box", "red", offset_x=0.03),
            ],
            question="A red small box is placed slightly off-center (3cm to the right) on a blue large box. Will it stay or fall?",
            difficulty="medium",
        ),

        # --- UNSTABLE ---
        StackingScenario(
            name="7. Ball on flat surface",
            objects=[
                StackObject("book", "cyan"),
                StackObject("sphere", "red", offset_x=0.07),
            ],
            question="A red ball is placed near the edge of a cyan book. Will the ball stay or roll off?",
            difficulty="medium",
        ),

        StackingScenario(
            name="8. Box hanging off edge",
            objects=[
                StackObject("small_box", "blue"),
                StackObject("small_box", "red", offset_x=0.06),
            ],
            question="A red small box is placed so it hangs mostly off the right edge of a blue small box. Will it stay or fall?",
            difficulty="easy",
        ),

        StackingScenario(
            name="9. Ball on ball",
            objects=[
                StackObject("sphere", "blue"),
                StackObject("small_sphere", "red", offset_x=0.01),
            ],
            question="A small red ball is placed slightly off-center on top of a larger blue ball. Will the small ball stay on top?",
            difficulty="easy",
        ),

        StackingScenario(
            name="10. Tall wobbly tower",
            objects=[
                StackObject("tall_cylinder", "blue"),
                StackObject("tiny_box", "green"),
                StackObject("tall_cylinder", "red", offset_x=0.02),
                StackObject("small_sphere", "yellow", offset_x=0.01),
            ],
            question="A tower: blue tall cylinder, green tiny box, red tall cylinder (offset 2cm right), yellow small ball (offset 1cm right). Will it stay standing?",
            difficulty="hard",
        ),
    ]


# ─── HTML Report ─────────────────────────────────────────────────────

def generate_stacking_demo_html(results: list[StackingResult]) -> str:
    """Generate interactive HTML showing all stacking results."""

    cards = ""
    for i, r in enumerate(results):
        s = r.scenario
        gt = "STABLE ✅" if r.stable else "UNSTABLE — TOPPLED ❌"
        gt_class = "stable" if r.stable else "unstable"

        # Object description
        obj_desc = " → ".join(
            f'<span class="obj-tag" style="border-color:{_css_color(o.color)}">'
            f'{STACKABLE_OBJECTS[o.obj_type]["label"]} ({o.color})'
            f'{"" if o.offset_x == 0 and o.offset_y == 0 else f" [offset: {o.offset_x*100:.0f}cm, {o.offset_y*100:.0f}cm]"}'
            f'</span>'
            for o in s.objects
        )

        before_imgs = "".join(
            f'<div><img src="{img}"/><span>{["Front","Angle","Top","Side"][j]}</span></div>'
            for j, img in enumerate(r.before_images)
        )
        after_imgs = "".join(
            f'<div><img src="{img}"/><span>{["Front","Angle","Top","Side"][j]}</span></div>'
            for j, img in enumerate(r.after_images)
        )

        # Position comparison
        pos_rows = ""
        for name in r.initial_positions:
            init = r.initial_positions[name]
            final = r.final_positions.get(name, [0,0,0])
            dx = abs(final[0] - init[0])
            dy = abs(final[1] - init[1])
            dz = init[2] - final[2]
            fell = name in r.fell_objects
            row_class = ' class="fell"' if fell else ""
            label = name.split("_", 2)[-1] if name.count("_") >= 2 else name
            pos_rows += (
                f'<tr{row_class}>'
                f'<td>{label}</td>'
                f'<td>[{init[0]:.3f}, {init[1]:.3f}, {init[2]:.3f}]</td>'
                f'<td>[{final[0]:.3f}, {final[1]:.3f}, {final[2]:.3f}]</td>'
                f'<td>{dx*100:.1f}cm</td><td>{dy*100:.1f}cm</td><td>{dz*100:.1f}cm</td>'
                f'<td>{"💥 FELL" if fell else "✓"}</td>'
                f'</tr>'
            )

        cards += f"""
        <div class="card">
            <div class="card-header {gt_class}">
                <div>
                    <h2>{s.name}</h2>
                    <span class="diff-badge {s.difficulty}">{s.difficulty}</span>
                </div>
                <div class="gt-badge">{gt}</div>
            </div>
            <div class="stack-viz">{obj_desc}</div>
            <div class="question">💬 {s.question}</div>
            <div class="gt-answer">
                <strong>MuJoCo Ground Truth:</strong>
                {"The stack is <b>stable</b>. All objects remained in place." if r.stable
                 else f"The stack is <b>unstable</b>. Object(s) fell: {', '.join(n.split('_',2)[-1] for n in r.fell_objects)}. Max displacement: {r.max_displacement*100:.1f}cm."}
            </div>

            <div class="views-row">
                <div class="views-col">
                    <h3>🔵 Initial Setup (VLM sees this)</h3>
                    <div class="img-grid">{before_imgs}</div>
                </div>
                <div class="views-col">
                    <h3>🔴 After {r.settle_time}s Simulation</h3>
                    <div class="img-grid">{after_imgs}</div>
                </div>
            </div>

            <details>
                <summary>📊 Position Data</summary>
                <table>
                    <tr><th>Object</th><th>Initial Pos</th><th>Final Pos</th><th>ΔX</th><th>ΔY</th><th>ΔZ (drop)</th><th>Status</th></tr>
                    {pos_rows}
                </table>
            </details>
        </div>
        """

    n_stable = sum(1 for r in results if r.stable)
    n_unstable = len(results) - n_stable

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>DISE E-D: Stacking Stability</title>
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
  .card-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
  .card-header h2 {{ font-size:1.15em; }}
  .gt-badge {{ font-size:1.05em; font-weight:bold; padding:5px 16px; border-radius:20px; }}
  .stable .gt-badge {{ background:#0d3320; color:#4ade80; }}
  .unstable .gt-badge {{ background:#3b1018; color:#f87171; }}
  .diff-badge {{ font-size:0.7em; padding:2px 8px; border-radius:10px; margin-left:8px; vertical-align:middle; }}
  .diff-badge.easy {{ background:#0d3320; color:#4ade80; }}
  .diff-badge.medium {{ background:#3b2e08; color:#fbbf24; }}
  .diff-badge.hard {{ background:#3b1018; color:#f87171; }}
  .stack-viz {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:10px; }}
  .obj-tag {{ background:#12141c; padding:4px 10px; border-radius:6px; font-size:0.85em; border-left:3px solid; }}
  .question {{ background:#12141c; padding:10px 14px; border-radius:8px; margin-bottom:10px; border-left:3px solid #6366f1; font-style:italic; line-height:1.5; }}
  .gt-answer {{ background:#12141c; padding:10px 14px; border-radius:8px; margin-bottom:15px; border-left:3px solid #f59e0b; }}
  .views-row {{ display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-bottom:10px; }}
  .views-col h3 {{ font-size:0.9em; color:#aaa; margin-bottom:6px; }}
  .img-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:5px; }}
  .img-grid div {{ text-align:center; }}
  .img-grid img {{ width:100%; border-radius:5px; border:1px solid #333; }}
  .img-grid span {{ font-size:0.7em; color:#777; }}
  details {{ margin-top:8px; }}
  summary {{ cursor:pointer; color:#6366f1; font-size:0.9em; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:0.8em; }}
  th,td {{ padding:5px 8px; border:1px solid #333; }}
  th {{ background:#12141c; }}
  .fell {{ background:#2a1015 !important; }}
  .how {{ background:#1a1d27; border-radius:12px; padding:20px; margin-bottom:25px; border:1px solid #2a2d37; }}
  .how h2 {{ margin-bottom:10px; }}
  .how ol {{ padding-left:20px; line-height:1.8; }}
  .how code {{ background:#12141c; padding:2px 6px; border-radius:4px; font-size:0.9em; }}
</style>
</head><body>
<h1>📦 DISE Extrinsic-Dynamic: Stacking Stability</h1>
<p class="sub">MuJoCo physics simulation determines ground truth. No math shortcuts.</p>

<div class="how">
  <h2>How Ground Truth Works</h2>
  <ol>
    <li>Objects are placed in stacked configuration in MuJoCo</li>
    <li><b>Before</b> images are rendered — this is what a VLM would see</li>
    <li>Physics simulation runs for 3 seconds (<code>mj_step()</code> × 3000)</li>
    <li>Final positions are compared to initial — did anything move &gt;5cm laterally or drop &gt;4cm?</li>
    <li><b>After</b> images show what actually happened</li>
    <li><b>Ground truth = did objects stay or topple?</b> Pure physics, no heuristics.</li>
  </ol>
</div>

<div class="stats">
  <div class="stat"><div class="n">{len(results)}</div><div class="l">Scenarios</div></div>
  <div class="stat"><div class="n" style="color:#4ade80">{n_stable}</div><div class="l">Stable ✅</div></div>
  <div class="stat"><div class="n" style="color:#f87171">{n_unstable}</div><div class="l">Unstable ❌</div></div>
</div>

{cards}
</body></html>"""


def _css_color(color_name: str) -> str:
    rgba = COLORS.get(color_name, "0.5 0.5 0.5 1")
    parts = [float(x) for x in rgba.split()]
    return f"rgb({int(parts[0]*255)},{int(parts[1]*255)},{int(parts[2]*255)})"
