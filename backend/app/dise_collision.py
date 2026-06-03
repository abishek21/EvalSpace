"""
DISE Extrinsic-Dynamic: Collision Prediction Tasks

"If object A is pushed, will it hit object B?"

Ground truth is determined ENTIRELY by MuJoCo physics simulation.
We place objects → apply a push force → simulate → check contacts.
No trajectory math. No heuristics. Pure physics.
"""
import io
import base64
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco


# ─── Object Primitives ──────────────────────────────────────────────

COLLISION_OBJECTS = {
    "box":          {"type": "box",      "size": [0.04, 0.04, 0.04], "mass": 0.5,  "label": "box"},
    "large_box":    {"type": "box",      "size": [0.07, 0.07, 0.04], "mass": 1.0,  "label": "large box"},
    "cylinder":     {"type": "cylinder", "size": [0.03, 0.05],       "mass": 0.4,  "label": "cylinder"},
    "sphere":       {"type": "sphere",   "size": [0.035],            "mass": 0.3,  "label": "ball"},
    "small_sphere": {"type": "sphere",   "size": [0.025],            "mass": 0.15, "label": "small ball"},
    "can":          {"type": "cylinder", "size": [0.033, 0.06],      "mass": 0.35, "label": "can"},
    "bottle":       {"type": "cylinder", "size": [0.025, 0.09],      "mass": 0.3,  "label": "bottle"},
    "wide_box":     {"type": "box",      "size": [0.10, 0.05, 0.03], "mass": 0.7,  "label": "wide box"},
    "heavy_block":  {"type": "box",      "size": [0.05, 0.05, 0.05], "mass": 2.0,  "label": "heavy block"},
    "puck":         {"type": "cylinder", "size": [0.04, 0.015],      "mass": 0.25, "label": "puck"},
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
class PlacedObject:
    """An object on the table."""
    obj_type: str       # key in COLLISION_OBJECTS
    color: str          # key in COLORS
    pos_x: float        # position on table (meters from center)
    pos_y: float
    role: str = "obstacle"  # "pushed", "target", or "obstacle"


@dataclass
class PushConfig:
    """How the pushed object is pushed."""
    direction_deg: float    # angle in degrees (0=+x, 90=+y)
    force: float            # impulse magnitude (N·s)


@dataclass
class CollisionScenario:
    """A complete collision prediction scenario."""
    name: str
    objects: list[PlacedObject]
    push: PushConfig
    question: str
    difficulty: str     # easy, medium, hard


@dataclass
class CollisionResult:
    """Result after MuJoCo simulation determines ground truth."""
    scenario: CollisionScenario
    hit_target: bool                # THE ground truth — did pushed object reach target?
    collision_events: list[dict]    # all detected collisions with timestamps
    pushed_trajectory: list[dict]   # [{t, x, y, z}, ...] path of pushed object
    target_moved: bool              # did the target get displaced?
    before_images: list[str]        # renders before push
    after_images: list[str]         # renders after simulation
    frames: list[str]              # animation frames during simulation
    sim_duration: float


# ─── MJCF Builder ────────────────────────────────────────────────────

TABLE_HEIGHT = 0.35
TABLE_SURFACE = TABLE_HEIGHT + 0.02  # top of tabletop


def _build_collision_xml(objects: list[PlacedObject], push_direction_deg: float = 0.0) -> str:
    """Build MJCF XML with objects placed on a table for collision sim."""

    mat_defs = ""
    for color_name, rgba in COLORS.items():
        mat_defs += f'    <material name="mat_{color_name}" rgba="{rgba}" shininess="0.6" specular="0.5" reflectance="0.1"/>\n'

    body_defs = ""
    for i, obj in enumerate(objects):
        preset = COLLISION_OBJECTS[obj.obj_type]
        geom_type = preset["type"]
        size = preset["size"]
        mass = preset["mass"]

        # Object sits on table surface
        if geom_type == "box":
            obj_z = TABLE_SURFACE + size[2]
            size_str = f"{size[0]} {size[1]} {size[2]}"
        elif geom_type == "cylinder":
            obj_z = TABLE_SURFACE + size[1]
            size_str = f"{size[0]} {size[1]}"
        elif geom_type == "sphere":
            obj_z = TABLE_SURFACE + size[0]
            size_str = f"{size[0]}"
        else:
            obj_z = TABLE_SURFACE + 0.04
            size_str = " ".join(str(s) for s in size)

        name = f"obj_{i}_{obj.role}"
        # Lower friction for pushed objects so they slide nicely
        friction = "0.25 0.005 0.001" if obj.role == "pushed" else "0.4 0.005 0.001"

        body_defs += f"""
    <body name="{name}" pos="{obj.pos_x} {obj.pos_y} {obj_z}">
      <freejoint name="{name}_jnt"/>
      <geom name="{name}_geom" type="{geom_type}" size="{size_str}" mass="{mass}"
            material="mat_{obj.color}" friction="{friction}" condim="4"
            solref="-10000 -200"/>
    </body>"""

    # Arrow indicator showing push direction (visual only, no collision)
    pushed = next((o for o in objects if o.role == "pushed"), None)
    arrow_def = ""
    if pushed:
        # Convert push direction to radians for euler rotation around z-axis
        arrow_angle_rad = math.radians(push_direction_deg)
        # Arrow tip: offset along push direction from pushed object
        tip_x = pushed.pos_x + 0.07 * math.cos(arrow_angle_rad)
        tip_y = pushed.pos_y + 0.07 * math.sin(arrow_angle_rad)
        # Build arrow as individual geoms positioned in world space
        # Shaft: cylinder from pushed object center outward along push direction
        shaft_len = 0.05
        shaft_cx = pushed.pos_x + (shaft_len / 2) * math.cos(arrow_angle_rad)
        shaft_cy = pushed.pos_y + (shaft_len / 2) * math.sin(arrow_angle_rad)
        shaft_ex = pushed.pos_x + shaft_len * math.cos(arrow_angle_rad)
        shaft_ey = pushed.pos_y + shaft_len * math.sin(arrow_angle_rad)
        sz = TABLE_SURFACE + 0.002
        # Arrowhead: 3 thin cylinders forming a filled triangle (tip + two sides + base)
        head_len = 0.02
        head_w = 0.012
        tip_x2 = shaft_ex + head_len * math.cos(arrow_angle_rad)
        tip_y2 = shaft_ey + head_len * math.sin(arrow_angle_rad)
        # perpendicular direction
        perp_x = -math.sin(arrow_angle_rad)
        perp_y = math.cos(arrow_angle_rad)
        wing_lx = shaft_ex + perp_x * head_w
        wing_ly = shaft_ey + perp_y * head_w
        wing_rx = shaft_ex - perp_x * head_w
        wing_ry = shaft_ey - perp_y * head_w
        arrow_def = f"""
    <!-- Push direction arrow (visual only) -->
    <geom name="arrow_shaft" type="cylinder" fromto="{pushed.pos_x} {pushed.pos_y} {sz} {shaft_ex} {shaft_ey} {sz}" size="0.003"
          rgba="1 0.15 0.15 0.9" contype="0" conaffinity="0"/>
    <geom name="arrow_head_l" type="cylinder" fromto="{tip_x2} {tip_y2} {sz} {wing_lx} {wing_ly} {sz}" size="0.003"
          rgba="1 0.15 0.15 0.9" contype="0" conaffinity="0"/>
    <geom name="arrow_head_r" type="cylinder" fromto="{tip_x2} {tip_y2} {sz} {wing_rx} {wing_ry} {sz}" size="0.003"
          rgba="1 0.15 0.15 0.9" contype="0" conaffinity="0"/>
    <geom name="arrow_head_b" type="cylinder" fromto="{wing_lx} {wing_ly} {sz} {wing_rx} {wing_ry} {sz}" size="0.003"
          rgba="1 0.15 0.15 0.9" contype="0" conaffinity="0"/>"""

    xml = f"""<mujoco model="collision_task">
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

    <body name="table" pos="0 0 {TABLE_HEIGHT}">
      <geom name="tabletop" type="box" size="0.5 0.35 0.02" material="table_mat" mass="10"
            friction="0.3 0.005 0.001"/>
      <geom name="leg1" type="cylinder" fromto="-0.45 -0.3 -{TABLE_HEIGHT}  -0.45 -0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg2" type="cylinder" fromto="0.45 -0.3 -{TABLE_HEIGHT}   0.45 -0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg3" type="cylinder" fromto="-0.45  0.3 -{TABLE_HEIGHT}  -0.45  0.3 0" size="0.025" material="table_mat"/>
      <geom name="leg4" type="cylinder" fromto="0.45  0.3 -{TABLE_HEIGHT}   0.45  0.3 0" size="0.025" material="table_mat"/>
    </body>
{body_defs}
{arrow_def}

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


def _render_frame(model, data, camera: str = "front", width: int = 800, height: int = 600) -> str:
    """Render a frame for animation sequences (high quality for playback)."""
    return _render(model, data, camera, width, height)


# ─── Core: Simulate Push and Determine Ground Truth ─────────────────

def _get_body_position(model, data, name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
        return [round(float(x), 4) for x in data.xpos[body_id].copy()]
    return [0, 0, 0]


def simulate_collision(
    scenario: CollisionScenario,
    sim_seconds: float = 3.0,
    frame_interval: float = 0.05,
    frame_camera: str = "front",
) -> CollisionResult:
    """
    THE KEY FUNCTION: MuJoCo decides whether the push causes a collision.

    1. Build scene with objects placed on table
    2. Render "before" views (what VLM sees — includes push direction annotation)
    3. Apply impulse to pushed object
    4. Step physics, recording frames + contacts
    5. Check if pushed object ever contacted target
    6. Render "after" views
    7. Return ground truth: hit or miss
    """
    xml = _build_collision_xml(scenario.objects, scenario.push.direction_deg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Find pushed and target objects
    pushed_idx = None
    target_idx = None
    for i, obj in enumerate(scenario.objects):
        if obj.role == "pushed":
            pushed_idx = i
        elif obj.role == "target":
            target_idx = i

    if pushed_idx is None or target_idx is None:
        raise ValueError("Scenario must have exactly one 'pushed' and one 'target' object")

    pushed_name = f"obj_{pushed_idx}_pushed"
    target_name = f"obj_{target_idx}_target"

    # Initialize
    mujoco.mj_forward(model, data)

    # Render BEFORE — what the VLM sees
    before_images = _render_views(model, data)

    # Record initial target position (to check if it moved)
    target_init_pos = _get_body_position(model, data, target_name)

    # ═══════════════════════════════════════════════════════════
    # APPLY THE PUSH — set initial velocity on the pushed object
    # ═══════════════════════════════════════════════════════════
    pushed_jnt_name = f"{pushed_name}_jnt"
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pushed_jnt_name)
    if jnt_id < 0:
        raise ValueError(f"Joint {pushed_jnt_name} not found")

    dof_adr = model.jnt_dofadr[jnt_id]

    # Convert push direction + force to velocity
    push = scenario.push
    angle_rad = math.radians(push.direction_deg)
    pushed_mass = COLLISION_OBJECTS[scenario.objects[pushed_idx].obj_type]["mass"]
    # v = impulse / mass
    vx = (push.force / pushed_mass) * math.cos(angle_rad)
    vy = (push.force / pushed_mass) * math.sin(angle_rad)

    data.qvel[dof_adr + 0] = vx     # x velocity
    data.qvel[dof_adr + 1] = vy     # y velocity
    data.qvel[dof_adr + 2] = 0.0    # no vertical velocity

    # ═══════════════════════════════════════════════════════════
    # SIMULATE — MuJoCo decides what happens
    # ═══════════════════════════════════════════════════════════
    n_steps = int(sim_seconds / model.opt.timestep)
    frame_step_interval = int(frame_interval / model.opt.timestep)

    pushed_geom = f"{pushed_name}_geom"
    target_geom = f"{target_name}_geom"

    collision_events = []
    pushed_trajectory = []
    frames = []
    hit_target = False

    for step in range(n_steps):
        mujoco.mj_step(model, data)

        # Check contacts every step
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if g1 and g2:
                pair = {g1, g2}
                # Check if pushed hit target
                if pushed_geom in pair and target_geom in pair:
                    hit_target = True
                    t = round(step * model.opt.timestep, 4)
                    # Avoid duplicate events within 0.05s
                    if not collision_events or (t - collision_events[-1]["time"]) > 0.05:
                        collision_events.append({
                            "time": t,
                            "type": "pushed_hit_target",
                            "position": [round(float(x), 4) for x in contact.pos],
                        })

                # Also record any other interesting collisions (pushed hitting obstacles)
                if pushed_geom in pair and target_geom not in pair:
                    other = g1 if g2 == pushed_geom else g2
                    if other not in ("tabletop", "floor"):
                        t = round(step * model.opt.timestep, 4)
                        if not any(e.get("other") == other and abs(e["time"] - t) < 0.1
                                   for e in collision_events):
                            collision_events.append({
                                "time": t,
                                "type": "pushed_hit_obstacle",
                                "other": other,
                                "position": [round(float(x), 4) for x in contact.pos],
                            })

        # Record trajectory + frames at intervals
        if step % frame_step_interval == 0:
            pos = _get_body_position(model, data, pushed_name)
            t = round(step * model.opt.timestep, 3)
            pushed_trajectory.append({"t": t, "x": pos[0], "y": pos[1], "z": pos[2]})

            # Record animation frame (limit to 60 frames for smooth playback)
            if len(frames) < 60:
                try:
                    frames.append(_render_frame(model, data, frame_camera))
                except Exception:
                    pass

    # Render AFTER
    after_images = _render_views(model, data)

    # Check if target moved
    target_final_pos = _get_body_position(model, data, target_name)
    target_disp = math.sqrt(
        (target_final_pos[0] - target_init_pos[0])**2 +
        (target_final_pos[1] - target_init_pos[1])**2
    )
    target_moved = target_disp > 0.02  # moved more than 2cm

    return CollisionResult(
        scenario=scenario,
        hit_target=hit_target,
        collision_events=collision_events,
        pushed_trajectory=pushed_trajectory,
        target_moved=target_moved,
        before_images=before_images,
        after_images=after_images,
        frames=frames,
        sim_duration=sim_seconds,
    )


# ─── Scenario Generators ────────────────────────────────────────────

def _random_scenario(idx: int) -> CollisionScenario:
    """Generate a random collision prediction scenario."""
    obj_types = list(COLLISION_OBJECTS.keys())
    color_names = list(COLORS.keys())

    # Pick pushed and target objects
    pushed_type = random.choice(obj_types)
    target_type = random.choice(obj_types)
    pushed_color = random.choice(color_names)
    target_color = random.choice([c for c in color_names if c != pushed_color])

    # Place pushed object — left side of table
    pushed_x = random.uniform(-0.38, -0.18)
    pushed_y = random.uniform(-0.12, 0.12)

    # Place target — right side, at varying distances
    distance = random.uniform(0.25, 0.55)
    target_angle_offset = random.uniform(-30, 30)  # degrees off from push direction

    # Push direction: generally toward right (+x) with some angle
    push_dir = random.uniform(-20, 20)  # degrees from +x axis
    push_force = random.uniform(0.15, 0.6)

    # Target position: along push direction + some offset
    effective_angle = math.radians(push_dir + target_angle_offset)
    target_x = pushed_x + distance * math.cos(effective_angle)
    target_y = pushed_y + distance * math.sin(effective_angle)

    # Clamp to table bounds (with padding so objects aren't on edge)
    target_x = max(-0.35, min(0.35, target_x))
    target_y = max(-0.20, min(0.20, target_y))

    objects = [
        PlacedObject(pushed_type, pushed_color, pushed_x, pushed_y, "pushed"),
        PlacedObject(target_type, target_color, target_x, target_y, "target"),
    ]

    # Add obstacles (0-2) for medium/hard scenarios
    n_obstacles = random.choices([0, 1, 2], weights=[0.3, 0.45, 0.25])[0]
    used_colors = {pushed_color, target_color}
    for obs_i in range(n_obstacles):
        obs_type = random.choice(obj_types)
        obs_color = random.choice([c for c in color_names if c not in used_colors])
        used_colors.add(obs_color)

        # Place obstacle clearly between pushed and target
        t = random.uniform(0.3, 0.7)
        obs_x = pushed_x + t * (target_x - pushed_x) + random.uniform(-0.05, 0.05)
        obs_y = pushed_y + t * (target_y - pushed_y) + random.uniform(-0.05, 0.05)
        obs_x = max(-0.35, min(0.35, obs_x))
        obs_y = max(-0.20, min(0.20, obs_y))

        objects.append(PlacedObject(obs_type, obs_color, obs_x, obs_y, "obstacle"))

    # Difficulty
    if n_obstacles == 0 and abs(target_angle_offset) < 10:
        difficulty = "easy"
    elif n_obstacles >= 2 or abs(target_angle_offset) > 20:
        difficulty = "hard"
    else:
        difficulty = "medium"

    # Build question with qualitative descriptions (no raw physics params)
    pushed_label = f"{pushed_color} {COLLISION_OBJECTS[pushed_type]['label']}"
    target_label = f"{target_color} {COLLISION_OBJECTS[target_type]['label']}"
    pushed_mass = COLLISION_OBJECTS[pushed_type]["mass"]

    # Qualitative force
    velocity = push_force / pushed_mass
    if velocity < 0.5:
        force_phrase = "given a very gentle tap"
    elif velocity < 1.0:
        force_phrase = "lightly nudged"
    elif velocity < 1.5:
        force_phrase = "pushed"
    elif velocity < 2.5:
        force_phrase = "firmly pushed"
    else:
        force_phrase = "shoved hard"

    # Qualitative direction
    if abs(push_dir) < 5:
        dir_phrase = "straight across the table"
    elif push_dir > 0:
        dir_phrase = "across the table at a slight upward angle"
    else:
        dir_phrase = "across the table at a slight downward angle"

    # Qualitative distance
    if distance < 0.15:
        dist_phrase = "which is sitting very close by"
    elif distance < 0.25:
        dist_phrase = "which is a short distance away"
    elif distance < 0.40:
        dist_phrase = "which is across the table"
    else:
        dist_phrase = "which is on the far side of the table"

    obstacle_desc = ""
    if n_obstacles > 0:
        obs_labels = [
            f"{o.color} {COLLISION_OBJECTS[o.obj_type]['label']}"
            for o in objects if o.role == "obstacle"
        ]
        obstacle_desc = f" There {'is' if len(obs_labels) == 1 else 'are'} {', '.join(obs_labels)} on the table between them."

    question = (
        f"A {pushed_label} is {force_phrase} {dir_phrase} toward a {target_label}, "
        f"{dist_phrase}.{obstacle_desc} "
        f"Will the {pushed_label} hit the {target_label}?"
    )

    return CollisionScenario(
        name=f"collision_{idx}",
        objects=objects,
        push=PushConfig(direction_deg=push_dir, force=push_force),
        question=question,
        difficulty=difficulty,
    )


def generate_10_scenarios() -> list[CollisionScenario]:
    """10 curated collision scenarios from obvious hit to clear miss."""
    return [
        # --- EASY HIT ---
        CollisionScenario(
            name="1. Direct hit (close)",
            objects=[
                PlacedObject("puck", "red", -0.25, 0.0, "pushed"),
                PlacedObject("box", "blue", 0.15, 0.0, "target"),
            ],
            push=PushConfig(direction_deg=0, force=0.3),
            question="A red puck is pushed directly toward a blue box that is close by. Will the puck hit the box?",
            difficulty="easy",
        ),

        CollisionScenario(
            name="2. Direct hit (far)",
            objects=[
                PlacedObject("sphere", "green", -0.35, 0.0, "pushed"),
                PlacedObject("can", "yellow", 0.30, 0.0, "target"),
            ],
            push=PushConfig(direction_deg=0, force=0.5),
            question="A green ball is pushed hard toward a yellow can across the full length of the table. Will it reach the can?",
            difficulty="easy",
        ),

        # --- EASY MISS ---
        CollisionScenario(
            name="3. Clear miss (perpendicular)",
            objects=[
                PlacedObject("box", "red", -0.30, -0.15, "pushed"),
                PlacedObject("cylinder", "blue", 0.25, 0.20, "target"),
            ],
            push=PushConfig(direction_deg=0, force=0.4),
            question="A red box is pushed straight to the right. A blue cylinder sits in the far corner. Will the box hit the cylinder?",
            difficulty="easy",
        ),

        CollisionScenario(
            name="4. Weak push (runs out of momentum)",
            objects=[
                PlacedObject("heavy_block", "purple", -0.35, 0.0, "pushed"),
                PlacedObject("can", "orange", 0.30, 0.0, "target"),
            ],
            push=PushConfig(direction_deg=0, force=0.1),
            question="A heavy purple block is given a gentle push toward an orange can on the far side of the table. Will it reach?",
            difficulty="easy",
        ),

        # --- MEDIUM ---
        CollisionScenario(
            name="5. Slight angle (near miss?)",
            objects=[
                PlacedObject("puck", "red", -0.30, 0.0, "pushed"),
                PlacedObject("box", "green", 0.25, 0.08, "target"),
            ],
            push=PushConfig(direction_deg=5, force=0.35),
            question="A red puck is pushed slightly upward to the right. A green box is offset slightly above the line of push. Will the puck hit or miss the box?",
            difficulty="medium",
        ),

        CollisionScenario(
            name="6. Obstacle blocks the path",
            objects=[
                PlacedObject("sphere", "red", -0.35, 0.0, "pushed"),
                PlacedObject("box", "blue", 0.30, 0.0, "target"),
                PlacedObject("large_box", "yellow", -0.02, 0.0, "obstacle"),
            ],
            push=PushConfig(direction_deg=0, force=0.4),
            question="A red ball is pushed toward a blue box, but a large yellow box sits directly in between. Will the red ball reach the blue box?",
            difficulty="medium",
        ),

        CollisionScenario(
            name="7. Obstacle deflection",
            objects=[
                PlacedObject("puck", "cyan", -0.32, 0.0, "pushed"),
                PlacedObject("can", "orange", 0.30, 0.12, "target"),
                PlacedObject("cylinder", "white", -0.02, 0.03, "obstacle"),
            ],
            push=PushConfig(direction_deg=2, force=0.4),
            question="A cyan puck is pushed to the right. A white cylinder is in the path, and an orange can is off to the side. Could the puck deflect off the cylinder and hit the can?",
            difficulty="medium",
        ),

        # --- HARD ---
        CollisionScenario(
            name="8. Chain collision (A→B→C?)",
            objects=[
                PlacedObject("sphere", "red", -0.35, 0.0, "pushed"),
                PlacedObject("box", "green", -0.05, 0.0, "obstacle"),
                PlacedObject("can", "blue", 0.30, 0.0, "target"),
            ],
            push=PushConfig(direction_deg=0, force=0.5),
            question="A red ball is pushed toward a green box. Behind the green box is a blue can. If the ball hits the green box, will the green box then hit the blue can?",
            difficulty="hard",
        ),

        CollisionScenario(
            name="9. Narrow gap",
            objects=[
                PlacedObject("small_sphere", "red", -0.32, 0.0, "pushed"),
                PlacedObject("can", "blue", 0.32, 0.0, "target"),
                PlacedObject("box", "yellow", 0.0, -0.07, "obstacle"),
                PlacedObject("box", "green", 0.0, 0.07, "obstacle"),
            ],
            push=PushConfig(direction_deg=0, force=0.35),
            question="A small red ball is pushed toward a blue can. Two boxes form a narrow gap in between. Will the ball pass through the gap and hit the can?",
            difficulty="hard",
        ),

        CollisionScenario(
            name="10. Mass mismatch",
            objects=[
                PlacedObject("small_sphere", "red", -0.30, 0.0, "pushed"),
                PlacedObject("heavy_block", "purple", 0.0, 0.0, "obstacle"),
                PlacedObject("can", "blue", 0.28, 0.0, "target"),
            ],
            push=PushConfig(direction_deg=0, force=0.3),
            question="A tiny red ball is pushed into a heavy purple block. Behind the block is a blue can. Will the ball's impact move the heavy block enough to hit the can?",
            difficulty="hard",
        ),
    ]
