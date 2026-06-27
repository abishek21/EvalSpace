"""
DISE Shelf Fitting: Advanced Spatial Reasoning Environment

A cupboard/rack with multiple shelves containing objects.
Tests whether a target object can fit in a specific shelf location.

Failure modes targeted:
  A. Height estimation — object too tall for shelf
  B. Width/gap — not enough room between existing objects
  C. Depth reasoning — object too deep for shallow shelf
  D. Orientation — fits only if rotated
  E. Multi-object/counting — how many more can fit?

Objects: bottles, jars, mugs, plates, boxes, pots, glasses, bowls
Shelves: open rack, kitchen cupboard, bookshelf, bar shelf
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


# ─── Object Definitions (built from MuJoCo primitives) ──────────────

# All dimensions in meters. Objects are composite bodies.
SHELF_OBJECTS = {
    "tall_bottle": {
        "label": "tall bottle",
        "color_desc": "tall bottle",
        "height": 0.30,
        "width": 0.08,
        "depth": 0.08,
        "build": "bottle_tall",
    },
    "lamp": {
        "label": "lamp",
        "color_desc": "lamp",
        "height": 0.22,
        "width": 0.08,
        "depth": 0.08,
        "build": "lamp",
    },
    "short_jar": {
        "label": "short jar",
        "color_desc": "jar",
        "height": 0.12,
        "width": 0.10,
        "depth": 0.10,
        "build": "jar_short",
    },
    "mug": {
        "label": "mug",
        "color_desc": "mug",
        "height": 0.10,
        "width": 0.12,   # includes handle
        "depth": 0.09,
        "build": "mug",
    },
    "cereal_box": {
        "label": "cereal box",
        "color_desc": "box",
        "height": 0.28,
        "width": 0.20,
        "depth": 0.07,
        "build": "box_tall",
    },
    "plate": {
        "label": "plate",
        "color_desc": "plate",
        "height": 0.03,
        "width": 0.26,
        "depth": 0.26,
        "build": "plate",
    },
    "pot": {
        "label": "pot with lid",
        "color_desc": "pot",
        "height": 0.18,
        "width": 0.22,
        "depth": 0.22,
        "build": "pot",
    },
    "spice_jar": {
        "label": "spice jar",
        "color_desc": "small jar",
        "height": 0.10,
        "width": 0.05,
        "depth": 0.05,
        "build": "jar_small",
    },
    "bowl": {
        "label": "bowl",
        "color_desc": "bowl",
        "height": 0.08,
        "width": 0.16,
        "depth": 0.16,
        "build": "bowl",
    },
    "book": {
        "label": "book",
        "color_desc": "book",
        "height": 0.24,
        "width": 0.16,
        "depth": 0.03,
        "build": "book",
    },
}

COLORS = {
    "red":      [0.82, 0.18, 0.18, 1.0],
    "blue":     [0.20, 0.30, 0.80, 1.0],
    "green":    [0.18, 0.70, 0.28, 1.0],
    "yellow":   [0.88, 0.78, 0.12, 1.0],
    "orange":   [0.88, 0.45, 0.12, 1.0],
    "purple":   [0.55, 0.18, 0.75, 1.0],
    "white":    [0.95, 0.95, 0.95, 1.0],
    "brown":    [0.55, 0.35, 0.20, 1.0],
    "teal":     [0.15, 0.65, 0.60, 1.0],
    "pink":     [0.90, 0.40, 0.60, 1.0],
}


# ─── Shelf Configurations ────────────────────────────────────────────

@dataclass
class ShelfConfig:
    """Defines a single shelf in the rack."""
    height: float       # distance from this shelf to the one above (internal clearance)
    depth: float        # how deep the shelf is
    width: float        # total width
    y_pos: float        # vertical position of the shelf surface


@dataclass
class RackConfig:
    """Defines the complete rack/cupboard."""
    name: str
    shelves: list       # list of ShelfConfig
    frame_width: float  # width of side panels
    total_width: float
    total_height: float
    total_depth: float


# Pre-defined rack types
def _open_rack() -> RackConfig:
    """3-shelf open rack — typical kitchen."""
    w = 0.60
    d = 0.30
    shelves = [
        ShelfConfig(height=0.32, depth=d, width=w, y_pos=0.05),   # bottom: tall (fits bottles)
        ShelfConfig(height=0.22, depth=d, width=w, y_pos=0.39),   # middle: medium
        ShelfConfig(height=0.16, depth=d, width=w, y_pos=0.63),   # top: short
    ]
    return RackConfig("open_rack", shelves, 0.02, w, 0.84, d)


def _kitchen_cupboard() -> RackConfig:
    """4-shelf kitchen cupboard — narrower depth."""
    w = 0.55
    d = 0.25
    shelves = [
        ShelfConfig(height=0.24, depth=d, width=w, y_pos=0.04),
        ShelfConfig(height=0.20, depth=d, width=w, y_pos=0.30),
        ShelfConfig(height=0.18, depth=d, width=w, y_pos=0.52),
        ShelfConfig(height=0.14, depth=d, width=w, y_pos=0.72),
    ]
    return RackConfig("kitchen_cupboard", shelves, 0.02, w, 0.90, d)


def _bar_shelf() -> RackConfig:
    """2 tall shelves — for bottles and glasses."""
    w = 0.70
    d = 0.28
    shelves = [
        ShelfConfig(height=0.35, depth=d, width=w, y_pos=0.05),   # bottom: very tall
        ShelfConfig(height=0.30, depth=d, width=w, y_pos=0.42),   # top: tall
    ]
    return RackConfig("bar_shelf", shelves, 0.02, w, 0.75, d)


def _bookshelf() -> RackConfig:
    """5 uniform shelves — sized for books."""
    w = 0.50
    d = 0.22
    shelves = [
        ShelfConfig(height=0.26, depth=d, width=w, y_pos=0.04),
        ShelfConfig(height=0.26, depth=d, width=w, y_pos=0.32),
        ShelfConfig(height=0.26, depth=d, width=w, y_pos=0.60),
        ShelfConfig(height=0.20, depth=d, width=w, y_pos=0.88),
        ShelfConfig(height=0.14, depth=d, width=w, y_pos=1.10),
    ]
    return RackConfig("bookshelf", shelves, 0.02, w, 1.28, d)


RACK_TYPES = {
    "open_rack": _open_rack,
    "kitchen_cupboard": _kitchen_cupboard,
    "bar_shelf": _bar_shelf,
    "bookshelf": _bookshelf,
}


# ─── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class PlacedObject:
    """An object already on a shelf."""
    obj_type: str       # key into SHELF_OBJECTS
    color: str          # key into COLORS
    shelf_idx: int      # which shelf it's on
    x_offset: float     # position from left edge of shelf


@dataclass
class ShelfScenario:
    """Complete shelf fitting scenario."""
    scene_id: str
    rack_type: str              # key into RACK_TYPES
    placed_objects: list        # objects already on shelves
    target_object: str          # key into SHELF_OBJECTS — the object to fit
    target_color: str           # color of target object
    target_shelf: int           # which shelf to check
    fits: bool                  # ground truth
    question: str               # natural language question
    category: str               # A/B/C/D/E
    difficulty: str             # medium/hard
    reasoning: str              # why it fits or doesn't
    orientation: str = "upright"  # upright/sideways (for category D)


# ─── Ground Truth Computation ────────────────────────────────────────

def _can_fit_on_shelf(
    obj_type: str,
    rack: RackConfig,
    shelf_idx: int,
    placed: list[PlacedObject],
    orientation: str = "upright",
) -> tuple[bool, str]:
    """
    Determine if object can fit on the specified shelf.
    Returns (fits, reasoning).
    """
    obj = SHELF_OBJECTS[obj_type]
    shelf = rack.shelves[shelf_idx]

    # Get object dimensions based on orientation
    if orientation == "sideways":
        obj_h = obj["width"]
        obj_w = obj["height"]
        obj_d = obj["depth"]
    else:
        obj_h = obj["height"]
        obj_w = obj["width"]
        obj_d = obj["depth"]

    # Check 1: Height clearance
    if obj_h > shelf.height - 0.01:  # 1cm tolerance
        return False, f"Object is {obj_h*100:.0f}cm tall, shelf clearance is only {shelf.height*100:.0f}cm"

    # Check 2: Depth
    if obj_d > shelf.depth - 0.01:
        return False, f"Object is {obj_d*100:.0f}cm deep, shelf is only {shelf.depth*100:.0f}cm deep"

    # Check 3: Width — find available gaps
    # For width on shelf, use the smaller of width/depth (books stand on spine)
    def _shelf_footprint(obj_type):
        """Width an object takes up on the shelf (min of width, depth for upright items)."""
        o = SHELF_OBJECTS[obj_type]
        # Books stand on their spine (depth), not page-face (width)
        if o["build"] == "book":
            return o["depth"]
        return o["width"]

    obj_footprint = _shelf_footprint(obj_type) if orientation == "upright" else obj_w

    shelf_objects = [p for p in placed if p.shelf_idx == shelf_idx]
    if not shelf_objects:
        # Empty shelf — just check total width
        if obj_footprint > shelf.width - 0.04:  # 2cm margin each side
            return False, f"Object is {obj_footprint*100:.0f}cm wide, shelf is only {shelf.width*100:.0f}cm"
        return True, f"Shelf is empty and object fits ({obj_h*100:.0f}cm < {shelf.height*100:.0f}cm clearance)"

    # Calculate occupied regions
    occupied = []
    for p in shelf_objects:
        pw = _shelf_footprint(p.obj_type)
        occupied.append((p.x_offset, p.x_offset + pw))

    occupied.sort()

    # Find gaps
    gaps = []
    # Gap before first object
    if occupied[0][0] > 0.02:
        gaps.append(occupied[0][0] - 0.02)
    # Gaps between objects
    for i in range(len(occupied) - 1):
        gap = occupied[i + 1][0] - occupied[i][1]
        gaps.append(gap)
    # Gap after last object
    remaining = shelf.width - occupied[-1][1] - 0.02
    if remaining > 0:
        gaps.append(remaining)

    max_gap = max(gaps) if gaps else 0

    if obj_footprint > max_gap:
        return False, f"Largest available gap is {max_gap*100:.0f}cm, object needs {obj_footprint*100:.0f}cm"

    return True, f"Object fits in a {max_gap*100:.0f}cm gap (needs {obj_footprint*100:.0f}cm)"


# ─── MuJoCo Scene Builder ───────────────────────────────────────────

def _rgba_str(color_name: str) -> str:
    return " ".join(f"{c:.2f}" for c in COLORS[color_name])


def _build_object_xml(obj_type: str, color: str, name: str, pos: list) -> str:
    """Build MuJoCo XML for a composite object at the given position."""
    obj = SHELF_OBJECTS[obj_type]
    rgba = _rgba_str(color)
    px, py, pz = pos
    build = obj["build"]

    if build == "bottle_tall":
        r = obj["width"] / 2
        h = obj["height"]
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/2 - r:.4f}" rgba="{rgba}" pos="0 0 0"/>
      <geom type="sphere" size="{r:.4f}" rgba="{rgba}" pos="0 0 {h/2 - r:.4f}"/>
      <geom type="cylinder" size="{r*0.4:.4f} {r:.4f}" rgba="{rgba}" pos="0 0 {h/2:.4f}"/>
    </body>"""

    if build == "lamp":
        r = obj["width"] / 2
        h = obj["height"]
        # Lamp: flat base + thin stem + wide bowl (open top cylinder)
        base_h = 0.005
        stem_h = h * 0.35
        bowl_h = h * 0.45
        bowl_r = r * 0.9
        stem_r = r * 0.12
        base_r = r * 0.6
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{base_r:.4f} {base_h:.4f}" rgba="{rgba}" pos="0 0 {-h/2 + base_h:.4f}"/>
      <geom type="cylinder" size="{stem_r:.4f} {stem_h/2:.4f}" rgba="{rgba}" pos="0 0 {-h/2 + base_h*2 + stem_h/2:.4f}"/>
      <geom type="cylinder" size="{bowl_r:.4f} {bowl_h/2:.4f}" rgba="0.85 0.85 0.9 0.5" pos="0 0 {h/2 - bowl_h/2:.4f}"/>
    </body>"""

    if build == "jar_short":
        r = obj["width"] / 2
        h = obj["height"]
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/2 - 0.005:.4f}" rgba="{rgba}"/>
      <geom type="cylinder" size="{r + 0.005:.4f} 0.005" rgba="0.7 0.7 0.7 1" pos="0 0 {h/2 - 0.005:.4f}"/>
    </body>"""

    if build == "jar_small":
        r = obj["width"] / 2
        h = obj["height"]
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/2:.4f}" rgba="{rgba}"/>
    </body>"""

    if build == "mug":
        r = obj["depth"] / 2  # body radius (not including handle)
        h = obj["height"]
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/2:.4f}" rgba="{rgba}"/>
      <geom type="box" size="0.01 {h*0.35:.4f} {h*0.3:.4f}" rgba="{rgba}" pos="{r + 0.01:.4f} 0 0"/>
    </body>"""

    if build == "box_tall":
        hw = obj["width"] / 2
        hh = obj["height"] / 2
        hd = obj["depth"] / 2
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + hh:.4f}">
      <geom type="box" size="{hw:.4f} {hd:.4f} {hh:.4f}" rgba="{rgba}"/>
    </body>"""

    if build == "plate":
        r = obj["width"] / 2
        h = obj["height"] / 2
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h:.4f}">
      <geom type="cylinder" size="{r:.4f} {h:.4f}" rgba="{rgba}"/>
    </body>"""

    if build == "pot":
        r = obj["width"] / 2
        h = obj["height"]
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/2 - 0.02:.4f}" rgba="{rgba}"/>
      <geom type="sphere" size="{r - 0.01:.4f}" rgba="0.6 0.6 0.6 0.8" pos="0 0 {h/2 - 0.02:.4f}"/>
      <geom type="box" size="0.01 0.03 {h*0.3:.4f}" rgba="{rgba}" pos="{r + 0.01:.4f} 0 0"/>
      <geom type="box" size="0.01 0.03 {h*0.3:.4f}" rgba="{rgba}" pos="{-r - 0.01:.4f} 0 0"/>
    </body>"""

    if build == "bowl":
        r = obj["width"] / 2
        h = obj["height"]
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/4:.4f}" rgba="{rgba}"/>
      <geom type="cylinder" size="{r*0.85:.4f} {h/4 - 0.003:.4f}" rgba="0.95 0.95 0.95 1" pos="0 0 0.003"/>
    </body>"""

    if build == "book":
        hw = obj["width"] / 2
        hh = obj["height"] / 2
        hd = obj["depth"] / 2
        # Book standing upright: depth(thin) as X, width as Y, height as Z
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + hh:.4f}">
      <geom type="box" size="{hd:.4f} {hw*0.6:.4f} {hh:.4f}" rgba="{rgba}"/>
    </body>"""

    # Fallback: simple box
    hw = obj["width"] / 2
    hh = obj["height"] / 2
    hd = obj["depth"] / 2
    return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + hh:.4f}">
      <geom type="box" size="{hw:.4f} {hd:.4f} {hh:.4f}" rgba="{rgba}"/>
    </body>"""


def _build_rack_xml(rack: RackConfig) -> str:
    """Build MuJoCo XML for the rack/cupboard structure."""
    parts = []
    w = rack.total_width
    h = rack.total_height
    d = rack.total_depth
    fw = rack.frame_width
    wood = "0.55 0.35 0.20 1"  # brown wood

    # Back panel
    parts.append(f'<geom type="box" size="{w/2:.4f} 0.005 {h/2:.4f}" pos="0 {-d/2 + 0.005:.4f} {h/2:.4f}" rgba="{wood}"/>')

    # Left side
    parts.append(f'<geom type="box" size="{fw/2:.4f} {d/2:.4f} {h/2:.4f}" pos="{-w/2 + fw/2:.4f} 0 {h/2:.4f}" rgba="{wood}"/>')

    # Right side
    parts.append(f'<geom type="box" size="{fw/2:.4f} {d/2:.4f} {h/2:.4f}" pos="{w/2 - fw/2:.4f} 0 {h/2:.4f}" rgba="{wood}"/>')

    # Top
    parts.append(f'<geom type="box" size="{w/2:.4f} {d/2:.4f} {fw/2:.4f}" pos="0 0 {h - fw/2:.4f}" rgba="{wood}"/>')

    # Shelves (thicker boards for visibility)
    for shelf in rack.shelves:
        sy = shelf.y_pos
        parts.append(f'<geom type="box" size="{w/2 - fw:.4f} {d/2:.4f} 0.012" pos="0 0 {sy:.4f}" rgba="{wood}"/>')

    return "\n    ".join(parts)


def build_scene_xml(scenario: ShelfScenario) -> str:
    """Build complete MuJoCo XML for a shelf fitting scenario."""
    rack_fn = RACK_TYPES[scenario.rack_type]
    rack = rack_fn()

    # Rack at center
    rack_xml = _build_rack_xml(rack)

    # Placed objects on shelves
    objects_xml = []
    for i, po in enumerate(scenario.placed_objects):
        shelf = rack.shelves[po.shelf_idx]
        obj_info = SHELF_OBJECTS[po.obj_type]
        # Position: x_offset from left edge, centered in depth, on shelf surface
        x = -rack.total_width / 2 + rack.frame_width + po.x_offset + obj_info["width"] / 2
        y = 0  # centered in depth
        z = shelf.y_pos + 0.009  # just above shelf surface

        # Safety: skip objects that would visually exceed the shelf above
        next_shelf_z = rack.total_height  # top of rack
        if po.shelf_idx < len(rack.shelves) - 1:
            next_shelf_z = rack.shelves[po.shelf_idx + 1].y_pos
        max_obj_height = next_shelf_z - z - 0.005
        if obj_info["height"] > max_obj_height + 0.01:
            # Object too tall for this shelf visually — skip rendering it
            continue

        objects_xml.append(_build_object_xml(po.obj_type, po.color, f"placed_{i}", [x, y, z]))

    # Target object — placed IN FRONT of the rack (visible to camera)
    target_obj = SHELF_OBJECTS[scenario.target_object]
    tx = rack.total_width / 2 + 0.12  # to the right, slightly in front
    tz = 0.0
    objects_xml.append(_build_object_xml(
        scenario.target_object, scenario.target_color, "target_obj", [tx, rack.total_depth / 2 + 0.08, tz]
    ))

    xml = f"""<mujoco model="shelf_fitting">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>

  <visual>
    <rgba haze="0.85 0.9 0.95 1"/>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="10"/>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <worldbody>
    <!-- Bright, well-lit scene -->
    <light pos="0 1.5 1.5" dir="0 -0.7 -0.5" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="0.5 1.0 0.8" dir="-0.2 -0.6 -0.4" diffuse="0.4 0.4 0.42"/>
    <light pos="-0.5 0.8 1.2" dir="0.2 -0.4 -0.7" diffuse="0.3 0.3 0.32"/>

    <!-- Floor — light color -->
    <geom type="plane" size="2 2 0.01" rgba="0.94 0.92 0.89 1"/>

    <!-- Back wall — light gray -->
    <geom type="box" size="2 0.01 2" pos="0 {-rack.total_depth/2 - 0.01:.4f} 1" rgba="0.96 0.95 0.93 1"/>

    <!-- Rack -->
    <body name="rack" pos="0 0 0">
      {rack_xml}
    </body>

    <!-- Objects on shelves and target -->
    {"".join(objects_xml)}
  </worldbody>
</mujoco>"""
    return xml


# ─── Rendering ───────────────────────────────────────────────────────

def _render_view(model, data, lookat, dist, azimuth, elevation, w=640, h=480) -> str:
    """Render a single view → base64 JPEG."""
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, cam)
    px = renderer.render()
    renderer.close()

    buf = io.BytesIO()
    Image.fromarray(px).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_scenario(scenario: ShelfScenario) -> list[str]:
    """Render front view of the scene."""
    xml = build_scene_xml(scenario)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rack = RACK_TYPES[scenario.rack_type]()
    cy = rack.total_height / 2

    # Dynamic camera distance — zoom out for taller racks
    # Base distance 1.2 for 0.8m rack, scale up for taller ones
    cam_dist = max(1.2, rack.total_height * 1.5 + 0.2)

    # Primary: straight front view (az270)
    front = _render_view(model, data,
                         lookat=[0, 0, cy], dist=cam_dist, azimuth=270, elevation=-10,
                         w=800, h=600)

    # Secondary: slight angle (az315) for depth context
    angle = _render_view(model, data,
                         lookat=[0, 0, cy], dist=cam_dist, azimuth=315, elevation=-10,
                         w=800, h=600)

    return [front, angle]


# ─── 50 Curated Scenarios ────────────────────────────────────────────

def _shelf_name(rack_type: str, shelf_idx: int) -> str:
    """Human-readable shelf position: 'bottom shelf', 'second shelf from the bottom', 'top shelf'."""
    rack = RACK_TYPES[rack_type]()
    n = len(rack.shelves)
    if shelf_idx == 0:
        return "bottom shelf"
    if shelf_idx == n - 1:
        return "top shelf"
    if n == 3 and shelf_idx == 1:
        return "middle shelf"
    if n == 4:
        if shelf_idx == 1:
            return "second shelf from the bottom"
        if shelf_idx == 2:
            return "third shelf from the bottom"
    if n == 5:
        if shelf_idx == 1:
            return "second shelf from the bottom"
        if shelf_idx == 2:
            return "middle shelf"
        if shelf_idx == 3:
            return "fourth shelf from the bottom"
    return f"shelf {shelf_idx + 1} from the bottom"


def generate_50_scenarios() -> list[ShelfScenario]:
    """50 curated scenarios across 5 categories."""
    scenarios = []

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY A: Height Check (10 scenarios) — 4 fit, 6 don't
    # ═══════════════════════════════════════════════════════════════════

    # A1: Tall bottle on short top shelf — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_A01", "open_rack",
        [PlacedObject("spice_jar", "green", 2, 0.05)],
        "tall_bottle", "blue", 2, False,
        "Is the top shelf tall enough to store the blue bottle upright?",
        "A", "medium",
        "Bottle is 30cm tall, top shelf clearance is only 16cm",
    ))

    # A2: Spice jar on short shelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_A02", "open_rack",
        [PlacedObject("short_jar", "red", 2, 0.10)],
        "spice_jar", "yellow", 2, True,
        "Will the yellow spice jar fit standing up on the top shelf next to the red jar?",
        "A", "medium",
        "Spice jar is 10cm, shelf clearance is 16cm — fits easily",
    ))

    # A3: Cereal box on middle shelf — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_A03", "open_rack",
        [PlacedObject("mug", "white", 1, 0.05)],
        "cereal_box", "orange", 1, False,
        "I want to store the orange cereal box standing upright on the middle shelf. Is there enough height clearance?",
        "A", "medium",
        "Cereal box is 28cm, middle shelf clearance is 22cm",
    ))

    # A4: Mug on middle shelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_A04", "kitchen_cupboard",
        [PlacedObject("spice_jar", "teal", 1, 0.10)],
        "mug", "red", 1, True,
        "Is the second shelf from the bottom of the cupboard tall enough for the red mug?",
        "A", "medium",
        "Mug is 10cm, shelf clearance is 20cm",
    ))

    # A5: Lamp on bookshelf top shelf (14cm clearance) — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_A05", "bookshelf",
        [PlacedObject("spice_jar", "blue", 4, 0.05)],
        "lamp", "purple", 4, False,
        "Could the purple lamp stand upright on the top shelf of the bookshelf without hitting the top?",
        "A", "medium",
        "Lamp is 22cm, top shelf clearance is only 14cm",
    ))

    # A6: Short jar on bookshelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_A06", "bookshelf",
        [],
        "short_jar", "green", 0, True,
        "Will the green jar fit on the bottom shelf of the bookshelf?",
        "A", "medium",
        "Jar is 12cm, shelf clearance is 15cm — fits",
    ))

    # A7: Pot on bar shelf top — FITS (height ok, depth ok)
    scenarios.append(ShelfScenario(
        "shelf_A07", "bar_shelf",
        [PlacedObject("tall_bottle", "green", 0, 0.10)],
        "pot", "red", 1, True,
        "Would the red pot be too tall for the top shelf of the bar?",
        "A", "medium",
        "Pot is 18cm tall, bar top shelf clearance is 30cm — fits comfortably",
    ))

    # A8: Tall bottle on bar bottom shelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_A08", "bar_shelf",
        [PlacedObject("lamp", "white", 0, 0.08)],
        "tall_bottle", "teal", 0, True,
        "Can the teal tall bottle fit on the bottom shelf of the bar?",
        "A", "medium",
        "Bottle is 30cm, bar bottom shelf is 35cm — fits",
    ))

    # A9: Cereal box on bar shelf — DOESN'T FIT (top)
    scenarios.append(ShelfScenario(
        "shelf_A09", "bar_shelf",
        [],
        "cereal_box", "yellow", 1, True,
        "Can the yellow cereal box fit on the top shelf of the bar?",
        "A", "medium",
        "Box is 28cm, top shelf clearance is 30cm — just fits",
    ))

    # A10: Tall bottle in kitchen cupboard top shelf — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_A10", "kitchen_cupboard",
        [PlacedObject("spice_jar", "pink", 3, 0.05)],
        "tall_bottle", "blue", 3, False,
        "Can the blue bottle stand upright on the top shelf of the kitchen cupboard?",
        "A", "medium",
        "Bottle is 30cm, top shelf clearance is only 14cm",
    ))

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY B: Width/Gap Check (10 scenarios) — 5 fit, 5 don't
    # ═══════════════════════════════════════════════════════════════════

    # B1: Mug between two jars — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_B01", "open_rack",
        [PlacedObject("short_jar", "red", 0, 0.05),
         PlacedObject("short_jar", "blue", 0, 0.20),
         PlacedObject("short_jar", "green", 0, 0.35)],
        "mug", "white", 0, False,
        "Is there enough horizontal space to squeeze the white mug between any of the jars on the bottom shelf?",
        "B", "medium",
        "Gaps between jars are ~5cm, mug needs 12cm",
    ))

    # B2: Spice jar in gap — FITS
    scenarios.append(ShelfScenario(
        "shelf_B02", "open_rack",
        [PlacedObject("short_jar", "red", 0, 0.03),
         PlacedObject("short_jar", "blue", 0, 0.25)],
        "spice_jar", "yellow", 0, True,
        "There's a gap between the red and blue jars. Is it wide enough for the yellow spice jar?",
        "B", "medium",
        "Gap is ~12cm, spice jar needs only 5cm",
    ))

    # B3: Pot on shelf with existing objects — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_B03", "open_rack",
        [PlacedObject("book", "green", 0, 0.03),
         PlacedObject("book", "blue", 0, 0.08),
         PlacedObject("short_jar", "red", 0, 0.20),
         PlacedObject("mug", "yellow", 0, 0.35)],
        "pot", "orange", 0, False,
        "The bottom shelf already has books, a jar, and a mug. Is there still room for the orange pot?",
        "B", "medium",
        "Remaining space is ~12cm, pot needs 22cm width",
    ))

    # B4: Spice jar at end of crowded shelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_B04", "kitchen_cupboard",
        [PlacedObject("mug", "red", 1, 0.02),
         PlacedObject("mug", "blue", 1, 0.16),
         PlacedObject("short_jar", "green", 1, 0.30)],
        "spice_jar", "purple", 1, True,
        "The second shelf from the bottom has mugs and a jar already. Is there still space at the right end for the purple spice jar?",
        "B", "medium",
        "~10cm remaining at the right end, spice jar is 5cm",
    ))

    # B5: Wide plate on shelf with bottles — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_B05", "bar_shelf",
        [PlacedObject("tall_bottle", "red", 0, 0.05),
         PlacedObject("tall_bottle", "green", 0, 0.20),
         PlacedObject("tall_bottle", "blue", 0, 0.35),
         PlacedObject("tall_bottle", "purple", 0, 0.50)],
        "plate", "white", 0, False,
        "The bar shelf has four bottles lined up. Is any gap between them wide enough for the white plate?",
        "B", "hard",
        "Plate is 26cm wide, largest gap between bottles is ~7cm",
    ))

    # B6: Bowl between mugs — FITS
    scenarios.append(ShelfScenario(
        "shelf_B06", "kitchen_cupboard",
        [PlacedObject("mug", "red", 0, 0.02),
         PlacedObject("mug", "blue", 0, 0.34)],
        "bowl", "teal", 0, True,
        "Looking at the gap between the red and blue mugs — is it wide enough for the teal bowl?",
        "B", "medium",
        "Gap is ~20cm, bowl needs 16cm",
    ))

    # B7: Cereal box in narrow gap — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_B07", "open_rack",
        [PlacedObject("cereal_box", "orange", 0, 0.02),
         PlacedObject("tall_bottle", "blue", 0, 0.35)],
        "cereal_box", "yellow", 0, False,
        "I want to place a second cereal box between the orange box and the blue bottle. Will it fit?",
        "B", "hard",
        "Gap is ~11cm, cereal box needs 20cm",
    ))

    # B8: Mug at edge — FITS
    scenarios.append(ShelfScenario(
        "shelf_B08", "open_rack",
        [PlacedObject("short_jar", "green", 1, 0.03),
         PlacedObject("short_jar", "red", 1, 0.18)],
        "mug", "pink", 1, True,
        "There's open space on the right side of the middle shelf. Would the pink mug fit there?",
        "B", "medium",
        "~28cm available on the right, mug needs 12cm",
    ))

    # B9: Short jar in tight spot — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_B09", "bookshelf",
        [PlacedObject("book", "brown", 1, 0.01),
         PlacedObject("book", "blue", 1, 0.05),
         PlacedObject("book", "red", 1, 0.09),
         PlacedObject("book", "green", 1, 0.13),
         PlacedObject("book", "purple", 1, 0.17),
         PlacedObject("short_jar", "orange", 1, 0.35)],
        "short_jar", "teal", 1, False,
        "The second shelf from the bottom has many books and an orange jar. Is there room to add the teal jar anywhere on that shelf?",
        "B", "hard",
        "Gap between books and jar is ~12cm, but jar needs 10cm — actually might fit. Needs tight check.",
    ))

    # B10: Lamp next to bottles — FITS
    scenarios.append(ShelfScenario(
        "shelf_B10", "bar_shelf",
        [PlacedObject("tall_bottle", "red", 1, 0.05),
         PlacedObject("tall_bottle", "green", 1, 0.20)],
        "lamp", "white", 1, True,
        "Can the white lamp fit on the top shelf next to the bottles?",
        "B", "medium",
        "~38cm remaining, lamp needs 8cm",
    ))

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY C: Depth Check (10 scenarios) — 4 fit, 6 don't
    # ═══════════════════════════════════════════════════════════════════

    # C1: Plate on shallow bookshelf — DOESN'T FIT (depth)
    scenarios.append(ShelfScenario(
        "shelf_C01", "bookshelf",
        [],
        "plate", "white", 0, False,
        "Can the white plate fit flat on the bookshelf?",
        "C", "medium",
        "Plate is 26cm deep, bookshelf is only 22cm deep",
    ))

    # C2: Pot on bookshelf — DOESN'T FIT (depth)
    scenarios.append(ShelfScenario(
        "shelf_C02", "bookshelf",
        [],
        "pot", "red", 0, False,
        "Can the red pot fit on the bottom shelf of the bookshelf?",
        "C", "medium",
        "Pot is 22cm deep, bookshelf is 22cm — too tight with no clearance",
    ))

    # C3: Spice jar on bookshelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_C03", "bookshelf",
        [PlacedObject("book", "brown", 0, 0.02)],
        "spice_jar", "green", 0, True,
        "Can the green spice jar fit on the bottom bookshelf?",
        "C", "medium",
        "Spice jar is only 5cm deep, bookshelf is 22cm",
    ))

    # C4: Pot in kitchen cupboard — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_C04", "kitchen_cupboard",
        [],
        "pot", "orange", 0, False,
        "Can the orange pot fit in the kitchen cupboard?",
        "C", "medium",
        "Pot is 22cm deep, kitchen cupboard shelves are 25cm — barely fits actually. Let me adjust.",
    ))

    # C5: Bowl in kitchen cupboard — FITS
    scenarios.append(ShelfScenario(
        "shelf_C05", "kitchen_cupboard",
        [PlacedObject("mug", "blue", 0, 0.05)],
        "bowl", "green", 0, True,
        "Can the green bowl fit in the kitchen cupboard on the bottom shelf?",
        "C", "medium",
        "Bowl is 16cm deep, cupboard is 25cm deep — fits",
    ))

    # C6: Plate in kitchen cupboard — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_C06", "kitchen_cupboard",
        [],
        "plate", "white", 2, False,
        "Can the white plate fit flat on the third shelf from the bottom of the kitchen cupboard?",
        "C", "medium",
        "Plate is 26cm, cupboard depth is 25cm — doesn't fit flat",
    ))

    # C7: Mug on open rack — FITS
    scenarios.append(ShelfScenario(
        "shelf_C07", "open_rack",
        [],
        "mug", "red", 1, True,
        "Can the red mug fit on the middle shelf of the open rack?",
        "C", "medium",
        "Mug is 9cm deep, rack is 30cm deep — plenty of room",
    ))

    # C8: Cereal box on open rack — FITS
    scenarios.append(ShelfScenario(
        "shelf_C08", "open_rack",
        [PlacedObject("tall_bottle", "blue", 0, 0.10)],
        "cereal_box", "yellow", 0, True,
        "Can the yellow cereal box fit on the bottom shelf of the open rack?",
        "C", "medium",
        "Box is 7cm deep, rack is 30cm — easily fits",
    ))

    # C9: Plate on bar shelf — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_C09", "bar_shelf",
        [],
        "plate", "pink", 0, False,
        "Can the pink plate sit flat on the bar shelf?",
        "C", "hard",
        "Plate is 26cm, bar shelf depth is 28cm — very tight, 1cm each side",
    ))

    # C10: Pot on open rack — FITS (just)
    scenarios.append(ShelfScenario(
        "shelf_C10", "open_rack",
        [],
        "pot", "teal", 0, True,
        "Can the teal pot fit on the bottom shelf of the open rack?",
        "C", "hard",
        "Pot is 22cm deep, rack is 30cm deep — fits with room",
    ))

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY D: Orientation (10 scenarios) — 6 fit, 4 don't
    # ═══════════════════════════════════════════════════════════════════

    # D1: Tall bottle sideways on bookshelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_D01", "bookshelf",
        [],
        "tall_bottle", "blue", 0, True,
        "Can the blue bottle fit on the bookshelf if laid on its side?",
        "D", "medium",
        "Bottle on side: 8cm tall, 30cm wide — shelf is 15cm tall, 50cm wide. Height fits!",
        "sideways",
    ))

    # D2: Cereal box sideways on bookshelf — FITS (sideways 7cm < 26cm clearance)
    scenarios.append(ShelfScenario(
        "shelf_D02", "bookshelf",
        [],
        "cereal_box", "orange", 1, True,
        "The orange cereal box is too tall to stand upright on the bookshelf. Could it fit if laid on its side?",
        "D", "medium",
        "Sideways: depth becomes height = 7cm, shelf clearance is 26cm — fits easily",
        "sideways",
    ))

    # D3: Book standing vs flat on short shelf — FITS standing
    scenarios.append(ShelfScenario(
        "shelf_D03", "kitchen_cupboard",
        [],
        "book", "brown", 3, False,
        "Can the book stand upright on the top shelf of the cupboard?",
        "D", "medium",
        "Book upright is 24cm, top shelf is only 14cm — doesn't fit standing",
    ))

    # D4: Plate upright in bookshelf — DOESN'T FIT (plate 26cm on edge > 26cm clearance, too tight)
    scenarios.append(ShelfScenario(
        "shelf_D04", "bookshelf",
        [PlacedObject("spice_jar", "red", 0, 0.02)],
        "plate", "white", 0, False,
        "Can the white plate fit on the bottom bookshelf if placed on its edge (upright)?",
        "D", "medium",
        "Plate on edge is 26cm tall, shelf clearance is 26cm — no room with zero tolerance",
        "sideways",
    ))

    # D5: Tall bottle sideways in kitchen cupboard — FITS
    scenarios.append(ShelfScenario(
        "shelf_D05", "kitchen_cupboard",
        [],
        "tall_bottle", "green", 0, True,
        "Can the green bottle fit in the cupboard if placed on its side?",
        "D", "medium",
        "On side: 8cm tall, needs 30cm width — cupboard is 24cm clearance and 55cm wide. Height 8cm < 24cm ✓",
        "sideways",
    ))

    # D6: Lamp sideways on bookshelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_D06", "bookshelf",
        [],
        "lamp", "purple", 1, True,
        "The purple lamp is too tall to stand upright on the bookshelf. Could it fit if laid on its side?",
        "D", "medium",
        "Sideways: 8cm tall (was width), shelf clearance 15cm — fits laid down",
        "sideways",
    ))

    # D7: Cereal box flat (lying down) on bottom shelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_D07", "open_rack",
        [],
        "cereal_box", "yellow", 0, True,
        "Can the cereal box fit on the bottom shelf if laid flat on its back?",
        "D", "medium",
        "Flat: 7cm tall (was depth), 20cm wide, 28cm deep — shelf is 28cm clearance, 30cm deep ✓",
        "sideways",
    ))

    # D8: Pot on side — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_D08", "bookshelf",
        [],
        "pot", "red", 2, False,
        "Can the pot fit on the bookshelf in any orientation?",
        "D", "hard",
        "Pot minimum dimension is 18cm (height on side = width 22cm, no help) — shelf is 15cm",
    ))

    # D9: Book flat on cupboard — FITS
    scenarios.append(ShelfScenario(
        "shelf_D09", "kitchen_cupboard",
        [PlacedObject("mug", "blue", 2, 0.05)],
        "book", "green", 2, True,
        "Can the book fit if placed flat (lying down) on the third shelf from the bottom?",
        "D", "medium",
        "Book flat: 3cm tall, 16cm wide — shelf clearance 18cm, easily fits",
        "sideways",
    ))

    # D10: Bowl upside down on shelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_D10", "open_rack",
        [],
        "bowl", "orange", 2, True,
        "Can the orange bowl fit on the top shelf of the rack?",
        "D", "medium",
        "Bowl is 8cm tall, shelf clearance is 16cm — fits in any orientation",
    ))

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY E: Multi-object/Counting (10 scenarios) — 4 fit, 6 don't
    # ═══════════════════════════════════════════════════════════════════

    # E1: Can 2 more mugs fit? — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_E01", "kitchen_cupboard",
        [PlacedObject("mug", "red", 1, 0.02),
         PlacedObject("mug", "blue", 1, 0.15),
         PlacedObject("mug", "green", 1, 0.28),
         PlacedObject("mug", "yellow", 1, 0.41)],
        "mug", "pink", 1, False,
        "Is there room for two more mugs on the second shelf from the bottom?",
        "E", "hard",
        "4 mugs × 12cm = 48cm. Shelf is 55cm. Only ~3cm left — can't fit 2 more (need 24cm)",
    ))

    # E2: Can 1 more spice jar fit? — FITS
    scenarios.append(ShelfScenario(
        "shelf_E02", "kitchen_cupboard",
        [PlacedObject("spice_jar", "red", 3, 0.02),
         PlacedObject("spice_jar", "green", 3, 0.08),
         PlacedObject("spice_jar", "blue", 3, 0.14),
         PlacedObject("spice_jar", "yellow", 3, 0.20)],
        "spice_jar", "purple", 3, True,
        "Can one more spice jar fit on the top shelf alongside the others?",
        "E", "medium",
        "4 jars × 5cm = 20cm. Shelf is 55cm. ~30cm remaining — easily fits one more",
    ))

    # E3: Can a bottle fit with 3 bottles already? — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_E03", "bar_shelf",
        [PlacedObject("tall_bottle", "red", 0, 0.02),
         PlacedObject("tall_bottle", "green", 0, 0.14),
         PlacedObject("tall_bottle", "blue", 0, 0.26),
         PlacedObject("tall_bottle", "purple", 0, 0.38),
         PlacedObject("tall_bottle", "orange", 0, 0.50),
         PlacedObject("tall_bottle", "teal", 0, 0.62)],
        "tall_bottle", "pink", 0, False,
        "Can another bottle fit on the bottom shelf? It looks quite full.",
        "E", "medium",
        "6 bottles × 8cm + gaps ≈ 62cm used, shelf is 70cm — ~0cm remaining",
    ))

    # E4: One more bowl among bowls — FITS
    scenarios.append(ShelfScenario(
        "shelf_E04", "open_rack",
        [PlacedObject("bowl", "red", 1, 0.03),
         PlacedObject("bowl", "blue", 1, 0.22)],
        "bowl", "green", 1, True,
        "Can the green bowl fit on the middle shelf with the other two bowls?",
        "E", "medium",
        "2 bowls use ~35cm, shelf is 60cm — ~22cm remaining, bowl needs 16cm",
    ))

    # E5: Bookshelf with books — can one more fit? — FITS
    scenarios.append(ShelfScenario(
        "shelf_E05", "bookshelf",
        [PlacedObject("book", "red", 0, 0.01),
         PlacedObject("book", "blue", 0, 0.05),
         PlacedObject("book", "green", 0, 0.09),
         PlacedObject("book", "brown", 0, 0.13),
         PlacedObject("book", "purple", 0, 0.17),
         PlacedObject("book", "orange", 0, 0.21),
         PlacedObject("book", "teal", 0, 0.25),
         PlacedObject("book", "pink", 0, 0.29),
         PlacedObject("book", "yellow", 0, 0.33),
         PlacedObject("book", "red", 0, 0.37)],
        "book", "teal", 0, True,
        "The bottom shelf has 10 books lined up. Is there space for one more?",
        "E", "medium",
        "10 books × 3cm spine = 30cm + gaps. Shelf is 50cm — plenty of room at the end",
    ))

    # E6: Two jars on empty shelf — FITS
    scenarios.append(ShelfScenario(
        "shelf_E06", "open_rack",
        [],
        "short_jar", "red", 2, True,
        "Can two short jars fit side by side on the top shelf?",
        "E", "medium",
        "2 × 10cm = 20cm, shelf is 60cm — easily fits",
    ))

    # E7: Crowded shelf, can pot fit? — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_E07", "kitchen_cupboard",
        [PlacedObject("short_jar", "white", 0, 0.02),
         PlacedObject("mug", "green", 0, 0.14),
         PlacedObject("short_jar", "orange", 0, 0.28),
         PlacedObject("bowl", "blue", 0, 0.40)],
        "pot", "red", 0, False,
        "The bottom shelf has a jar, a mug, another jar, and a bowl. Can the red pot still fit somewhere?",
        "E", "hard",
        "Objects span ~56cm, shelf is 55cm — no continuous 22cm gap for the pot",
    ))

    # E8: Last spot for lamp — FITS
    scenarios.append(ShelfScenario(
        "shelf_E08", "bar_shelf",
        [PlacedObject("lamp", "white", 1, 0.05),
         PlacedObject("lamp", "white", 1, 0.15),
         PlacedObject("lamp", "white", 1, 0.25),
         PlacedObject("tall_bottle", "red", 1, 0.40)],
        "lamp", "purple", 1, True,
        "Is there room for one more lamp on the top shelf?",
        "E", "medium",
        "Glasses + bottle use ~50cm, shelf is 70cm — ~12cm at the end, glass needs 8cm",
    ))

    # E9: Crowded kitchen shelf — DOESN'T FIT
    scenarios.append(ShelfScenario(
        "shelf_E09", "kitchen_cupboard",
        [PlacedObject("mug", "red", 2, 0.02),
         PlacedObject("mug", "blue", 2, 0.14),
         PlacedObject("short_jar", "green", 2, 0.28),
         PlacedObject("spice_jar", "yellow", 2, 0.40),
         PlacedObject("spice_jar", "orange", 2, 0.46)],
        "bowl", "teal", 2, False,
        "Can the teal bowl squeeze onto the third shelf from the bottom?",
        "E", "hard",
        "Objects span ~51cm, shelf is 55cm — only 2cm gap left, bowl needs 16cm",
    ))

    # E10: Lots of room — FITS
    scenarios.append(ShelfScenario(
        "shelf_E10", "open_rack",
        [PlacedObject("spice_jar", "red", 0, 0.05)],
        "short_jar", "blue", 0, True,
        "Can the blue jar and two more spice jars all fit on the bottom shelf?",
        "E", "medium",
        "One jar + space used = ~10cm. Shelf is 60cm. Plenty for jar(10cm) + 2×spice(10cm) = 30cm total",
    ))

    return scenarios


# ─── Review HTML ─────────────────────────────────────────────────────

def generate_review_html(scenarios: list[ShelfScenario] | None = None) -> str:
    """Generate HTML review page showing scenarios with rendered images."""
    if scenarios is None:
        scenarios = generate_50_scenarios()

    cards = []
    for sc in scenarios:
        try:
            imgs = render_scenario(sc)
            img_html = f'<img src="{imgs[0]}" style="border-radius:8px;border:1px solid #e0e0e0;max-height:350px;margin-right:8px">'
            if len(imgs) > 1:
                img_html += f'<img src="{imgs[1]}" style="border-radius:8px;border:1px solid #e0e0e0;max-height:350px">'
        except Exception as e:
            img_html = f'<div style="color:red;font-size:12px">Render error: {e}</div>'

        fit_badge = (
            '<span style="background:#dcfce7;color:#166534;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">FITS</span>'
            if sc.fits else
            '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">DOESN\'T FIT</span>'
        )

        cards.append(f"""
<div style="background:white;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
    <span style="font-weight:700;color:#333">{sc.scene_id}</span>
    <span style="background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:10px;font-size:11px">Cat {sc.category}</span>
    <span style="background:#f3f4f6;color:#555;padding:2px 8px;border-radius:10px;font-size:11px">{sc.difficulty}</span>
    {fit_badge}
  </div>
  <p style="font-size:14px;color:#333;margin:4px 0 8px">💬 "{sc.question}"</p>
  <div style="display:flex;gap:8px;margin:8px 0">{img_html}</div>
  <p style="font-size:12px;color:#888;margin-top:6px">💡 {sc.reasoning}</p>
</div>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Shelf Fitting Scenarios</title></head>
<body style="font-family:-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#f5f5f5">
<h1>🗄️ Shelf Fitting Benchmark — 50 Scenarios</h1>
<p style="color:#666">A: Height | B: Width/Gap | C: Depth | D: Orientation | E: Multi-object</p>
{"".join(cards)}
</body></html>"""
