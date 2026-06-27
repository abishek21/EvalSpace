"""
DISE Door Passage: Can furniture fit through the doorway?

Tests spatial reasoning about whether objects can pass through door openings.
Varies door sizes, furniture types, and orientation questions.

Camera: az=315, el=-20, dist=5.0
"""

import io
import base64
from dataclasses import dataclass, field

import numpy as np
import mujoco
from PIL import Image


# ─── Door Configurations ─────────────────────────────────────────────

DOOR_TYPES = {
    "standard":    {"width": 0.90, "height": 2.0,  "label": "standard doorway"},
    "wide":        {"width": 1.10, "height": 2.0,  "label": "wide doorway"},
    "narrow":      {"width": 0.75, "height": 2.0,  "label": "narrow doorway"},
    "short":       {"width": 0.90, "height": 1.75, "label": "short doorway"},
}


# ─── Furniture Definitions ────────────────────────────────────────────

# Dimensions: width (longest), depth (shortest horizontal), height
FURNITURE = {
    "couch": {
        "label": "couch",
        "width": 1.80, "depth": 0.80, "height": 0.85,
        "color": [0.25, 0.45, 0.70, 1.0],
        "build": "couch",
    },
    "armchair": {
        "label": "armchair",
        "width": 0.80, "depth": 0.75, "height": 0.90,
        "color": [0.65, 0.30, 0.30, 1.0],
        "build": "armchair",
    },
    "dining_table": {
        "label": "dining table",
        "width": 1.40, "depth": 0.80, "height": 0.75,
        "color": [0.55, 0.35, 0.20, 1.0],
        "build": "table",
    },
    "bookshelf": {
        "label": "bookshelf",
        "width": 0.80, "depth": 0.30, "height": 1.80,
        "color": [0.45, 0.30, 0.18, 1.0],
        "build": "bookshelf",
    },
    "wardrobe": {
        "label": "wardrobe",
        "width": 1.20, "depth": 0.55, "height": 1.90,
        "color": [0.50, 0.35, 0.22, 1.0],
        "build": "wardrobe",
    },
    "desk": {
        "label": "desk",
        "width": 1.20, "depth": 0.60, "height": 0.75,
        "color": [0.60, 0.40, 0.25, 1.0],
        "build": "table",
    },
    "single_bed": {
        "label": "single bed frame",
        "width": 2.00, "depth": 0.90, "height": 0.45,
        "color": [0.70, 0.55, 0.35, 1.0],
        "build": "bed",
    },
    "coffee_table": {
        "label": "coffee table",
        "width": 1.00, "depth": 0.50, "height": 0.40,
        "color": [0.50, 0.33, 0.18, 1.0],
        "build": "table",
    },
    "dresser": {
        "label": "dresser",
        "width": 1.00, "depth": 0.45, "height": 1.10,
        "color": [0.55, 0.38, 0.22, 1.0],
        "build": "wardrobe",
    },
    "stool": {
        "label": "stool",
        "width": 0.40, "depth": 0.40, "height": 0.50,
        "color": [0.60, 0.45, 0.30, 1.0],
        "build": "stool",
    },
}

COLORS = {
    "blue":   [0.25, 0.45, 0.70, 1.0],
    "red":    [0.70, 0.25, 0.25, 1.0],
    "green":  [0.25, 0.60, 0.35, 1.0],
    "brown":  [0.55, 0.35, 0.20, 1.0],
    "gray":   [0.50, 0.50, 0.50, 1.0],
    "beige":  [0.75, 0.65, 0.50, 1.0],
    "orange": [0.80, 0.45, 0.15, 1.0],
    "purple": [0.50, 0.30, 0.60, 1.0],
}


# ─── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class DoorScenario:
    scene_id: str
    door_type: str          # key into DOOR_TYPES
    furniture_type: str     # key into FURNITURE
    furniture_color: str    # key into COLORS
    fits: bool              # ground truth
    question: str
    difficulty: str         # medium/hard
    reasoning: str


# ─── Ground Truth ─────────────────────────────────────────────────────

def _can_fit(furniture_type: str, door_type: str, orientation: str = "normal") -> tuple[bool, str]:
    """Check if furniture can fit through door in given orientation."""
    furn = FURNITURE[furniture_type]
    door = DOOR_TYPES[door_type]

    fw, fd, fh = furn["width"], furn["depth"], furn["height"]
    dw, dh = door["width"], door["height"]

    if orientation == "normal":
        fits_w = fw <= dw - 0.02
        fits_h = fh <= dh - 0.02
        if fits_w and fits_h:
            return True, f"{furn['label']} is {fw*100:.0f}cm wide × {fh*100:.0f}cm tall, door is {dw*100:.0f}cm × {dh*100:.0f}cm — fits"
        if not fits_w:
            return False, f"{furn['label']} is {fw*100:.0f}cm wide, door is only {dw*100:.0f}cm wide"
        return False, f"{furn['label']} is {fh*100:.0f}cm tall, door is only {dh*100:.0f}cm tall"

    elif orientation == "sideways":
        fits_w = fd <= dw - 0.02
        fits_h = fh <= dh - 0.02
        if fits_w and fits_h:
            return True, f"Turned sideways: {fd*100:.0f}cm deep < {dw*100:.0f}cm door width, {fh*100:.0f}cm tall < {dh*100:.0f}cm door height"
        if not fits_w:
            return False, f"Even sideways: {fd*100:.0f}cm deep > {dw*100:.0f}cm door width"
        return False, f"Even sideways: {fh*100:.0f}cm tall > {dh*100:.0f}cm door height"

    elif orientation == "any":
        # Check all orientations: normal, sideways, on-side, tilted
        dims = sorted([fw, fd, fh])  # smallest to largest
        # Best case: two smallest dimensions go through the door
        if dims[0] <= dw - 0.02 and dims[1] <= dh - 0.02:
            return True, f"Can fit in some orientation: smallest cross-section {dims[0]*100:.0f}cm × {dims[1]*100:.0f}cm fits through {dw*100:.0f}cm × {dh*100:.0f}cm door"
        if dims[0] <= dh - 0.02 and dims[1] <= dw - 0.02:
            return True, f"Can fit in some orientation: {dims[1]*100:.0f}cm × {dims[0]*100:.0f}cm fits through {dw*100:.0f}cm × {dh*100:.0f}cm door"
        return False, f"No orientation works: smallest cross-section is {dims[0]*100:.0f}cm × {dims[1]*100:.0f}cm, door is {dw*100:.0f}cm × {dh*100:.0f}cm"


# ─── Furniture XML Builders ───────────────────────────────────────────

def _rgba(color_name: str) -> str:
    c = COLORS.get(color_name, COLORS["brown"])
    return " ".join(f"{v:.2f}" for v in c)


def _build_furniture_xml(ftype: str, color: str, pos: list) -> str:
    furn = FURNITURE[ftype]
    rgba = _rgba(color)
    px, py, pz = pos
    hw, hd, hh = furn["width"]/2, furn["depth"]/2, furn["height"]/2
    build = furn["build"]

    if build == "couch":
        return f"""
    <body name="furniture" pos="{px:.3f} {py:.3f} {pz:.3f}">
      <geom type="box" size="{hw:.3f} {hd:.3f} {hh*0.55:.3f}" rgba="{rgba}" pos="0 0 {hh*0.55:.3f}"/>
      <geom type="box" size="{hw:.3f} 0.06 {hh*0.45:.3f}" rgba="{rgba}" pos="0 {-hd+0.06:.3f} {hh*1.1 + hh*0.45:.3f}"/>
      <geom type="box" size="0.06 {hd*0.85:.3f} {hh*0.3:.3f}" rgba="{rgba}" pos="{-hw+0.06:.3f} 0.04 {hh*1.1 + hh*0.15:.3f}"/>
      <geom type="box" size="0.06 {hd*0.85:.3f} {hh*0.3:.3f}" rgba="{rgba}" pos="{hw-0.06:.3f} 0.04 {hh*1.1 + hh*0.15:.3f}"/>
    </body>"""

    if build == "armchair":
        return f"""
    <body name="furniture" pos="{px:.3f} {py:.3f} {pz:.3f}">
      <geom type="box" size="{hw:.3f} {hd:.3f} {hh*0.5:.3f}" rgba="{rgba}" pos="0 0 {hh*0.5:.3f}"/>
      <geom type="box" size="{hw:.3f} 0.06 {hh*0.5:.3f}" rgba="{rgba}" pos="0 {-hd+0.06:.3f} {hh*1.0 + hh*0.25:.3f}"/>
      <geom type="box" size="0.06 {hd*0.85:.3f} {hh*0.25:.3f}" rgba="{rgba}" pos="{-hw+0.06:.3f} 0.04 {hh*1.0 + hh*0.12:.3f}"/>
      <geom type="box" size="0.06 {hd*0.85:.3f} {hh*0.25:.3f}" rgba="{rgba}" pos="{hw-0.06:.3f} 0.04 {hh*1.0 + hh*0.12:.3f}"/>
    </body>"""

    if build == "table":
        return f"""
    <body name="furniture" pos="{px:.3f} {py:.3f} {pz:.3f}">
      <geom type="box" size="{hw:.3f} {hd:.3f} 0.02" rgba="{rgba}" pos="0 0 {furn['height'] - 0.02:.3f}"/>
      <geom type="cylinder" size="0.03 {hh - 0.02:.3f}" rgba="{rgba}" pos="{hw-0.06:.3f} {hd-0.06:.3f} {hh - 0.02:.3f}"/>
      <geom type="cylinder" size="0.03 {hh - 0.02:.3f}" rgba="{rgba}" pos="{-hw+0.06:.3f} {hd-0.06:.3f} {hh - 0.02:.3f}"/>
      <geom type="cylinder" size="0.03 {hh - 0.02:.3f}" rgba="{rgba}" pos="{hw-0.06:.3f} {-hd+0.06:.3f} {hh - 0.02:.3f}"/>
      <geom type="cylinder" size="0.03 {hh - 0.02:.3f}" rgba="{rgba}" pos="{-hw+0.06:.3f} {-hd+0.06:.3f} {hh - 0.02:.3f}"/>
    </body>"""

    if build == "bookshelf" or build == "wardrobe":
        return f"""
    <body name="furniture" pos="{px:.3f} {py:.3f} {pz:.3f}">
      <geom type="box" size="{hw:.3f} {hd:.3f} {hh:.3f}" rgba="{rgba}" pos="0 0 {hh:.3f}"/>
    </body>"""

    if build == "bed":
        return f"""
    <body name="furniture" pos="{px:.3f} {py:.3f} {pz:.3f}">
      <geom type="box" size="{hw:.3f} {hd:.3f} {hh*0.4:.3f}" rgba="{rgba}" pos="0 0 {hh*0.4:.3f}"/>
      <geom type="box" size="0.04 {hd:.3f} {hh:.3f}" rgba="{rgba}" pos="{hw-0.04:.3f} 0 {hh:.3f}"/>
      <geom type="box" size="0.04 {hd:.3f} {hh*0.7:.3f}" rgba="{rgba}" pos="{-hw+0.04:.3f} 0 {hh*0.7:.3f}"/>
    </body>"""

    if build == "stool":
        return f"""
    <body name="furniture" pos="{px:.3f} {py:.3f} {pz:.3f}">
      <geom type="cylinder" size="{hw:.3f} 0.025" rgba="{rgba}" pos="0 0 {furn['height'] - 0.025:.3f}"/>
      <geom type="cylinder" size="0.02 {hh - 0.025:.3f}" rgba="{rgba}" pos="{hw*0.6:.3f} {hw*0.6:.3f} {hh - 0.025:.3f}"/>
      <geom type="cylinder" size="0.02 {hh - 0.025:.3f}" rgba="{rgba}" pos="{-hw*0.6:.3f} {hw*0.6:.3f} {hh - 0.025:.3f}"/>
      <geom type="cylinder" size="0.02 {hh - 0.025:.3f}" rgba="{rgba}" pos="{hw*0.6:.3f} {-hw*0.6:.3f} {hh - 0.025:.3f}"/>
      <geom type="cylinder" size="0.02 {hh - 0.025:.3f}" rgba="{rgba}" pos="{-hw*0.6:.3f} {-hw*0.6:.3f} {hh - 0.025:.3f}"/>
    </body>"""

    # Fallback box
    return f"""
    <body name="furniture" pos="{px:.3f} {py:.3f} {pz:.3f}">
      <geom type="box" size="{hw:.3f} {hd:.3f} {hh:.3f}" rgba="{rgba}" pos="0 0 {hh:.3f}"/>
    </body>"""


# ─── Scene Builder ────────────────────────────────────────────────────

def _build_door_xml(door_type: str) -> str:
    door = DOOR_TYPES[door_type]
    dw = door["width"] / 2
    dh = door["height"]
    wall_w = 0.60
    wall_color = "0.85 0.82 0.78 1"
    frame_color = "0.55 0.35 0.20 1"

    return f"""
    <!-- Wall left -->
    <body name="wall_left" pos="{-dw - wall_w:.3f} 0 {(dh + 0.4)/2:.3f}">
      <geom type="box" size="{wall_w:.3f} 0.1 {(dh + 0.4)/2:.3f}" rgba="{wall_color}"/>
    </body>
    <!-- Wall right -->
    <body name="wall_right" pos="{dw + wall_w:.3f} 0 {(dh + 0.4)/2:.3f}">
      <geom type="box" size="{wall_w:.3f} 0.1 {(dh + 0.4)/2:.3f}" rgba="{wall_color}"/>
    </body>
    <!-- Wall top -->
    <body name="wall_top" pos="0 0 {dh + 0.2:.3f}">
      <geom type="box" size="{dw + wall_w * 2:.3f} 0.1 0.2" rgba="{wall_color}"/>
    </body>
    <!-- Frame left -->
    <body name="frame_left" pos="{-dw:.3f} 0.05 {dh/2:.3f}">
      <geom type="box" size="0.03 0.06 {dh/2:.3f}" rgba="{frame_color}"/>
    </body>
    <!-- Frame right -->
    <body name="frame_right" pos="{dw:.3f} 0.05 {dh/2:.3f}">
      <geom type="box" size="0.03 0.06 {dh/2:.3f}" rgba="{frame_color}"/>
    </body>
    <!-- Frame top -->
    <body name="frame_top" pos="0 0.05 {dh:.3f}">
      <geom type="box" size="{dw + 0.03:.3f} 0.06 0.03" rgba="{frame_color}"/>
    </body>"""


def build_scene_xml(scenario: DoorScenario) -> str:
    door_xml = _build_door_xml(scenario.door_type)
    furniture_xml = _build_furniture_xml(
        scenario.furniture_type, scenario.furniture_color, [0, 1.5, 0]
    )

    return f"""<mujoco model="door_passage">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <rgba haze="0.85 0.9 0.95 1"/>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="20"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <worldbody>
    <light pos="0 2 4" dir="0 -0.5 -0.8" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="-2 -1 3" dir="0.3 0.3 -0.6" diffuse="0.35 0.35 0.38"/>
    <light pos="3 0 3" dir="-0.3 0 -0.7" diffuse="0.3 0.3 0.32"/>
    <geom type="plane" size="5 5 0.01" rgba="0.92 0.90 0.87 1"/>
    {door_xml}
    {furniture_xml}
  </worldbody>
</mujoco>"""


# ─── Rendering ────────────────────────────────────────────────────────

def render_scenario(scenario: DoorScenario) -> list[str]:
    xml = build_scene_xml(scenario)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0.2, 0.9]
    cam.distance = 5.0
    cam.azimuth = 315
    cam.elevation = -20
    renderer.update_scene(data, cam)
    px = renderer.render()
    renderer.close()

    buf = io.BytesIO()
    Image.fromarray(px).save(buf, format="JPEG", quality=88)
    return ["data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()]


# ─── 20 Curated Scenarios ────────────────────────────────────────────

def generate_20_scenarios() -> list[DoorScenario]:
    scenarios = []

    # ── 1-5: Normal orientation questions ──

    # 1. Couch through standard door — NO (180cm > 90cm)
    scenarios.append(DoorScenario(
        "door_01", "standard", "couch", "blue", False,
        "Can the blue couch fit through the doorway without rotating it?",
        "medium",
        "Couch is 180cm wide, door is 90cm — too wide without rotation",
    ))

    # 2. Armchair through standard door — YES (80cm < 90cm)
    scenarios.append(DoorScenario(
        "door_02", "standard", "armchair", "red", True,
        "Can the red armchair pass through the doorway without rotating it?",
        "medium",
        "Armchair is 80cm wide, door is 90cm — fits without rotation",
    ))

    # 3. Dining table through standard door — NO (140cm > 90cm)
    scenarios.append(DoorScenario(
        "door_03", "standard", "dining_table", "brown", False,
        "Can the brown dining table fit through the doorway without rotating it?",
        "medium",
        "Table is 140cm wide, door is 90cm — too wide without rotation",
    ))

    # 4. Stool through standard door — YES (40cm < 90cm)
    scenarios.append(DoorScenario(
        "door_04", "standard", "stool", "gray", True,
        "Can the gray stool fit through the doorway?",
        "medium",
        "Stool is 40cm wide, door is 90cm — fits easily",
    ))

    # 5. Bookshelf through standard door — YES (80cm < 90cm, 180cm < 200cm)
    fits, reason = _can_fit("bookshelf", "standard", "normal")
    scenarios.append(DoorScenario(
        "door_05", "standard", "bookshelf", "brown", fits,
        "Can the brown bookshelf fit through the doorway upright?",
        "medium", reason,
    ))

    # ── 6-10: Different door sizes ──

    # 6. Armchair through narrow door — NO (85cm > 75cm)
    scenarios.append(DoorScenario(
        "door_06", "narrow", "armchair", "red", False,
        "Can the red armchair fit through the doorway?",
        "medium",
        "Armchair is 85cm wide, door is only 75cm",
    ))

    # 7. Coffee table through wide door — YES (100cm < 110cm)
    scenarios.append(DoorScenario(
        "door_07", "wide", "coffee_table", "brown", True,
        "Can the brown coffee table fit through the doorway?",
        "medium",
        "Coffee table is 100cm wide, door is 110cm — fits",
    ))

    # 8. Wardrobe through standard door — NO (120cm > 90cm)
    scenarios.append(DoorScenario(
        "door_08", "standard", "wardrobe", "brown", False,
        "Can the brown wardrobe fit through the doorway without rotating it?",
        "medium",
        "Wardrobe is 120cm wide, door is 90cm — too wide without rotation",
    ))

    # 9. Desk through wide door — NO (120cm > 110cm)
    fits, reason = _can_fit("desk", "wide", "normal")
    scenarios.append(DoorScenario(
        "door_09", "wide", "desk", "beige", fits,
        "Can the beige desk fit through the doorway without rotating it?",
        "medium", reason,
    ))

    # 10. Wardrobe through short door — NO (190cm > 175cm height)
    scenarios.append(DoorScenario(
        "door_10", "short", "wardrobe", "brown", False,
        "Can the brown wardrobe fit through the doorway upright?",
        "medium",
        "Wardrobe is 190cm tall, door is only 175cm",
    ))

    # ── 11-15: Sideways/orientation questions ──

    # 11. Couch sideways through standard — YES (80cm < 90cm)
    fits, reason = _can_fit("couch", "standard", "sideways")
    scenarios.append(DoorScenario(
        "door_11", "standard", "couch", "blue", fits,
        "Can the blue couch fit through the doorway if turned sideways?",
        "medium", reason,
    ))

    # 12. Dining table sideways through standard — YES (80cm < 90cm)
    fits, reason = _can_fit("dining_table", "standard", "sideways")
    scenarios.append(DoorScenario(
        "door_12", "standard", "dining_table", "brown", fits,
        "Can the brown dining table fit through the doorway if turned sideways?",
        "medium", reason,
    ))

    # 13. Wardrobe sideways through standard — YES (55cm < 90cm)
    fits, reason = _can_fit("wardrobe", "standard", "sideways")
    scenarios.append(DoorScenario(
        "door_13", "standard", "wardrobe", "brown", fits,
        "Can the wardrobe fit through the doorway if turned sideways?",
        "medium", reason,
    ))

    # 14. Single bed through narrow door any orientation — YES (45cm × 90cm < 75cm × 200cm)
    fits, reason = _can_fit("single_bed", "narrow", "any")
    scenarios.append(DoorScenario(
        "door_14", "narrow", "single_bed", "beige", fits,
        "Can the bed frame fit through the doorway in any orientation?",
        "hard", reason,
    ))

    # 15. Desk through narrow door any orientation — YES (60cm × 75cm < 75cm × 200cm)
    fits, reason = _can_fit("desk", "narrow", "any")
    scenarios.append(DoorScenario(
        "door_15", "narrow", "desk", "brown", fits,
        "Can the desk fit through the doorway in any orientation?",
        "medium", reason,
    ))

    # ── 16-20: "Any orientation" questions ──

    # 16. Couch through narrow door in any way
    fits, reason = _can_fit("couch", "narrow", "any")
    scenarios.append(DoorScenario(
        "door_16", "narrow", "couch", "blue", fits,
        "Is there any orientation in which the blue couch can pass through the doorway?",
        "hard", reason,
    ))

    # 17. Dresser through standard door any way — YES
    fits, reason = _can_fit("dresser", "standard", "any")
    scenarios.append(DoorScenario(
        "door_17", "standard", "dresser", "orange", fits,
        "Is there any way to get the orange dresser through the doorway?",
        "hard", reason,
    ))

    # 18. Single bed through standard any way — YES
    fits, reason = _can_fit("single_bed", "standard", "any")
    scenarios.append(DoorScenario(
        "door_18", "standard", "single_bed", "beige", fits,
        "Is there any orientation in which the bed frame can pass through the doorway?",
        "hard", reason,
    ))

    # 19. Wardrobe through narrow any way
    fits, reason = _can_fit("wardrobe", "narrow", "any")
    scenarios.append(DoorScenario(
        "door_19", "narrow", "wardrobe", "brown", fits,
        "Is there any way to get the wardrobe through the doorway?",
        "hard", reason,
    ))

    # 20. Bookshelf through short door any way
    fits, reason = _can_fit("bookshelf", "short", "any")
    scenarios.append(DoorScenario(
        "door_20", "short", "bookshelf", "brown", fits,
        "Is there any orientation in which the bookshelf can fit through the doorway?",
        "hard", reason,
    ))

    return scenarios
    scenarios.append(DoorScenario(
        "door_18", "standard", "single_bed", "beige", fits,
        "Is there any way to move the bed frame through the standard doorway?",
        "hard", reason,
    ))

    # 19. Wardrobe through narrow any way — check
    fits, reason = _can_fit("wardrobe", "narrow", "any")
    scenarios.append(DoorScenario(
        "door_19", "narrow", "wardrobe", "brown", fits,
        "Can the wardrobe fit through the narrow doorway in any orientation?",
        "hard", reason,
    ))

    # 20. Bookshelf through short door any way
    fits, reason = _can_fit("bookshelf", "short", "any")
    scenarios.append(DoorScenario(
        "door_20", "short", "bookshelf", "brown", fits,
        "Is there any way to get the bookshelf through this short doorway?",
        "hard", reason,
    ))

    return scenarios


# ─── Review HTML ──────────────────────────────────────────────────────

def generate_review_html(scenarios: list[DoorScenario] | None = None) -> str:
    if scenarios is None:
        scenarios = generate_20_scenarios()

    cards = []
    for sc in scenarios:
        try:
            imgs = render_scenario(sc)
            img_html = f'<img src="{imgs[0]}" style="border-radius:8px;border:1px solid #e0e0e0;max-height:350px">'
        except Exception as e:
            img_html = f'<div style="color:red">Render error: {e}</div>'

        fit_badge = (
            '<span style="background:#dcfce7;color:#166534;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">FITS</span>'
            if sc.fits else
            '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">DOESN\'T FIT</span>'
        )

        door = DOOR_TYPES[sc.door_type]
        furn = FURNITURE[sc.furniture_type]

        cards.append(f"""
<div style="background:white;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
    <span style="font-weight:700;color:#333">{sc.scene_id}</span>
    <span style="background:#f3f4f6;color:#555;padding:2px 8px;border-radius:10px;font-size:11px">{sc.difficulty}</span>
    {fit_badge}
    <span style="font-size:11px;color:#888">Door: {door['label']} ({door['width']*100:.0f}cm × {door['height']*100:.0f}cm) | {furn['label']} ({furn['width']*100:.0f}×{furn['depth']*100:.0f}×{furn['height']*100:.0f}cm)</span>
  </div>
  <p style="font-size:14px;color:#333;margin:4px 0 8px">💬 "{sc.question}"</p>
  <div style="display:flex;gap:8px;margin:8px 0">{img_html}</div>
  <p style="font-size:12px;color:#888;margin-top:6px">💡 {sc.reasoning}</p>
</div>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Door Passage Scenarios</title></head>
<body style="font-family:-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f5f5f5">
<h1>🚪 Door Passage Benchmark — 20 Scenarios</h1>
<p style="color:#666">Can the furniture fit through the doorway?</p>
{"".join(cards)}
</body></html>"""
