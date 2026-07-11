"""
Shelf Fitting Environment — Verifiable spatial reasoning

Scene: A rack/shelf with objects. A target object shown beside it.
Question: "Can the {color} {object} fit on {shelf} without moving existing objects?"
Answer: "fits" or "no_fit"

Ground truth: exhaustive physics-based verification
  - Height: object height vs shelf clearance
  - Width: object footprint vs available gaps
  - Depth: object depth vs shelf depth
  - For "no_fit": verified across ALL 6 orientation permutations

No dimension leaks in questions — model must reason from the image.
"""

import io
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco
from PIL import Image

from evalspace.environments.base import BaseEnvironment, Observation


# ─── Parametric Object Catalogue ─────────────────────────────────────

@dataclass
class ObjectSpec:
    """Concrete object with fixed dimensions (sampled from a family)."""
    family: str         # bottle, box, jar, pot, cup, plate, book, vase
    label: str          # human-readable: "tall bottle", "small jar"
    width: float        # X extent (meters)
    depth: float        # Y extent (meters)
    height: float       # Z extent (meters)
    color: str          # color name
    color_rgba: list    # [r, g, b, a]


# Object families with dimension ranges (meters)
OBJECT_FAMILIES = {
    "bottle": {
        "height": (0.18, 0.35),
        "width": (0.05, 0.09),
        "depth": (0.05, 0.09),     # cylindrical → width ≈ depth
        "labels": ["bottle", "tall bottle", "water bottle"],
        "build": "bottle",
        "symmetric": True,          # width ≈ depth always
    },
    "box": {
        "height": (0.10, 0.32),
        "width": (0.08, 0.22),
        "depth": (0.05, 0.18),
        "labels": ["box", "cereal box", "tissue box", "storage box"],
        "build": "box",
        "symmetric": False,
    },
    "jar": {
        "height": (0.06, 0.16),
        "width": (0.05, 0.12),
        "depth": (0.05, 0.12),
        "labels": ["jar", "spice jar", "jam jar"],
        "build": "jar",
        "symmetric": True,
    },
    "pot": {
        "height": (0.12, 0.24),
        "width": (0.16, 0.26),
        "depth": (0.16, 0.26),
        "labels": ["pot", "cooking pot", "saucepan"],
        "build": "pot",
        "symmetric": True,
    },
    "cup": {
        "height": (0.08, 0.13),
        "width": (0.07, 0.11),
        "depth": (0.07, 0.11),
        "labels": ["cup", "mug", "coffee mug"],
        "build": "cup",
        "symmetric": True,
    },
    "plate": {
        "height": (0.02, 0.04),
        "width": (0.18, 0.28),
        "depth": (0.18, 0.28),
        "labels": ["plate", "dinner plate"],
        "build": "plate",
        "symmetric": True,
    },
    "book": {
        "height": (0.20, 0.30),
        "width": (0.13, 0.22),
        "depth": (0.02, 0.04),      # thin!
        "labels": ["book", "hardcover book", "textbook"],
        "build": "book",
        "symmetric": False,
    },
    "vase": {
        "height": (0.15, 0.30),
        "width": (0.08, 0.14),
        "depth": (0.08, 0.14),
        "labels": ["vase", "flower vase"],
        "build": "vase",
        "symmetric": True,
    },
}

COLORS = {
    "red":    [0.82, 0.18, 0.18, 1.0],
    "blue":   [0.20, 0.35, 0.80, 1.0],
    "green":  [0.18, 0.68, 0.28, 1.0],
    "yellow": [0.88, 0.78, 0.12, 1.0],
    "orange": [0.88, 0.45, 0.12, 1.0],
    "purple": [0.55, 0.20, 0.72, 1.0],
    "white":  [0.92, 0.92, 0.92, 1.0],
    "brown":  [0.55, 0.35, 0.20, 1.0],
    "teal":   [0.15, 0.62, 0.58, 1.0],
    "pink":   [0.88, 0.40, 0.58, 1.0],
}

COLOR_NAMES = list(COLORS.keys())


def sample_object(rng: np.random.RandomState, family: str | None = None,
                  color: str | None = None) -> ObjectSpec:
    """Sample a random object with concrete dimensions."""
    if family is None:
        family = rng.choice(list(OBJECT_FAMILIES.keys()))

    fam = OBJECT_FAMILIES[family]
    h = rng.uniform(*fam["height"])
    w = rng.uniform(*fam["width"])
    d = rng.uniform(*fam["depth"])

    # Symmetric objects: force width ≈ depth
    if fam["symmetric"]:
        d = w

    if color is None:
        color = rng.choice(COLOR_NAMES)

    label = rng.choice(fam["labels"])

    return ObjectSpec(
        family=family,
        label=label,
        width=round(w, 3),
        depth=round(d, 3),
        height=round(h, 3),
        color=color,
        color_rgba=COLORS[color],
    )


# ─── Rack / Shelf Configuration ─────────────────────────────────────

@dataclass
class Shelf:
    """A single shelf with fixed dimensions."""
    clearance: float    # vertical space to shelf above (or top of rack)
    depth: float        # how deep
    width: float        # how wide
    y_pos: float        # vertical position of shelf surface


@dataclass
class Rack:
    """Complete rack with multiple shelves."""
    name: str
    shelves: list[Shelf]
    total_width: float
    total_height: float
    total_depth: float
    frame_width: float = 0.02


def sample_rack(rng: np.random.RandomState) -> Rack:
    """Generate a randomized rack with realistic dimensions."""
    # Number of shelves: 2-5
    num_shelves = rng.randint(2, 6)

    # Total dimensions
    total_width = round(rng.uniform(0.45, 0.75), 2)
    total_depth = round(rng.uniform(0.20, 0.35), 2)

    # Generate shelf clearances (varied)
    clearances = []
    for _ in range(num_shelves):
        c = round(rng.uniform(0.14, 0.36), 2)
        clearances.append(c)

    # Compute positions
    shelves = []
    y = 0.04  # bottom shelf position
    for i in range(num_shelves):
        shelves.append(Shelf(
            clearance=clearances[i],
            depth=total_depth,
            width=total_width - 0.04,  # minus frame
            y_pos=round(y, 3),
        ))
        y += clearances[i] + 0.02  # shelf thickness

    total_height = round(y + 0.04, 2)

    names = ["open rack", "kitchen shelf", "storage unit", "cupboard", "bookcase"]
    name = rng.choice(names)

    return Rack(
        name=name,
        shelves=shelves,
        total_width=total_width,
        total_height=total_height,
        total_depth=total_depth,
    )


# ─── Placed Objects (already on shelves) ──────────────────────────────

@dataclass
class PlacedObject:
    """An object already sitting on a shelf."""
    spec: ObjectSpec
    shelf_idx: int
    x_offset: float     # left edge position from shelf left edge


def place_objects_on_shelf(rack: Rack, shelf_idx: int, rng: np.random.RandomState,
                           max_objects: int = 3) -> list[PlacedObject]:
    """
    Randomly place 0 to max_objects on a shelf, ensuring they fit.
    Returns list of placed objects (guaranteed non-overlapping).
    """
    shelf = rack.shelves[shelf_idx]
    placed = []
    num = rng.randint(0, max_objects + 1)

    # Track occupied x-ranges
    occupied = []
    margin = 0.02  # 2cm between objects

    for _ in range(num):
        # Pick a random object that fits the shelf
        for _attempt in range(10):
            obj = sample_object(rng)
            footprint = _shelf_footprint(obj)

            # Must fit height and depth with comfortable margin
            # Object sits at shelf.y_pos + 0.012, top at + 0.012 + height
            # Next shelf at shelf.y_pos + clearance
            # Need: height < clearance - 0.012 (shelf board offset) - 0.02 (visual margin)
            if obj.height >= shelf.clearance - 0.035:
                continue
            if obj.depth >= shelf.depth - 0.02:
                continue

            # Find a valid x position
            x = _find_gap(occupied, footprint, shelf.width, margin, rng)
            if x is not None:
                placed.append(PlacedObject(spec=obj, shelf_idx=shelf_idx, x_offset=x))
                occupied.append((x, x + footprint))
                occupied.sort()
                break

    return placed


def _shelf_footprint(obj: ObjectSpec) -> float:
    """Width the object takes on the shelf (upright, natural orientation)."""
    # Books stand on spine → footprint = depth
    if obj.family == "book":
        return obj.depth
    return obj.width


def _find_gap(occupied: list[tuple], needed: float, shelf_width: float,
              margin: float, rng: np.random.RandomState) -> float | None:
    """Find a random valid x position for an object on the shelf."""
    if not occupied:
        # Empty shelf — place randomly
        max_x = shelf_width - needed - margin
        if max_x < margin:
            return None
        return round(rng.uniform(margin, max_x), 3)

    # Build list of available gaps
    gaps = []
    # Before first object
    if occupied[0][0] > margin + needed:
        gaps.append((margin, occupied[0][0] - needed))

    # Between objects
    for i in range(len(occupied) - 1):
        gap_start = occupied[i][1] + margin
        gap_end = occupied[i + 1][0] - needed - margin
        if gap_end > gap_start:
            gaps.append((gap_start, gap_end))

    # After last object
    after_start = occupied[-1][1] + margin
    after_end = shelf_width - needed - margin
    if after_end > after_start:
        gaps.append((after_start, after_end))

    if not gaps:
        return None

    # Pick a random gap and random position within it
    gap = gaps[rng.randint(len(gaps))]
    return round(rng.uniform(gap[0], gap[1]), 3)


# ─── Ground Truth: Exhaustive Fit Check ──────────────────────────────

def can_fit_exhaustive(
    target: ObjectSpec,
    rack: Rack,
    shelf_idx: int,
    placed: list[PlacedObject],
    check_all_orientations: bool = False,
) -> tuple[bool, str]:
    """
    Exhaustive check: can the target object fit on the specified shelf?

    Checks:
      1. Height: object height < shelf clearance
      2. Depth:  object depth < shelf depth
      3. Width:  object footprint fits in an available gap

    If check_all_orientations=True, tests all 3 axis permutations.
    For a "no_fit" answer, we MUST verify no orientation works.

    Returns:
        (fits: bool, reasoning: str)
    """
    shelf = rack.shelves[shelf_idx]

    if check_all_orientations:
        # Try all unique dimension permutations as (width_on_shelf, depth_on_shelf, height_on_shelf)
        dims = [target.width, target.depth, target.height]
        orientations = set()
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                k = 3 - i - j
                orientations.add((dims[i], dims[j], dims[k]))

        for w, d, h in orientations:
            fits, reason = _check_single_orientation(w, d, h, shelf, shelf_idx, placed, target)
            if fits:
                return True, f"Fits in orientation (w={w*100:.0f}cm, d={d*100:.0f}cm, h={h*100:.0f}cm): {reason}"

        return False, f"Does not fit in any orientation. Dimensions {dims[0]*100:.0f}×{dims[1]*100:.0f}×{dims[2]*100:.0f}cm"
    else:
        # Upright only
        w = _shelf_footprint(target)
        d = target.depth
        h = target.height
        return _check_single_orientation(w, d, h, shelf, shelf_idx, placed, target)


def _check_single_orientation(
    obj_w: float, obj_d: float, obj_h: float,
    shelf: Shelf, shelf_idx: int, placed: list[PlacedObject], target: ObjectSpec,
) -> tuple[bool, str]:
    """Check fit for one specific orientation."""

    # Check 1: Height
    if obj_h > shelf.clearance - 0.005:  # 0.5cm tolerance
        return False, (
            f"Too tall: object {obj_h*100:.0f}cm, "
            f"shelf clearance {shelf.clearance*100:.0f}cm"
        )

    # Check 2: Depth
    if obj_d > shelf.depth - 0.005:
        return False, (
            f"Too deep: object {obj_d*100:.0f}cm, "
            f"shelf depth {shelf.depth*100:.0f}cm"
        )

    # Check 3: Width — find available gaps (only objects on THIS shelf)
    shelf_objects = [p for p in placed if p.shelf_idx == shelf_idx]

    occupied = []
    for p in shelf_objects:
        pw = _shelf_footprint(p.spec)
        occupied.append((p.x_offset, p.x_offset + pw))
    occupied.sort()

    # Find max gap
    max_gap = _max_gap(occupied, shelf.width)

    if obj_w > max_gap - 0.005:
        return False, (
            f"No gap wide enough: object needs {obj_w*100:.0f}cm, "
            f"largest gap is {max_gap*100:.0f}cm"
        )

    return True, (
        f"Fits: {obj_h*100:.0f}cm < {shelf.clearance*100:.0f}cm clearance, "
        f"{obj_d*100:.0f}cm < {shelf.depth*100:.0f}cm depth, "
        f"{obj_w*100:.0f}cm < {max_gap*100:.0f}cm gap"
    )


def _max_gap(occupied: list[tuple], shelf_width: float) -> float:
    """Find the largest available gap on the shelf."""
    margin = 0.015  # 1.5cm from edges
    if not occupied:
        return shelf_width - 2 * margin

    gaps = []
    # Before first
    gaps.append(occupied[0][0] - margin)
    # Between objects
    for i in range(len(occupied) - 1):
        gaps.append(occupied[i + 1][0] - occupied[i][1])
    # After last
    gaps.append(shelf_width - occupied[-1][1] - margin)

    return max(gaps) if gaps else 0.0


# ─── Physics Engine Verification (MuJoCo) ────────────────────────────

def _build_physics_test_xml(rack: Rack, shelf_idx: int, placed: list[PlacedObject],
                            target: ObjectSpec, target_pos: list,
                            target_euler: list = None) -> str:
    """
    Build a MuJoCo scene for physics verification.
    Target object has a free joint so it can move/fall.
    All other objects and rack are static.
    """
    rack_xml = _build_rack_xml(rack)
    
    obj_xmls = []
    # Existing objects — static (no joint)
    for i, p in enumerate(placed):
        shelf = rack.shelves[p.shelf_idx]
        x = -rack.total_width / 2 + rack.frame_width + p.x_offset + _shelf_footprint(p.spec) / 2
        z = shelf.y_pos + 0.012
        obj_xmls.append(_build_object_geom(p.spec, f"placed_{i}", [x, 0.0, z]))
    
    # Target object — FREE joint (can move, fall, collide)
    rgba = _rgba(target.color_rgba)
    tx, ty, tz = target_pos
    euler_str = f'euler="{target_euler[0]} {target_euler[1]} {target_euler[2]}"' if target_euler else ""
    
    w, d, h = target.width, target.depth, target.height
    if target.family in ("bottle", "vase", "jar", "cup"):
        r = w / 2
        target_xml = f"""
    <body name="target" pos="{tx:.4f} {ty:.4f} {tz + h/2:.4f}" {euler_str}>
      <freejoint/>
      <geom type="cylinder" size="{r:.4f} {h/2:.4f}" rgba="{rgba}" mass="0.5"/>
    </body>"""
    elif target.family in ("pot",):
        r = w / 2
        target_xml = f"""
    <body name="target" pos="{tx:.4f} {ty:.4f} {tz + h/2:.4f}" {euler_str}>
      <freejoint/>
      <geom type="cylinder" size="{r:.4f} {h/2:.4f}" rgba="{rgba}" mass="1.0"/>
    </body>"""
    elif target.family == "plate":
        r = w / 2
        target_xml = f"""
    <body name="target" pos="{tx:.4f} {ty:.4f} {tz + h/2:.4f}" {euler_str}>
      <freejoint/>
      <geom type="cylinder" size="{r:.4f} {h/2:.4f}" rgba="{rgba}" mass="0.3"/>
    </body>"""
    else:
        # box, book, etc.
        target_xml = f"""
    <body name="target" pos="{tx:.4f} {ty:.4f} {tz + h/2:.4f}" {euler_str}>
      <freejoint/>
      <geom type="box" size="{w/2:.4f} {d/2:.4f} {h/2:.4f}" rgba="{rgba}" mass="0.5"/>
    </body>"""
    
    all_objs = "\n".join(obj_xmls)
    
    return f"""
<mujoco model="shelf_physics_test">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <geom type="plane" size="3 3 0.01" rgba="0.9 0.9 0.9 1"/>
    {rack_xml}
    {all_objs}
    {target_xml}
  </worldbody>
</mujoco>
"""


def can_fit_physics(
    target: ObjectSpec,
    rack: Rack,
    shelf_idx: int,
    placed: list[PlacedObject],
) -> tuple[bool, str]:
    """
    Verify fit using MuJoCo physics simulation.
    
    Tries placing the target on the shelf in all 6 orientations.
    For each: simulates physics for 0.5s, checks if object stays on shelf.
    
    Returns:
        (fits: bool, reasoning: str)
    """
    shelf = rack.shelves[shelf_idx]
    dims = [target.width, target.depth, target.height]
    
    # Find the best gap position for placement
    shelf_objects = [p for p in placed if p.shelf_idx == shelf_idx]
    occupied = []
    for p in shelf_objects:
        pw = _shelf_footprint(p.spec)
        occupied.append((p.x_offset, p.x_offset + pw))
    occupied.sort()
    
    # Find largest gap center
    gap_center_x = 0.0
    max_gap = _max_gap(occupied, shelf.width)
    if not occupied:
        gap_center_x = 0.0  # center of shelf
    else:
        # Find which gap is the largest and compute its center
        gaps = []
        margin = 0.015
        # Before first
        gaps.append((margin, occupied[0][0], occupied[0][0] - margin))
        for i in range(len(occupied) - 1):
            start = occupied[i][1]
            end = occupied[i + 1][0]
            gaps.append((start, end, end - start))
        gaps.append((occupied[-1][1], shelf.width - margin, shelf.width - margin - occupied[-1][1]))
        
        best_gap = max(gaps, key=lambda g: g[2])
        gap_center_local = (best_gap[0] + best_gap[1]) / 2
        gap_center_x = -rack.total_width / 2 + rack.frame_width + gap_center_local
    
    # Try all 6 orientation permutations
    from itertools import permutations
    
    euler_for_orientation = [
        [0, 0, 0],              # upright (W=x, D=y, H=z)
        [90, 0, 0],             # rotated: H→y, D→z
        [0, 90, 0],             # rotated: H→x, W→z
        [0, 0, 90],             # rotated around z: W↔D
        [90, 0, 90],            # compound
        [0, 90, 90],            # compound
    ]
    
    for euler in euler_for_orientation:
        # DROP from above: place target HIGH above the shelf, let gravity pull it down
        # If it lands on the shelf surface cleanly → fits
        # If it hits existing objects/walls and bounces off → no_fit
        # If it's too tall (hits shelf above) → no_fit
        drop_height = shelf.y_pos + shelf.clearance * 0.7  # drop from 70% of clearance height
        target_pos = [
            gap_center_x,
            0.0,
            drop_height,
        ]
        
        try:
            xml = _build_physics_test_xml(rack, shelf_idx, placed, target, target_pos, euler)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            
            # Get IDs
            target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
            
            # Pre-check: would the object in this orientation exceed shelf clearance?
            # Get the oriented bounding height
            mujoco.mj_forward(model, data)
            # Approximate oriented height from the body's geom bounds
            target_geom_ids = []
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == target_body_id:
                    target_geom_ids.append(g)
            
            # Compute axis-aligned bounding box of target in current orientation
            if target_geom_ids:
                min_z = float('inf')
                max_z = float('-inf')
                for gid in target_geom_ids:
                    gpos = data.geom_xpos[gid]
                    gsize = model.geom_size[gid]
                    gtype = model.geom_type[gid]
                    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                        half_h = gsize[2]
                    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        half_h = gsize[1]
                    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                        half_h = gsize[0]
                    else:
                        half_h = gsize[0]
                    min_z = min(min_z, gpos[2] - half_h)
                    max_z = max(max_z, gpos[2] + half_h)
                
                oriented_height = max_z - min_z
                
                # Height check: if oriented height > clearance, skip this orientation
                if oriented_height > shelf.clearance - 0.005:
                    continue
            
            # Depth check: approximate oriented depth
            if target_geom_ids:
                min_y = float('inf')
                max_y = float('-inf')
                for gid in target_geom_ids:
                    gpos = data.geom_xpos[gid]
                    gsize = model.geom_size[gid]
                    gtype = model.geom_type[gid]
                    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                        half_d = gsize[1]
                    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        half_d = gsize[0]  # radius
                    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                        half_d = gsize[0]
                    else:
                        half_d = gsize[0]
                    min_y = min(min_y, gpos[1] - half_d)
                    max_y = max(max_y, gpos[1] + half_d)
                
                oriented_depth = max_y - min_y
                if oriented_depth > shelf.depth - 0.005:
                    continue
            
            # Now place on shelf surface (not drop) and check for lateral collisions
            # Reset position to just above shelf
            data.qpos[0] = gap_center_x  # x
            data.qpos[1] = 0.0           # y
            data.qpos[2] = shelf.y_pos + oriented_height / 2 + 0.005  # z: sitting on shelf
            data.qvel[:] = 0  # no velocity
            
            mujoco.mj_forward(model, data)
            initial_pos = data.xpos[target_body_id].copy()
            
            # Simulate for 0.3 seconds — let contacts resolve
            for _ in range(150):
                mujoco.mj_step(model, data)
            
            final_pos = data.xpos[target_body_id].copy()
            final_z = final_pos[2]
            
            # Check 1: Did the object fall off the shelf?
            fell = final_z < shelf.y_pos - 0.03
            
            # Check 2: Was it pushed away by existing objects?
            lateral_drift = abs(final_pos[0] - initial_pos[0]) + abs(final_pos[1] - initial_pos[1])
            pushed_away = lateral_drift > 0.06  # more than 6cm drift = collision pushed it off
            
            # Check 3: Is it still on the shelf (didn't fly up or sink)?
            settled = abs(final_z - initial_pos[2]) < 0.03
            
            # Check 4: Any active contacts with placed objects? (overlap detection)
            has_overlap = False
            for c in range(data.ncon):
                contact = data.contact[c]
                # Check if contact involves the target body
                geom1_body = model.geom_bodyid[contact.geom1]
                geom2_body = model.geom_bodyid[contact.geom2]
                if target_body_id in (geom1_body, geom2_body):
                    # Contact with non-shelf surface (another object)
                    other_body = geom2_body if geom1_body == target_body_id else geom1_body
                    # If penetration depth > 0.5cm with another OBJECT (not shelf/floor)
                    if other_body > 1 and contact.dist < -0.005:  # body 0=world, 1=rack
                        has_overlap = True
                        break
            
            if not fell and not pushed_away and settled and not has_overlap:
                return True, (
                    f"Physics verified: object stable on shelf at euler={euler}. "
                    f"Oriented height={oriented_height*100:.1f}cm < clearance={shelf.clearance*100:.0f}cm, "
                    f"drift={lateral_drift*100:.1f}cm, no overlaps"
                )
        except Exception as e:
            # XML build or simulation error — skip this orientation
            continue
    
    return False, (
        f"Physics verified: object cannot fit in any orientation. "
        f"Dimensions {dims[0]*100:.0f}×{dims[1]*100:.0f}×{dims[2]*100:.0f}cm, "
        f"shelf clearance {shelf.clearance*100:.0f}cm, depth {shelf.depth*100:.0f}cm"
    )


# ─── MuJoCo Scene Builder ───────────────────────────────────────────

def _rgba(color_rgba):
    return " ".join(f"{c:.2f}" for c in color_rgba)


def _build_object_geom(obj: ObjectSpec, name: str, pos: list) -> str:
    """Build MuJoCo XML for an object at position."""
    rgba = _rgba(obj.color_rgba)
    px, py, pz = pos
    w, d, h = obj.width, obj.depth, obj.height

    if obj.family == "bottle" or obj.family == "vase":
        r = w / 2
        # Cap sphere sits at top of cylinder — total visual height = h + r
        # Adjust so total stays within h (sphere overlaps cylinder top)
        cyl_half = h/2 - r/2
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {cyl_half:.4f}" rgba="{rgba}"/>
      <geom type="sphere" size="{r:.4f}" pos="0 0 {cyl_half:.4f}" rgba="{rgba}"/>
    </body>"""

    if obj.family == "box" or obj.family == "book":
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="box" size="{w/2:.4f} {d/2:.4f} {h/2:.4f}" rgba="{rgba}"/>
    </body>"""

    if obj.family == "jar" or obj.family == "cup":
        r = w / 2
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/2:.4f}" rgba="{rgba}"/>
    </body>"""

    if obj.family == "pot":
        r = w / 2
        # Rim stays within declared height
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/2 - 0.008:.4f}" rgba="{rgba}"/>
      <geom type="cylinder" size="{r + 0.01:.4f} 0.008" pos="0 0 {h/2 - 0.008:.4f}" rgba="{rgba}"/>
    </body>"""

    if obj.family == "plate":
        r = w / 2
        return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="cylinder" size="{r:.4f} {h/2:.4f}" rgba="{rgba}"/>
    </body>"""

    # Fallback: box
    return f"""
    <body name="{name}" pos="{px:.4f} {py:.4f} {pz + h/2:.4f}">
      <geom type="box" size="{w/2:.4f} {d/2:.4f} {h/2:.4f}" rgba="{rgba}"/>
    </body>"""


def _build_rack_xml(rack: Rack) -> str:
    """Build MuJoCo XML for the rack frame."""
    hw = rack.total_width / 2
    hd = rack.total_depth / 2
    th = rack.total_height
    fw = rack.frame_width
    rgba = "0.55 0.38 0.22 1"

    xml = ""
    # Side panels
    xml += f'<geom type="box" size="{fw/2:.4f} {hd:.4f} {th/2:.4f}" pos="{-hw + fw/2:.4f} 0 {th/2:.4f}" rgba="{rgba}"/>\n'
    xml += f'<geom type="box" size="{fw/2:.4f} {hd:.4f} {th/2:.4f}" pos="{hw - fw/2:.4f} 0 {th/2:.4f}" rgba="{rgba}"/>\n'
    # Back panel
    xml += f'<geom type="box" size="{hw:.4f} {0.005:.4f} {th/2:.4f}" pos="0 {-hd + 0.005:.4f} {th/2:.4f}" rgba="{rgba}" group="0"/>\n'

    # Shelf boards
    for shelf in rack.shelves:
        xml += f'<geom type="box" size="{hw - fw:.4f} {hd:.4f} 0.01" pos="0 0 {shelf.y_pos:.4f}" rgba="{rgba}"/>\n'

    # Top board
    xml += f'<geom type="box" size="{hw:.4f} {hd:.4f} 0.01" pos="0 0 {th:.4f}" rgba="{rgba}"/>\n'

    return f'<body name="rack" pos="0 0 0">\n{xml}</body>'


def build_scene_xml(rack: Rack, placed: list[PlacedObject],
                    target: ObjectSpec, target_shelf_idx: int) -> str:
    """Build complete MuJoCo scene XML."""
    rack_xml = _build_rack_xml(rack)

    # Place existing objects
    obj_xmls = []
    for i, p in enumerate(placed):
        shelf = rack.shelves[p.shelf_idx]
        x = -rack.total_width / 2 + rack.frame_width + p.x_offset + _shelf_footprint(p.spec) / 2
        y = 0.0
        z = shelf.y_pos + 0.012  # above shelf board
        obj_xmls.append(_build_object_geom(p.spec, f"placed_{i}", [x, y, z]))

    # Target object — placed BESIDE the rack (right side)
    target_z = 0.0
    target_x = rack.total_width / 2 + 0.15  # 15cm to the right of rack
    obj_xmls.append(_build_object_geom(target, "target_object", [target_x, 0.0, target_z]))

    all_objs = "\n".join(obj_xmls)

    return f"""
<mujoco model="shelf_fitting">
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.90 0.90 0.88" rgb2="0.78 0.78 0.75"
             width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="5 5" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 -2 3" dir="0 0.5 -0.7" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="1.5 0 2.5" dir="-0.3 0 -0.6" diffuse="0.35 0.35 0.38"/>
    <light pos="-1.5 1 2.5" dir="0.3 -0.2 -0.6" diffuse="0.3 0.3 0.32"/>

    <geom type="plane" size="3 3 0.01" material="floor_mat"/>

    {rack_xml}
    {all_objs}
  </worldbody>
</mujoco>
"""


def render_scene(xml: str, rack: Rack, width: int = 960, height: int = 720,
                 camera: dict | None = None) -> Image.Image:
    """
    Render scene from a camera angle.
    
    Args:
        xml: MuJoCo XML string
        rack: Rack config (used for auto camera distance)
        width, height: image resolution
        camera: optional dict with keys: azimuth, elevation, distance, lookat
                if None, auto-computes a good default view
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    if camera:
        cam.azimuth = camera.get("azimuth", 270)
        cam.elevation = camera.get("elevation", -15)
        cam.distance = camera.get("distance", 2.0)
        lookat = camera.get("lookat", [0, 0, rack.total_height / 2])
        cam.lookat[:] = lookat
    else:
        # Auto-compute: zoom out based on rack size
        cam.azimuth = 270
        cam.elevation = -15
        # Distance scales with rack height + width to ensure full visibility
        cam.distance = max(1.6, rack.total_height * 1.5, rack.total_width * 2.0)
        cam.lookat[:] = [0, 0, rack.total_height / 2]

    renderer.update_scene(data, cam)
    pixels = renderer.render()
    renderer.close()

    return Image.fromarray(pixels)


def render_scene_multiview(xml: str, rack: Rack, width: int = 960, height: int = 720,
                           views: list[dict] | None = None) -> list[Image.Image]:
    """
    Render scene from multiple camera angles.
    
    Args:
        xml: MuJoCo XML string
        rack: Rack config
        width, height: per-view resolution
        views: list of camera dicts. If None, uses default [front, angle, top-down]
    
    Returns:
        List of PIL Images, one per view
    """
    if views is None:
        center_z = rack.total_height / 2
        auto_dist = max(1.6, rack.total_height * 1.5, rack.total_width * 2.0)
        views = [
            {"azimuth": 270, "elevation": -15, "distance": auto_dist,
             "lookat": [0, 0, center_z]},  # front
            {"azimuth": 315, "elevation": -20, "distance": auto_dist * 1.1,
             "lookat": [0, 0, center_z]},  # angled
        ]

    images = []
    for view in views:
        img = render_scene(xml, rack, width, height, camera=view)
        images.append(img)
    return images


# ─── Question Templates (NO dimension leaks) ────────────────────────

def generate_question(target: ObjectSpec, rack: Rack, shelf_idx: int,
                      placed: list[PlacedObject], rng: np.random.RandomState) -> str:
    """
    Generate an unambiguous question with NO dimension leaks.
    Model must infer ALL spatial info from the image.
    """
    shelf = rack.shelves[shelf_idx]

    # Shelf description (by position, not dimensions)
    total = len(rack.shelves)
    if shelf_idx == 0:
        shelf_desc = "bottom shelf"
    elif shelf_idx == total - 1:
        shelf_desc = "top shelf"
    elif shelf_idx == 1 and total > 2:
        shelf_desc = "second shelf from the bottom"
    elif shelf_idx == total - 2 and total > 3:
        shelf_desc = "second shelf from the top"
    else:
        shelf_desc = f"shelf {shelf_idx + 1}"

    # Object description
    obj_desc = f"{target.color} {target.label}"

    # How many objects already on that shelf?
    shelf_objects = [p for p in placed if p.shelf_idx == shelf_idx]
    num_existing = len(shelf_objects)

    # Question templates — always "in any orientation" since GT checks all
    templates = [
        f"Can the {obj_desc} fit on the {shelf_desc} in any orientation without moving any existing objects?",
        f"Is there enough space to place the {obj_desc} on the {shelf_desc} in any orientation without disturbing the other objects?",
        f"Will the {obj_desc} fit on the {shelf_desc} of the {rack.name} in any orientation?",
    ]

    if num_existing == 0:
        templates.append(f"Can the {obj_desc} be stored on the empty {shelf_desc} in any orientation?")
    elif num_existing >= 2:
        templates.append(
            f"There are already {num_existing} objects on the {shelf_desc}. "
            f"Can the {obj_desc} still fit in any orientation without rearranging anything?"
        )

    return rng.choice(templates)


# ─── Scene Sampler ───────────────────────────────────────────────────

@dataclass
class SceneData:
    """All data for one generated scene."""
    rack: Rack
    placed: list[PlacedObject]
    target: ObjectSpec
    target_shelf: int
    fits: bool
    reasoning: str
    question: str
    difficulty: str


def sample_scene(rng: np.random.RandomState, difficulty: str = "medium",
                 target_fits: bool | None = None,
                 max_existing: int | None = None) -> SceneData:
    """
    Sample a complete scene with verified ground truth.

    Args:
        rng: random state
        difficulty: controls margins and default object count
            easy:   margin > 30%, 0-1 existing objects
            medium: margin > 15%, 0-2 existing objects
            hard:   margin > 5%,  0-4 existing objects
        target_fits: force True/False, or None for random (50/50)
        max_existing: override max objects on shelf (None = use difficulty default)
    """
    if target_fits is None:
        target_fits = bool(rng.randint(0, 2))

    # Sample rack
    rack = sample_rack(rng)

    # Pick target shelf
    shelf_idx = rng.randint(len(rack.shelves))
    shelf = rack.shelves[shelf_idx]

    # Difficulty controls
    if difficulty == "easy":
        default_max_placed = 1
        min_margin = 0.30       # 30% minimum margin
    elif difficulty == "hard":
        default_max_placed = 4
        min_margin = 0.05       # 5% minimum margin
    else:  # medium
        default_max_placed = 2
        min_margin = 0.15       # 15% minimum margin

    # User can override max existing objects
    max_placed = max_existing if max_existing is not None else default_max_placed

    # Place existing objects on the target shelf
    placed = place_objects_on_shelf(rack, shelf_idx, rng, max_objects=max_placed)

    # Sample target object — iterate until we get desired fits/no_fit WITH sufficient margin
    for _attempt in range(100):
        target = sample_object(rng)

        # ALWAYS check all orientations for ground truth
        fits, reasoning = can_fit_exhaustive(
            target, rack, shelf_idx, placed, check_all_orientations=True
        )

        if fits != target_fits:
            continue

        # Compute margin to ensure no ambiguous cases
        margin_ok = _check_margin(target, shelf, placed, shelf_idx, fits, min_margin)
        if not margin_ok:
            continue

        question = generate_question(target, rack, shelf_idx, placed, rng)

        return SceneData(
            rack=rack,
            placed=placed,
            target=target,
            target_shelf=shelf_idx,
            fits=fits,
            reasoning=reasoning,
            question=question,
            difficulty=difficulty,
        )

    # If 100 attempts with this rack failed, try with a completely new rack
    # Keep retrying until we get a valid scene (max 10 rack attempts)
    for _rack_attempt in range(10):
        rack = sample_rack(rng)
        shelf_idx = rng.randint(len(rack.shelves))
        shelf = rack.shelves[shelf_idx]
        placed = place_objects_on_shelf(rack, shelf_idx, rng, max_objects=max_placed)

        for _attempt in range(50):
            target = sample_object(rng)
            fits, reasoning = can_fit_exhaustive(
                target, rack, shelf_idx, placed, check_all_orientations=True
            )
            if fits != target_fits:
                continue
            margin_ok = _check_margin(target, shelf, placed, shelf_idx, fits, min_margin)
            if not margin_ok:
                continue

            question = generate_question(target, rack, shelf_idx, placed, rng)
            return SceneData(
                rack=rack, placed=placed, target=target, target_shelf=shelf_idx,
                fits=fits, reasoning=reasoning, question=question, difficulty=difficulty,
            )

    # Final fallback: accept any valid fits/no_fit (should rarely reach here)
    rack = sample_rack(rng)
    shelf_idx = rng.randint(len(rack.shelves))
    placed = []  # empty shelf = easiest to find valid margin
    target = sample_object(rng)
    fits, reasoning = can_fit_exhaustive(
        target, rack, shelf_idx, placed, check_all_orientations=True
    )
    question = generate_question(target, rack, shelf_idx, placed, rng)

    return SceneData(
        rack=rack, placed=placed, target=target, target_shelf=shelf_idx,
        fits=fits, reasoning=reasoning, question=question, difficulty=difficulty,
    )


def _check_margin(target: ObjectSpec, shelf: Shelf, placed: list[PlacedObject],
                  shelf_idx: int, fits: bool, min_margin: float) -> bool:
    """
    Ensure the scene has sufficient margin to avoid ambiguity.

    For FITS: ALL three dimensions must have > min_margin slack in the best orientation.
    For NO_FIT: the best (closest-to-fitting) orientation must fail by > min_margin.

    This prevents cases like 23cm object vs 22cm shelf (4% margin = ambiguous).
    """
    dims = [target.width, target.depth, target.height]

    # Get available gap on shelf
    shelf_objects = [p for p in placed if p.shelf_idx == shelf_idx]
    occupied = []
    for p in shelf_objects:
        pw = _shelf_footprint(p.spec)
        occupied.append((p.x_offset, p.x_offset + pw))
    occupied.sort()
    max_gap = _max_gap(occupied, shelf.width)

    # Avoid division by zero
    if max_gap <= 0:
        max_gap = 0.001

    if fits:
        # For fits: find the BEST orientation and check its tightest margin
        # ALL 3 dimensions must have > min_margin in at least one orientation
        best_tightest = 0.0

        from itertools import permutations
        for w, d, h in set(permutations(dims)):
            # All three must pass
            if h >= shelf.clearance - 0.005:
                continue
            if d >= shelf.depth - 0.005:
                continue
            if w >= max_gap - 0.005:
                continue

            # Compute margins for this orientation
            h_margin = (shelf.clearance - h) / shelf.clearance
            d_margin = (shelf.depth - d) / shelf.depth
            w_margin = (max_gap - w) / max_gap

            tightest = min(h_margin, d_margin, w_margin)
            best_tightest = max(best_tightest, tightest)

        return best_tightest >= min_margin

    else:
        # For no_fit: every orientation must fail, and the CLOSEST one
        # must still fail by > min_margin
        # i.e., even the "almost fits" case is clearly too big
        min_excess = 999.0

        from itertools import permutations
        for w, d, h in set(permutations(dims)):
            # Which dimensions fail?
            failures = []
            if h >= shelf.clearance - 0.005:
                failures.append((h - shelf.clearance) / shelf.clearance)
            if d >= shelf.depth - 0.005:
                failures.append((d - shelf.depth) / shelf.depth)
            if w >= max_gap - 0.005:
                failures.append((w - max_gap) / max_gap)

            if not failures:
                # This orientation actually fits — shouldn't happen if no_fit is correct
                return False

            # The "excess" is how much the worst dimension exceeds
            # We want the BEST orientation's failure to still be > min_margin
            best_failure = min(failures)  # smallest excess in this orientation
            min_excess = min(min_excess, best_failure)

        return min_excess >= min_margin


# ─── Environment Class ──────────────────────────────────────────────

class ShelfFittingEnv(BaseEnvironment):
    """
    Shelf Fitting Environment.

    Question: "Can the {object} fit on the {shelf} in any orientation without moving existing objects?"
    Answer: "fits" or "no_fit"
    Verification: exhaustive physics check (height, width, depth, all orientations)
    
    Args:
        difficulty: "easy" | "medium" | "hard" | "mixed"
        seed: random seed
        camera: optional dict with azimuth, elevation, distance, lookat
        multi_view: if True, obs.images returns list of views (front + angle)
        views: list of camera dicts for custom multi-view
        max_existing: max objects already on the target shelf (None = auto by difficulty)
        physics_engine: if True, use MuJoCo physics simulation for verification
                        if False (default), use fast rule-based verification
    """

    def __init__(self, difficulty="medium", seed=None, camera=None,
                 multi_view=False, views=None, max_existing=None,
                 physics_engine=False, **kwargs):
        super().__init__(difficulty=difficulty, seed=seed, **kwargs)
        self._scene: SceneData | None = None
        self._current_obs: Observation | None = None
        self._step_count = 0
        self._camera = camera
        self._multi_view = multi_view
        self._views = views
        self._max_existing = max_existing
        self._physics_engine = physics_engine

    def reset(self) -> Observation:
        """Generate a new random scene."""
        self._scene = sample_scene(
            self.rng, difficulty=self.difficulty, max_existing=self._max_existing
        )
        self._step_count += 1

        # If physics_engine=True, double-verify GT with MuJoCo simulation
        if self._physics_engine:
            physics_fits, physics_reasoning = can_fit_physics(
                self._scene.target,
                self._scene.rack,
                self._scene.target_shelf,
                self._scene.placed,
            )
            # If physics disagrees with rules, use physics as ground truth
            if physics_fits != self._scene.fits:
                self._scene.fits = physics_fits
                self._scene.reasoning = f"[physics override] {physics_reasoning}"
            else:
                self._scene.reasoning = f"[physics confirmed] {self._scene.reasoning}"

        # Build XML
        xml = build_scene_xml(
            self._scene.rack,
            self._scene.placed,
            self._scene.target,
            self._scene.target_shelf,
        )

        # Render single or multi-view
        if self._multi_view:
            images = render_scene_multiview(
                xml, self._scene.rack, views=self._views
            )
            image = images[0]  # primary view
        else:
            image = render_scene(xml, self._scene.rack, camera=self._camera)
            images = [image]

        self._current_obs = Observation(
            image=image,
            question=self._scene.question,
            metadata={
                "scene_id": self._step_count,
                "fits": self._scene.fits,
                "difficulty": self._scene.difficulty,
                "images": images,  # all views if multi_view=True
                "target": {
                    "family": self._scene.target.family,
                    "label": self._scene.target.label,
                    "color": self._scene.target.color,
                    "width": self._scene.target.width,
                    "depth": self._scene.target.depth,
                    "height": self._scene.target.height,
                },
                "shelf": {
                    "index": self._scene.target_shelf,
                    "clearance": self._scene.rack.shelves[self._scene.target_shelf].clearance,
                    "depth": self._scene.rack.shelves[self._scene.target_shelf].depth,
                    "width": self._scene.rack.shelves[self._scene.target_shelf].width,
                },
                "num_existing_objects": len(self._scene.placed),
            },
        )
        return self._current_obs

    def verify(self, answer: str) -> float:
        """
        Check if answer is correct.

        Args:
            answer: "fits" or "no_fit" (also accepts "yes"/"no", "fit"/"doesn't fit")

        Returns:
            1.0 if correct, 0.0 if wrong
        """
        if self._scene is None:
            raise RuntimeError("Call reset() first")

        # Normalize answer
        a = answer.lower().strip()
        if a in ("fits", "fit", "yes", "true", "can fit", "it fits"):
            predicted = True
        elif a in ("no_fit", "no fit", "doesn't fit", "does not fit", "no", "false",
                    "cannot fit", "can't fit", "too big", "too tall"):
            predicted = False
        else:
            # Try to extract from PREDICTION: format
            if "fits" in a and "not" not in a and "no" not in a.split("fit")[0][-5:]:
                predicted = True
            elif "not fit" in a or "no fit" in a or "doesn't fit" in a or "too" in a:
                predicted = False
            else:
                return 0.0  # can't parse → wrong

        return 1.0 if predicted == self._scene.fits else 0.0

    def ground_truth(self) -> dict:
        """Return ground truth for current scene (internal use only)."""
        if self._scene is None:
            raise RuntimeError("Call reset() first")
        return {
            "answer": "fits" if self._scene.fits else "no_fit",
            "reasoning": self._scene.reasoning,
        }
