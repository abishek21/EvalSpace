"""
MuJoCo Scene Randomizer — generates diverse tabletop scenes for VLM spatial reasoning.

Replaces ScanNet as a data source: instead of real scans, we procedurally generate
physics-grounded scenes where we know exact ground truth positions.

Flow:
  1. Randomly place N objects on table
  2. Settle physics (objects fall/stack naturally)
  3. Render multi-view images
  4. Record ground-truth positions
  5. Return scene data ready for VLM question generation
"""
import random
import math
from dataclasses import dataclass, field

import numpy as np
import mujoco

from app.mujoco_sim import (
    build_scene_xml,
    render_multi_view,
    SceneObject,
    OBJECT_PRESETS,
    MATERIALS,
)


# Table surface bounds (x: [-0.5, 0.5], y: [-0.3, 0.3], z: ~0.37 = table top)
TABLE_X_RANGE = (-0.45, 0.45)
TABLE_Y_RANGE = (-0.28, 0.28)
TABLE_Z = 0.42  # just above table surface

# Named objects for natural language
OBJECT_NAMES = {
    "cup": ["red cup", "blue cup", "green cup", "yellow cup", "cup"],
    "bowl": ["large bowl", "small bowl", "mixing bowl", "bowl"],
    "box": ["cardboard box", "small box", "wooden block", "box"],
    "bottle": ["water bottle", "tall bottle", "bottle"],
    "can": ["soda can", "tin can", "metal can", "can"],
    "ball": ["tennis ball", "red ball", "small ball", "ball"],
    "plate": ["dinner plate", "round plate", "plate"],
    "book": ["textbook", "notebook", "thick book", "book"],
}


@dataclass
class GroundTruthObject:
    """Ground truth info for a placed object — used to verify VLM answers."""
    name: str
    label: str  # human-readable, e.g. "red cup"
    preset: str  # e.g. "cup"
    initial_pos: list[float]
    settled_pos: list[float] = field(default_factory=list)
    material: str = ""


@dataclass
class GeneratedScene:
    """A complete generated scene ready for VLM inference."""
    id: str
    xml: str
    objects: list[GroundTruthObject]
    images_b64: list[str]  # multi-view renders
    descriptions: list[str]  # text descriptions for each object
    ground_truth: dict = field(default_factory=dict)  # positions, relations, counts


def _random_table_pos(existing_positions: list[list[float]], min_dist: float = 0.1) -> list[float]:
    """Random position on table that doesn't overlap existing objects."""
    for _ in range(50):
        x = random.uniform(*TABLE_X_RANGE)
        y = random.uniform(*TABLE_Y_RANGE)
        pos = [x, y, TABLE_Z]

        # Check minimum distance from existing objects
        too_close = False
        for ep in existing_positions:
            dist = math.sqrt((x - ep[0])**2 + (y - ep[1])**2)
            if dist < min_dist:
                too_close = True
                break

        if not too_close:
            return pos

    # Fallback: just place it somewhere
    return [random.uniform(*TABLE_X_RANGE), random.uniform(*TABLE_Y_RANGE), TABLE_Z]


def generate_random_scene(
    scene_id: str,
    num_objects: int = 4,
    cameras: list[str] = None,
    image_width: int = 480,
    image_height: int = 360,
) -> GeneratedScene:
    """
    Generate a random tabletop scene:
    1. Pick random objects and positions
    2. Build MJCF XML
    3. Simulate physics (let objects settle)
    4. Render multi-view images
    5. Compute ground-truth spatial relations
    """
    if cameras is None:
        cameras = ["front", "top", "side"]

    # Pick random object types (allow duplicates)
    presets = list(OBJECT_PRESETS.keys())
    chosen_presets = random.choices(presets, k=num_objects)

    # Place objects with unique names
    positions = []
    scene_objects = []
    gt_objects = []
    used_names = set()

    for i, preset in enumerate(chosen_presets):
        # Generate unique name
        possible_labels = OBJECT_NAMES.get(preset, [preset])
        label = random.choice(possible_labels)
        name = f"{preset}_{i}"

        # Random position
        pos = _random_table_pos(positions)
        positions.append(pos)

        mat = MATERIALS[i % len(MATERIALS)]

        scene_objects.append(SceneObject(
            name=name,
            preset=preset,
            pos=pos,
            material=mat,
        ))

        gt_objects.append(GroundTruthObject(
            name=name,
            label=label,
            preset=preset,
            initial_pos=pos.copy(),
            material=mat,
        ))

    # Build scene XML
    xml = build_scene_xml("tabletop", scene_objects)

    # Simulate physics — let objects settle
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    for _ in range(1000):  # ~2 seconds of sim time
        mujoco.mj_step(model, data)

    # Read settled positions
    for gt_obj in gt_objects:
        jnt_name = f"{gt_obj.name}_jnt"
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id >= 0:
            qpos_adr = model.jnt_qposadr[jnt_id]
            settled = data.qpos[qpos_adr:qpos_adr + 3].tolist()
            gt_obj.settled_pos = [round(x, 4) for x in settled]

    # Render multi-view images
    images_b64 = render_multi_view(model, data, cameras, image_width, image_height)

    # Build text descriptions
    descriptions = []
    for gt_obj in gt_objects:
        pos = gt_obj.settled_pos or gt_obj.initial_pos
        descriptions.append(
            f"{gt_obj.label} at position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        )

    # Compute ground-truth spatial relations
    ground_truth = _compute_ground_truth(gt_objects)

    return GeneratedScene(
        id=scene_id,
        xml=xml,
        objects=gt_objects,
        images_b64=images_b64,
        descriptions=descriptions,
        ground_truth=ground_truth,
    )


def _compute_ground_truth(objects: list[GroundTruthObject]) -> dict:
    """Compute verifiable ground-truth facts about the scene."""
    gt = {
        "object_count": len(objects),
        "objects": {},
        "spatial_relations": [],
        "categories": {},
    }

    # Object positions
    for obj in objects:
        pos = obj.settled_pos or obj.initial_pos
        gt["objects"][obj.name] = {
            "label": obj.label,
            "type": obj.preset,
            "position": pos,
        }

    # Count by type
    from collections import Counter
    type_counts = Counter(obj.preset for obj in objects)
    gt["categories"] = dict(type_counts)

    # Pairwise spatial relations
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i >= j:
                continue
            pa = a.settled_pos or a.initial_pos
            pb = b.settled_pos or b.initial_pos

            dx = pb[0] - pa[0]
            dy = pb[1] - pa[1]
            dz = pb[2] - pa[2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            # Determine spatial relation
            relations = []
            if dx > 0.05:
                relations.append(f"{b.label} is to the right of {a.label}")
            elif dx < -0.05:
                relations.append(f"{b.label} is to the left of {a.label}")
            if dy > 0.05:
                relations.append(f"{b.label} is behind {a.label}")
            elif dy < -0.05:
                relations.append(f"{b.label} is in front of {a.label}")
            if dz > 0.05:
                relations.append(f"{b.label} is above {a.label}")
            if dist < 0.15:
                relations.append(f"{a.label} and {b.label} are close together")

            gt["spatial_relations"].extend(relations)

    return gt


def generate_batch(
    num_scenes: int = 5,
    objects_per_scene: int = 4,
    cameras: list[str] = None,
) -> list[GeneratedScene]:
    """Generate a batch of random scenes."""
    scenes = []
    for i in range(num_scenes):
        n_objects = random.randint(max(2, objects_per_scene - 1), objects_per_scene + 2)
        scene = generate_random_scene(
            scene_id=f"mujoco_scene_{i:03d}",
            num_objects=n_objects,
            cameras=cameras,
        )
        scenes.append(scene)
    return scenes
