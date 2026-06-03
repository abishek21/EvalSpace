"""
MuJoCo API routes — physics-grounded verification for RLHF spatial reasoning.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

import mujoco

from app.mujoco_sim import (
    build_scene_xml,
    create_quick_scene,
    render_multi_view,
    simulate_trajectory,
    SceneObject,
    Waypoint,
    OBJECT_PRESETS,
)
from app.mujoco_scene_gen import generate_random_scene, generate_batch

router = APIRouter(prefix="/api/mujoco", tags=["mujoco"])


# ─── Request / Response models ──────────────────────────────────────

class ObjectDef(BaseModel):
    name: str
    type: str  # key in OBJECT_PRESETS
    pos: list[float]  # [x, y, z]
    material: Optional[str] = None
    size: Optional[str] = None


class CreateSceneRequest(BaseModel):
    base_scene: str = "tabletop"
    objects: list[ObjectDef]


class SimulateRequest(BaseModel):
    base_scene: str = "tabletop"
    objects: list[ObjectDef]
    target_object: str
    waypoints: list[dict]  # [{x, y, z, t}, ...]
    record_cameras: list[str] = []


class RenderRequest(BaseModel):
    base_scene: str = "tabletop"
    objects: list[ObjectDef]
    cameras: list[str] = ["front", "top", "side"]
    width: int = 480
    height: int = 360


# ─── Endpoints ───────────────────────────────────────────────────────

@router.get("/presets")
async def list_presets():
    """List available object presets."""
    return {
        "presets": {k: v for k, v in OBJECT_PRESETS.items()},
        "materials": ["obj_red", "obj_green", "obj_blue", "obj_yellow"],
    }


@router.post("/render")
async def render_scene_endpoint(req: RenderRequest):
    """Create a scene and render multi-view images."""
    try:
        obj_dicts = [{"name": o.name, "type": o.type, "pos": o.pos,
                       "material": o.material, "size": o.size} for o in req.objects]
        xml, _ = create_quick_scene(obj_dicts, req.base_scene)

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Let objects settle
        for _ in range(500):
            mujoco.mj_step(model, data)

        images = render_multi_view(model, data, req.cameras, req.width, req.height)

        return {"images": images, "cameras": req.cameras}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate")
async def simulate_endpoint(req: SimulateRequest):
    """Simulate a trajectory and return collision/physics info."""
    try:
        obj_dicts = [{"name": o.name, "type": o.type, "pos": o.pos,
                       "material": o.material, "size": o.size} for o in req.objects]
        xml, _ = create_quick_scene(obj_dicts, req.base_scene)

        waypoints = [Waypoint(x=w["x"], y=w["y"], z=w["z"], t=w["t"]) for w in req.waypoints]

        result = simulate_trajectory(
            scene_xml=xml,
            target_object=req.target_object,
            waypoints=waypoints,
            record_cameras=req.record_cameras or None,
        )

        return {
            "success": result.success,
            "collisions": result.collisions,
            "trajectory_actual": result.trajectory_actual,
            "physics_plausible": result.physics_plausible,
            "reason": result.reason,
            "frames": result.frames,
            "num_collisions": len(result.collisions),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-pair")
async def verify_pair(req: SimulateRequest):
    """
    Run simulation and return a verification verdict suitable for
    attaching to an RLHF pair as physics metadata.
    """
    try:
        obj_dicts = [{"name": o.name, "type": o.type, "pos": o.pos,
                       "material": o.material, "size": o.size} for o in req.objects]
        xml, _ = create_quick_scene(obj_dicts, req.base_scene)

        waypoints = [Waypoint(x=w["x"], y=w["y"], z=w["z"], t=w["t"]) for w in req.waypoints]

        result = simulate_trajectory(
            scene_xml=xml,
            target_object=req.target_object,
            waypoints=waypoints,
        )

        # Render final state for visual confirmation
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        # Replay to final state
        for _ in range(500):
            mujoco.mj_step(model, data)
        final_views = render_multi_view(model, data, ["front", "top"], 320, 240)

        return {
            "verified": result.success,
            "physics_plausible": result.physics_plausible,
            "collision_count": len(result.collisions),
            "collisions": result.collisions[:10],  # cap for payload size
            "reason": result.reason,
            "final_views": final_views,
            "metadata": {
                "target_object": req.target_object,
                "num_waypoints": len(req.waypoints),
                "objects_in_scene": [o.name for o in req.objects],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Scene Generation ───────────────────────────────────────────────

class GenerateSceneRequest(BaseModel):
    num_objects: int = 4
    cameras: list[str] = ["front", "top", "side"]
    image_width: int = 480
    image_height: int = 360


class GenerateBatchRequest(BaseModel):
    num_scenes: int = 5
    objects_per_scene: int = 4
    cameras: list[str] = ["front", "top", "side"]


@router.post("/generate-scene")
async def generate_scene_endpoint(req: GenerateSceneRequest):
    """Generate a single random tabletop scene with rendered views + ground truth."""
    try:
        import uuid
        scene = generate_random_scene(
            scene_id=f"mujoco_{uuid.uuid4().hex[:8]}",
            num_objects=req.num_objects,
            cameras=req.cameras,
            image_width=req.image_width,
            image_height=req.image_height,
        )
        return {
            "id": scene.id,
            "images": scene.images_b64,
            "descriptions": scene.descriptions,
            "ground_truth": scene.ground_truth,
            "objects": [
                {"name": o.name, "label": o.label, "type": o.preset,
                 "position": o.settled_pos or o.initial_pos, "material": o.material}
                for o in scene.objects
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-batch")
async def generate_batch_endpoint(req: GenerateBatchRequest):
    """Generate multiple random scenes for VLM inference."""
    try:
        scenes = generate_batch(req.num_scenes, req.objects_per_scene, req.cameras)
        return {
            "count": len(scenes),
            "scenes": [
                {
                    "id": s.id,
                    "images": s.images_b64,
                    "descriptions": s.descriptions,
                    "ground_truth": s.ground_truth,
                    "num_objects": len(s.objects),
                }
                for s in scenes
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))