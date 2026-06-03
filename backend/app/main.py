"""
FastAPI main application — serves as the backend for the RLHF annotation tool.
Next.js frontend calls these endpoints.
"""
import asyncio
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import db, IMAGES_DIR
from app.models import JobConfig
from app.gpu_client import check_health
from app.job_processor import process_job
from app.mujoco_job_processor import process_mujoco_job
from app.dise_stacking_job import process_stacking_job
from app.gt_generator import process_gt_job
from app.eval_runner import process_eval_job
from app.rlhf_generator import process_rlhf_job
from app.mujoco_routes import router as mujoco_router

app = FastAPI(title="RLHF Annotation Tool API", version="0.1.0")
app.include_router(mujoco_router)

# Allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3002", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved scene images from disk (stored outside repo)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# --- Health ---

@app.get("/health")
async def health():
    return {"status": "ok", "service": "rlhf-backend"}


@app.get("/gpu/health")
async def gpu_health():
    """Check if GPU worker is reachable."""
    try:
        info = await check_health()
        return {"status": "connected", **info}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# --- Jobs ---

@app.get("/api/jobs")
async def list_jobs(status: str | None = None):
    return db.get_jobs(status)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs", status_code=201)
async def create_job(config: JobConfig):
    job = {
        "id": str(uuid4()),
        "status": "queued",
        "config": config.model_dump(),
        "progress": {"scenes_processed": 0, "questions_generated": 0, "pairs_generated": 0},
        "created_at": datetime.now().isoformat(),
    }
    db.create_job(job)
    return job


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.delete_job(job_id)
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/run")
async def run_job(job_id: str, background_tasks: BackgroundTasks):
    """Start processing a queued job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "queued":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not queued")

    # Check GPU worker — skip if not needed
    config = job["config"]
    answer_model = config.get("model", "")
    job_type = config.get("job_type", "")
    needs_gpu = (
        answer_model not in ("gpt-4o", "none", "")
        and job_type not in ("generate_gt", "rlhf")
    )
    if needs_gpu:
        last_err = None
        for _attempt in range(3):
            try:
                await check_health()
                last_err = None
                break
            except Exception as e:
                last_err = e
                import asyncio
                await asyncio.sleep(2)
        if last_err:
            raise HTTPException(
                status_code=503,
                detail=f"GPU worker not reachable after 3 attempts. Set GPU_WORKER_URL or RUNPOD_ENDPOINT_ID+RUNPOD_API_KEY. Error: {last_err}",
            )

    # Run in background — choose processor based on job type / dataset source
    job_type = config.get("job_type", "")
    dataset = config.get("dataset", "")

    if job_type == "generate_gt":
        processor = process_gt_job
    elif job_type == "evaluate":
        processor = process_eval_job
    elif job_type == "rlhf":
        processor = process_rlhf_job
    elif dataset == "mujoco:stacking":
        processor = process_stacking_job
    elif dataset.startswith("mujoco:"):
        processor = process_mujoco_job
    else:
        processor = process_job
    background_tasks.add_task(processor, job_id, config)
    return {"status": "started", "job_id": job_id}


# --- Projects ---

@app.get("/api/projects")
async def list_projects():
    projects = db.get_projects()
    result = []
    for p in projects:
        pairs = db.get_pairs(p["id"])
        result.append({
            **p,
            "total": len(pairs),
            "annotated": sum(1 for pr in pairs if pr.get("status") == "annotated"),
            "pending": sum(1 for pr in pairs if pr.get("status") == "pending"),
        })
    return result


class UploadRequest(BaseModel):
    name: str
    pairs: list[dict]
    source: dict | None = None
    generation: dict | None = None


@app.post("/api/projects", status_code=201)
async def create_project(req: UploadRequest):
    project_id = str(uuid4())
    db.create_project({
        "id": project_id,
        "name": req.name or "Untitled",
        "created_at": datetime.now().isoformat(),
    })

    db_pairs = []
    for p in req.pairs:
        db_pairs.append({
            "id": str(uuid4()),
            "project_id": project_id,
            "prompt": p.get("prompt", ""),
            "chosen": p.get("chosen", ""),
            "rejected": p.get("rejected", ""),
            "scene_id": p.get("scene_id", p.get("sceneId", "")),
            "category": p.get("category", ""),
            "difficulty": p.get("difficulty", ""),
            "status": "pending",
            "source": p.get("source") or req.source,
            "generation": p.get("generation") or req.generation,
        })
    db.add_pairs(db_pairs)

    return {"id": project_id, "pair_count": len(db_pairs)}


# --- Pairs (annotation) ---

@app.get("/api/pairs")
async def get_next_pair(projectId: str, review: str = "false", index: int = 0, type: str = "project"):
    # For eval runs, load eval_result pairs instead of project pairs
    if type == "eval":
        pairs = db.get_eval_results(projectId)
    elif review == "true":
        pairs = db.get_pairs(projectId)
    else:
        pairs = db.get_pairs(projectId, status="pending")

    if review == "true" or type == "eval":
        if not pairs or index >= len(pairs):
            return {"done": True, "pair": None, "total": len(pairs), "index": index}
        return {"done": False, "pair": pairs[index], "total": len(pairs), "index": index}

    if not pairs:
        return {"done": True, "pair": None}
    return {"done": False, "pair": pairs[0]}


class AnnotateRequest(BaseModel):
    id: str
    preference: str
    rationale: str = ""


@app.patch("/api/pairs")
async def annotate_pair(req: AnnotateRequest):
    db.update_pair(req.id, {
        "preference": req.preference,
        "rationale": req.rationale or None,
        "status": "annotated",
        "annotated_at": datetime.now().isoformat(),
    })
    return {"status": "ok"}


# --- Export ---

# --- Datasets (Stage 1 GT collections) ---

@app.get("/api/datasets")
async def list_datasets():
    datasets = db.get_datasets()
    for ds in datasets:
        scenarios = db.get_dataset_scenarios(ds["id"])
        ds["scenario_count"] = len(scenarios)
    return datasets


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    ds = db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    scenarios = db.get_dataset_scenarios(dataset_id)
    ds["scenarios"] = scenarios
    ds["scenario_count"] = len(scenarios)
    return ds


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    ds = db.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete_dataset(dataset_id)
    return {"deleted": dataset_id}


# --- Eval Runs (Stage 2) ---

@app.get("/api/eval-runs")
async def list_eval_runs(dataset_id: str | None = None):
    runs = db.get_eval_runs(dataset_id)
    return runs


@app.get("/api/eval-runs/{run_id}")
async def get_eval_run(run_id: str):
    run = db.get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    results = db.get_eval_results(run_id)
    run["results"] = results
    return run


@app.delete("/api/eval-runs/{run_id}")
async def delete_eval_run(run_id: str):
    run = db.get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    db.delete_eval_run(run_id)
    return {"deleted": run_id}


# --- Export (existing) ---

@app.get("/api/export")
async def export_pairs(projectId: str, format: str = "full", all: str = "false"):
    if all == "true":
        pairs = db.get_pairs(projectId)  # all statuses
    else:
        pairs = db.get_pairs(projectId, status="annotated")

    exported = []
    for p in pairs:
        swapped = p.get("preference") == "rejected"
        chosen = p["rejected"] if swapped else p["chosen"]
        rejected = p["chosen"] if swapped else p["rejected"]

        if format in ("dpo", "trl"):
            exported.append({
                "prompt": p["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "images": (p.get("source") or {}).get("images", []),
            })
        else:
            exported.append({
                "prompt": p["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "source": p.get("source") or {"dataset": "unknown", "scene_id": p.get("scene_id", "")},
                "generation": p.get("generation"),
                "verification": p.get("verification"),
                "annotation": {
                    "preference": p.get("preference"),
                    "rationale": p.get("rationale"),
                    "annotated_at": p.get("annotated_at"),
                },
                "category": p.get("category"),
                "difficulty": p.get("difficulty"),
            })

    projects = db.get_projects()
    project = next((pr for pr in projects if pr["id"] == projectId), None)

    import json as json_mod
    result = {
        "version": "1.0",
        "project": project,
        "exported_at": datetime.now().isoformat(),
        "total_pairs": len(exported),
        "format": format,
        "data": exported,
    }
    filename = f"rlhf_export_{projectId[:8]}_{format}.json"
    return Response(
        content=json_mod.dumps(result, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
