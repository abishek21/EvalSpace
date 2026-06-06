"""
Simple JSON file database.
Collections: jobs, datasets, eval_runs, projects
Pairs are stored in separate files per dataset/eval_run to avoid loading 800MB on every request.
"""
import json
from pathlib import Path
from threading import Lock

DB_PATH = Path(__file__).parent.parent / "data" / "db.json"
PAIRS_DIR = Path(__file__).parent.parent / "data" / "pairs"
_lock = Lock()

COLLECTIONS = ["projects", "jobs", "datasets", "eval_runs"]


def _read() -> dict:
    """Read the main db.json (small — no pairs)."""
    if not DB_PATH.exists():
        return {c: [] for c in COLLECTIONS}
    with open(DB_PATH) as f:
        raw = json.load(f)
    return {c: raw.get(c, []) for c in COLLECTIONS}


def _write(data: dict):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Never write pairs to main db
    clean = {c: data.get(c, []) for c in COLLECTIONS}
    with open(DB_PATH, "w") as f:
        json.dump(clean, f, indent=2)


def _pairs_path(owner_id: str) -> Path:
    """Get path for a pairs file: data/pairs/{owner_id}.json"""
    return PAIRS_DIR / f"{owner_id}.json"


def _read_pairs(owner_id: str) -> list[dict]:
    p = _pairs_path(owner_id)
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def _write_pairs(owner_id: str, pairs: list[dict]):
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_pairs_path(owner_id), "w") as f:
        json.dump(pairs, f)


# --- Generic helpers ---

def _get_all(collection: str, **filters) -> list[dict]:
    data = _read()
    items = data.get(collection, [])
    for k, v in filters.items():
        if v is not None:
            items = [i for i in items if i.get(k) == v]
    return items


def _get_one(collection: str, item_id: str) -> dict | None:
    for item in _read().get(collection, []):
        if item["id"] == item_id:
            return item
    return None


def _create(collection: str, item: dict) -> dict:
    with _lock:
        data = _read()
        data.setdefault(collection, []).append(item)
        _write(data)
    return item


def _update(collection: str, item_id: str, updates: dict):
    with _lock:
        data = _read()
        for item in data.get(collection, []):
            if item["id"] == item_id:
                item.update(updates)
                break
        _write(data)


def _delete(collection: str, item_id: str):
    with _lock:
        data = _read()
        data[collection] = [i for i in data.get(collection, []) if i["id"] != item_id]
        _write(data)


# --- Jobs ---

def get_jobs(status=None):
    return _get_all("jobs", status=status) if status else _get_all("jobs")

def get_job(job_id):
    return _get_one("jobs", job_id)

def create_job(job):
    return _create("jobs", job)

def delete_job(job_id):
    _delete("jobs", job_id)

def update_job(job_id, updates):
    _update("jobs", job_id, updates)

def update_job_progress(job_id, progress):
    with _lock:
        data = _read()
        for j in data["jobs"]:
            if j["id"] == job_id:
                j.setdefault("progress", {}).update(progress)
                break
        _write(data)


# --- Datasets (Ground Truth collections) ---

def get_datasets():
    return _get_all("datasets")

def get_dataset(dataset_id):
    return _get_one("datasets", dataset_id)

def create_dataset(ds):
    return _create("datasets", ds)

def update_dataset(dataset_id, updates):
    _update("datasets", dataset_id, updates)

def delete_dataset(dataset_id):
    _delete("datasets", dataset_id)
    p = _pairs_path(dataset_id)
    if p.exists():
        p.unlink()

def get_dataset_scenarios(dataset_id):
    return _read_pairs(dataset_id)

def add_scenarios(scenarios):
    if not scenarios:
        return
    dataset_id = scenarios[0].get("dataset_id", "unknown")
    with _lock:
        existing = _read_pairs(dataset_id)
        existing.extend(scenarios)
        _write_pairs(dataset_id, existing)


# --- Eval Runs ---

def get_eval_runs(dataset_id=None):
    if dataset_id:
        return _get_all("eval_runs", dataset_id=dataset_id)
    return _get_all("eval_runs")

def get_eval_run(run_id):
    return _get_one("eval_runs", run_id)

def create_eval_run(run):
    return _create("eval_runs", run)

def update_eval_run(run_id, updates):
    _update("eval_runs", run_id, updates)

def delete_eval_run(run_id):
    _delete("eval_runs", run_id)
    p = _pairs_path(run_id)
    if p.exists():
        p.unlink()

def get_eval_results(run_id):
    return _read_pairs(run_id)

def add_eval_results(results):
    if not results:
        return
    run_id = results[0].get("eval_run_id", "unknown")
    with _lock:
        existing = _read_pairs(run_id)
        existing.extend(results)
        _write_pairs(run_id, existing)


# --- Projects (legacy + RLHF preference data) ---

def get_projects():
    return _get_all("projects")

def get_project(project_id):
    return _get_one("projects", project_id)

def create_project(project):
    return _create("projects", project)

def update_project(project_id, updates):
    _update("projects", project_id, updates)

def delete_project(project_id):
    _delete("projects", project_id)


# --- Pairs (generic / legacy) ---

def get_pairs(project_id, status=None):
    items = _read_pairs(project_id)
    if status:
        items = [p for p in items if p.get("status") == status]
    return items

def add_pairs(pairs):
    if not pairs:
        return
    project_id = pairs[0].get("projectId", "unknown")
    with _lock:
        existing = _read_pairs(project_id)
        existing.extend(pairs)
        _write_pairs(project_id, existing)

save_pairs = add_pairs  # alias for rlhf_generator

def update_pair(pair_id, updates):
    _update("pairs", pair_id, updates)
