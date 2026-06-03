"""
Simple JSON file database.
Collections: jobs, datasets, eval_runs, projects, pairs
"""
import json
from pathlib import Path
from threading import Lock

DB_PATH = Path(__file__).parent.parent / "data" / "db.json"
_lock = Lock()

COLLECTIONS = ["projects", "pairs", "jobs", "datasets", "eval_runs"]


def _read() -> dict:
    if not DB_PATH.exists():
        return {c: [] for c in COLLECTIONS}
    with open(DB_PATH) as f:
        raw = json.load(f)
    return {c: raw.get(c, []) for c in COLLECTIONS}


def _write(data: dict):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)


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

def get_dataset_scenarios(dataset_id):
    data = _read()
    return [p for p in data.get("pairs", [])
            if p.get("dataset_id") == dataset_id and p.get("pair_type") == "ground_truth"]

def add_scenarios(scenarios):
    with _lock:
        data = _read()
        data["pairs"].extend(scenarios)
        _write(data)


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

def get_eval_results(run_id):
    data = _read()
    return [p for p in data.get("pairs", [])
            if p.get("eval_run_id") == run_id and p.get("pair_type") == "eval_result"]

def add_eval_results(results):
    with _lock:
        data = _read()
        data["pairs"].extend(results)
        _write(data)


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


# --- Pairs (generic) ---

def get_pairs(project_id, status=None):
    items = _get_all("pairs", project_id=project_id)
    if status:
        items = [p for p in items if p.get("status") == status]
    return items

def add_pairs(pairs):
    with _lock:
        data = _read()
        data["pairs"].extend(pairs)
        _write(data)

save_pairs = add_pairs  # alias for rlhf_generator

def update_pair(pair_id, updates):
    _update("pairs", pair_id, updates)
