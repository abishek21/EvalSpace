"""
Stage 3: RLHF Preference Data Generator

Takes two eval runs (preferred + rejected model) and creates
DPO-style preference pairs for training.
"""
from datetime import datetime
from uuid import uuid4

from app import db


async def process_rlhf_job(job_id: str, config: dict):
    """
    Stage 3: Generate RLHF preference pairs from two eval runs.
    - preferred_run_id: eval run from the better model (e.g. GPT-4o)
    - rejected_run_id: eval run from the weaker model (e.g. Qwen-3B)
    - strategy: 'all' | 'disagreements' | 'correct_preferred'
    """
    try:
        preferred_run_id = config["preferred_run_id"]
        rejected_run_id = config["rejected_run_id"]
        dataset_id = config["dataset_id"]
        strategy = config.get("strategy", "correct_preferred")
        project_name = config.get("name", "RLHF Pairs")

        db.update_job(job_id, {
            "status": "generating",
            "started_at": datetime.now().isoformat(),
        })

        # Load eval results
        preferred_results = db.get_eval_results(preferred_run_id)
        rejected_results = db.get_eval_results(rejected_run_id)

        preferred_run = db.get_eval_run(preferred_run_id)
        rejected_run = db.get_eval_run(rejected_run_id)

        # Index by scenario_id
        preferred_map = {r["scenario_id"]: r for r in preferred_results}
        rejected_map = {r["scenario_id"]: r for r in rejected_results}

        common_ids = set(preferred_map.keys()) & set(rejected_map.keys())
        print(f"\n{'='*55}")
        print(f"Generating RLHF pairs: {preferred_run['model']} vs {rejected_run['model']}")
        print(f"Common scenarios: {len(common_ids)}, Strategy: {strategy}")

        # Create project
        project_id = str(uuid4())
        db.create_project({
            "id": project_id,
            "name": project_name,
            "dataset_id": dataset_id,
            "preferred_model": preferred_run["model"],
            "rejected_model": rejected_run["model"],
            "preferred_run_id": preferred_run_id,
            "rejected_run_id": rejected_run_id,
            "strategy": strategy,
            "created_at": datetime.now().isoformat(),
            "job_id": job_id,
        })

        pairs = []
        for scenario_id in sorted(common_ids):
            pref = preferred_map[scenario_id]
            rej = rejected_map[scenario_id]

            # Apply strategy filter
            if strategy == "correct_preferred":
                # Only include if preferred model got it right
                if not pref.get("correct"):
                    continue
            elif strategy == "disagreements":
                # Only include if models disagree
                if pref.get("prediction") == rej.get("prediction"):
                    continue
            # 'all' strategy includes everything

            gt = pref.get("ground_truth", {})
            pair = {
                "id": str(uuid4()),
                "project_id": project_id,
                "dataset_id": dataset_id,
                "pair_type": "preference",
                "scenario_id": scenario_id,
                "scene_id": pref.get("scene_id", ""),
                "prompt": pref.get("prompt", ""),
                "category": "stacking_stability",
                "difficulty": pref.get("difficulty", ""),
                "ground_truth": gt,
                "chosen": {
                    "model": preferred_run["model"],
                    "response": pref.get("model_response", ""),
                    "prediction": pref.get("prediction", ""),
                    "reasoning": pref.get("reasoning", ""),
                    "correct": pref.get("correct", False),
                },
                "rejected": {
                    "model": rejected_run["model"],
                    "response": rej.get("model_response", ""),
                    "prediction": rej.get("prediction", ""),
                    "reasoning": rej.get("reasoning", ""),
                    "correct": rej.get("correct", False),
                },
                "source": pref.get("source", {}),
                "status": "pending",  # for human annotation
            }
            pairs.append(pair)

        # Save pairs
        if pairs:
            db.save_pairs(pairs)

        db.update_project(project_id, {"pair_count": len(pairs)})

        db.update_job(job_id, {
            "status": "completed",
            "project_id": project_id,
            "completed_at": datetime.now().isoformat(),
        })

        db.update_job_progress(job_id, {
            "pairs_generated": len(pairs),
            "total_common": len(common_ids),
        })

        print(f"✅ Generated {len(pairs)} preference pairs")
        print(f"   Project: {project_id[:8]}")
        print(f"   Strategy '{strategy}' filtered {len(common_ids)} → {len(pairs)}")

    except Exception as e:
        db.update_job(job_id, {"status": "failed", "error": str(e)})
        print(f"❌ RLHF job {job_id[:8]} failed: {e}")
        raise
