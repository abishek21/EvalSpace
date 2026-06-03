"""
Job processor — orchestrates the full pipeline:
1. Download dataset scenes
2. Generate questions via GPU worker
3. Generate answer pairs (chosen + rejected)
4. Save results as a project
"""
import json
import re
from datetime import datetime
from uuid import uuid4

from app import db
from app.gpu_client import infer
from app.datasets_service import load_scenes


async def process_job(job_id: str, config: dict):
    """Run the full generation pipeline for a job."""
    try:
        # Step 1: Download scenes
        db.update_job(job_id, {"status": "downloading", "started_at": datetime.now().isoformat()})

        scenes = load_scenes(
            dataset_name=config["dataset"],
            split=config["split"],
            num_scenes=config["num_scenes"],
            max_views=config["max_views"],
            image_resolution=config["image_resolution"],
            job_id=job_id,
            column_map=config.get("column_map"),
        )

        # Step 2: Generate
        db.update_job(job_id, {"status": "generating"})

        project_id = str(uuid4())
        all_pairs = []

        for scene_idx, scene in enumerate(scenes):
            # Generate questions
            questions = await generate_questions(scene, config)
            db.update_job_progress(job_id, {
                "scenes_processed": scene_idx + 1,
                "questions_generated": sum(1 for _ in all_pairs) + len(questions),
            })

            # Generate answer pairs
            for q in questions:
                try:
                    pair = await generate_pair(scene, q, config)
                    all_pairs.append(pair)
                    db.update_job_progress(job_id, {"pairs_generated": len(all_pairs)})
                except Exception as e:
                    print(f"  ⚠️ Pair error: {e}")
                    continue

        # Step 3: Save as project
        db.create_project({
            "id": project_id,
            "name": f"Generated: {config['dataset'].split('/')[-1]} ({len(scenes)} scenes)",
            "created_at": datetime.now().isoformat(),
            "job_id": job_id,
        })

        db_pairs = []
        for p in all_pairs:
            db_pairs.append({
                "id": str(uuid4()),
                "project_id": project_id,
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
                "scene_id": p["scene_id"],
                "category": p["category"],
                "difficulty": p["difficulty"],
                "status": "pending",
                "source": p["source"],
                "generation": p["generation"],
            })
        db.add_pairs(db_pairs)

        db.update_job(job_id, {
            "status": "completed",
            "project_id": project_id,
            "completed_at": datetime.now().isoformat(),
        })

        print(f"✅ Job {job_id} completed: {len(all_pairs)} pairs in project {project_id}")

    except Exception as e:
        db.update_job(job_id, {"status": "failed", "error": str(e)})
        print(f"❌ Job {job_id} failed: {e}")
        raise


async def generate_questions(scene: dict, config: dict) -> list[dict]:
    """Generate spatial reasoning questions for a scene."""
    descriptions = "; ".join(d[:100] for d in scene["descriptions"] if d)
    categories = ", ".join(config["categories"])

    template = config.get("question_user_prompt_template",
        "Scene: {descriptions}\n\nGenerate {n} questions across these categories: {categories}.\n"
        "Vary difficulty: easy, medium, hard.\n\n"
        'Output JSON:\n{{"questions": [{{"id": 1, "text": "...", "category": "...", "difficulty": "..."}}]}}')

    prompt = template.format(
        descriptions=descriptions,
        n=config["questions_per_scene"],
        categories=categories,
    )

    response = await infer(
        system_prompt=config.get("question_system_prompt", "Generate spatial reasoning questions about 3D scenes for robotics RLHF. Output ONLY valid JSON."),
        user_prompt=prompt,
        images=scene["images_b64"],
        max_tokens=1500,
        temperature=0.7,
    )

    # Parse JSON
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            try:
                data = json.loads(match.group())
            except:
                return []
        else:
            return []

    return data.get("questions", [])


async def generate_pair(scene: dict, question: dict, config: dict) -> dict:
    """Generate a chosen + rejected answer pair."""
    descriptions = "; ".join(d for d in scene["descriptions"][:3] if d)
    q_text = question.get("text", str(question))

    # Chosen: multi-view CoT with images
    chosen_template = config.get("chosen_user_prompt_template",
        "Scene annotations: {descriptions}\n\nQuestion: {question}\n\n"
        "Look at the provided camera views carefully and respond with step-by-step spatial reasoning (Observe → Reason → Answer).")
    chosen = await infer(
        system_prompt=config.get("answer_system_prompt", "You are a helpful spatial reasoning assistant."),
        user_prompt=chosen_template.format(descriptions=descriptions, question=q_text),
        images=scene["images_b64"],
        max_tokens=600,
        temperature=0.3,
    )

    # Rejected: text-only, no images, high temp
    rejected_template = config.get("rejected_user_prompt_template",
        "Room with objects: {descriptions_brief}\nQuestion: {question}\nAnswer:")
    rejected = await infer(
        system_prompt=config.get("rejected_system_prompt", "Answer questions about rooms briefly and directly. Don't overthink it."),
        user_prompt=rejected_template.format(descriptions_brief=descriptions[:60], question=q_text),
        images=None,
        max_tokens=150,
        temperature=0.9,
    )

    return {
        "prompt": f"User: A robot is observing a room. {q_text}",
        "chosen": chosen,
        "rejected": rejected,
        "scene_id": scene["id"],
        "category": question.get("category", ""),
        "difficulty": question.get("difficulty", ""),
        "source": {
            "dataset": config["dataset"],
            "split": config["split"],
            "row_indices": scene["row_indices"],
            "scene_id": scene["id"],
            "images": scene.get("image_paths", scene["images_b64"]),
        },
        "generation": {
            "model": config["model"],
            "chosen_strategy": "multi-view-cot",
            "rejected_strategy": "text-only-shallow",
            "chosen_temperature": 0.3,
            "rejected_temperature": 0.9,
            "num_views": len(scene["images_b64"]),
            "image_resolution": config["image_resolution"],
            "generated_at": datetime.now().isoformat(),
        },
    }
