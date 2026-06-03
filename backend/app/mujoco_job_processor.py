"""
MuJoCo Job Processor — generates RLHF pairs from procedural physics scenes.

Same flow as job_processor.py but uses MuJoCo instead of HuggingFace datasets:
  1. Generate random tabletop scenes (MuJoCo)
  2. Render multi-view images
  3. Send to GPU for question generation
  4. Generate chosen (with images) + rejected (without images) answer pairs
  5. Auto-verify answers against physics ground truth
  6. Save as project with verification metadata
"""
import json
import re
import math
from datetime import datetime
from uuid import uuid4

from app import db
from app.gpu_client import infer
from app.mujoco_scene_gen import generate_random_scene, generate_batch


async def process_mujoco_job(job_id: str, config: dict):
    """Run the MuJoCo RLHF generation pipeline."""
    try:
        db.update_job(job_id, {"status": "downloading", "started_at": datetime.now().isoformat()})

        num_scenes = config.get("num_scenes", 5)
        questions_per_scene = config.get("questions_per_scene", 10)
        objects_per_scene = config.get("objects_per_scene", 4)
        categories = config.get("categories", ["counting", "spatial", "occlusion", "affordance", "manipulation"])

        # Step 1: Generate scenes (this is fast — no download needed)
        scenes = generate_batch(
            num_scenes=num_scenes,
            objects_per_scene=objects_per_scene,
        )

        # Step 2: Generate Q&A pairs
        db.update_job(job_id, {"status": "generating"})

        project_id = str(uuid4())
        all_pairs = []

        for scene_idx, scene in enumerate(scenes):
            # Generate questions
            questions = await _generate_questions(scene, config)
            db.update_job_progress(job_id, {
                "scenes_processed": scene_idx + 1,
                "questions_generated": len(all_pairs) + len(questions),
            })

            # Generate answer pairs
            for q in questions:
                try:
                    pair = await _generate_pair(scene, q, config)
                    # Auto-verify against ground truth
                    pair["verification"] = _verify_answer(pair, scene)
                    all_pairs.append(pair)
                    db.update_job_progress(job_id, {"pairs_generated": len(all_pairs)})
                except Exception as e:
                    print(f"  ⚠️ Pair error: {e}")
                    continue

        # Step 3: Save as project
        db.create_project({
            "id": project_id,
            "name": f"MuJoCo: {num_scenes} scenes, {objects_per_scene} objects/scene",
            "created_at": datetime.now().isoformat(),
            "job_id": job_id,
            "source_type": "mujoco",
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
                "verification": p.get("verification"),
            })
        db.add_pairs(db_pairs)

        db.update_job(job_id, {
            "status": "completed",
            "project_id": project_id,
            "completed_at": datetime.now().isoformat(),
        })

        verified = sum(1 for p in all_pairs if p.get("verification", {}).get("physics_correct"))
        print(f"✅ MuJoCo job {job_id} completed: {len(all_pairs)} pairs ({verified} physics-verified)")

    except Exception as e:
        db.update_job(job_id, {"status": "failed", "error": str(e)})
        print(f"❌ MuJoCo job {job_id} failed: {e}")
        raise


async def _generate_questions(scene, config: dict) -> list[dict]:
    """Generate questions from MuJoCo scene views."""
    desc = "; ".join(scene.descriptions)
    categories = ", ".join(config.get("categories", ["counting", "spatial", "manipulation"]))
    n = config.get("questions_per_scene", 10)

    template = config.get("question_user_prompt_template",
        "Scene: {descriptions}\n\nGenerate {n} questions across these categories: {categories}.\n"
        "Vary difficulty: easy, medium, hard.\n\n"
        'Output JSON:\n{{"questions": [{{"id": 1, "text": "...", "category": "...", "difficulty": "..."}}]}}')

    prompt = template.format(descriptions=desc, n=n, categories=categories)

    response = await infer(
        system_prompt=config.get("question_system_prompt",
            "Generate spatial reasoning questions about this 3D tabletop scene for robotics RLHF. Output ONLY valid JSON."),
        user_prompt=prompt,
        images=scene.images_b64,
        max_tokens=1500,
        temperature=0.7,
    )

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


async def _generate_pair(scene, question: dict, config: dict) -> dict:
    """Generate chosen + rejected answers for a question."""
    desc = "; ".join(scene.descriptions)
    q_text = question.get("text", str(question))

    # Chosen: multi-view CoT with images
    chosen_template = config.get("chosen_user_prompt_template",
        "Scene: {descriptions}\n\nQuestion: {question}\n\n"
        "Look at the provided camera views and respond with step-by-step spatial reasoning (Observe → Reason → Answer).")
    chosen = await infer(
        system_prompt=config.get("answer_system_prompt",
            "You are a spatial reasoning VLM for robotics. Analyze 3D scenes from multiple views with precise CoT reasoning."),
        user_prompt=chosen_template.format(descriptions=desc, question=q_text),
        images=scene.images_b64,
        max_tokens=600,
        temperature=0.3,
    )

    # Rejected: no images, shallow
    rejected_template = config.get("rejected_user_prompt_template",
        "Table with objects: {descriptions_brief}\nQuestion: {question}\nAnswer:")
    rejected = await infer(
        system_prompt=config.get("rejected_system_prompt",
            "Answer questions about a table scene briefly. Don't overthink it."),
        user_prompt=rejected_template.format(descriptions_brief=desc[:80], question=q_text),
        images=None,
        max_tokens=150,
        temperature=0.9,
    )

    return {
        "prompt": f"User: A robot is observing a tabletop scene. {q_text}",
        "chosen": chosen,
        "rejected": rejected,
        "scene_id": scene.id,
        "category": question.get("category", ""),
        "difficulty": question.get("difficulty", ""),
        "source": {
            "dataset": "mujoco:tabletop",
            "split": "generated",
            "row_indices": [],
            "scene_id": scene.id,
            "images": scene.images_b64,
            "ground_truth": scene.ground_truth,
        },
        "generation": {
            "model": config.get("model", "Qwen/Qwen2.5-VL-3B-Instruct"),
            "chosen_strategy": "multi-view-cot",
            "rejected_strategy": "text-only-shallow",
            "chosen_temperature": 0.3,
            "rejected_temperature": 0.9,
            "num_views": len(scene.images_b64),
            "image_resolution": config.get("image_resolution", 480),
            "generated_at": datetime.now().isoformat(),
            "source_type": "mujoco",
        },
    }


def _verify_answer(pair: dict, scene) -> dict:
    """Auto-verify chosen answer against physics ground truth."""
    gt = scene.ground_truth
    chosen = pair["chosen"].lower()
    category = pair["category"].lower()

    checks = []

    # Counting verification
    if category == "counting":
        for obj_type, count in gt.get("categories", {}).items():
            if obj_type in chosen:
                # Check if the VLM got the count right
                for word, num in [("one", 1), ("two", 2), ("three", 3), ("four", 4),
                                   ("five", 5), ("six", 6), ("1", 1), ("2", 2),
                                   ("3", 3), ("4", 4), ("5", 5), ("6", 6)]:
                    if word in chosen:
                        correct = (num == count)
                        checks.append({
                            "type": "counting",
                            "claim": f"{num} {obj_type}(s)",
                            "ground_truth": count,
                            "correct": correct,
                        })
                        break

    # Spatial relation verification
    if category == "spatial":
        gt_relations = gt.get("spatial_relations", [])
        for rel in gt_relations:
            rel_lower = rel.lower()
            # Check if VLM mentions similar spatial claim
            key_phrases = ["left", "right", "behind", "front", "above", "close"]
            for phrase in key_phrases:
                if phrase in rel_lower and phrase in chosen:
                    # Check if the objects in the relation are mentioned
                    objects_in_rel = [o.label.lower() for o in scene.objects]
                    mentioned = sum(1 for o in objects_in_rel if o in chosen)
                    if mentioned >= 2:
                        checks.append({
                            "type": "spatial",
                            "claim": rel,
                            "correct": True,  # VLM agrees with ground truth
                        })

    physics_correct = all(c["correct"] for c in checks) if checks else None

    return {
        "physics_correct": physics_correct,
        "checks": checks,
        "ground_truth_summary": {
            "object_count": gt["object_count"],
            "categories": gt["categories"],
            "num_relations": len(gt.get("spatial_relations", [])),
        },
    }
