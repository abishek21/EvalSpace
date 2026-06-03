"""
Stage 2: Model Evaluation Runner

Runs a VLM model against an existing ground truth dataset.
Compares predictions to MuJoCo ground truth and generates metrics.
"""
import base64
import io
import os
from datetime import datetime
from uuid import uuid4
from PIL import Image

from app import db
from app.gpu_client import infer


def _downscale_images(images: list[str], max_size: int = 512) -> list[str]:
    """Downscale base64 images to max_size for faster GPU inference."""
    result = []
    for img in images:
        try:
            if img.startswith("data:"):
                header, b64 = img.split(",", 1)
            else:
                header, b64 = "data:image/png;base64", img
            raw = base64.b64decode(b64)
            pil = Image.open(io.BytesIO(raw))
            pil.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=70)
            small_b64 = base64.b64encode(buf.getvalue()).decode()
            result.append(f"data:image/jpeg;base64,{small_b64}")
        except Exception:
            result.append(img)  # fallback: send original
    return result


# ─── VLM Prompts ────────────────────────────────────────────────────

# -- Stacking Stability --
STABILITY_SYSTEM = """You are a physics reasoning assistant for robotics. You analyze images of stacked objects and predict whether the stack is physically stable.

Look at the images carefully and consider:
- Support area: Is the top object's center of mass over the base?
- Shape compatibility: Flat-on-flat is stable, round-on-round is not
- Height vs base width: Tall narrow stacks are less stable
- Offsets: Objects placed off-center may fall

Answer with EXACTLY this format:
PREDICTION: STABLE or UNSTABLE
REASONING: <your reasoning in 2-3 sentences>"""

STABILITY_USER = """Look at these images of objects stacked on a table.

{description}

Question: Will this stack remain stable, or will objects fall/topple?

Answer with PREDICTION: STABLE or UNSTABLE, then REASONING."""

# -- Collision Prediction --
COLLISION_SYSTEM = """You are a physics reasoning assistant for robotics. You analyze images of objects on a table and predict whether a pushed object will collide with a target object.

Look at the images carefully and consider:
- Distance between the pushed object and the target
- Direction of the push relative to the target's position
- Any obstacles in the path that could block or deflect the pushed object
- The force of the push — will it be enough to reach the target?
- Object sizes and masses — heavy objects are harder to deflect

Answer with EXACTLY this format:
PREDICTION: HIT or MISS
REASONING: <your reasoning in 2-3 sentences>"""

COLLISION_USER = """Look at these images of objects on a table. One object is about to be pushed.

{question}

Answer with PREDICTION: HIT or MISS, then REASONING."""


def _parse_vlm_prediction(response: str, task_type: str = "stacking_stability") -> tuple[str, str]:
    response_upper = response.upper()
    prediction = "unknown"

    if task_type == "collision_prediction":
        # Parse HIT / MISS
        if "PREDICTION: HIT" in response_upper or "PREDICTION:HIT" in response_upper:
            prediction = "hit"
        elif "PREDICTION: MISS" in response_upper or "PREDICTION:MISS" in response_upper:
            prediction = "miss"
        elif "MISS" in response_upper and "HIT" not in response_upper:
            prediction = "miss"
        elif "HIT" in response_upper and "MISS" not in response_upper:
            prediction = "hit"
    else:
        # Parse STABLE / UNSTABLE
        if "PREDICTION: STABLE" in response_upper or "PREDICTION:STABLE" in response_upper:
            idx = response_upper.find("PREDICTION:")
            after = response_upper[idx+11:idx+25].strip()
            if after.startswith("UNSTABLE"):
                prediction = "unstable"
            else:
                prediction = "stable"
        elif "PREDICTION: UNSTABLE" in response_upper or "PREDICTION:UNSTABLE" in response_upper:
            prediction = "unstable"
        elif "UNSTABLE" in response_upper and "STABLE" not in response_upper.replace("UNSTABLE", ""):
            prediction = "unstable"
        elif "STABLE" in response_upper:
            prediction = "stable"

    reasoning = response
    if "REASONING:" in response.upper():
        idx = response.upper().find("REASONING:")
        reasoning = response[idx + 10:].strip()

    return prediction, reasoning


async def _azure_vision_infer(
    endpoint: str, api_key: str, system_prompt: str, user_prompt: str,
    images: list[str], max_tokens: int = 300, temperature: float = 0.3,
) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=endpoint, api_key=api_key)

    content_parts = [{"type": "text", "text": user_prompt}]
    for img_path in images:
        if img_path.startswith("data:"):
            content_parts.append({"type": "image_url", "image_url": {"url": img_path}})
        elif os.path.isfile(img_path):
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        else:
            content_parts.append({"type": "image_url", "image_url": {"url": img_path}})

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ─── Main Eval Runner ───────────────────────────────────────────────

async def process_eval_job(job_id: str, config: dict):
    """
    Stage 2: Evaluate a model against a ground truth dataset.
    Loads scenarios from dataset → sends to VLM → compares to GT → saves metrics.
    """
    try:
        dataset_id = config["dataset_id"]
        model = config.get("model", "qwen2.5-vl-3b")

        # Normalize Azure config
        azure_cfg = config.get("azure_config")
        if azure_cfg:
            azure_cfg["api_key"] = azure_cfg.get("api_key") or azure_cfg.get("apiKey", "")
        use_gpt = model == "gpt-4o" and azure_cfg and azure_cfg.get("api_key")

        db.update_job(job_id, {
            "status": "generating",
            "started_at": datetime.now().isoformat(),
        })

        # Load dataset scenarios
        scenarios = db.get_dataset_scenarios(dataset_id)
        if not scenarios:
            raise ValueError(f"Dataset {dataset_id} has no scenarios")

        dataset = db.get_dataset(dataset_id)
        task_type = dataset.get("task_type", "stacking_stability")
        is_collision = task_type == "collision_prediction"
        num_scenarios = len(scenarios)
        print(f"\n{'='*55}")
        print(f"Evaluating {model} on '{dataset['name']}' ({num_scenarios} scenarios, {task_type})")

        # Pick prompts based on task type
        if is_collision:
            system_prompt = COLLISION_SYSTEM
            positive_label = "hit"
            negative_label = "miss"
        else:
            system_prompt = STABILITY_SYSTEM
            positive_label = "stable"
            negative_label = "unstable"

        eval_run_id = str(uuid4())
        all_results = []
        correct_count = 0
        positive_correct = 0
        positive_total = 0
        negative_correct = 0
        negative_total = 0

        for idx, scenario in enumerate(scenarios):
            gt_data = scenario.get("ground_truth", {})
            gt_answer = gt_data.get("answer", "")
            images = gt_data.get("before_images", [])
            question = scenario.get("prompt", "")
            objects = gt_data.get("objects", [])

            stack_desc = ", ".join(
                f"{o['color']} {o['label']}"
                + (f" (offset {abs(o.get('offset_x', 0))*100:.0f}cm)" if o.get('offset_x', 0) != 0 else "")
                for o in objects
            )

            print(f"\n[{idx+1}/{num_scenarios}] {scenario.get('scene_id', f'scenario_{idx}')}")
            print(f"  GT: {gt_answer.upper()}")

            # Build VLM prompt based on task type
            if is_collision:
                # Only give the VLM the natural language question — NO physics params
                vlm_prompt = COLLISION_USER.format(question=question)
            else:
                vlm_prompt = STABILITY_USER.format(
                    description=f"Stack (bottom to top): {stack_desc}\n\n{question}"
                )

            if use_gpt:
                print(f"  Querying GPT-4o...")
                vlm_response = await _azure_vision_infer(
                    endpoint=azure_cfg["endpoint"],
                    api_key=azure_cfg["api_key"],
                    system_prompt=system_prompt,
                    user_prompt=vlm_prompt,
                    images=images,
                )
            else:
                print(f"  Querying {model}...")
                # Send front and top views at 384px to keep payload small for ngrok
                front_and_top = [images[0], images[2]] if len(images) > 2 else images[:2]
                small_images = _downscale_images(front_and_top, max_size=384)
                vlm_response = await infer(
                    system_prompt=system_prompt,
                    user_prompt=vlm_prompt,
                    images=small_images,
                    max_tokens=300,
                    temperature=0.3,
                )

            prediction, reasoning = _parse_vlm_prediction(vlm_response, task_type)
            correct = prediction == gt_answer
            if correct:
                correct_count += 1

            if gt_answer == positive_label:
                positive_total += 1
                if correct:
                    positive_correct += 1
            else:
                negative_total += 1
                if correct:
                    negative_correct += 1

            icon = "✅" if correct else "❌"
            print(f"  Prediction: {prediction.upper()} {icon}")

            # Save eval result
            result = {
                "id": str(uuid4()),
                "eval_run_id": eval_run_id,
                "dataset_id": dataset_id,
                "pair_type": "eval_result",
                "scenario_id": scenario["id"],
                "scene_id": scenario.get("scene_id", ""),
                "prompt": question,
                "category": task_type,
                "difficulty": scenario.get("difficulty", ""),
                "ground_truth": gt_data,
                "model_response": vlm_response,
                "prediction": prediction,
                "reasoning": reasoning,
                "correct": correct,
                "source": scenario.get("source", {}),
                "status": "pending",
            }
            all_results.append(result)

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "pairs_generated": len(all_results),
                "correct": correct_count,
                "total": len(all_results),
                "scenarios_total": num_scenarios,
            })

        # Compute metrics
        accuracy = round(correct_count / num_scenarios * 100, 1) if num_scenarios > 0 else 0
        metrics = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": num_scenarios,
            "task_type": task_type,
            # Generic positive/negative labels
            "positive_label": positive_label,
            "negative_label": negative_label,
            "positive_accuracy": round(positive_correct / positive_total * 100, 1) if positive_total > 0 else 0,
            "negative_accuracy": round(negative_correct / negative_total * 100, 1) if negative_total > 0 else 0,
            "positive_correct": positive_correct,
            "positive_total": positive_total,
            "negative_correct": negative_correct,
            "negative_total": negative_total,
            # Keep backward-compatible aliases for stacking
            "stable_accuracy": round(positive_correct / positive_total * 100, 1) if positive_total > 0 else 0,
            "unstable_accuracy": round(negative_correct / negative_total * 100, 1) if negative_total > 0 else 0,
            "stable_correct": positive_correct,
            "stable_total": positive_total,
            "unstable_correct": negative_correct,
            "unstable_total": negative_total,
        }

        # Save eval run
        db.create_eval_run({
            "id": eval_run_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset["name"],
            "model": model,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
            "job_id": job_id,
        })
        db.add_eval_results(all_results)

        db.update_job(job_id, {
            "status": "completed",
            "eval_run_id": eval_run_id,
            "completed_at": datetime.now().isoformat(),
        })

        print(f"\n{'='*55}")
        print(f"✅ Eval run {eval_run_id[:8]} completed")
        print(f"   {model}: {correct_count}/{num_scenarios} ({accuracy}%)")

    except Exception as e:
        import traceback
        error_detail = f"{e.__class__.__name__}: {e}"
        print(f"❌ Eval job {job_id[:8]} failed: {error_detail}")
        traceback.print_exc()
        db.update_job(job_id, {"status": "failed", "error": error_detail})
        raise
