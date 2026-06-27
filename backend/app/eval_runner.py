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
from app.dise_ui import Constraint, evaluate_constraints, screenshot as ui_screenshot


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

# -- Spatial Fitting --
FITTING_SYSTEM = """You are a spatial reasoning assistant. You analyze images of an object next to a wall with an opening, and determine whether the object could fit through the opening.

Look at the images carefully and consider:
- The size of the object relative to the opening
- Whether the object could be rotated or reoriented to fit
- The shape of the object vs the shape of the opening

Answer with EXACTLY this format:
PREDICTION: FITS or DOES NOT FIT
REASONING: <your reasoning in 2-3 sentences>"""

FITTING_USER = """Look at these images showing an object and a wall with an opening.

{question}

Answer with PREDICTION: FITS or DOES NOT FIT, then REASONING."""

# -- UI Visual Coding --
UI_SYSTEM = """You are a frontend developer assistant. You will be shown a screenshot of a UI and its HTML/CSS source code.

You will receive an instruction to modify the layout. Your job is to output the COMPLETE modified HTML/CSS code that implements the requested change.

Rules:
- Output ONLY the complete HTML document (<!DOCTYPE html> to </html>)
- Keep all existing elements — only change what the instruction asks
- Use absolute positioning (the original code uses it)
- Do NOT add explanations — just output the code"""

UI_USER = """Here is a screenshot of a UI and its source code.

INSTRUCTION: {instruction}

CURRENT HTML/CSS CODE:
```html
{code}
```

Output the COMPLETE modified HTML/CSS code that implements the instruction. Output ONLY the code, nothing else."""

# -- Perspective Taking & Allocentric --
PERSPECTIVE_SYSTEM = """You are a spatial reasoning expert. Look at the image carefully and answer the question.

Answer with EXACTLY this format:
PREDICTION: <object name or left/right>
REASONING: <your reasoning in 1-2 sentences>"""

PERSPECTIVE_USER = """{question}"""


def _parse_vlm_prediction(response: str, task_type: str = "stacking_stability") -> tuple[str, str]:
    response_upper = response.upper()
    prediction = "unknown"

    if task_type in ("perspective_taking", "allocentric"):
        # Extract PREDICTION: ... line, match only against that
        prediction = response.strip()
        reasoning = response
        
        resp_upper = response.upper()
        if "PREDICTION:" in resp_upper:
            idx = resp_upper.find("PREDICTION:")
            after = response[idx + 11:].strip()
            # Take everything up to next newline or REASONING:
            end = len(after)
            for delim in ["\n", "REASONING:", "reasoning:"]:
                pos = after.upper().find(delim.upper())
                if pos > 0:
                    end = min(end, pos)
            prediction = after[:end].strip().strip("*").strip()
        
        if "REASONING:" in resp_upper:
            idx = resp_upper.find("REASONING:")
            reasoning = response[idx + 10:].strip()
        
        return prediction, reasoning

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
    elif task_type in ("spatial_fitting", "shelf_fitting", "door_passage"):
        # Parse FITS / DOES NOT FIT
        if "DOES NOT FIT" in response_upper or "DOESN'T FIT" in response_upper:
            prediction = "no_fit"
        elif "PREDICTION: FITS" in response_upper or "PREDICTION:FITS" in response_upper:
            prediction = "fits"
        elif "NOT FIT" in response_upper or "NO FIT" in response_upper or "CANNOT FIT" in response_upper:
            prediction = "no_fit"
        elif "FITS" in response_upper:
            prediction = "fits"
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
    model_name: str = "gpt-4o",
    timeout: int = 120,
) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=endpoint, api_key=api_key, timeout=timeout)

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

    # GPT-5 / o-series: different API params
    is_new_api = any(x in model_name for x in ["gpt-5", "o1", "o3", "o4"])
    # GPT-5.5 needs more tokens due to internal CoT
    if "gpt-5.5" in model_name:
        max_tokens = max(max_tokens, 1000)
    extra_params = {}
    if is_new_api:
        extra_params["max_completion_tokens"] = max_tokens
    else:
        extra_params["max_tokens"] = max_tokens
        extra_params["temperature"] = temperature

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        **extra_params,
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
        use_gpt = azure_cfg and azure_cfg.get("api_key")

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

        # Route UI visual to separate handler
        if task_type == "ui_visual":
            await _process_ui_eval(job_id, config, dataset, scenarios, model, azure_cfg, use_gpt)
            return

        is_collision = task_type == "collision_prediction"
        num_scenarios = len(scenarios)
        print(f"\n{'='*55}")
        print(f"Evaluating {model} on '{dataset['name']}' ({num_scenarios} scenarios, {task_type})")

        # Pick prompts based on task type
        if task_type == "collision_prediction":
            system_prompt = COLLISION_SYSTEM
            positive_label = "hit"
            negative_label = "miss"
        elif task_type in ("spatial_fitting", "shelf_fitting", "door_passage"):
            system_prompt = FITTING_SYSTEM
            positive_label = "fits"
            negative_label = "no_fit"
        elif task_type in ("perspective_taking", "allocentric"):
            system_prompt = PERSPECTIVE_SYSTEM
            positive_label = "correct"
            negative_label = "incorrect"
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
            # Fallback: perspective_taking/allocentric store images in source.images
            if not images:
                images = scenario.get("source", {}).get("images", [])
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
            if task_type == "collision_prediction":
                vlm_prompt = COLLISION_USER.format(question=question)
            elif task_type in ("spatial_fitting", "shelf_fitting", "door_passage"):
                vlm_prompt = FITTING_USER.format(question=question)
            elif task_type in ("perspective_taking", "allocentric"):
                vlm_prompt = PERSPECTIVE_USER.format(question=question)
            else:
                vlm_prompt = STABILITY_USER.format(
                    description=f"Stack (bottom to top): {stack_desc}\n\n{question}"
                )

            if use_gpt:
                print(f"  Querying {model} (Azure)...")
                try:
                    vlm_response = await _azure_vision_infer(
                        endpoint=azure_cfg["endpoint"],
                        api_key=azure_cfg["api_key"],
                        system_prompt=system_prompt,
                        user_prompt=vlm_prompt,
                        images=images,
                        model_name=model,
                    )
                except Exception as e:
                    print(f"  ⚠️ API call failed (timeout/error): {e}")
                    vlm_response = ""
            else:
                print(f"  Querying {model}...")
                # Send front and top views at 384px to keep payload small for ngrok
                front_and_top = [images[0], images[2]] if len(images) > 2 else images[:2]
                small_images = _downscale_images(front_and_top, max_size=384)
                try:
                    vlm_response = await infer(
                        system_prompt=system_prompt,
                        user_prompt=vlm_prompt,
                        images=small_images,
                        max_tokens=300,
                        temperature=0.3,
                    )
                except Exception as e:
                    print(f"  ⚠️ GPU call failed: {e}")
                    vlm_response = ""

            prediction, reasoning = _parse_vlm_prediction(vlm_response, task_type)
            
            # Correctness check
            if task_type in ("perspective_taking", "allocentric"):
                # Flexible free-form matching:
                # 1. Direct match (all GT words in response)
                # 2. Color match + synonym check (e.g., "blue drum" matches "blue barrel")
                resp_lower = prediction.lower()
                gt_lower = gt_answer.lower()
                gt_words = gt_lower.split()
                
                # Synonyms: model may describe objects differently
                synonyms = {
                    "box": ["box", "cube", "block", "crate"],
                    "ball": ["ball", "sphere", "orb"],
                    "barrel": ["barrel", "drum", "cylinder", "canister"],
                    "cone": ["cone", "traffic cone", "pylon"],
                    "pillar": ["pillar", "column", "cylinder", "pole", "post"],
                    "pyramid": ["pyramid", "stepped", "ziggurat"],
                    "cylinder": ["cylinder", "tube", "pillar", "column"],
                }
                
                # Check 1: exact word match
                correct = all(w in resp_lower for w in gt_words)
                
                if not correct:
                    # Check 2: color match + shape synonym
                    gt_color = gt_words[0] if len(gt_words) >= 2 else ""
                    gt_shape = gt_words[1] if len(gt_words) >= 2 else gt_words[0]
                    
                    color_match = gt_color in resp_lower if gt_color else True
                    shape_match = False
                    for syn in synonyms.get(gt_shape, [gt_shape]):
                        if syn in resp_lower:
                            shape_match = True
                            break
                    correct = color_match and shape_match
                
                if not correct and gt_lower in ("left", "right"):
                    # Direction questions: just check if the direction word appears
                    correct = gt_lower in resp_lower
            else:
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


# ─── UI Visual Coding Eval ──────────────────────────────────────────

def _extract_html(response: str) -> str:
    """Extract HTML code from VLM response (may be wrapped in ```html blocks)."""
    text = response.strip()
    # Remove markdown code fences
    if "```html" in text:
        text = text.split("```html", 1)[1]
        if "```" in text:
            text = text.split("```")[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```")[0]
    text = text.strip()
    # Ensure it's valid HTML
    if not text.startswith("<!DOCTYPE") and not text.startswith("<html"):
        if "<body" in text or "<div" in text:
            text = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>*{{margin:0;padding:0;box-sizing:border-box}}body{{width:1280px;height:720px;background:#f8fafc;font-family:-apple-system,sans-serif;position:relative;overflow:hidden}}</style></head>{text}</html>"
    return text


async def _process_ui_eval(job_id, config, dataset, scenarios, model, azure_cfg, use_gpt):
    """Evaluate a VLM on UI visual coding tasks."""
    try:
        dataset_id = dataset["id"]
        num_scenarios = len(scenarios)
        eval_run_id = str(uuid4())
        all_results = []
        correct_count = 0

        print(f"\n{'='*55}")
        print(f"🖥️ UI Visual Eval: {model} on '{dataset['name']}' ({num_scenarios} scenarios)")

        for idx, scenario in enumerate(scenarios):
            gt_data = scenario.get("ground_truth", {})
            html_code = gt_data.get("html_code", "")
            images = gt_data.get("before_images", [])
            instruction = scenario.get("prompt", "")
            constraint_defs = gt_data.get("constraints", [])

            print(f"\n[{idx+1}/{num_scenarios}] {scenario.get('scene_id', '')} — {instruction[:50]}")

            # Build VLM prompt
            vlm_prompt = UI_USER.format(instruction=instruction, code=html_code)

            # Call VLM
            if use_gpt:
                print(f"  Querying {model} (Azure)...")
                try:
                    vlm_response = await _azure_vision_infer(
                        endpoint=azure_cfg["endpoint"],
                        api_key=azure_cfg["api_key"],
                        system_prompt=UI_SYSTEM,
                        user_prompt=vlm_prompt,
                        images=images,
                        model_name=model,
                        max_tokens=4000,
                    )
                except Exception as e:
                    print(f"  ⚠️ API call failed (timeout/error): {e}")
                    vlm_response = ""
            else:
                print(f"  Querying {model}...")
                small_images = _downscale_images(images[:1], max_size=512)
                try:
                    vlm_response = await infer(
                        system_prompt=UI_SYSTEM,
                        user_prompt=vlm_prompt,
                        images=small_images,
                        max_tokens=4000,
                        temperature=0.3,
                    )
                except Exception as e:
                    print(f"  ⚠️ GPU call failed: {e}")
                    vlm_response = ""

            # Extract HTML from response
            vlm_html = _extract_html(vlm_response)

            # Evaluate constraints
            constraints = [
                Constraint(
                    type=c["type"],
                    selector=c["selector"],
                    params=c.get("params", {}),
                    description=c.get("description", ""),
                )
                for c in constraint_defs
            ]

            try:
                constraint_results = evaluate_constraints(vlm_html, constraints)
                passed = sum(1 for r in constraint_results if r["passed"])
                total = len(constraint_results)
                correct = passed == total
            except Exception as e:
                print(f"  ⚠️ Constraint eval failed: {e}")
                constraint_results = [{"constraint": "eval_error", "passed": False, "detail": str(e)}]
                passed = 0
                total = len(constraints)
                correct = False

            if correct:
                correct_count += 1

            # Render after screenshot
            try:
                after_img = ui_screenshot(vlm_html)
            except Exception:
                after_img = ""

            icon = "✅" if correct else "❌"
            print(f"  {icon} {passed}/{total} constraints passed")

            result = {
                "id": str(uuid4()),
                "eval_run_id": eval_run_id,
                "dataset_id": dataset_id,
                "pair_type": "eval_result",
                "scenario_id": scenario["id"],
                "scene_id": scenario.get("scene_id", ""),
                "prompt": instruction,
                "category": "ui_visual",
                "difficulty": scenario.get("difficulty", ""),
                "ground_truth": {
                    "answer": "pass",
                    "before_images": images,
                    "after_images": [after_img] if after_img else [],
                    "constraints": constraint_defs,
                    "html_code": html_code,
                },
                "model_response": vlm_response[:5000],
                "prediction": "pass" if correct else "fail",
                "reasoning": json_safe_constraints(constraint_results),
                "correct": correct,
                "source": scenario.get("source", {}),
                "status": "pending",
            }
            all_results.append(result)

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
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
            "task_type": "ui_visual",
            "positive_label": "pass",
            "negative_label": "fail",
            "positive_accuracy": accuracy,
            "negative_accuracy": 0,
            "positive_correct": correct_count,
            "positive_total": num_scenarios,
            "negative_correct": 0,
            "negative_total": 0,
            # backward compat
            "stable_accuracy": accuracy,
            "unstable_accuracy": 0,
            "stable_correct": correct_count,
            "stable_total": num_scenarios,
            "unstable_correct": 0,
            "unstable_total": 0,
        }

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
        print(f"✅ UI Eval: {model} — {correct_count}/{num_scenarios} ({accuracy}%)")

    except Exception as e:
        import traceback
        error_detail = f"{e.__class__.__name__}: {e}"
        print(f"❌ UI eval failed: {error_detail}")
        traceback.print_exc()
        db.update_job(job_id, {"status": "failed", "error": error_detail})
        raise


def json_safe_constraints(results: list[dict]) -> str:
    """Convert constraint results to a readable string."""
    lines = []
    for r in results:
        icon = "✅" if r.get("passed") else "❌"
        lines.append(f"{icon} {r.get('constraint', '?')}: {r.get('detail', '')}")
    return "\n".join(lines)
