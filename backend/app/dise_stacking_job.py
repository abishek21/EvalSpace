"""
Stacking Stability Job Processor

Pipeline:
1. Generate N stacking scenarios (randomized)
2. Simulate each in MuJoCo → get ground truth (stable/unstable)
3. Send "before" renders to VLM → ask "Is this stable?"
4. Compare VLM answer to MuJoCo ground truth
5. Save as project with verification metadata
"""
import random
from datetime import datetime
from uuid import uuid4

import httpx

from app import db
from app.gpu_client import infer
from app.dise_stacking import (
    StackingScenario,
    StackObject,
    simulate_stacking,
    generate_10_scenarios,
    STACKABLE_OBJECTS,
    COLORS,
)

# ─── Random scenario generator ──────────────────────────────────────

def _random_scenario(idx: int) -> StackingScenario:
    """Generate a random stacking scenario with 2-4 objects."""
    n_objects = random.choice([2, 2, 2, 3, 3, 4])  # bias toward 2-3
    obj_types = list(STACKABLE_OBJECTS.keys())
    color_names = list(COLORS.keys())

    objects = []
    for i in range(n_objects):
        obj_type = random.choice(obj_types)
        color = random.choice(color_names)

        # Random offset — more likely on higher objects
        offset_x = 0.0
        offset_y = 0.0
        if i > 0 and random.random() < 0.4:
            offset_x = random.uniform(-0.06, 0.06)
        if i > 0 and random.random() < 0.2:
            offset_y = random.uniform(-0.04, 0.04)

        objects.append(StackObject(obj_type, color, offset_x, offset_y))

    # Build question
    obj_descriptions = []
    for i, o in enumerate(objects):
        label = STACKABLE_OBJECTS[o.obj_type]["label"]
        pos = "bottom" if i == 0 else ("top" if i == len(objects) - 1 else "middle")
        offset_desc = ""
        if o.offset_x != 0 or o.offset_y != 0:
            offset_desc = f" (offset {abs(o.offset_x)*100:.0f}cm)"
        obj_descriptions.append(f"{o.color} {label}{offset_desc} ({pos})")

    stack_desc = ", ".join(obj_descriptions)
    question = f"Objects are stacked on a table: {stack_desc}. Will this stack remain stable or will it topple?"

    return StackingScenario(
        name=f"scenario_{idx}",
        objects=objects,
        question=question,
        difficulty=random.choice(["easy", "medium", "hard"]),
    )


# ─── Azure OpenAI question enhancer ─────────────────────────────────

QUESTION_GEN_PROMPT = """You are generating natural-language spatial reasoning questions for a robotics RLHF dataset.

Given a description of objects stacked on a table, generate a clear, specific question asking whether the stack will remain stable or topple. 

Make the question:
- Natural and varied (don't always use the same phrasing)
- Mention the specific objects and their arrangement
- Sometimes ask about specific failure modes (e.g. "will the top object slide off?")

Object stack (bottom to top): {stack_description}

Respond with ONLY the question, nothing else."""


async def _generate_azure_question(
    stack_desc: str, endpoint: str, api_key: str
) -> str | None:
    """Use Azure OpenAI to generate a richer question for the scenario."""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": QUESTION_GEN_PROMPT.format(stack_description=stack_desc)},
            ],
            max_tokens=150,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Azure question gen failed: {e}")
        return None


async def _azure_vision_infer(
    endpoint: str, api_key: str, system_prompt: str, user_prompt: str,
    images: list[str], max_tokens: int = 300, temperature: float = 0.3,
) -> str:
    """Call Azure OpenAI GPT-4o with vision via openai library."""
    import base64, os
    from openai import OpenAI

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )

    # Build image content parts
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


# ─── VLM Prompts ────────────────────────────────────────────────────

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


def _parse_vlm_prediction(response: str) -> tuple[str, str]:
    """Extract prediction and reasoning from VLM response."""
    response_upper = response.upper()
    
    prediction = "unknown"
    if "PREDICTION: STABLE" in response_upper or "PREDICTION:STABLE" in response_upper:
        # Check it's not "UNSTABLE"
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

    # Extract reasoning
    reasoning = response
    if "REASONING:" in response.upper():
        idx = response.upper().find("REASONING:")
        reasoning = response[idx + 10:].strip()

    return prediction, reasoning


# ─── Main Job Processor ─────────────────────────────────────────────

async def process_stacking_job(job_id: str, config: dict):
    """
    Full stacking stability pipeline:
    MuJoCo scenarios → simulate → render → VLM predicts → compare to ground truth
    """
    try:
        num_scenarios = config.get("num_scenes", 10)
        use_curated = config.get("use_curated", False)
        target_stable = config.get("num_stable", None)     # e.g. 5
        target_unstable = config.get("num_unstable", None)  # e.g. 5

        db.update_job(job_id, {
            "status": "generating",
            "started_at": datetime.now().isoformat(),
        })

        # Step 1: Generate scenarios
        if use_curated:
            scenarios = generate_10_scenarios()[:num_scenarios]
        elif target_stable is not None and target_unstable is not None:
            # Generate-and-filter: simulate until we hit desired counts
            scenarios = []
            stable_count = 0
            unstable_count = 0
            attempts = 0
            max_attempts = num_scenarios * 5  # safety limit
            print(f"  Targeting {target_stable} stable + {target_unstable} unstable")
            while (stable_count < target_stable or unstable_count < target_unstable) and attempts < max_attempts:
                candidate = _random_scenario(attempts)
                result_preview = simulate_stacking(candidate, settle_seconds=3.0)
                if result_preview.stable and stable_count < target_stable:
                    scenarios.append(candidate)
                    stable_count += 1
                    print(f"  Found stable #{stable_count} (attempt {attempts+1})")
                elif not result_preview.stable and unstable_count < target_unstable:
                    scenarios.append(candidate)
                    unstable_count += 1
                    print(f"  Found unstable #{unstable_count} (attempt {attempts+1})")
                attempts += 1
            num_scenarios = len(scenarios)
            print(f"  Generated {num_scenarios} scenarios ({stable_count} stable, {unstable_count} unstable) in {attempts} attempts")
        else:
            scenarios = [_random_scenario(i) for i in range(num_scenarios)]

        # Check if Azure OpenAI question generation is configured
        q_model = config.get("question_model")
        if q_model:
            # Normalize camelCase keys from frontend
            q_model["api_key"] = q_model.get("api_key") or q_model.get("apiKey", "")
        use_azure_questions = (
            q_model
            and q_model.get("provider") == "azure-openai"
            and q_model.get("api_key")
        )
        if use_azure_questions:
            print(f"  Using Azure OpenAI for question generation")

        # Determine answer model
        answer_model = config.get("model", "qwen2.5-vl-3b")
        azure_cfg = config.get("azure_config")
        if azure_cfg:
            # Normalize camelCase keys from frontend
            azure_cfg["api_key"] = azure_cfg.get("api_key") or azure_cfg.get("apiKey", "")
        use_gpt_answer = answer_model == "gpt-4o" and azure_cfg and azure_cfg.get("api_key")

        if use_gpt_answer:
            print(f"  Answer model: GPT-4o via Azure OpenAI")
        else:
            print(f"  Answer model: {answer_model} via GPU server")

        project_id = str(uuid4())
        all_pairs = []

        for idx, scenario in enumerate(scenarios):
            print(f"\n[{idx+1}/{num_scenarios}] {scenario.name}")
            obj_names = [f"{STACKABLE_OBJECTS[o.obj_type]['label']}({o.color})" for o in scenario.objects]
            print(f"  Stack: {' → '.join(obj_names)}")

            # Step 2: MuJoCo simulation → GROUND TRUTH
            print(f"  Simulating physics (3s)...")
            result = simulate_stacking(scenario, settle_seconds=3.0)
            gt = "stable" if result.stable else "unstable"
            print(f"  MuJoCo ground truth: {gt.upper()}")

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "scenarios_total": num_scenarios,
            })

            # Step 3: Build question — optionally use Azure OpenAI
            stack_desc = ", ".join(
                f"{o.color} {STACKABLE_OBJECTS[o.obj_type]['label']}"
                + (f" (offset {abs(o.offset_x)*100:.0f}cm)" if o.offset_x != 0 else "")
                for o in scenario.objects
            )

            question = scenario.question  # default template question
            if use_azure_questions:
                print(f"  Generating question via Azure OpenAI...")
                azure_q = await _generate_azure_question(
                    stack_desc, q_model["endpoint"], q_model["api_key"]
                )
                if azure_q:
                    question = azure_q
                    print(f"  Question: {azure_q[:80]}...")

            # Step 4: Send "before" images to VLM
            vlm_prompt = STABILITY_USER.format(description=f"Stack (bottom to top): {stack_desc}\n\n{question}")
            if use_gpt_answer:
                print(f"  Querying GPT-4o...")
                vlm_response = await _azure_vision_infer(
                    endpoint=azure_cfg["endpoint"],
                    api_key=azure_cfg["api_key"],
                    system_prompt=STABILITY_SYSTEM,
                    user_prompt=vlm_prompt,
                    images=result.before_images,
                    max_tokens=300,
                    temperature=0.3,
                )
            else:
                print(f"  Querying Qwen VLM...")
                vlm_response = await infer(
                    system_prompt=STABILITY_SYSTEM,
                    user_prompt=vlm_prompt,
                    images=result.before_images,
                    max_tokens=300,
                    temperature=0.3,
                )

            prediction, reasoning = _parse_vlm_prediction(vlm_response)
            correct = prediction == gt
            status_icon = "✅" if correct else "❌"
            print(f"  VLM prediction: {prediction.upper()} {status_icon}")
            print(f"  Reasoning: {reasoning[:100]}...")

            # Step 5: Build pair
            pair = {
                "id": str(uuid4()),
                "project_id": project_id,
                "prompt": question,
                "chosen": vlm_response,  # VLM's full response
                "rejected": "",  # Not applicable for eval mode
                "scene_id": scenario.name,
                "category": "stacking_stability",
                "difficulty": scenario.difficulty,
                "status": "pending",
                "source": {
                    "dataset": "mujoco:stacking",
                    "scene_id": scenario.name,
                    "images": result.before_images,
                },
                "generation": {
                    "model": "qwen2.5-vl-3b",
                    "task_type": "stacking_stability",
                    "generated_at": datetime.now().isoformat(),
                },
                "verification": {
                    "mujoco_ground_truth": gt,
                    "vlm_prediction": prediction,
                    "vlm_reasoning": reasoning,
                    "correct": correct,
                    "max_displacement_cm": round(result.max_displacement * 100, 1),
                    "fell_objects": [n.split("_", 2)[-1] for n in result.fell_objects],
                    "initial_positions": result.initial_positions,
                    "final_positions": result.final_positions,
                },
                "ground_truth": {
                    "stable": result.stable,
                    "objects": [
                        {
                            "type": o.obj_type,
                            "color": o.color,
                            "label": STACKABLE_OBJECTS[o.obj_type]["label"],
                            "offset_x": o.offset_x,
                            "offset_y": o.offset_y,
                        }
                        for o in scenario.objects
                    ],
                    "before_images": result.before_images,
                    "after_images": result.after_images,
                },
            }
            all_pairs.append(pair)

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "pairs_generated": len(all_pairs),
                "correct": sum(1 for p in all_pairs if p["verification"]["correct"]),
                "total": len(all_pairs),
            })

        # Step 5: Save project
        correct_count = sum(1 for p in all_pairs if p["verification"]["correct"])
        project_name = config.get("name") or f"Stacking Stability ({num_scenarios} scenarios)"
        db.create_project({
            "id": project_id,
            "name": f"{project_name} — {correct_count}/{num_scenarios} correct",
            "created_at": datetime.now().isoformat(),
            "job_id": job_id,
            "task_type": "stacking_stability",
        })
        db.add_pairs(all_pairs)

        db.update_job(job_id, {
            "status": "completed",
            "project_id": project_id,
            "completed_at": datetime.now().isoformat(),
        })

        print(f"\n{'='*55}")
        print(f"✅ Job {job_id[:8]} completed")
        print(f"   {num_scenarios} scenarios, VLM accuracy: {correct_count}/{num_scenarios}")
        print(f"   Project: {project_id}")

    except Exception as e:
        db.update_job(job_id, {"status": "failed", "error": str(e)})
        print(f"❌ Job {job_id[:8]} failed: {e}")
        raise
