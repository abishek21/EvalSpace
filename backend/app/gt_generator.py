"""
Stage 1: Ground Truth Generator

Generates scenarios for ANY environment, simulates in MuJoCo, renders images,
and saves as a reusable dataset. No VLM evaluation here.

Supported environments:
  - stacking_stability: Will a stack of objects remain stable?
  - collision_prediction: If object A is pushed, will it hit object B?
"""
import math
import random
from datetime import datetime
from uuid import uuid4

from app import db
from app.dise_stacking import (
    StackingScenario,
    StackObject,
    simulate_stacking,
    generate_10_scenarios as stacking_curated_10,
    STACKABLE_OBJECTS,
    COLORS,
)
from app.dise_collision import (
    CollisionScenario,
    PlacedObject,
    PushConfig,
    simulate_collision,
    generate_10_scenarios as collision_curated_10,
    COLLISION_OBJECTS,
    COLORS as COLLISION_COLORS,
    _random_scenario as _random_collision_scenario,
)
from app.dise_fitting import (
    FittingScenario,
    render_scenario as render_fitting_scenario,
    generate_10_scenarios as fitting_curated_10,
    _can_fit,
    FITTING_OBJECTS,
    GAP_TYPES,
    COLORS as FITTING_COLORS,
)


# ─── Random scenario generator ──────────────────────────────────────

def _random_scenario(idx: int) -> StackingScenario:
    """Generate a random stacking scenario with 2-4 objects."""
    n_objects = random.choice([2, 2, 2, 3, 3, 4])
    obj_types = list(STACKABLE_OBJECTS.keys())
    color_names = list(COLORS.keys())

    objects = []
    for i in range(n_objects):
        obj_type = random.choice(obj_types)
        color = random.choice(color_names)
        offset_x = 0.0
        offset_y = 0.0
        if i > 0 and random.random() < 0.4:
            offset_x = random.uniform(-0.06, 0.06)
        if i > 0 and random.random() < 0.2:
            offset_y = random.uniform(-0.04, 0.04)
        objects.append(StackObject(obj_type, color, offset_x, offset_y))

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

COLLISION_QUESTION_GEN_PROMPT = """You are generating natural-language spatial reasoning questions for a robotics RLHF dataset.

Given a description of a collision prediction scenario on a table, generate a clear, natural question asking whether the pushed object will hit the target.

Key rules:
- NEVER mention exact force values, angles in degrees, or N·s units
- Instead use qualitative descriptions: "gently nudged", "firmly pushed", "shoved hard", "given a light tap"
- Describe direction naturally: "pushed toward", "pushed roughly in the direction of", "pushed to the right"
- Mention obstacles if present: "there's a large box between them"
- Make it sound like something a person would ask looking at the scene
- Vary your phrasing — don't always use the same structure

Scenario details:
- Pushed object: {pushed_obj}
- Target object: {target_obj}
- Push strength: {force_desc}
- Obstacles: {obstacles_desc}
- Distance: {distance_desc}

Respond with ONLY the question, nothing else."""


def _force_to_qualitative(force: float, mass: float) -> str:
    """Convert force magnitude to a human-readable description."""
    # velocity = force / mass
    v = force / max(mass, 0.1)
    if v < 0.5:
        return "very gently tapped"
    elif v < 1.0:
        return "lightly nudged"
    elif v < 1.5:
        return "pushed"
    elif v < 2.5:
        return "firmly pushed"
    else:
        return "shoved hard"


def _distance_to_qualitative(dist: float) -> str:
    if dist < 0.15:
        return "very close together"
    elif dist < 0.25:
        return "a short distance apart"
    elif dist < 0.40:
        return "a moderate distance apart"
    else:
        return "far apart, across the table"


async def _generate_azure_question(stack_desc: str, endpoint: str, api_key: str) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=endpoint, api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": QUESTION_GEN_PROMPT.format(stack_description=stack_desc)}],
            max_tokens=150,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Azure question gen failed: {e}")
        return None


async def _generate_collision_question(prompt: str, endpoint: str, api_key: str) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=endpoint, api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Azure collision question gen failed: {e}")
        return None


# ─── Main GT Generation Job ─────────────────────────────────────────

async def process_gt_job(job_id: str, config: dict):
    """
    Stage 1: Generate ground truth dataset.
    Routes to the appropriate environment based on config.
    """
    environment = config.get("environment", "stacking_stability")

    if environment == "collision_prediction":
        await _process_collision_gt(job_id, config)
    elif environment == "spatial_fitting":
        await _process_fitting_gt(job_id, config)
    else:
        await _process_stacking_gt(job_id, config)


# ─── Spatial Fitting GT ──────────────────────────────────────────────

def _random_fitting_scenario(idx: int) -> FittingScenario:
    """Generate a random spatial fitting scenario."""
    obj_keys = list(FITTING_OBJECTS.keys())
    gap_keys = list(GAP_TYPES.keys())
    color_names = list(FITTING_COLORS.keys())

    obj_key = random.choice(obj_keys)
    gap_key = random.choice(gap_keys)
    color = random.choice(color_names)
    fits, reasoning = _can_fit(obj_key, gap_key)
    obj = FITTING_OBJECTS[obj_key]
    gap = GAP_TYPES[gap_key]

    question = f"Can the {color} {obj['vlm_shape']} fit through the {gap['vlm_label']} in the wall?"

    return FittingScenario(
        name=f"scenario_{idx}",
        object_type=obj_key,
        object_color=color,
        gap_type=gap_key,
        fits=fits,
        best_orientation=reasoning,
        question=question,
        difficulty="medium",
        reasoning=reasoning,
    )


FITTING_QUESTION_GEN_PROMPT = """You are generating spatial reasoning questions for a VLM evaluation dataset.

An object is placed next to a wall with an opening. The question is whether the object can fit through.

CRITICAL RULES:
- NEVER mention sizes, dimensions, measurements, or numbers
- NEVER use words like "large", "small", "tiny", "wide", "narrow", "big"
- ONLY refer to the object by its COLOR and SHAPE (e.g. "the red ball", "the blue cube")
- ONLY refer to the opening by its SHAPE (e.g. "the circular opening", "the rectangular slot")
- The VLM must judge sizes purely from the image

Object: {color} {shape}
Opening: {gap_shape}

Generate a clear, natural question asking whether the object can pass through the opening.
Vary your phrasing. Respond with ONLY the question."""


async def _generate_fitting_question(color: str, shape: str, gap_shape: str,
                                      endpoint: str, api_key: str) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=endpoint, api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": FITTING_QUESTION_GEN_PROMPT.format(
                color=color, shape=shape, gap_shape=gap_shape
            )}],
            max_tokens=100,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Azure fitting question gen failed: {e}")
        return None


async def _process_fitting_gt(job_id: str, config: dict):
    """Generate spatial fitting ground truth dataset."""
    try:
        num_scenarios = config.get("num_scenes", 10)
        use_curated = config.get("use_curated", False)
        target_fits = config.get("num_stable", None)      # reuse: stable=fits
        target_nofits = config.get("num_unstable", None)   # unstable=doesn't fit

        db.update_job(job_id, {
            "status": "generating",
            "started_at": datetime.now().isoformat(),
        })

        # Generate scenarios
        if use_curated:
            scenarios = fitting_curated_10()[:num_scenarios]
        elif target_fits is not None and target_nofits is not None:
            scenarios = []
            fits_count = 0
            nofits_count = 0
            attempts = 0
            max_attempts = num_scenarios * 8
            print(f"  Targeting {target_fits} fits + {target_nofits} doesn't fit")
            while (fits_count < target_fits or nofits_count < target_nofits) and attempts < max_attempts:
                candidate = _random_fitting_scenario(attempts)
                if candidate.fits and fits_count < target_fits:
                    scenarios.append(candidate)
                    fits_count += 1
                    print(f"  Found FITS #{fits_count} (attempt {attempts+1})")
                elif not candidate.fits and nofits_count < target_nofits:
                    scenarios.append(candidate)
                    nofits_count += 1
                    print(f"  Found NO FIT #{nofits_count} (attempt {attempts+1})")
                attempts += 1
            num_scenarios = len(scenarios)
            print(f"  Generated {num_scenarios} scenarios in {attempts} attempts")
        else:
            scenarios = [_random_fitting_scenario(i) for i in range(num_scenarios)]

        # Azure question generation
        q_model = config.get("question_model")
        if q_model:
            q_model["api_key"] = q_model.get("api_key") or q_model.get("apiKey", "")
        use_azure_questions = q_model and q_model.get("provider") == "azure-openai" and q_model.get("api_key")

        dataset_id = str(uuid4())
        all_scenarios = []
        fits_total = 0
        nofits_total = 0

        for idx, scenario in enumerate(scenarios):
            obj = FITTING_OBJECTS[scenario.object_type]
            gap = GAP_TYPES[scenario.gap_type]
            print(f"\n[{idx+1}/{num_scenarios}] {scenario.name}")
            print(f"  {scenario.object_color} {obj['label']} → {gap['label']}")

            # Render the scene
            print(f"  Rendering...")
            result = render_fitting_scenario(scenario)
            gt = "fits" if scenario.fits else "no_fit"
            if scenario.fits:
                fits_total += 1
            else:
                nofits_total += 1
            print(f"  Ground truth: {gt.upper()} — {scenario.reasoning}")

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "scenarios_total": num_scenarios,
            })

            question = scenario.question
            if use_azure_questions:
                print(f"  Generating question via Azure OpenAI...")
                azure_q = await _generate_fitting_question(
                    scenario.object_color, obj['vlm_shape'], gap['vlm_label'],
                    q_model["endpoint"], q_model["api_key"]
                )
                if azure_q:
                    question = azure_q
                    print(f"  Question: {azure_q[:80]}...")

            scenario_record = {
                "id": str(uuid4()),
                "dataset_id": dataset_id,
                "pair_type": "ground_truth",
                "scene_id": scenario.name,
                "prompt": question,
                "category": "spatial_fitting",
                "difficulty": scenario.difficulty,
                "ground_truth": {
                    "answer": gt,
                    "fits": scenario.fits,
                    "reasoning": scenario.reasoning,
                    "object": {
                        "type": scenario.object_type,
                        "color": scenario.object_color,
                        "label": obj["label"],
                        "vlm_shape": obj["vlm_shape"],
                    },
                    "gap": {
                        "type": scenario.gap_type,
                        "label": gap["label"],
                        "vlm_label": gap["vlm_label"],
                    },
                    "before_images": result.images,
                },
                "source": {
                    "dataset": "mujoco:fitting",
                    "scene_id": scenario.name,
                    "images": result.images,
                },
                "status": "ready",
            }
            all_scenarios.append(scenario_record)

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "pairs_generated": len(all_scenarios),
            })

        dataset_name = config.get("name") or f"Fitting {num_scenarios} scenarios"
        db.create_dataset({
            "id": dataset_id,
            "name": dataset_name,
            "task_type": "spatial_fitting",
            "scenario_count": num_scenarios,
            "stable_count": fits_total,       # reuse: fits
            "unstable_count": nofits_total,   # doesn't fit
            "created_at": datetime.now().isoformat(),
            "job_id": job_id,
            "config": {
                "environment": "spatial_fitting",
                "mode": "curated" if use_curated else "random",
                "question_source": "azure-openai" if use_azure_questions else "template",
            },
        })
        db.add_scenarios(all_scenarios)

        db.update_job(job_id, {
            "status": "completed",
            "dataset_id": dataset_id,
            "completed_at": datetime.now().isoformat(),
        })

        print(f"\n{'='*55}")
        print(f"✅ Fitting dataset {dataset_id[:8]} created")
        print(f"   {num_scenarios} scenarios ({fits_total} fits, {nofits_total} no-fit)")

    except Exception as e:
        db.update_job(job_id, {"status": "failed", "error": str(e)})
        print(f"❌ Fitting GT job {job_id[:8]} failed: {e}")
        raise


# ─── Collision Prediction GT ────────────────────────────────────────

async def _process_collision_gt(job_id: str, config: dict):
    """Generate collision prediction ground truth dataset."""
    try:
        num_scenarios = config.get("num_scenes", 10)
        use_curated = config.get("use_curated", False)
        target_hit = config.get("num_stable", None)      # reuse fields: stable=hit
        target_miss = config.get("num_unstable", None)    # unstable=miss

        db.update_job(job_id, {
            "status": "generating",
            "started_at": datetime.now().isoformat(),
        })

        # Generate scenarios
        if use_curated:
            scenarios = collision_curated_10()[:num_scenarios]
        elif target_hit is not None and target_miss is not None:
            scenarios = []
            hit_count = 0
            miss_count = 0
            attempts = 0
            max_attempts = num_scenarios * 8
            print(f"  Targeting {target_hit} hit + {target_miss} miss")
            while (hit_count < target_hit or miss_count < target_miss) and attempts < max_attempts:
                candidate = _random_collision_scenario(attempts)
                result_preview = simulate_collision(candidate, sim_seconds=3.0)
                if result_preview.hit_target and hit_count < target_hit:
                    scenarios.append(candidate)
                    hit_count += 1
                    print(f"  Found HIT #{hit_count} (attempt {attempts+1})")
                elif not result_preview.hit_target and miss_count < target_miss:
                    scenarios.append(candidate)
                    miss_count += 1
                    print(f"  Found MISS #{miss_count} (attempt {attempts+1})")
                attempts += 1
            num_scenarios = len(scenarios)
            print(f"  Generated {num_scenarios} scenarios in {attempts} attempts")
        else:
            scenarios = [_random_collision_scenario(i) for i in range(num_scenarios)]

        # Azure question generation
        q_model = config.get("question_model")
        if q_model:
            q_model["api_key"] = q_model.get("api_key") or q_model.get("apiKey", "")
        use_azure_questions = q_model and q_model.get("provider") == "azure-openai" and q_model.get("api_key")

        dataset_id = str(uuid4())
        all_scenarios = []
        hit_total = 0
        miss_total = 0

        for idx, scenario in enumerate(scenarios):
            print(f"\n[{idx+1}/{num_scenarios}] {scenario.name}")
            pushed = next(o for o in scenario.objects if o.role == "pushed")
            target = next(o for o in scenario.objects if o.role == "target")
            obstacles = [o for o in scenario.objects if o.role == "obstacle"]
            print(f"  Push: {pushed.color} {COLLISION_OBJECTS[pushed.obj_type]['label']} → "
                  f"{target.color} {COLLISION_OBJECTS[target.obj_type]['label']}"
                  f" ({len(obstacles)} obstacles)")

            print(f"  Simulating physics (3s)...")
            result = simulate_collision(scenario, sim_seconds=3.0)
            gt = "hit" if result.hit_target else "miss"
            if result.hit_target:
                hit_total += 1
            else:
                miss_total += 1
            print(f"  Ground truth: {gt.upper()}"
                  f" ({len(result.collision_events)} collision events, "
                  f"{len(result.frames)} frames)")

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "scenarios_total": num_scenarios,
            })

            question = scenario.question
            if use_azure_questions:
                pushed_label = f"{pushed.color} {COLLISION_OBJECTS[pushed.obj_type]['label']}"
                target_label = f"{target.color} {COLLISION_OBJECTS[target.obj_type]['label']}"
                pushed_mass = COLLISION_OBJECTS[pushed.obj_type]["mass"]
                force_desc = _force_to_qualitative(scenario.push.force, pushed_mass)
                dist = math.sqrt((pushed.pos_x - target.pos_x)**2 + (pushed.pos_y - target.pos_y)**2)
                distance_desc = _distance_to_qualitative(dist)
                obs_labels = [f"{o.color} {COLLISION_OBJECTS[o.obj_type]['label']}" for o in obstacles]
                obstacles_desc = ", ".join(obs_labels) if obs_labels else "none"

                collision_prompt = COLLISION_QUESTION_GEN_PROMPT.format(
                    pushed_obj=pushed_label,
                    target_obj=target_label,
                    force_desc=force_desc,
                    obstacles_desc=obstacles_desc,
                    distance_desc=distance_desc,
                )
                azure_q = await _generate_collision_question(collision_prompt, q_model["endpoint"], q_model["api_key"])
                if azure_q:
                    question = azure_q
                    print(f"  Question: {azure_q[:80]}...")

            scenario_record = {
                "id": str(uuid4()),
                "dataset_id": dataset_id,
                "pair_type": "ground_truth",
                "scene_id": scenario.name,
                "prompt": question,
                "category": "collision_prediction",
                "difficulty": scenario.difficulty,
                "ground_truth": {
                    "answer": gt,
                    "hit": result.hit_target,
                    "objects": [
                        {
                            "type": o.obj_type,
                            "color": o.color,
                            "label": COLLISION_OBJECTS[o.obj_type]["label"],
                            "role": o.role,
                            "pos_x": o.pos_x,
                            "pos_y": o.pos_y,
                        }
                        for o in scenario.objects
                    ],
                    "push": {
                        "direction_deg": scenario.push.direction_deg,
                        "force": scenario.push.force,
                    },
                    "collision_events": result.collision_events,
                    "target_moved": result.target_moved,
                    "pushed_trajectory": result.pushed_trajectory,
                    "before_images": result.before_images,
                    "after_images": result.after_images,
                    "frames": result.frames,
                },
                "source": {
                    "dataset": "mujoco:collision",
                    "scene_id": scenario.name,
                    "images": result.before_images,
                },
                "status": "ready",
            }
            all_scenarios.append(scenario_record)

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "pairs_generated": len(all_scenarios),
            })

        dataset_name = config.get("name") or f"Collision {num_scenarios} scenarios"
        db.create_dataset({
            "id": dataset_id,
            "name": dataset_name,
            "task_type": "collision_prediction",
            "scenario_count": num_scenarios,
            "stable_count": hit_total,       # reuse field: hit
            "unstable_count": miss_total,     # reuse field: miss
            "created_at": datetime.now().isoformat(),
            "job_id": job_id,
            "config": {
                "environment": "collision_prediction",
                "mode": "curated" if config.get("use_curated") else "random",
                "question_source": "azure-openai" if use_azure_questions else "template",
            },
        })
        db.add_scenarios(all_scenarios)

        db.update_job(job_id, {
            "status": "completed",
            "dataset_id": dataset_id,
            "completed_at": datetime.now().isoformat(),
        })

        print(f"\n{'='*55}")
        print(f"✅ Collision dataset {dataset_id[:8]} created")
        print(f"   {num_scenarios} scenarios ({hit_total} hit, {miss_total} miss)")

    except Exception as e:
        db.update_job(job_id, {"status": "failed", "error": str(e)})
        print(f"❌ Collision GT job {job_id[:8]} failed: {e}")
        raise


# ─── Stacking Stability GT ──────────────────────────────────────────

async def _process_stacking_gt(job_id: str, config: dict):
    try:
        num_scenarios = config.get("num_scenes", 10)
        use_curated = config.get("use_curated", False)
        target_stable = config.get("num_stable", None)
        target_unstable = config.get("num_unstable", None)

        db.update_job(job_id, {
            "status": "generating",
            "started_at": datetime.now().isoformat(),
        })

        # Step 1: Generate scenarios
        if use_curated:
            scenarios = stacking_curated_10()[:num_scenarios]
        elif target_stable is not None and target_unstable is not None:
            scenarios = []
            stable_count = 0
            unstable_count = 0
            attempts = 0
            max_attempts = num_scenarios * 5
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
            print(f"  Generated {num_scenarios} scenarios in {attempts} attempts")
        else:
            scenarios = [_random_scenario(i) for i in range(num_scenarios)]

        # Check for Azure question generation
        q_model = config.get("question_model")
        if q_model:
            q_model["api_key"] = q_model.get("api_key") or q_model.get("apiKey", "")
        use_azure_questions = q_model and q_model.get("provider") == "azure-openai" and q_model.get("api_key")

        dataset_id = str(uuid4())
        all_scenarios = []
        stable_total = 0
        unstable_total = 0

        for idx, scenario in enumerate(scenarios):
            print(f"\n[{idx+1}/{num_scenarios}] {scenario.name}")
            obj_names = [f"{STACKABLE_OBJECTS[o.obj_type]['label']}({o.color})" for o in scenario.objects]
            print(f"  Stack: {' → '.join(obj_names)}")

            # Step 2: MuJoCo simulation
            print(f"  Simulating physics (3s)...")
            result = simulate_stacking(scenario, settle_seconds=3.0)
            gt = "stable" if result.stable else "unstable"
            if result.stable:
                stable_total += 1
            else:
                unstable_total += 1
            print(f"  Ground truth: {gt.upper()}")

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "scenarios_total": num_scenarios,
            })

            # Step 3: Build question
            stack_desc = ", ".join(
                f"{o.color} {STACKABLE_OBJECTS[o.obj_type]['label']}"
                + (f" (offset {abs(o.offset_x)*100:.0f}cm)" if o.offset_x != 0 else "")
                for o in scenario.objects
            )

            question = scenario.question
            if use_azure_questions:
                print(f"  Generating question via Azure OpenAI...")
                azure_q = await _generate_azure_question(stack_desc, q_model["endpoint"], q_model["api_key"])
                if azure_q:
                    question = azure_q
                    print(f"  Question: {azure_q[:80]}...")

            # Step 4: Save scenario as ground truth record
            scenario_record = {
                "id": str(uuid4()),
                "dataset_id": dataset_id,
                "pair_type": "ground_truth",
                "scene_id": scenario.name,
                "prompt": question,
                "category": "stacking_stability",
                "difficulty": scenario.difficulty,
                "ground_truth": {
                    "answer": gt,
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
                    "max_displacement_cm": round(result.max_displacement * 100, 1),
                    "fell_objects": [n.split("_", 2)[-1] for n in result.fell_objects],
                    "before_images": result.before_images,
                    "after_images": result.after_images,
                },
                "source": {
                    "dataset": "mujoco:stacking",
                    "scene_id": scenario.name,
                    "images": result.before_images,
                },
                "status": "ready",
            }
            all_scenarios.append(scenario_record)

            db.update_job_progress(job_id, {
                "scenes_processed": idx + 1,
                "pairs_generated": len(all_scenarios),
            })

        # Step 5: Save dataset
        dataset_name = config.get("name") or f"Stacking {num_scenarios} scenarios"
        db.create_dataset({
            "id": dataset_id,
            "name": dataset_name,
            "task_type": "stacking_stability",
            "scenario_count": num_scenarios,
            "stable_count": stable_total,
            "unstable_count": unstable_total,
            "created_at": datetime.now().isoformat(),
            "job_id": job_id,
            "config": {
                "mode": "curated" if use_curated else "random",
                "question_source": "azure-openai" if use_azure_questions else "template",
            },
        })
        db.add_scenarios(all_scenarios)

        db.update_job(job_id, {
            "status": "completed",
            "dataset_id": dataset_id,
            "completed_at": datetime.now().isoformat(),
        })

        print(f"\n{'='*55}")
        print(f"✅ Dataset {dataset_id[:8]} created")
        print(f"   {num_scenarios} scenarios ({stable_total} stable, {unstable_total} unstable)")

    except Exception as e:
        db.update_job(job_id, {"status": "failed", "error": str(e)})
        print(f"❌ GT job {job_id[:8]} failed: {e}")
        raise
