# EvalSpace

Verifiable visual-spatial reasoning environments for evaluating and training VLMs.

Physics-engine-verified ground truth. No ambiguity. Infinite scale.

## Quick Start

```python
import evalspace as es

# Create environment
env = es.make(task="shelf_fitting", difficulty="medium", seed=42)

# Generate a scene
obs = env.reset()
obs.image          # PIL Image — the rendered scene
obs.question       # "Can the blue bottle fit on the top shelf in any orientation?"

# Check ground truth
env.ground_truth() # {"answer": "fits", "reasoning": "..."}

# Verify a model's answer
env.verify("fits")   # 1.0 (correct)
env.verify("no_fit") # 0.0 (wrong)

# Gym-compatible interface for RL
obs, reward, done, info = env.step("fits")
```

## Generate Datasets

```python
suite = es.generate(task="shelf_fitting", num_scenes=5000, seed=42)
print(suite)
# EvalSuite(task='shelf_fitting', scenes=5000, fits=2500, no_fit=2500)

# Save locally
suite.save("./my_dataset")

# Push to HuggingFace
suite.push_to_hub("your-org/shelf-fitting-5k")
```

## Parameters

```python
env = es.make(
    task="shelf_fitting",       # environment type
    difficulty="hard",          # easy | medium | hard
    seed=42,                    # reproducibility
    physics_engine=True,        # MuJoCo physics verification (default: rule-based)
    multi_view=True,            # multiple camera angles
    max_existing=3,             # max objects already on the shelf
    camera={"azimuth": 270, "elevation": -15, "distance": 2.0},  # custom camera
)
```

## Difficulty Levels

| Difficulty | Margin | Existing Objects | Description |
|------------|--------|-----------------|-------------|
| easy | >30% | 0-1 | Obvious fit or clear miss |
| medium | >15% | 0-2 | Visible difference |
| hard | >5% | 0-4 | Tight but unambiguous |

## Verification Modes

```python
# Fast rule-based (default) — dimension checks, all orientations
env = es.make(task="shelf_fitting")

# Physics engine — MuJoCo simulation, drop test
env = es.make(task="shelf_fitting", physics_engine=True)
```

## Available Environments

| Task | Description | Answer Format |
|------|-------------|---------------|
| `shelf_fitting` | Can the object fit on the shelf? | fits / no_fit |

More environments coming: `perspective_taking`, `allocentric`, `door_passage`, `box_packing`

## Install

```bash
pip install evalspace

# With HuggingFace support
pip install evalspace[hub]

# With model evaluation
pip install evalspace[eval]

# Everything
pip install evalspace[all]
```
