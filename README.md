# 3D Spatial Reasoning RLHF Dataset Generator

Validation pipeline for generating RLHF training data from 3D scenes (ScanNet).

## Pipeline

```
ScanNet Scenes → Question Generation (LLM) → Answer Generation (LLM) → RLHF Pairs → Human Ranking
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python run_pipeline.py --scenes 10 --questions-per-scene 20
```

## Output

RLHF pairs in `output/` as JSON, ready for human review.
