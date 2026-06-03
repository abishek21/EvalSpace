"""
RLHF Annotation Tool — Python Backend

Handles:
- Job orchestration (create, run, monitor)
- HuggingFace dataset download + scene preparation
- GPU worker communication (Colab/Modal inference)
- Serves prepared data to Next.js frontend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend root
load_dotenv(Path(__file__).parent.parent / ".env")

# Dirs
DATA_DIR = Path(__file__).parent.parent / "data"  # backend/data (db.json etc)
DATA_DIR.mkdir(exist_ok=True)

# Image dumps go outside the repo to avoid bloating git
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", str(Path.home() / ".rlhf" / "images")))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

GPU_WORKER_URL = os.getenv("GPU_WORKER_URL", "")
