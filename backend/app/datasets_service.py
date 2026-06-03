"""
Dataset service — downloads and prepares scenes from HuggingFace.

Supports arbitrary datasets via column mapping:
  - image_column: which column has the PIL image
  - text_column: which column has the text description
  - scene_id_column: optional grouping column (e.g., scene name)

Images are saved to disk as JPEG files instead of base64 in JSON.
"""
import base64
import os
from io import BytesIO
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from collections import defaultdict

from app import IMAGES_DIR


def images_to_base64(images: list[Image.Image], max_size: int = 480) -> list[str]:
    """Convert PIL images to base64 data URIs."""
    encoded = []
    for img in images:
        thumb = img.copy()
        thumb.thumbnail((max_size, max_size))
        buf = BytesIO()
        thumb.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        encoded.append(f"data:image/jpeg;base64,{b64}")
    return encoded


def save_images_to_disk(
    images: list[Image.Image],
    scene_id: str,
    job_id: str,
    max_size: int = 480,
) -> list[str]:
    """Save PIL images to disk, return relative paths."""
    scene_dir = IMAGES_DIR / job_id / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, img in enumerate(images):
        thumb = img.copy()
        thumb.thumbnail((max_size, max_size))
        filename = f"view_{i:02d}.jpg"
        filepath = scene_dir / filename
        thumb.save(filepath, format="JPEG", quality=75)
        rel_path = f"/images/{job_id}/{scene_id}/{filename}"
        paths.append(rel_path)

    return paths


def detect_columns(sample: dict) -> dict:
    """Auto-detect image and text columns from a dataset sample."""
    image_col = None
    text_col = None
    scene_col = None

    for key, value in sample.items():
        if isinstance(value, Image.Image) and image_col is None:
            image_col = key
        elif isinstance(value, str):
            lower = key.lower()
            if any(kw in lower for kw in ["scene", "scan", "room", "group"]):
                scene_col = key
            elif text_col is None and any(kw in lower for kw in ["text", "desc", "caption", "label", "prompt"]):
                text_col = key

    # Fallback: first string column as text
    if text_col is None:
        for key, value in sample.items():
            if isinstance(value, str) and key != scene_col:
                text_col = key
                break

    return {
        "image_column": image_col,
        "text_column": text_col,
        "scene_id_column": scene_col,
    }


def load_scenes(
    dataset_name: str,
    split: str,
    num_scenes: int,
    max_views: int,
    image_resolution: int = 480,
    job_id: str = "default",
    column_map: dict | None = None,
) -> list[dict]:
    """
    Download dataset from HuggingFace and group into scenes.

    Args:
        column_map: {"image_column": "image", "text_column": "text", "scene_id_column": None}
                    If None, auto-detects from the first sample.

    Returns list of scene dicts with both base64 (for GPU) and disk paths (for frontend).
    """
    print(f"📥 Loading {dataset_name} (split={split}, streaming)...")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    rows_needed = num_scenes * max_views * 2
    raw_samples = []
    for i, sample in enumerate(ds):
        raw_samples.append(sample)
        if i >= rows_needed - 1:
            break

    print(f"  Downloaded {len(raw_samples)} rows")

    # Detect or use provided column mapping
    if column_map is None or column_map.get("image_column") is None:
        detected = detect_columns(raw_samples[0])
        print(f"  🔍 Auto-detected columns: {detected}")
        col_map = {**(column_map or {})}
        for k, v in detected.items():
            if col_map.get(k) is None and v is not None:
                col_map[k] = v
    else:
        col_map = column_map

    img_col = col_map.get("image_column", "image")
    text_col = col_map.get("text_column", "text")
    scene_col = col_map.get("scene_id_column")

    print(f"  📋 Using columns: image='{img_col}', text='{text_col}', scene='{scene_col or 'auto-batch'}'")

    # Group by scene ID if available, otherwise batch sequentially
    if scene_col:
        groups = defaultdict(lambda: {"images": [], "descriptions": [], "row_indices": []})
        for i, sample in enumerate(raw_samples):
            scene_key = str(sample.get(scene_col, f"scene_{i}"))
            img = sample.get(img_col)
            desc = sample.get(text_col, "")
            if img is not None and isinstance(img, Image.Image):
                groups[scene_key]["images"].append(img)
                groups[scene_key]["descriptions"].append(desc or "")
                groups[scene_key]["row_indices"].append(i)

        scenes = []
        for scene_key, data in groups.items():
            if len(scenes) >= num_scenes:
                break
            if data["images"]:
                imgs = data["images"][:max_views]
                sid = scene_key[:60]
                scenes.append({
                    "id": sid,
                    "images_b64": images_to_base64(imgs, max_size=image_resolution),
                    "image_paths": save_images_to_disk(imgs, sid, job_id, max_size=image_resolution),
                    "descriptions": data["descriptions"][:max_views],
                    "row_indices": data["row_indices"][:max_views],
                })
    else:
        scenes = []
        for i in range(0, len(raw_samples), max_views):
            if len(scenes) >= num_scenes:
                break
            batch = raw_samples[i : i + max_views]
            images, descriptions, row_indices = [], [], []
            for j, sample in enumerate(batch):
                img = sample.get(img_col)
                desc = sample.get(text_col, "")
                if img is not None and isinstance(img, Image.Image):
                    images.append(img)
                    row_indices.append(i + j)
                    descriptions.append(desc or "")
            if images:
                scene_id = f"scene_{len(scenes):03d}"
                scenes.append({
                    "id": scene_id,
                    "images_b64": images_to_base64(images, max_size=image_resolution),
                    "image_paths": save_images_to_disk(images, scene_id, job_id, max_size=image_resolution),
                    "descriptions": descriptions,
                    "row_indices": row_indices,
                })

    print(f"  ✅ Prepared {len(scenes)} scenes with ~{max_views} views each")
    print(f"  💾 Images saved to {IMAGES_DIR / job_id}/")
    return scenes
