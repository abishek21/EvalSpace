"""EvalSuite — a generated dataset of scenarios."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalSuite:
    """A static dataset of generated scenarios."""
    task: str
    scenarios: list[dict]
    difficulty: str = "mixed"
    seed: int = 42

    def __len__(self):
        return len(self.scenarios)

    def __repr__(self):
        fits = sum(1 for s in self.scenarios if s["ground_truth"]["answer"] == "fits")
        nofits = len(self) - fits
        return (
            f"EvalSuite(task='{self.task}', scenes={len(self)}, "
            f"fits={fits}, no_fit={nofits}, difficulty='{self.difficulty}')"
        )

    def save(self, path: str):
        """Save to local directory."""
        import json, os
        os.makedirs(path, exist_ok=True)

        # Save metadata
        meta = {
            "task": self.task,
            "num_scenes": len(self),
            "difficulty": self.difficulty,
            "seed": self.seed,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Save scenarios (images saved separately)
        for i, sc in enumerate(self.scenarios):
            sc["image"].save(os.path.join(path, f"scene_{i:04d}.jpg"), quality=90)
            record = {k: v for k, v in sc.items() if k != "image"}
            record["image_file"] = f"scene_{i:04d}.jpg"
            with open(os.path.join(path, f"scene_{i:04d}.json"), "w") as f:
                json.dump(record, f, indent=2, default=str)

        print(f"Saved {len(self)} scenarios to {path}/")

    def push_to_hub(self, repo_id: str, token: str | None = None):
        """Push to HuggingFace Hub as a dataset."""
        try:
            from datasets import Dataset, Features, Value, Image as HFImage
        except ImportError:
            raise ImportError("pip install datasets huggingface_hub")

        records = []
        for sc in self.scenarios:
            records.append({
                "image": sc["image"],
                "question": sc["question"],
                "answer": sc["ground_truth"]["answer"],
                "reasoning": sc["ground_truth"]["reasoning"],
                "difficulty": sc["metadata"].get("difficulty", ""),
            })

        ds = Dataset.from_list(records)
        ds = ds.cast_column("image", HFImage())
        ds.push_to_hub(repo_id, token=token)
        print(f"Pushed {len(self)} scenarios to huggingface.co/datasets/{repo_id}")
