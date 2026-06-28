"""EvalResults — output from evaluating a model on a suite."""

from dataclasses import dataclass, field


@dataclass
class EvalResults:
    """Results from evaluating a model."""
    task: str
    model: str
    accuracy: float
    correct: int
    total: int
    results: list[dict] = field(default_factory=list)

    def __repr__(self):
        return (
            f"EvalResults(task='{self.task}', model='{self.model}', "
            f"accuracy={self.accuracy:.1f}%, correct={self.correct}/{self.total})"
        )

    def summary(self):
        print(f"Model: {self.model}")
        print(f"Task:  {self.task}")
        print(f"Score: {self.correct}/{self.total} ({self.accuracy:.1f}%)")

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({
                "task": self.task,
                "model": self.model,
                "accuracy": self.accuracy,
                "correct": self.correct,
                "total": self.total,
                "results": self.results,
            }, f, indent=2, default=str)
        print(f"Saved results to {path}")
