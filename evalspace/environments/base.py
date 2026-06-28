"""Base environment class — all environments implement this interface."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from PIL import Image


@dataclass
class Observation:
    """What the model receives."""
    image: Image.Image          # rendered scene
    question: str               # natural language question
    metadata: dict = field(default_factory=dict)  # extra info (scene_id, etc.)


class BaseEnvironment:
    """
    Base class for all EvalSpace environments.

    Implements a Gym-like interface:
        obs = env.reset()           # new scene
        obs, reward, done, info = env.step(action)
        reward = env.verify(answer)  # stateless check
    """

    def __init__(self, difficulty: str = "medium", seed: int | None = None, **kwargs):
        self.difficulty = difficulty
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._current_scene = None

    def reset(self) -> Observation:
        """Generate a new scene. Returns observation."""
        raise NotImplementedError

    def step(self, action: str) -> tuple[Observation, float, bool, dict]:
        """
        Gym-compatible interface.
        For single-step envs: always returns done=True.

        Args:
            action: model's answer (e.g., "fits" or "no_fit")

        Returns:
            (observation, reward, done, info)
        """
        reward = self.verify(action)
        gt = self.ground_truth()
        info = {
            "ground_truth": gt,
            "correct": reward > 0.5,
        }
        return self._current_obs, reward, True, info

    def verify(self, answer: str) -> float:
        """
        Check if the answer is correct for the current scene.

        Returns:
            1.0 if correct, 0.0 if wrong
        """
        raise NotImplementedError

    def ground_truth(self) -> dict:
        """Return ground truth for current scene."""
        raise NotImplementedError

    def render(self) -> Image.Image:
        """Return the current scene image."""
        if self._current_obs:
            return self._current_obs.image
        raise RuntimeError("Call reset() first")

    def question(self) -> str:
        """Return the current question."""
        if self._current_obs:
            return self._current_obs.question
        raise RuntimeError("Call reset() first")
