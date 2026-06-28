"""
EvalSpace — Verifiable Visual-Spatial Reasoning Environments

Physics-engine-verified environments for evaluating and training
VLMs on spatial reasoning tasks.

Usage:
    import evalspace as es

    env = es.make(task="shelf_fitting", difficulty="medium", seed=42)
    obs = env.reset()
    reward = env.verify("fits")
"""

from evalspace.core import make, generate, evaluate
from evalspace.suite import EvalSuite
from evalspace.results import EvalResults

__version__ = "0.1.0"
__all__ = ["make", "generate", "evaluate", "EvalSuite", "EvalResults"]
