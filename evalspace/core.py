"""Core API: es.make(), es.generate(), es.evaluate()"""

from evalspace.suite import EvalSuite
from evalspace.results import EvalResults


# Registry of available environments
ENVIRONMENTS = {
    "shelf_fitting": "evalspace.environments.shelf_fitting:ShelfFittingEnv",
}


def make(task: str, difficulty: str = "medium", seed: int | None = None, **kwargs):
    """
    Create an environment instance.

    Args:
        task: Environment name (e.g., "shelf_fitting")
        difficulty: "easy" | "medium" | "hard" | "mixed"
        seed: Random seed for reproducibility
        **kwargs: Environment-specific config

    Returns:
        Environment instance with reset()/step()/verify() methods
    """
    if task not in ENVIRONMENTS:
        available = ", ".join(ENVIRONMENTS.keys())
        raise ValueError(f"Unknown task '{task}'. Available: {available}")

    # Lazy import
    module_path, class_name = ENVIRONMENTS[task].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    env_class = getattr(module, class_name)

    return env_class(difficulty=difficulty, seed=seed, **kwargs)


def generate(task: str, num_scenes: int = 100, difficulty: str = "mixed",
             seed: int = 42, **kwargs) -> EvalSuite:
    """
    Generate a static dataset of scenes.

    Args:
        task: Environment name
        num_scenes: Number of scenarios to generate
        difficulty: Difficulty distribution
        seed: Random seed for reproducibility

    Returns:
        EvalSuite containing all generated scenarios
    """
    env = make(task=task, difficulty=difficulty, seed=seed, **kwargs)
    scenarios = []

    for i in range(num_scenes):
        obs = env.reset()
        gt = env.ground_truth()
        scenarios.append({
            "id": i,
            "image": obs.image,
            "question": obs.question,
            "ground_truth": gt,
            "metadata": obs.metadata,
        })

    return EvalSuite(
        task=task,
        scenarios=scenarios,
        difficulty=difficulty,
        seed=seed,
    )


def evaluate(suite: EvalSuite, model: str, api_key: str | None = None,
             system_prompt: str | None = None, **kwargs) -> EvalResults:
    """
    Evaluate a model on a suite of scenarios.

    Args:
        suite: EvalSuite from generate()
        model: Model identifier. Supported formats:
            - "openrouter/x-ai/grok-4.3" → OpenRouter API
            - "openrouter/openai/gpt-5.5" → OpenRouter API
            - "openrouter/google/gemini-2.5-pro" → OpenRouter API
        api_key: API key for the model provider
        system_prompt: Optional custom system prompt (default: spatial reasoning expert)

    Returns:
        EvalResults with accuracy, per-scenario results, etc.
    """
    import asyncio

    if not model.startswith("openrouter/"):
        raise ValueError(
            f"Model format not supported: '{model}'. "
            "Use 'openrouter/<provider>/<model>' (e.g., 'openrouter/x-ai/grok-4.3')"
        )

    if not api_key:
        raise ValueError("api_key is required for OpenRouter models")

    # Extract the actual model ID (remove 'openrouter/' prefix)
    model_id = model[len("openrouter/"):]

    if system_prompt is None:
        system_prompt = (
            "You are a spatial reasoning expert. Look at the image carefully "
            "and answer the question.\n\n"
            "Answer with EXACTLY this format:\n"
            "PREDICTION: FITS or DOES NOT FIT\n"
            "REASONING: <your reasoning in 1-2 sentences>"
        )

    async def _run():
        import httpx
        import io
        import base64

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        results_list = []
        correct = 0

        for i, sc in enumerate(suite.scenarios):
            # Convert PIL image to base64
            buf = io.BytesIO()
            sc["image"].save(buf, format="JPEG", quality=85)
            img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": img_b64}},
                    {"type": "text", "text": sc["question"]},
                ]},
            ]

            async with httpx.AsyncClient(timeout=120) as client:
                try:
                    resp = await client.post(url, headers=headers, json={
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": 300,
                        "temperature": 0.0,
                    })
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                except Exception as e:
                    content = f"[error: {e}]"

            # Parse prediction
            answer = content.lower()
            gt = sc["ground_truth"]["answer"]

            if "does not fit" in answer or "no fit" in answer or "doesn't fit" in answer:
                predicted = "no_fit"
            elif "fits" in answer:
                predicted = "fits"
            else:
                predicted = "unknown"

            is_correct = predicted == gt
            if is_correct:
                correct += 1

            results_list.append({
                "id": i,
                "question": sc["question"],
                "ground_truth": gt,
                "prediction": predicted,
                "correct": is_correct,
                "model_response": content[:500],
            })

            status = "✅" if is_correct else "❌"
            print(f"  [{i+1}/{len(suite)}] {status} GT={gt:6s} pred={predicted:6s} | {sc['question'][:50]}")

        accuracy = round(correct / len(suite) * 100, 1) if suite else 0

        return EvalResults(
            task=suite.task,
            model=model,
            accuracy=accuracy,
            correct=correct,
            total=len(suite),
            results=results_list,
        )

    return asyncio.run(_run())
