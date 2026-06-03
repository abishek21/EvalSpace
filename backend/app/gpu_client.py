"""
GPU Worker client — calls RunPod Serverless endpoint for VLM inference.
"""
import httpx
import os

def _get_endpoint_id():
    return os.getenv("RUNPOD_ENDPOINT_ID", "")

def _get_api_key():
    return os.getenv("RUNPOD_API_KEY", "")

TIMEOUT = 300.0  # 5 min — generous for VLM inference
MAX_RETRIES = 5
RUNPOD_BASE = "https://api.runpod.ai/v2"


async def check_health() -> dict:
    """Check if RunPod endpoint is reachable."""
    endpoint_id = _get_endpoint_id()
    api_key = _get_api_key()
    if not endpoint_id or not api_key:
        raise RuntimeError("RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY not set")
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(f"{RUNPOD_BASE}/{endpoint_id}/health", headers=headers)
        res.raise_for_status()
        return res.json()


async def infer(
    system_prompt: str,
    user_prompt: str,
    images: list[str] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Call RunPod Serverless endpoint for VLM inference.
    
    Args:
        system_prompt: System message for the VLM
        user_prompt: User message
        images: Optional list of base64 data URI strings
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text from the VLM
    """
    endpoint_id = _get_endpoint_id()
    api_key = _get_api_key()
    if not endpoint_id or not api_key:
        raise RuntimeError("RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY not set")

    payload = {
        "input": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "images": images or [],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    }

    import asyncio

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{RUNPOD_BASE}/{endpoint_id}/runsync"
    payload_kb = sum(len(i) for i in (images or [])) // 1024
    print(f"  📡 RunPod request: {len(images or [])} images ({payload_kb}KB) → {endpoint_id}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(MAX_RETRIES):
            try:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code != 200:
                    print(f"  ⚠️  RunPod returned HTTP {res.status_code}: {res.text[:200]}")
                res.raise_for_status()
                data = res.json()

                # RunPod wraps output: {"id": "...", "status": "COMPLETED", "output": {...}}
                if data.get("status") == "FAILED":
                    error = data.get("error", "Unknown error")
                    raise RuntimeError(f"RunPod job failed: {error}")

                output = data.get("output", {})
                if "error" in output:
                    raise RuntimeError(f"Handler error: {output['error']}")

                return output["text"]
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                    httpx.RemoteProtocolError, httpx.ConnectTimeout,
                    httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
                    httpx.HTTPStatusError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  ⚠️  RunPod request failed (attempt {attempt+1}/{MAX_RETRIES}): {e.__class__.__name__}: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  ❌ RunPod request failed after {MAX_RETRIES} attempts: {e.__class__.__name__}: {e}")
                    raise
