"""
GPU Worker client — calls the Colab/Modal inference server.
"""
import httpx
import os

def _get_url():
    return os.getenv("GPU_WORKER_URL", "")

TIMEOUT = 300.0  # 5 min — generous for VLM inference over ngrok
MAX_RETRIES = 5
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}


async def check_health() -> dict:
    """Check if GPU worker is reachable."""
    url = _get_url()
    if not url:
        raise RuntimeError("GPU_WORKER_URL not set")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(f"{url}/health", headers=NGROK_HEADERS)
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
    Call GPU worker for VLM inference.
    
    Args:
        system_prompt: System message for the VLM
        user_prompt: User message
        images: Optional list of base64 data URI strings
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text from the VLM
    """
    if not _get_url():
        raise RuntimeError("GPU_WORKER_URL not set. Start Colab server and set env var.")

    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "images": images or [],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    import asyncio

    url = _get_url()
    payload_kb = sum(len(i) for i in (images or [])) // 1024
    print(f"  📡 GPU request: {len(images or [])} images ({payload_kb}KB) → {url}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(MAX_RETRIES):
            try:
                res = await client.post(f"{url}/infer", json=payload, headers=NGROK_HEADERS)
                if res.status_code != 200:
                    print(f"  ⚠️  GPU returned HTTP {res.status_code}: {res.text[:200]}")
                res.raise_for_status()
                data = res.json()
                return data["text"]
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                    httpx.RemoteProtocolError, httpx.ConnectTimeout,
                    httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
                    httpx.HTTPStatusError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  ⚠️  GPU request failed (attempt {attempt+1}/{MAX_RETRIES}): {e.__class__.__name__}: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  ❌ GPU request failed after {MAX_RETRIES} attempts: {e.__class__.__name__}: {e}")
                    raise
