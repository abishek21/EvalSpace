"""
GPU Worker client — supports both direct (ngrok/Colab) and RunPod Serverless modes.

Mode is auto-detected from env vars:
  - Direct mode: set GPU_WORKER_URL=https://xxxx.ngrok-free.app
  - RunPod mode: set RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY
"""
import httpx
import os
import asyncio
import time

TIMEOUT = 300.0
MAX_RETRIES = 5
RUNPOD_BASE = "https://api.runpod.ai/v2"

RETRY_ERRORS = (
    httpx.ConnectError, httpx.ReadError, httpx.WriteError,
    httpx.RemoteProtocolError, httpx.ConnectTimeout,
    httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
    httpx.HTTPStatusError,
)


def _mode():
    """Return 'runpod' or 'direct' based on env vars."""
    if os.getenv("RUNPOD_ENDPOINT_ID") and os.getenv("RUNPOD_API_KEY"):
        return "runpod"
    if os.getenv("GPU_WORKER_URL"):
        return "direct"
    raise RuntimeError(
        "No GPU backend configured. Set GPU_WORKER_URL (ngrok) "
        "or RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY (RunPod)."
    )


# ─── Health check ────────────────────────────────────────────────────

async def check_health() -> dict:
    mode = _mode()
    if mode == "runpod":
        eid = os.getenv("RUNPOD_ENDPOINT_ID")
        headers = {"Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.get(f"{RUNPOD_BASE}/{eid}/health", headers=headers)
            res.raise_for_status()
            return res.json()
    else:
        url = os.getenv("GPU_WORKER_URL")
        headers = {"ngrok-skip-browser-warning": "true"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.get(f"{url}/health", headers=headers)
            res.raise_for_status()
            return res.json()


# ─── Inference ───────────────────────────────────────────────────────

async def infer(
    system_prompt: str,
    user_prompt: str,
    images: list[str] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    mode = _mode()
    if mode == "runpod":
        return await _infer_runpod(system_prompt, user_prompt, images, max_tokens, temperature)
    else:
        return await _infer_direct(system_prompt, user_prompt, images, max_tokens, temperature)


async def _infer_direct(system_prompt, user_prompt, images, max_tokens, temperature):
    """Call ngrok/Colab Flask server directly."""
    url = os.getenv("GPU_WORKER_URL")
    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "images": images or [],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"ngrok-skip-browser-warning": "true"}
    payload_kb = sum(len(i) for i in (images or [])) // 1024
    print(f"  📡 Direct GPU request: {len(images or [])} images ({payload_kb}KB) → {url}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                res = await client.post(f"{url}/infer", json=payload, headers=headers)
                elapsed = time.time() - t0
                print(f"  📡 Response: HTTP {res.status_code} in {elapsed:.1f}s")
                if res.status_code != 200:
                    print(f"  ⚠️  GPU returned HTTP {res.status_code}: {res.text[:200]}")
                res.raise_for_status()
                data = res.json()
                return data["text"]
            except RETRY_ERRORS as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  ⚠️  GPU request failed (attempt {attempt+1}/{MAX_RETRIES}): {e.__class__.__name__}: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  ❌ GPU request failed after {MAX_RETRIES} attempts: {e.__class__.__name__}: {e}")
                    raise


async def _infer_runpod(system_prompt, user_prompt, images, max_tokens, temperature):
    """Call RunPod Serverless /runsync endpoint."""
    eid = os.getenv("RUNPOD_ENDPOINT_ID")
    payload = {
        "input": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "images": images or [],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}",
        "Content-Type": "application/json",
    }
    url = f"{RUNPOD_BASE}/{eid}/runsync"
    payload_kb = sum(len(i) for i in (images or [])) // 1024
    print(f"  📡 RunPod request: {len(images or [])} images ({payload_kb}KB) → {eid}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(MAX_RETRIES):
            try:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code != 200:
                    print(f"  ⚠️  RunPod returned HTTP {res.status_code}: {res.text[:200]}")
                res.raise_for_status()
                data = res.json()
                if data.get("status") == "FAILED":
                    raise RuntimeError(f"RunPod job failed: {data.get('error', 'Unknown')}")
                output = data.get("output", {})
                if "error" in output:
                    raise RuntimeError(f"Handler error: {output['error']}")
                return output["text"]
            except RETRY_ERRORS as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  ⚠️  RunPod request failed (attempt {attempt+1}/{MAX_RETRIES}): {e.__class__.__name__}: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  ❌ RunPod request failed after {MAX_RETRIES} attempts: {e.__class__.__name__}: {e}")
                    raise
