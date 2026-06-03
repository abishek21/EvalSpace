# GPU Inference Server

Serves a Qwen2.5-VL model as an HTTP API for spatial reasoning evaluation.

## RunPod Deployment

1. Push Docker image:
```bash
cd gpu-server
docker build -t your-dockerhub/spatial-gpu-server:latest .
docker push your-dockerhub/spatial-gpu-server:latest
```

2. Create a RunPod **Serverless** or **Pod** endpoint:
   - Template: `your-dockerhub/spatial-gpu-server:latest`
   - GPU: T4 (3B model) or A100 (7B model)
   - Env vars (optional):
     - `MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct` (default: 3B)
     - `QUANTIZE=4bit` (default: auto)

3. Set `GPU_WORKER_URL` in your backend `.env`:
```
GPU_WORKER_URL=https://your-runpod-endpoint.runpod.ai
```

## Local Testing

```bash
pip install -r requirements.txt
python server.py --model Qwen/Qwen2.5-VL-3B-Instruct --port 5000
```

## API

### `GET /health`
```json
{"status": "ready", "model": "Qwen/Qwen2.5-VL-3B-Instruct", "gpu": "Tesla T4", "gpu_memory_gb": 15.6}
```

### `POST /infer`
```json
{
  "system_prompt": "You are a spatial reasoning expert.",
  "user_prompt": "Will the ball hit the box?",
  "images": ["data:image/jpeg;base64,..."],
  "max_tokens": 300,
  "temperature": 0.3
}
```
Response:
```json
{"text": "HIT. The ball is...", "model": "Qwen/Qwen2.5-VL-3B-Instruct", "tokens_generated": 42}
```
