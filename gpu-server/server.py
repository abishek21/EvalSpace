"""
GPU Inference Server for Spatial Reasoning VLM evaluation.

Loads a Qwen2.5-VL model and exposes /health and /infer endpoints.
Designed to run on RunPod, Colab, or any GPU machine.

Usage:
    python server.py                           # Default: Qwen2.5-VL-3B
    python server.py --model Qwen/Qwen2.5-VL-7B-Instruct
    python server.py --port 8080
"""
import argparse
import base64
import os
import torch
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# ─── Parse args ──────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="GPU VLM Inference Server")
parser.add_argument("--model", type=str,
                    default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"),
                    help="HuggingFace model ID")
parser.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "5000")),
                    help="Server port")
parser.add_argument("--quantize", type=str,
                    default=os.environ.get("QUANTIZE", "auto"),
                    choices=["auto", "4bit", "none"],
                    help="Quantization: auto (based on GPU), 4bit, or none")
args = parser.parse_args()

MODEL_ID = args.model
PORT = args.port

# ─── Load model ──────────────────────────────────────────────────────

print(f"Loading model: {MODEL_ID}")
print(f"CUDA available: {torch.cuda.is_available()}")

gpu_name = "none"
gpu_mem_gb = 0
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    gpu_mem_gb = props.total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

# Decide quantization
use_4bit = False
if args.quantize == "4bit":
    use_4bit = True
elif args.quantize == "auto":
    # 4-bit if <24GB GPU or using 7B+ model
    if gpu_mem_gb < 24 or "7B" in MODEL_ID or "7b" in MODEL_ID:
        use_4bit = True

if use_4bit:
    print("Using 4-bit quantization")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto"
    )
else:
    print("Using float16 (no quantization)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
)
print(f"✅ {MODEL_ID} loaded and ready")

# ─── Inference helper ────────────────────────────────────────────────

IMG_SIZE = 480 if gpu_mem_gb < 40 else 720


def run_inference(system_prompt: str, user_text: str, images: list = None,
                  max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Run VLM inference with optional multi-image input."""
    user_content = []
    if images:
        for img in images:
            thumb = img.copy()
            thumb.thumbnail((IMG_SIZE, IMG_SIZE))
            user_content.append({"type": "image", "image": thumb})
        user_content.append({
            "type": "text",
            "text": f"[The above are {len(images)} different camera views of the same scene.]\n\n{user_text}"
        })
    else:
        user_content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if images:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           return_tensors="pt", padding=True).to(model.device)
    else:
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=temperature, do_sample=(temperature > 0)
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    torch.cuda.empty_cache()
    return processor.decode(generated, skip_special_tokens=True)


# ─── Flask API ───────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready",
        "model": MODEL_ID,
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem_gb, 1),
    })


@app.route("/infer", methods=["POST"])
def infer():
    """
    POST /infer
    {
        "system_prompt": "...",
        "user_prompt": "...",
        "images": ["data:image/jpeg;base64,..."],  // optional
        "max_tokens": 512,
        "temperature": 0.7
    }
    """
    try:
        data = request.json
        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        user_prompt = data.get("user_prompt", "")
        image_b64s = data.get("images", [])
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.7)

        # Decode base64 images
        pil_images = []
        for img_str in image_b64s:
            if "," in img_str:
                img_str = img_str.split(",", 1)[1]
            img_bytes = base64.b64decode(img_str)
            pil_images.append(Image.open(BytesIO(img_bytes)))

        result = run_inference(
            system_prompt=system_prompt,
            user_text=user_prompt,
            images=pil_images if pil_images else None,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return jsonify({
            "text": result,
            "model": MODEL_ID,
            "tokens_generated": len(result.split()),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🚀 GPU Inference Server starting on port {PORT}")
    print(f"   Model: {MODEL_ID}")
    print(f"   GPU:   {gpu_name} ({gpu_mem_gb:.1f}GB)")
    print(f"   Test:  curl http://localhost:{PORT}/health\n")
    app.run(host="0.0.0.0", port=PORT)
