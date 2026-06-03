FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (torch first for CUDA)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY server.py .

# RunPod exposes port 8000 by default
ENV PORT=8000
ENV MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
ENV QUANTIZE=auto

EXPOSE 8000

CMD ["python3", "server.py"]
