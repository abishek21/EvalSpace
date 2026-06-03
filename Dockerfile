FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Install Python deps (torch first for CUDA)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY gpu-server/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY gpu-server/server.py handler.py

ENV MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
ENV QUANTIZE=auto

CMD ["python3", "-u", "/handler.py"]
