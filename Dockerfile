ARG CUDA_IMAGE=12.1.1-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libusb-1.0-0 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

ARG TORCH_INDEX
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url "${TORCH_INDEX}"

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r /app/requirements-docker.txt

COPY . /app

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "pipeline.web_gui:app", "--host", "0.0.0.0", "--port", "7860"]