ARG CUDA_IMAGE=12.8.0-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    git \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements-docker.txt /app/requirements-docker.txt

RUN pip install --no-cache-dir -r /app/requirements-docker.txt

RUN pip install --no-cache-dir --force-reinstall --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

COPY . /app

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "pipeline.gui_app:app", "--host", "0.0.0.0", "--port", "7860"]