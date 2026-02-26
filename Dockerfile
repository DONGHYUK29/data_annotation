FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

#시간대 설정 질문 방지 및 필수 라이브러리 설치
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    -r requirements.txt

COPY . .
CMD ["python", "main.py"]