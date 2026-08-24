# ===== Base =====
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ===== system OpenCV 제거 =====
RUN apt-get update && \
    apt-get remove -y \
        python3-opencv \
        libopencv* \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ===== pip =====
RUN python3 -m pip install --upgrade pip setuptools wheel

# ===== OpenCV (runtime에서만 사용, build 중 import 금지) =====
RUN pip uninstall -y \
        opencv-python \
        opencv-python-headless \
        opencv-contrib-python || true

RUN pip install opencv-python-headless==4.8.1.78

# ===== YOLO =====
RUN pip install ultralytics

RUN pip uninstall -y opencv opencv-python opencv-python-headless opencv-contrib-python
RUN find /usr/local/lib/python3.10/dist-packages -name "cv2*" -type d -exec rm -rf {} +
RUN pip install opencv-python-headless==4.8.1.78
