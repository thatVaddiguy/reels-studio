FROM python:3.12-slim

# ffmpeg (with libass for captions) and a bold font for subtitles
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core fontconfig curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# JS runtime that yt-dlp needs for YouTube
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CUDA libraries for GPU transcription (ignored when no GPU is attached)
ARG GPU=1
RUN if [ "$GPU" = "1" ]; then pip install --no-cache-dir nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"; fi
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

# Face detection model
RUN curl -fsSL -o /app/face_detection_yunet_2023mar.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

COPY app/ /app/

ENV DATA_DIR=/data/jobs \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
