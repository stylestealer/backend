FROM nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /backend

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
