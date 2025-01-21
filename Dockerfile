FROM nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

COPY requirements.txt /backend/requirements.txt

WORKDIR /backend

RUN pip install --no-cache-dir -r requirements.txt

COPY . /backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6

CMD ["python3", "app.py"]