FROM nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

COPY requirements.txt /backend/requirements.txt

WORKDIR /backend

RUN pip install -r requirements.txt

COPY . /backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='franciszzj/Leffa', local_dir='/backend/ckpts')"

CMD ["python3", "app.py"]