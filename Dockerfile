FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch first so iopaint does not pull CUDA wheels
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir iopaint fastapi "uvicorn[standard]" python-multipart

# Pre-bake the LaMa model (avoids first-request download and cold-start latency)
RUN python - <<'PY'
from iopaint.helper import download_model
download_model("https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt")
print("model cached")
PY

WORKDIR /srv
COPY app.py /srv/app.py

ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
