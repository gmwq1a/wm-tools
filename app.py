# 万事屋去水印后端:图片去水印(IOPaint LaMa)+ 视频打码(ffmpeg boxblur)
# 单进程 FastAPI;LaMa 模型懒加载(免费实例 512MB,端口必须先起来过健康检查)。
import io
import json
import os
import subprocess
import tempfile
import time
from collections import defaultdict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

ALLOWED_ORIGINS = [
    "https://wanshiwu.surge.sh",
    "https://geshi-gongfang.surge.sh",
]
MAX_IMAGE_SIDE = 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 150 * 1024 * 1024
MAX_VIDEO_SECONDS = 300
RATE_WINDOW = 60          # seconds
RATE_IMAGE = 10           # per window per IP
RATE_VIDEO = 3

app = FastAPI(title="wm-tools")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_hits = defaultdict(list)


def rate_ok(ip: str, bucket: str, limit: int) -> bool:
    now = time.time()
    key = f"{bucket}:{ip}"
    _hits[key] = [t for t in _hits[key] if now - t < RATE_WINDOW]
    if len(_hits[key]) >= limit:
        return False
    _hits[key].append(now)
    return True


_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(1)
        from iopaint.model_manager import ModelManager
        from iopaint.schema import Device
        _model = ModelManager(name="lama", device=Device.cpu)
    return _model


@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


@app.post("/api/image/dewatermark")
async def image_dewatermark(request: Request, image: UploadFile = File(...), mask: UploadFile = File(...)):
    ip = request.client.host if request.client else "?"
    if not rate_ok(ip, "img", RATE_IMAGE):
        return JSONResponse({"success": False, "error": "rate_limited"}, 429)
    img_bytes = await image.read()
    mask_bytes = await mask.read()
    if len(img_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse({"success": False, "error": "image_too_large"}, 413)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    msk = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        return JSONResponse({"success": False, "error": "bad_image_or_mask"}, 400)
    h, w = img.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, msk = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
    msk = cv2.dilate(msk, np.ones((5, 5), np.uint8), iterations=1)
    from iopaint.schema import InpaintRequest
    result = get_model()(img, msk, InpaintRequest())  # output is BGR, encode directly
    ok, buf = cv2.imencode(".png", result.astype(np.uint8))
    if not ok:
        return JSONResponse({"success": False, "error": "encode_failed"}, 500)
    return Response(content=buf.tobytes(), media_type="image/png")


def _probe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=30)
        info = json.loads(out.stdout or "{}")
    except Exception:
        return None, None
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    width = 0
    height = 0
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            width = int(s.get("width", 0)); height = int(s.get("height", 0))
            break
    return dur, (width, height)


@app.post("/api/video/mosaic")
async def video_mosaic(request: Request, video: UploadFile = File(...), regions: str = Form(default=""),
                       strength: int = Form(default=12)):
    ip = request.client.host if request.client else "?"
    if not rate_ok(ip, "vid", RATE_VIDEO):
        return JSONResponse({"success": False, "error": "rate_limited"}, 429)
    data = await video.read()
    if len(data) > MAX_VIDEO_BYTES:
        return JSONResponse({"success": False, "error": "video_too_large"}, 413)
    strength = max(4, min(30, int(strength)))
    tmpdir = tempfile.mkdtemp(prefix="wmtools-")
    src = os.path.join(tmpdir, "in" + os.path.splitext(video.filename or ".mp4")[1][:8] or ".mp4")
    dst = os.path.join(tmpdir, "out.mp4")
    with open(src, "wb") as f:
        f.write(data)
    dur, wh = _probe(src)
    if dur is None:
        return JSONResponse({"success": False, "error": "probe_failed"}, 400)
    if dur > MAX_VIDEO_SECONDS:
        return JSONResponse({"success": False, "error": f"video_too_long_max_{MAX_VIDEO_SECONDS}s"}, 413)
    W, H = wh
    # 默认区域:抖音双角落水印(右上昵称/抖音号、左下 @账号),按比例
    if regions.strip():
        try:
            boxes = json.loads(regions)
            boxes = [(int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])) for b in boxes][:6]
        except Exception:
            return JSONResponse({"success": False, "error": "bad_regions_json"}, 400)
    else:
        boxes = [
            (int(W * 0.66), int(H * 0.015), int(W * 0.325), int(H * 0.085)),
            (int(W * 0.015), int(H * 0.90), int(W * 0.40), int(H * 0.085)),
        ]
    boxes = [(max(0, x), max(0, y), min(w, W - x), min(h, H - y)) for x, y, w, h in boxes if w > 4 and h > 4]
    if not boxes:
        return JSONResponse({"success": False, "error": "no_valid_regions"}, 400)
    n = len(boxes)
    parts = [f"[0:v]split={n + 1}" + "".join(f"[s{i}]" for i in range(n + 1))]
    last = f"[s{n}]"
    for i, (x, y, w, h) in enumerate(boxes):
        parts.append(f"[s{i}]crop={w}:{h}:{x}:{y},boxblur={strength}[b{i}]")
        parts.append(f"{last}[b{i}]overlay={x}:{y}[v{i}]")
        last = f"[v{i}]"
    cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex", ";".join(parts),
           "-map", last, "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-c:a", "copy", "-movflags", "+faststart", dst]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=560)
    except subprocess.TimeoutExpired:
        return JSONResponse({"success": False, "error": "ffmpeg_timeout"}, 500)
    if proc.returncode != 0 or not os.path.exists(dst):
        return JSONResponse({"success": False, "error": "ffmpeg_failed", "tail": (proc.stderr or "")[-300:]}, 500)
    return FileResponse(dst, media_type="video/mp4", filename="mosaicked.mp4")
