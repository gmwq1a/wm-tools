# 万事屋去水印后端:图片去水印(IOPaint LaMa)+ 视频打码(ffmpeg boxblur)
# 单进程 FastAPI;LaMa 模型懒加载(免费实例 512MB,端口必须先起来过健康检查)。
# 视频打码为异步 job 模式:POST 立即返回 job_id,后台线程跑 ffmpeg,
# 页面轮询 GET /api/video/mosaic/{job_id} 拿状态,完成后 /result 取片。
# (同步长请求会被中间代理在 ~60s 掐断,浏览器表现为 Failed to fetch。)
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections import defaultdict

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

ALLOWED_ORIGINS = [
    "https://wanshiwu.surge.sh",
    "https://geshi-gongfang.surge.sh",
    "https://wanshiwu-preview.onrender.com",
]
MAX_IMAGE_SIDE = 768
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 150 * 1024 * 1024
MAX_VIDEO_SECONDS = 300
RATE_WINDOW = 60          # seconds
RATE_IMAGE = 10           # per window per IP
RATE_VIDEO = 3
JOB_TTL = 900             # 结果保留 15 分钟

app = FastAPI(title="wm-tools")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_hits = defaultdict(list)
_jobs = {}                # job_id -> dict(status, error, dst, tmpdir, created, done_at)
_jobs_lock = threading.Lock()


def rate_ok(ip: str, bucket: str, limit: int) -> bool:
    now = time.time()
    key = f"{bucket}:{ip}"
    _hits[key] = [t for t in _hits[key] if now - t < RATE_WINDOW]
    if len(_hits[key]) >= limit:
        return False
    _hits[key].append(now)
    return True


def _sweep_jobs():
    import shutil
    now = time.time()
    with _jobs_lock:
        old = [k for k, j in _jobs.items() if now - j["created"] > JOB_TTL]
        dirs = [_jobs.pop(k)["tmpdir"] for k in old]
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


_model = None


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


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


def _run_mosaic(job_id, src, dst, boxes, strength, need_scale):
    try:
        n = len(boxes)
        pre = "[0:v]scale=1280:-2[sc];" if need_scale else ""
        srcv = "[sc]" if need_scale else "[0:v]"
        parts = [pre + f"{srcv}split={n + 1}" + "".join(f"[s{i}]" for i in range(n + 1))]
        last = f"[s{n}]"
        for i, (x, y, w, h) in enumerate(boxes):
            # boxblur radius is capped per plane: chroma (4:2:0) is half-size,
            # so max radius = (min(w,h)/2 - 1) / 2. Clamp or ffmpeg EINVALs on
            # small regions (e.g. 208x31 douyin corner boxes at strength 12).
            r = max(1, min(strength, (min(w, h) - 2) // 4))
            parts.append(f"[s{i}]crop={w}:{h}:{x}:{y},boxblur={r}[b{i}]")
            parts.append(f"{last}[b{i}]overlay={x}:{y}[v{i}]")
            last = f"[v{i}]"
        cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex", ";".join(parts),
               "-map", last, "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-c:a", "copy", "-movflags", "+faststart", dst]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=560)
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j is None:
                return
            if proc.returncode != 0 or not os.path.exists(dst):
                j["status"] = "error"
                j["error"] = "ffmpeg_failed"
                j["tail"] = (proc.stderr or "")[-300:]
            else:
                j["status"] = "done"
                j["done_at"] = time.time()
    except subprocess.TimeoutExpired:
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["error"] = "ffmpeg_timeout"
    except Exception as e:
        with _jobs_lock:
            j = _jobs.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["error"] = "worker_error"
                j["tail"] = str(e)[-200:]


@app.post("/api/video/jobs")
async def video_mosaic_job(request: Request, video: UploadFile = File(...), regions: str = Form(default=""),
                       strength: int = Form(default=12)):
    _sweep_jobs()
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
    need_scale = W > 1280
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
    if need_scale:
        sx = 1280 / W
        boxes = [(int(x * sx), int(y * sx), max(6, int(w * sx)), max(6, int(h * sx))) for x, y, w, h in boxes]
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "error": None, "dst": dst,
                         "tmpdir": tmpdir, "created": time.time(), "done_at": None}
    t = threading.Thread(target=_run_mosaic, args=(job_id, src, dst, boxes, strength, need_scale), daemon=True)
    t.start()
    return {"success": True, "job_id": job_id}


@app.get("/api/video/jobs/{job_id}")
def mosaic_status(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            return JSONResponse({"success": False, "error": "job_not_found"}, 404)
        out = {"success": True, "status": j["status"], "elapsed": int(time.time() - j["created"])}
        if j["status"] == "error":
            out["error"] = j.get("error") or "unknown"
            if j.get("tail"):
                out["tail"] = j["tail"]
        return out


@app.get("/api/video/jobs/{job_id}/result")
def mosaic_result(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            return JSONResponse({"success": False, "error": "job_not_found"}, 404)
        if j["status"] != "done":
            return JSONResponse({"success": False, "error": "not_ready"}, 409)
        dst = j["dst"]
    return FileResponse(dst, media_type="video/mp4", filename="mosaicked.mp4")

@app.post("/api/video/mosaic")
async def video_mosaic_sync(request: Request, video: UploadFile = File(...), regions: str = Form(default=""),
                            strength: int = Form(default=12)):
    """兼容旧版同步页面:内部走 job,原地等结果后直接返回 MP4。
    注意:长任务(>60s)仍可能被中间代理掐断,新页面请用 /api/video/jobs 异步模式。"""
    resp = await video_mosaic_job(request, video, regions, strength)
    if isinstance(resp, JSONResponse):
        return resp
    job_id = resp["job_id"]
    deadline = time.time() + 580
    while time.time() < deadline:
        with _jobs_lock:
            j = _jobs.get(job_id)
        if j is None:
            return JSONResponse({"success": False, "error": "job_lost"}, 500)
        if j["status"] == "done":
            return FileResponse(j["dst"], media_type="video/mp4", filename="mosaicked.mp4")
        if j["status"] == "error":
            out = {"success": False, "error": j.get("error") or "unknown"}
            if j.get("tail"):
                out["tail"] = j["tail"]
            return JSONResponse(out, 500)
        import asyncio
        await asyncio.sleep(2)
    return JSONResponse({"success": False, "error": "timeout"}, 504)

