import os, shutil, subprocess, tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from isl_raw import get_predictor

# PLACEHOLDERS: tune on idle / no-sign clips BEFORE the demo
CONF_MIN = float(os.environ.get("CONF_MIN", "0.70"))
ANY_RATE_MIN = float(os.environ.get("ANY_RATE_MIN", "0.30"))
HAS_FFMPEG = shutil.which("ffmpeg") is not None

app = FastAPI(title="ISL Emergency Communicator API (landmark-only)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _load():
    get_predictor()


@app.get("/health")
def health():
    return {"ok": True, "ffmpeg": HAS_FFMPEG, "conf_min": CONF_MIN, "any_rate_min": ANY_RATE_MIN}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    tmp = tempfile.mkdtemp()
    try:
        ext = os.path.splitext(file.filename or "")[1] or ".webm"
        raw = os.path.join(tmp, "in" + ext)
        with open(raw, "wb") as f:
            shutil.copyfileobj(file.file, f)

        path, fps_override = raw, 30.0       # fallback: decode directly, assume 30 fps
        if HAS_FFMPEG:                       # same normalisation as data collection
            mp4 = os.path.join(tmp, "norm.mp4")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", raw, "-t", "5",
                 "-vf", "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2",
                 "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", mp4],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode == 0 and os.path.isfile(mp4):
                path, fps_override = mp4, None

        d = get_predictor().predict_detailed(path, fps_override=fps_override)

        if d["no_hand"] or d["any_rate"] < ANY_RATE_MIN:
            return {"status": "no_sign",
                    "message": "No hands detected clearly. Please sign properly with both hands visible and try again."}
        if d["confidence"] < CONF_MIN:
            return {"status": "no_sign", "confidence": round(d["confidence"], 3),
                    "message": "No clear sign detected. Please sign properly and try again."}
        return {"status": "ok", "label": d["label"], "confidence": round(d["confidence"], 3)}
    except Exception as e:
        return {"status": "error", "message": f"Server error: {type(e).__name__}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
