"""
ISL inference, LANDMARK-ONLY (raw 126-D) + Transformer, run with ONNX Runtime.
No torch, no ViT. Feature extraction is copied verbatim from isl_inference.py / notebook Cell C;
the model input is the first 126 columns of the 450-D kinematic block (= raw xyz of both hands).
"""
import os
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import onnxruntime as ort

SEQ_LEN = 30
NUM_LANDMARKS = 21
COORDS = 3
LANDMARKS_PER_HAND = NUM_LANDMARKS * COORDS
XYZ_DIM = 126
ANGLE_DIM = 30
SPEED_DIM = 42
KINEMATIC_DIM = 450
RAW_DIM = 126
CLASSES = ["help", "doctor", "pain", "call", "accident"]

ANGLE_TRIPLETS = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
    (0, 5, 9), (0, 9, 13), (0, 13, 17),
    (5, 9, 13), (9, 13, 17),
]

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("ISL_MODEL_DIR", BASE_DIR / "model"))
DEFAULT_ONNX = MODEL_DIR / "raw_transformer.onnx"
DEFAULT_SCALER = MODEL_DIR / "raw_scaler_parameters.npz"
DEFAULT_HAND = MODEL_DIR / "hand_landmarker.task"

# ----------------------------------------------------------------------
# Helpers (identical logic to Cell C)
# ----------------------------------------------------------------------
def sample_frame_indices(total_frames, target=SEQ_LEN):
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {total_frames}")
    return np.rint(np.linspace(0, total_frames - 1, target)).astype(np.int64)


def compute_hand_angles(coords):
    angles = np.zeros(15, dtype=np.float32)
    for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        v1 = coords[a] - coords[b]
        v2 = coords[c] - coords[b]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles[i] = np.arccos(cosang)
    return angles


def make_landmarker(hand_model_path):
    base_options = mp_python.BaseOptions(model_asset_path=str(hand_model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.05,
        min_hand_presence_confidence=0.05,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def read_sampled_frames(video_path, flip=False):
    """
    Decode the whole video, sample SEQ_LEN frames uniformly (rint(linspace)).
    Uses the DECODED frame count (equals the reported count for clean mp4s,
    which is what training required). Returns (frames_bgr, indices, fps, total, mismatch).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
    finally:
        cap.release()

    total = len(frames)
    if total == 0:
        raise RuntimeError(f"No frames decoded: {video_path}")
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    indices = sample_frame_indices(total)
    sampled = []
    for fi in indices:
        fr = frames[int(fi)]
        if flip:
            fr = cv2.flip(fr, 1)
        sampled.append(fr)
    return sampled, indices, fps, total, (reported != total)


def kinematic_from_frames(sampled, indices, fps, landmarker):
    """Returns kinematic (SEQ_LEN, 450) float32, presence (SEQ_LEN, 2) bool."""
    detections = {}
    for t, fi in enumerate(indices):
        fi = int(fi)
        if fi in detections:
            continue
        rgb = cv2.cvtColor(sampled[t], cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detections[fi] = landmarker.detect(mp_image)

    raw = np.zeros((SEQ_LEN, XYZ_DIM), dtype=np.float32)
    angles = np.zeros((SEQ_LEN, ANGLE_DIM), dtype=np.float32)
    presence = np.zeros((SEQ_LEN, 2), dtype=bool)

    for t, fi in enumerate(indices):
        result = detections[int(fi)]
        if not result.hand_landmarks:
            continue
        detected = []
        for hand_lm in result.hand_landmarks:
            coords = np.asarray([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)
            if coords.shape != (NUM_LANDMARKS, COORDS) or not np.isfinite(coords).all():
                continue
            detected.append({"coords": coords, "wrist_x": float(coords[0, 0])})
        if not detected:
            continue
        detected.sort(key=lambda d: d["wrist_x"])
        detected = detected[:2]
        if len(detected) == 1:
            d = detected[0]
            slot_assignments = [(0 if d["wrist_x"] < 0.5 else 1, d)]
        else:
            slot_assignments = [(slot, d) for slot, d in enumerate(detected)]
        for slot, d in slot_assignments:
            coords = d["coords"]
            start = slot * LANDMARKS_PER_HAND
            end = start + LANDMARKS_PER_HAND
            raw[t, start:end] = coords.reshape(-1)
            rel = coords - coords[0]
            angles[t, slot * 15: slot * 15 + 15] = compute_hand_angles(rel)
            presence[t, slot] = True

    times = indices.astype(np.float64) / fps

    velocity = np.zeros_like(raw)
    for t in range(1, SEQ_LEN):
        delta_t = times[t] - times[t - 1]
        if delta_t <= 0:
            continue
        for slot in range(2):
            if not (presence[t, slot] and presence[t - 1, slot]):
                continue
            s, e = slot * LANDMARKS_PER_HAND, (slot + 1) * LANDMARKS_PER_HAND
            velocity[t, s:e] = (raw[t, s:e] - raw[t - 1, s:e]) / delta_t

    velocity_3d = velocity.reshape(SEQ_LEN, 2, NUM_LANDMARKS, COORDS)
    speed = np.linalg.norm(velocity_3d, axis=-1).reshape(SEQ_LEN, SPEED_DIM).astype(np.float32)

    acceleration = np.zeros_like(raw)
    for t in range(2, SEQ_LEN):
        dt_prev = times[t - 1] - times[t - 2]
        dt_curr = times[t] - times[t - 1]
        if dt_prev <= 0 or dt_curr <= 0:
            continue
        denom = dt_prev + dt_curr
        for slot in range(2):
            if not (presence[t, slot] and presence[t - 1, slot] and presence[t - 2, slot]):
                continue
            s, e = slot * LANDMARKS_PER_HAND, (slot + 1) * LANDMARKS_PER_HAND
            v_curr = (raw[t, s:e] - raw[t - 1, s:e]) / dt_curr
            v_prev = (raw[t - 1, s:e] - raw[t - 2, s:e]) / dt_prev
            acceleration[t, s:e] = 2.0 * (v_curr - v_prev) / denom

    kinematic = np.concatenate([raw, angles, velocity, speed, acceleration], axis=1).astype(np.float32)
    assert kinematic.shape == (SEQ_LEN, KINEMATIC_DIM)
    return kinematic, presence




class ISLRawPredictor:
    def __init__(self, onnx_path=DEFAULT_ONNX, scaler_path=DEFAULT_SCALER, hand_model_path=DEFAULT_HAND):
        for p in (onnx_path, scaler_path, hand_model_path):
            if not Path(p).is_file():
                raise FileNotFoundError(f"Missing file: {p}")
        z = np.load(str(scaler_path))
        self.mean = z["mean"].astype(np.float64)
        self.scale = z["scale"].astype(np.float64)
        assert self.mean.shape == (RAW_DIM,) and self.scale.shape == (RAW_DIM,)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.landmarker = make_landmarker(hand_model_path)
        self.classes = list(CLASSES)

    def extract_raw(self, video_path, flip=False, fps_override=None):
        sampled, indices, fps, total, mismatch = read_sampled_frames(video_path, flip=flip)
        if fps_override:
            fps = float(fps_override)
        kin, presence = kinematic_from_frames(sampled, indices, fps, self.landmarker)
        diag = {
            "frames_decoded": int(total), "fps": float(fps), "frame_count_mismatch": bool(mismatch),
            "left_rate": float(presence[:, 0].mean()), "right_rate": float(presence[:, 1].mean()),
            "any_rate": float(presence.any(axis=1).mean()), "both_rate": float(presence.all(axis=1).mean()),
        }
        return kin[:, :RAW_DIM].astype(np.float32), diag

    def predict_detailed(self, video_path, flip=False, fps_override=None):
        raw, diag = self.extract_raw(video_path, flip=flip, fps_override=fps_override)
        xs = ((raw.astype(np.float64) - self.mean) / self.scale).astype(np.float32)
        diag["max_abs_scaled"] = float(np.abs(xs).max())
        logits = self.sess.run(None, {self.in_name: xs[None]})[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        k = int(probs.argmax())
        diag.update({"label": self.classes[k], "confidence": float(probs[k]),
                     "probs": {c: float(p) for c, p in zip(self.classes, probs)},
                     "no_hand": diag["any_rate"] == 0.0})
        return diag


_singleton = None


def get_predictor():
    global _singleton
    if _singleton is None:
        _singleton = ISLRawPredictor()
    return _singleton
