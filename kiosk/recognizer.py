"""
kiosk/recognizer.py
───────────────────
Face detection + recognition + anti-spoofing using InsightFace + MediaPipe.

Changed from dlib/face_recognition  →  InsightFace (ONNX Runtime)
  • No dlib dependency — works on Python 3.10 / 3.11 / 3.12 / 3.14
  • SCRFD detector + ArcFace 512-d embeddings (more accurate than 128-d dlib)
  • Cosine distance matching (L2-normalised embeddings → dot product)

Liveness — MediaPipe Tasks API (NOT the legacy mp.solutions.face_mesh)
  Google ended support for MediaPipe's legacy "Solutions" API
  (mp.solutions.*) back in March 2023, and recent pip wheels (0.10.31+)
  have stopped shipping it entirely — importing it now raises
  `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
  This file uses the modern replacement, the Tasks API FaceLandmarker
  (mediapipe.tasks.python.vision), which auto-downloads its model bundle
  (~3.6 MB) to ~/.face_attendance_models/ on first run, the same pattern
  InsightFace already uses for its own models.

Per-frame output
────────────────
    Recognizer.process(bgr_frame) → list[dict]

    Each dict:
        box        : (x1, y1, x2, y2)  pixel coords in the original frame
        student_id : str | None
        live       : bool
        prompt     : str | None         amber overlay text when not live
"""
from __future__ import annotations

import math
import time
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from shared.encoder import face_distance

# ── Liveness tuning ───────────────────────────────────────────────────────────
EAR_CLOSED       = 0.25    # EAR below this → eye closed
BLINK_MIN_FRAMES = 2       # consecutive closed frames = one blink
LIVENESS_WINDOW  = 3.0     # seconds; need ≥1 blink inside window to be "live"

# ── Head-yaw challenge ────────────────────────────────────────────────────────
CHALLENGE_ENABLED = True
CHALLENGE_RESET   = 60     # seconds between challenge resets
YAW_THRESHOLD     = 15     # degrees off-centre = turned enough

# ── InsightFace ArcFace cosine distance threshold ─────────────────────────────
# 0 = identical, 1 = orthogonal.
# Same person:      distance < 0.4   (cosine similarity > 0.6)
# Different person: distance > 0.55
MATCH_DISTANCE = 0.40

# ── MediaPipe eye landmark indices (refined mesh) ─────────────────────────────
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]
_NOSE_TIP  = 1
_LEFT_EAR  = 234
_RIGHT_EAR = 454


# ─────────────────────────────────────────────────────────────────────────────
# Face Landmarker model bundle — auto-download (mirrors InsightFace's pattern)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_DIR = Path.home() / ".face_attendance_models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_landmarker_model() -> str:
    """Download the Face Landmarker model bundle on first run. Returns its path."""
    if not _MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[Recognizer] Downloading Face Landmarker model to {_MODEL_PATH} …")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the MediaPipe Face Landmarker model "
                f"({exc}). Download it manually from {_MODEL_URL} and save "
                f"it to {_MODEL_PATH}."
            ) from exc
    return str(_MODEL_PATH)


_landmarker = None

def _get_landmarker():
    """Lazy singleton — one FaceLandmarker instance reused across all frames."""
    global _landmarker
    if _landmarker is None:
        model_path = _ensure_landmarker_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options = mp_python.BaseOptions(model_asset_path=model_path),
            running_mode = mp_vision.RunningMode.IMAGE,
            num_faces    = 5,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _landmarker


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (unchanged — landmark.x/.y/.z interface is identical
# between the old Solutions API and the new Tasks API)
# ─────────────────────────────────────────────────────────────────────────────

def _ear(lms, indices: list[int], w: int, h: int) -> float:
    """Eye Aspect Ratio from 6 MediaPipe landmark indices."""
    pts = [(lms[i].x * w, lms[i].y * h) for i in indices]
    def _d(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
    A = _d(pts[1], pts[5]); B = _d(pts[2], pts[4]); C = _d(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 1.0


def _yaw_degrees(lms, w: int, h: int) -> float:
    """Yaw angle in degrees. Positive = looking right."""
    nose  = lms[_NOSE_TIP]
    l_ear = lms[_LEFT_EAR]
    r_ear = lms[_RIGHT_EAR]
    mid_x = (l_ear.x + r_ear.x) / 2.0
    span  = abs(r_ear.x - l_ear.x)
    if span < 1e-6: return 0.0
    ratio = (nose.x - mid_x) / span
    return math.degrees(math.asin(max(-1.0, min(1.0, ratio))))


# ─────────────────────────────────────────────────────────────────────────────
# Per-face liveness state (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class _FaceState:
    def __init__(self):
        self.consec_closed = 0
        self.blink_times: list[float] = []
        self.challenge_dir = ""
        self.challenge_met = False
        self.challenge_ts  = 0.0
        self._init_challenge()

    def _init_challenge(self):
        self.challenge_dir = random.choice(["left", "right"])
        self.challenge_met = False
        self.challenge_ts  = time.time()

    def update_blink(self, avg_ear: float) -> bool:
        blinked = False
        if avg_ear < EAR_CLOSED:
            self.consec_closed += 1
        else:
            if self.consec_closed >= BLINK_MIN_FRAMES:
                self.blink_times.append(time.time())
                blinked = True
            self.consec_closed = 0
        now = time.time()
        self.blink_times = [t for t in self.blink_times if now - t <= LIVENESS_WINDOW]
        return blinked

    def is_live(self) -> bool:
        now = time.time()
        self.blink_times = [t for t in self.blink_times if now - t <= LIVENESS_WINDOW]
        return bool(self.blink_times)

    def update_challenge(self, yaw: float) -> bool:
        if time.time() - self.challenge_ts > CHALLENGE_RESET:
            self._init_challenge()
        if not self.challenge_met:
            if self.challenge_dir == "left"  and yaw < -YAW_THRESHOLD:
                self.challenge_met = True
            if self.challenge_dir == "right" and yaw >  YAW_THRESHOLD:
                self.challenge_met = True
        return self.challenge_met


# ─────────────────────────────────────────────────────────────────────────────
# InsightFace singleton
# ─────────────────────────────────────────────────────────────────────────────

_insight_app = None

def _get_insight():
    global _insight_app
    if _insight_app is None:
        from insightface.app import FaceAnalysis
        _insight_app = FaceAnalysis(
            name      = "buffalo_sc",
            providers = ["CPUExecutionProvider"],
        )
        _insight_app.prepare(ctx_id=0, det_size=(640, 480))
    return _insight_app


# ─────────────────────────────────────────────────────────────────────────────
# Public Recognizer
# ─────────────────────────────────────────────────────────────────────────────

class Recognizer:
    """
    Detection + recognition + liveness — one object, one call per frame.

    Parameters
    ----------
    known_encodings : list of numpy 512-d arrays (InsightFace ArcFace)
    known_ids       : list of student ID strings (same order)
    """

    def __init__(self, known_encodings: list, known_ids: list):
        self.known_encodings = known_encodings
        self.known_ids       = known_ids
        self._states: dict[str, _FaceState] = {}

    def update_encodings(self, known_encodings: list, known_ids: list) -> None:
        """Hot-reload encodings without restarting."""
        self.known_encodings = known_encodings
        self.known_ids       = known_ids

    def process(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Run InsightFace detection + ArcFace encoding + MediaPipe liveness
        on one BGR frame.

        Returns list of dicts:
            box        : (x1, y1, x2, y2)
            student_id : str | None
            live       : bool
            prompt     : str | None
        """
        h, w = bgr_frame.shape[:2]
        out  = []

        # ── 1. InsightFace: detect all faces + get 512-d ArcFace embeddings ───
        #    InsightFace takes BGR natively — no colour conversion needed.
        try:
            insight_faces = _get_insight().get(bgr_frame)
        except Exception:
            insight_faces = []

        # ── 2. MediaPipe Tasks API: full-frame Face Landmarker for EAR + yaw ──
        rgb_full = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_full)
        try:
            mesh_result = _get_landmarker().detect(mp_image)
            mesh_faces  = mesh_result.face_landmarks   # list[list[NormalizedLandmark]]
        except Exception:
            mesh_faces = []

        for idx, iface in enumerate(insight_faces):

            # Bounding box (float → int)
            x1, y1, x2, y2 = (int(v) for v in iface.bbox)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            enc = iface.normed_embedding          # 512-d, L2-normalised

            # ── Match ─────────────────────────────────────────────────────────
            student_id: str | None = None
            if self.known_encodings and enc is not None:
                dists = face_distance(self.known_encodings, enc)
                best  = int(np.argmin(dists))
                if dists[best] < MATCH_DISTANCE:
                    student_id = self.known_ids[best]

            face_key = student_id or f"anon_{idx}"
            if face_key not in self._states:
                self._states[face_key] = _FaceState()

            state  = self._states[face_key]
            live   = False
            prompt: str | None = None

            # ── Liveness (MediaPipe mesh) ──────────────────────────────────────
            if idx < len(mesh_faces):
                lms   = mesh_faces[idx]   # Tasks API: list IS the landmarks (no .landmark needed)
                l_ear = _ear(lms, _LEFT_EYE,  w, h)
                r_ear = _ear(lms, _RIGHT_EYE, w, h)
                state.update_blink((l_ear + r_ear) / 2.0)

                if not state.is_live():
                    prompt = "Please blink to verify"
                elif CHALLENGE_ENABLED:
                    yaw = _yaw_degrees(lms, w, h)
                    state.update_challenge(yaw)
                    if not state.challenge_met:
                        prompt = f"Look {state.challenge_dir}"
                    else:
                        live = True
                else:
                    live = True
            else:
                prompt = "Please blink to verify"

            out.append({
                "box":        (x1, y1, x2, y2),
                "student_id": student_id,
                "live":       live,
                "prompt":     prompt,
            })

        return out
