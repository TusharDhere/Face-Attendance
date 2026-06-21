"""
shared/encoder.py
─────────────────
Face-encoding helpers — powered by InsightFace + ONNX Runtime.

Why InsightFace instead of face_recognition / dlib?
  • dlib has no pre-built wheels for Python 3.12+ / 3.14
  • InsightFace uses ONNX Runtime: pure Python install, no C++ compiler
  • ArcFace (512-d) is significantly more accurate than dlib's 128-d model
  • The buffalo_sc model downloads automatically on first run (~85 MB)

Firestore schema (unchanged from v1 — only vector length changes 128→512):
    students/{student_id}/face_encoding → list[float]  (512 values, L2-normalised)

⚠️  If you have existing 128-d dlib encodings in Firestore, run
    "Sync Encodings" from the admin panel to regenerate them with InsightFace.
"""
from __future__ import annotations

import cv2
import numpy as np

from shared.firebase_config import get_firebase

# ── InsightFace lazy singleton ────────────────────────────────────────────────
_face_app = None

def _get_app():
    """
    Initialise InsightFace FaceAnalysis once, reuse for all calls.
    Model: buffalo_sc  (SCRFD-500MF detection + MBF recognition, ~85 MB total)
    For higher accuracy swap to 'buffalo_l' (~700 MB).
    """
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name      = "buffalo_sc",
            providers = ["CPUExecutionProvider"],   # use "CUDAExecutionProvider" for GPU
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 480))
    return _face_app


# ── Distance helper (replaces face_recognition.face_distance) ─────────────────

def face_distance(known_encodings: list[np.ndarray],
                  encoding: np.ndarray) -> np.ndarray:
    """
    Cosine distance between *encoding* and each vector in *known_encodings*.
    Because InsightFace returns L2-normalised embeddings, cosine similarity
    equals the dot product, and distance = 1 - similarity.

    Returns distances in [0, 2]:  0 = identical,  1 = orthogonal,  2 = opposite.
    Typical same-person distance: < 0.4
    Typical different-person distance: > 0.6
    """
    if not known_encodings:
        return np.array([])
    stack = np.stack(known_encodings)          # (N, 512)
    dots  = stack @ encoding                   # cosine similarity (already normalised)
    return (1.0 - dots).astype(np.float64)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_and_store_encoding(student_id: str, image_path: str) -> list[float]:
    """
    Detect the first face in *image_path*, compute its 512-d ArcFace embedding,
    and write it to Firestore > students > {student_id} > face_encoding.

    Returns the encoding list.

    Raises
    ------
    FileNotFoundError  if the image cannot be opened.
    RuntimeError       if no face is found in the image.
    """
    _, _, fs = get_firebase()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path!r}")

    faces = _get_app().get(img)
    if not faces:
        raise RuntimeError(
            f"No face detected in {image_path!r}. "
            "Use a well-lit, frontal photo with a single face."
        )

    # If multiple faces are detected, use the largest (most prominent) one
    face     = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    encoding = face.normed_embedding.tolist()     # 512 floats, already L2-normalised

    fs.collection("students").document(student_id).set(
        {"face_encoding": encoding}, merge=True
    )
    return encoding


def load_all_encodings() -> tuple[list[np.ndarray], list[str]]:
    """
    Fetch all Firestore student documents that have a face_encoding field.
    Returns (encoding_array_list, student_id_list).
    Called at kiosk startup instead of loading EncodeFile.p.
    """
    _, _, fs = get_firebase()

    enc_list: list[np.ndarray] = []
    id_list:  list[str]        = []

    for doc in fs.collection("students").stream():
        data = doc.to_dict()
        if data and "face_encoding" in data and data["face_encoding"]:
            enc_list.append(np.array(data["face_encoding"], dtype=np.float64))
            id_list.append(doc.id)

    return enc_list, id_list
