"""
shared/storage.py
─────────────────
Cloudinary photo storage — replaces Firebase Storage (no billing required).

Free tier: 25 GB storage + 25 GB bandwidth/month, no credit card needed.
Sign up → https://cloudinary.com/users/register/free

Add to .env:
    CLOUDINARY_CLOUD_NAME = your_cloud_name
    CLOUDINARY_API_KEY    = your_api_key
    CLOUDINARY_API_SECRET = your_api_secret

Public API
──────────
  upload_student_photo(student_id, local_path) → secure_url
  fetch_student_photo(student_id, size)         → BGR np.ndarray | None
  download_student_photo_to_file(student_id)    → temp_path | None
  delete_student_photo(student_id)              → None
"""
from __future__ import annotations

import os
import tempfile
import urllib.request

import cv2
import numpy as np
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary import CloudinaryImage
from dotenv import load_dotenv

load_dotenv()

_FOLDER     = "face_attendance/students"
_configured = False


def _cfg() -> None:
    """Lazy-initialise Cloudinary once from .env values."""
    global _configured
    if not _configured:
        name   = os.getenv("CLOUDINARY_CLOUD_NAME")
        key    = os.getenv("CLOUDINARY_API_KEY")
        secret = os.getenv("CLOUDINARY_API_SECRET")
        if not (name and key and secret):
            raise EnvironmentError(
                "Cloudinary credentials missing. Set CLOUDINARY_CLOUD_NAME, "
                "CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET in .env"
            )
        cloudinary.config(cloud_name=name, api_key=key, api_secret=secret, secure=True)
        _configured = True


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

def upload_student_photo(student_id: str, local_path: str) -> str:
    """
    Upload *local_path* to Cloudinary under face_attendance/students/{student_id}.
    Overwrites any previous photo for the same student.
    Returns the secure HTTPS URL.
    """
    _cfg()
    result = cloudinary.uploader.upload(
        local_path,
        public_id   = f"{_FOLDER}/{student_id}",
        overwrite   = True,
        resource_type = "image",
        tags        = ["face_attendance", "student"],
    )
    return result["secure_url"]


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_student_photo_url(student_id: str, width: int = 80, height: int = 80) -> str:
    """
    Return a Cloudinary transformation URL: face-centred square crop.
    Cloudinary auto-formats (WebP / AVIF) for optimal delivery.
    """
    _cfg()
    return CloudinaryImage(f"{_FOLDER}/{student_id}").build_url(
        width       = width,
        height      = height,
        crop        = "fill",
        gravity     = "face",
        fetch_format= "auto",
        quality     = "auto",
    )


def fetch_student_photo(student_id: str, size: int = 80) -> np.ndarray | None:
    """
    Download the student's avatar from Cloudinary and return as a BGR
    numpy array suitable for the kiosk card display.
    Returns None on any error (network, missing image, decode failure).
    """
    try:
        _cfg()
        url = get_student_photo_url(student_id, size, size)
        req = urllib.request.Request(
            url, headers={"User-Agent": "FaceAttendance/2.0"}
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            arr = np.frombuffer(resp.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        return None


def download_student_photo_to_file(student_id: str, size: int = 400) -> str | None:
    """
    Download a higher-resolution version to a temporary file.
    Returns the absolute path of the temp file (caller must delete it),
    or None if the download fails.
    Used by the sync-encodings job.
    """
    try:
        _cfg()
        url = get_student_photo_url(student_id, size, size)
        req = urllib.request.Request(
            url, headers={"User-Agent": "FaceAttendance/2.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(data)
            return tmp.name
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Delete
# ─────────────────────────────────────────────────────────────────────────────

def delete_student_photo(student_id: str) -> None:
    """Remove the student's photo from Cloudinary (best-effort, silent on error)."""
    try:
        _cfg()
        cloudinary.uploader.destroy(f"{_FOLDER}/{student_id}")
    except Exception:
        pass
