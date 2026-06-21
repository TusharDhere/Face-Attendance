"""
kiosk/registration.py
─────────────────────
Standalone Dear PyGui student-registration window.

Changes from v1:
  • face_recognition / dlib  → InsightFace (Python 3.14 compatible)
  • Firebase Storage         → Cloudinary (free, no billing)
  • Averages 10 ArcFace embeddings and re-normalises before storing

Run standalone:
    python -m kiosk.registration
"""
from __future__ import annotations

import os
import sys
import time
import threading
import tempfile

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.firebase_config import get_firebase
from shared.storage import upload_student_photo

# ── Constants ──────────────────────────────────────────────────────────────────
CAM_W, CAM_H   = 320, 240
WIN_W, WIN_H   = 740, 560
CAPTURE_FRAMES = 10
CAPTURE_DELAY  = 0.5

# ── DPG colours ────────────────────────────────────────────────────────────────
C_BG    = (248, 249, 250, 255)
C_PRI   = (15,  23,  42,  255)
C_SEC   = (100, 116, 139, 255)
C_GREEN = (34,  197,  94, 255)
C_RED   = (239,  68,  68, 255)
C_AMBER = (245, 158,  11, 255)
C_BTN   = (15,  23,  42,  255)
C_BTN_T = (255, 255, 255, 255)

# ── InsightFace singleton ──────────────────────────────────────────────────────
_iface_app = None

def _get_iface():
    global _iface_app
    if _iface_app is None:
        from insightface.app import FaceAnalysis
        _iface_app = FaceAnalysis(name="buffalo_sc",
                                  providers=["CPUExecutionProvider"])
        # det_size MUST have both dimensions divisible by 32 — SCRFD's three
        # feature-pyramid strides (8/16/32) generate mismatched anchor counts
        # otherwise (e.g. 240 → ValueError: shapes (140,) (160,) mismatch).
        # InsightFace internally letterboxes the actual camera frame to fit
        # this size, so it's fine for det_size to exceed CAM_W/CAM_H.
        _iface_app.prepare(ctx_id=0, det_size=(640, 480))
    return _iface_app


class RegistrationWindow:
    """GUI registration workflow — no terminal needed."""

    def __init__(self):
        self._db, _, self._fs = get_firebase()
        self._cap: cv2.VideoCapture | None = None
        self._lock        = threading.Lock()
        self._tex_flat    = np.zeros(CAM_H * CAM_W * 4, dtype=np.float32)
        self._cam_running = False

    # ── Camera preview ─────────────────────────────────────────────────────────

    def _preview_loop(self):
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        while self._cam_running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.033); continue
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            flat = (rgba.astype(np.float32) / 255.0).flatten()
            with self._lock:
                self._tex_flat = flat
        self._cap.release(); self._cap = None

    def _start_preview(self):
        if not self._cam_running:
            self._cam_running = True
            threading.Thread(target=self._preview_loop, daemon=True).start()

    def _stop_preview(self):
        self._cam_running = False

    def _tick_preview(self):
        if self._cam_running and dpg.does_item_exist("reg_cam_tex"):
            with self._lock:
                dpg.set_value("reg_cam_tex", self._tex_flat.copy())

    # ── Status helper ──────────────────────────────────────────────────────────

    def _status(self, msg: str, colour=None):
        colour = colour or C_SEC
        if dpg.does_item_exist("reg_status"):
            dpg.configure_item("reg_status", default_value=msg, color=colour)

    # ── Capture + encode + upload ──────────────────────────────────────────────

    def _run_capture(self, sid: str, name: str, major: str,
                     year: int, starting_year: int,
                     email: str, parent_email: str):
        """Background thread: capture frames → InsightFace encode → Cloudinary upload → write DB."""

        self._start_preview()
        time.sleep(0.8)   # camera warm-up

        embeddings: list[np.ndarray] = []
        best_frame: np.ndarray | None = None

        app = _get_iface()

        for i in range(CAPTURE_FRAMES):
            self._status(f"Capturing frame {i+1}/{CAPTURE_FRAMES}…", C_AMBER)
            time.sleep(CAPTURE_DELAY)

            with self._lock:
                flat = self._tex_flat.copy()

            rgba  = (flat.reshape(CAM_H, CAM_W, 4) * 255).astype(np.uint8)
            frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            faces = app.get(frame)

            if faces:
                f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                embeddings.append(f.normed_embedding.copy())
                if best_frame is None:
                    best_frame = frame.copy()

        self._stop_preview()

        if not embeddings:
            self._status("No face detected. Ensure good lighting, face the camera.", C_RED)
            return

        # Average embeddings and re-normalise (required for cosine distance)
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm
        encoding: list[float] = avg.tolist()

        # Save best frame to temp file
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, best_frame)

            # Upload to Cloudinary
            self._status("Uploading photo to Cloudinary…", C_AMBER)
            upload_student_photo(sid, tmp_path)

            # Store 512-d encoding in Firestore
            self._status("Storing face encoding…", C_AMBER)
            self._fs.collection("students").document(sid).set(
                {"face_encoding": encoding}, merge=True
            )

            # Write student record to Realtime DB
            self._status("Writing student record…", C_AMBER)
            self._db.reference(f"Students/{sid}").set({
                "name":                 name,
                "major":                major,
                "year":                 year,
                "starting year":        starting_year,
                "total_attendance":     0,
                "standing":             "G",
                "last_attendance_time": "2000-01-01 00:00:00",
                "email":                email,
                "parent_email":         parent_email,
            })

            n = len(embeddings)
            self._status(f"✓ {name} registered! ({n}/{CAPTURE_FRAMES} frames used)", C_GREEN)

            # Clear form fields
            for tag in ("reg_sid","reg_name","reg_major","reg_email","reg_pemail"):
                if dpg.does_item_exist(tag): dpg.set_value(tag, "")

        except Exception as exc:
            self._status(f"Error: {exc}", C_RED)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Submit callback ────────────────────────────────────────────────────────

    def _on_submit(self, sender, app_data, user_data):
        sid          = (dpg.get_value("reg_sid")   or "").strip()
        name         = (dpg.get_value("reg_name")  or "").strip()
        major        = (dpg.get_value("reg_major") or "").strip()
        year         = int(dpg.get_value("reg_year") or 1)
        starting_yr  = int(dpg.get_value("reg_syr")  or 2024)
        email        = (dpg.get_value("reg_email")  or "").strip()
        parent_email = (dpg.get_value("reg_pemail") or "").strip()

        if not sid:   self._status("Student ID is required.", C_RED); return
        if not name:  self._status("Full name is required.", C_RED);  return

        self._status("Starting capture…", C_AMBER)
        threading.Thread(
            target=self._run_capture,
            args=(sid, name, major, year, starting_yr, email, parent_email),
            daemon=True,
        ).start()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def open(self):
        if dpg.does_item_exist("reg_win"): dpg.delete_item("reg_win")

        blank = np.zeros(CAM_H * CAM_W * 4, dtype=np.float32)
        if not dpg.does_item_exist("reg_cam_tex"):
            with dpg.texture_registry():
                dpg.add_raw_texture(width=CAM_W, height=CAM_H,
                                    default_value=blank.tolist(),
                                    format=dpg.mvFormat_Float_rgba,
                                    tag="reg_cam_tex")

        with dpg.window(label="Register New Student", tag="reg_win",
                        width=WIN_W, height=WIN_H, pos=(WIN_W//6, 60),
                        no_collapse=True, on_close=self._stop_preview):
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):

                # Form
                with dpg.group(width=370):
                    def _row(lbl, tag, hint=""):
                        dpg.add_text(lbl, color=C_SEC)
                        dpg.add_input_text(tag=tag, hint=hint, width=340)
                        dpg.add_spacer(height=6)

                    _row("Student ID *",  "reg_sid",   "e.g. 254527")
                    _row("Full Name *",   "reg_name",  "e.g. Aarav Shah")
                    _row("Major",         "reg_major", "e.g. Computer Science")
                    dpg.add_text("Year", color=C_SEC)
                    dpg.add_input_int(tag="reg_year", default_value=1, min_value=1, max_value=6, width=110)
                    dpg.add_spacer(height=6)
                    dpg.add_text("Starting Year", color=C_SEC)
                    dpg.add_input_int(tag="reg_syr", default_value=2024, min_value=2000, max_value=2100, width=110)
                    dpg.add_spacer(height=6)
                    _row("Student Email (optional)",  "reg_email",  "student@uni.edu")
                    _row("Parent Email (optional)",   "reg_pemail", "parent@email.com")
                    dpg.add_spacer(height=10)

                    dpg.add_button(label="  Start Capture (10 frames)  ", tag="reg_btn",
                                   width=340, height=36, callback=self._on_submit)

                    with dpg.theme() as _bt:
                        with dpg.theme_component(dpg.mvButton):
                            dpg.add_theme_color(dpg.mvThemeCol_Button,        C_BTN)
                            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (40,60,80,255))
                            dpg.add_theme_color(dpg.mvThemeCol_Text,          C_BTN_T)
                            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                    dpg.bind_item_theme("reg_btn", _bt)

                    dpg.add_spacer(height=10)
                    dpg.add_text("", tag="reg_status", color=C_SEC, wrap=340)

                # Preview
                with dpg.group():
                    dpg.add_spacer(width=12)
                    dpg.add_text("Live Preview", color=C_SEC)
                    dpg.add_spacer(height=4)
                    dpg.add_image("reg_cam_tex", width=CAM_W, height=CAM_H, tag="reg_preview")
                    dpg.add_spacer(height=6)
                    dpg.add_text(f"Auto-captures {CAPTURE_FRAMES} frames · "
                                 f"{CAPTURE_DELAY:.1f}s apart", color=C_SEC)
                    dpg.add_text("Uses InsightFace ArcFace (512-d)", color=C_SEC)


# ── Standalone entry point ─────────────────────────────────────────────────────

def main():
    dpg.create_context()
    dpg.create_viewport(title="Student Registration", width=WIN_W+40, height=WIN_H+40)
    dpg.setup_dearpygui(); dpg.show_viewport()
    win = RegistrationWindow(); win.open()
    while dpg.is_dearpygui_running():
        win._tick_preview()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
