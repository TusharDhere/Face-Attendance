"""
kiosk/main.py
─────────────
Dear PyGui face-attendance kiosk.

Layout
──────
  ┌──────────────────────────────┬──────────────────┐
  │  Left panel  60%             │  Right panel 40% │
  │  Live camera feed            │  Student card    │
  │  (boxes drawn on OpenCV      │  or idle state   │
  │   frame before upload)       │                  │
  └──────────────────────────────┴──────────────────┘

Run from the project root:
    python -m kiosk.main
"""
from __future__ import annotations

import os
import sys
import time
import threading
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiosk.recognizer import Recognizer
from shared.firebase_config import get_firebase
from shared.encoder import load_all_encodings

# ── Layout ───────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 720
L_W          = int(WIN_W * 0.6)          # 768 – left (camera) panel
R_W          = WIN_W - L_W               # 512 – right (info) panel
CAM_W, CAM_H = 640, 480
IMG_W        = L_W                       # camera display width
IMG_H        = int(L_W * CAM_H / CAM_W) # maintain 4:3 aspect ratio

# ── Cooldown ─────────────────────────────────────────────────────────────────
COOLDOWN_SECS   = 30     # seconds between attendance marks per student
CARD_HOLD_SECS  = 6      # seconds to hold the info card visible

# ── OpenCV draw colours (BGR) ─────────────────────────────────────────────────
CV_GREEN = (34, 197, 94)
CV_RED   = (68,  68, 239)
CV_AMBER = (11, 158, 245)

# ── Dear PyGui colours (RGBA 0-255) ──────────────────────────────────────────
C_BG       = (248, 249, 250, 255)
C_BORDER   = (226, 232, 240, 255)
C_PRI      = (15,  23,  42,  255)
C_SEC      = (100, 116, 139, 255)
C_GREEN    = (34,  197,  94, 255)
C_AMBER    = (245, 158,  11, 255)
C_RED      = (239,  68,  68, 255)
C_WHITE    = (255, 255, 255, 255)


def _circular_rgba(img_bgr: np.ndarray, size: int = 80) -> np.ndarray:
    """Resize to size×size and apply a circular alpha mask. Returns RGBA float32 flat."""
    resized = cv2.resize(img_bgr, (size, size))
    mask    = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 1, 255, -1)
    rgba    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    return (rgba.astype(np.float32) / 255.0).flatten()


class KioskApp:
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self):
        print("[Kiosk] Connecting to Firebase …")
        self._db, _, self._fs = get_firebase()   # storage unused — photos via Cloudinary

        print("[Kiosk] Loading face encodings from Firestore …")
        enc, ids = load_all_encodings()
        self._recognizer = Recognizer(enc, ids)
        print(f"[Kiosk] Loaded {len(ids)} encoding(s): {ids}")

        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

        # ── Shared state (guarded by _lock) ──────────────────────────────────
        self._lock          = threading.Lock()
        self._tex_flat      = np.zeros(CAM_H * CAM_W * 4, dtype=np.float32)
        self._face_results: list[dict] = []
        self._card_info: dict | None   = None
        self._card_timer: float        = 0.0

        # ── Rate-limiting ─────────────────────────────────────────────────────
        self._last_att:    dict[str, datetime] = {}   # last attendance mark
        self._last_card:   dict[str, float]    = {}   # last card display

        # ── Pulse animation ───────────────────────────────────────────────────
        self._pulse = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Camera / recognition thread
    # ─────────────────────────────────────────────────────────────────────────

    def _camera_loop(self):
        while True:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.033)
                continue

            results = self._recognizer.process(frame)
            drawn   = frame.copy()

            for r in results:
                x1, y1, x2, y2 = (int(v) for v in r["box"])
                sid  = r["student_id"]
                live = r["live"]
                pmt  = r["prompt"]

                # Bounding-box colour
                if sid and live:
                    colour = CV_GREEN
                elif pmt:
                    colour = CV_AMBER
                else:
                    colour = CV_RED

                cv2.rectangle(drawn, (x1, y1), (x2, y2), colour, 2)

                label = (sid if sid and live
                         else (pmt if pmt
                               else "Unknown"))
                if label:
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(drawn,
                                  (x1, y1 - th - 10), (x1 + tw + 8, y1),
                                  colour, cv2.FILLED)
                    cv2.putText(drawn, label,
                                (x1 + 4, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)

                if sid and live:
                    self._schedule_match(sid)

            # Convert to RGBA float32 for DPG texture
            rgba = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGBA)
            flat = (rgba.astype(np.float32) / 255.0).flatten()

            with self._lock:
                self._tex_flat    = flat
                self._face_results = results

    # ─────────────────────────────────────────────────────────────────────────
    # Attendance logic
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_match(self, student_id: str):
        """Rate-limit card display to once per CARD_HOLD_SECS per student."""
        now_ts = time.time()
        if student_id in self._last_card and \
                (now_ts - self._last_card[student_id]) < CARD_HOLD_SECS:
            return
        self._last_card[student_id] = now_ts

        now    = datetime.now()
        last   = self._last_att.get(student_id)
        in_cd  = last and (now - last).total_seconds() < COOLDOWN_SECS

        if not in_cd:
            self._last_att[student_id] = now
            threading.Thread(
                target=self._write_attendance,
                args=(student_id, now),
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._fetch_card,
                args=(student_id, True),
                daemon=True,
            ).start()

    def _write_attendance(self, student_id: str, ts: datetime):
        try:
            ref  = self._db.reference(f"Students/{student_id}")
            info = ref.get()
            if not info:
                print(f"[Attendance] No DB record for {student_id}")
                return

            info["total_attendance"] = info.get("total_attendance", 0) + 1
            ref.child("total_attendance").set(info["total_attendance"])
            ref.child("last_attendance_time").set(ts.strftime("%Y-%m-%d %H:%M:%S"))

            # Log to Firestore attendance collection
            self._fs.collection("attendance").add({
                "student_id": student_id,
                "name":       info.get("name", ""),
                "date":       ts.strftime("%Y-%m-%d"),
                "time":       ts.strftime("%H:%M:%S"),
                "status":     "Present",
            })

            photo = self._fetch_photo(student_id)
            with self._lock:
                self._card_info  = {**info, "id": student_id,
                                    "photo": photo, "already_marked": False}
                self._card_timer = time.time()

            # Email notification (Phase 2)
            self._send_email(info, ts)

        except Exception as exc:
            print(f"[Attendance] Error for {student_id}: {exc}")

    def _fetch_card(self, student_id: str, already_marked: bool):
        try:
            info  = self._db.reference(f"Students/{student_id}").get() or {}
            photo = self._fetch_photo(student_id)
            with self._lock:
                self._card_info  = {**info, "id": student_id,
                                    "photo": photo, "already_marked": already_marked}
                self._card_timer = time.time()
        except Exception as exc:
            print(f"[Card] Fetch error for {student_id}: {exc}")

    def _fetch_photo(self, student_id: str) -> np.ndarray | None:
        """Download student avatar from Cloudinary (replaces Firebase Storage)."""
        try:
            from shared.storage import fetch_student_photo
            return fetch_student_photo(student_id, size=80)
        except Exception:
            return None

    def _send_email(self, info: dict, ts: datetime):
        host = os.getenv("SMTP_HOST")
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USER")
        pwd  = os.getenv("SMTP_PASS")
        to   = info.get("email") or info.get("parent_email")

        if not (host and user and pwd and to):
            return   # Email not configured — skip silently

        try:
            msg = MIMEText(
                f"Hello {info.get('name', '')},\n\n"
                f"Your attendance has been recorded.\n"
                f"Date:  {ts.strftime('%d %B %Y')}\n"
                f"Time:  {ts.strftime('%I:%M %p')}\n"
                f"Total sessions attended: {info.get('total_attendance', 0)}\n\n"
                f"— Face Attendance System"
            )
            msg["Subject"] = f"Attendance marked — {ts.strftime('%d %B %Y')}"
            msg["From"]    = user
            msg["To"]      = to

            with smtplib.SMTP(host, port) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(user, pwd)
                smtp.sendmail(user, to, msg.as_string())

            print(f"[Email] Sent to {to}")
        except Exception as exc:
            print(f"[Email] Failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Dear PyGui UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Global theme ──────────────────────────────────────────────────────
        with dpg.theme() as g_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,     C_BG)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,      C_BG)
                dpg.add_theme_color(dpg.mvThemeCol_Text,         C_PRI)
                dpg.add_theme_color(dpg.mvThemeCol_Border,       C_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, C_GREEN)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,  0, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,  8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,    8, 4)
        dpg.bind_theme(g_theme)

        # ── Texture registry ──────────────────────────────────────────────────
        blank_cam = np.zeros(CAM_H * CAM_W * 4, dtype=np.float32)
        blank_av  = np.ones(80 * 80 * 4, dtype=np.float32) * 0.78

        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=CAM_W, height=CAM_H,
                default_value=blank_cam.tolist(),
                format=dpg.mvFormat_Float_rgba,
                tag="cam_tex",
            )
            dpg.add_raw_texture(
                width=80, height=80,
                default_value=blank_av.tolist(),
                format=dpg.mvFormat_Float_rgba,
                tag="av_tex",
            )

        # ── Main window ───────────────────────────────────────────────────────
        with dpg.window(
            tag="main_win",
            no_title_bar=True, no_move=True, no_resize=True,
            no_scrollbar=True, no_scroll_with_mouse=True,
            width=WIN_W, height=WIN_H, pos=(0, 0),
        ):
            with dpg.group(horizontal=True):
                self._build_left_panel()
                self._build_right_panel()

    def _build_left_panel(self):
        with dpg.child_window(
            tag="left_panel",
            width=L_W, height=WIN_H,
            no_scrollbar=True, border=False,
        ):
            dpg.add_image(
                "cam_tex",
                width=IMG_W, height=IMG_H,
                tag="cam_image",
            )
            # Status bar
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=12)
                dpg.add_text("●", tag="status_dot",  color=C_GREEN)
                dpg.add_spacer(width=4)
                dpg.add_text("Scanning…", tag="status_txt", color=C_SEC)

    def _build_right_panel(self):
        with dpg.child_window(
            tag="right_panel",
            width=R_W - 2, height=WIN_H,
            no_scrollbar=True, border=True,
        ):
            # ── Idle state ────────────────────────────────────────────────────
            with dpg.group(tag="idle_grp"):
                dpg.add_spacer(height=180)
                _cx = R_W // 2 - 70
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=_cx)
                    with dpg.group():
                        dpg.add_text("Face Attendance",
                                     tag="idle_title", color=C_PRI)
                        dpg.add_spacer(height=4)
                        dpg.add_text("System Ready", color=C_SEC)
                dpg.add_spacer(height=24)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=_cx + 8)
                    dpg.add_text("● Scanning…", tag="idle_dot", color=C_GREEN)

            # ── Student card ──────────────────────────────────────────────────
            with dpg.group(tag="card_grp", show=False):
                # Avatar (circular)
                dpg.add_spacer(height=24)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=R_W // 2 - 44)
                    dpg.add_image("av_tex", width=80, height=80, tag="card_av")

                dpg.add_spacer(height=14)

                # Name
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=16)
                    dpg.add_text("", tag="card_name",
                                 color=C_PRI)

                dpg.add_spacer(height=2)

                # ID · Major · Year
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=16)
                    dpg.add_text("", tag="card_meta", color=C_SEC)

                dpg.add_spacer(height=18)

                # Divider
                with dpg.theme() as sep_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_color(dpg.mvThemeCol_Separator, C_BORDER)
                dpg.add_separator(tag="card_sep")
                dpg.bind_item_theme("card_sep", sep_theme)

                dpg.add_spacer(height=18)

                # Attendance count
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=16)
                    dpg.add_text("Total attendance:", color=C_SEC)
                    dpg.add_spacer(width=6)
                    dpg.add_text("", tag="card_count", color=C_PRI)

                dpg.add_spacer(height=8)

                # Progress bar
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=16)
                    dpg.add_progress_bar(
                        tag="card_bar",
                        default_value=0.0,
                        width=R_W - 48, height=10,
                    )

                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=16)
                    dpg.add_text("", tag="card_pct", color=C_SEC)

                dpg.add_spacer(height=28)

                # Status badge
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=R_W // 2 - 56)
                    dpg.add_text("", tag="card_badge", color=C_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-frame UI update (called from main thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _update(self):
        with self._lock:
            tex   = self._tex_flat
            card  = self._card_info
            ctimer = self._card_timer

        # Camera texture
        dpg.set_value("cam_tex", tex)

        # Pulse animation on idle dot
        self._pulse += 0.06
        alpha = int(100 + 155 * abs(np.sin(self._pulse)))
        dpg.configure_item("idle_dot", color=(34, 197, 94, alpha))
        dpg.configure_item("status_dot", color=(34, 197, 94, alpha))

        # Auto-hide card after CARD_HOLD_SECS
        if card and (time.time() - ctimer) > CARD_HOLD_SECS:
            with self._lock:
                self._card_info = None
            card = None

        if card:
            self._render_card(card)
        else:
            dpg.configure_item("idle_grp", show=True)
            dpg.configure_item("card_grp", show=False)

    def _render_card(self, info: dict):
        dpg.configure_item("idle_grp", show=False)
        dpg.configure_item("card_grp", show=True)

        # Avatar
        photo = info.get("photo")
        if photo is not None:
            av_flat = _circular_rgba(photo, 80)
            dpg.set_value("av_tex", av_flat)
        else:
            dpg.set_value("av_tex", np.ones(80*80*4, dtype=np.float32) * 0.78)

        # Text fields
        name  = info.get("name", "Unknown")
        sid   = info.get("id", "")
        major = info.get("major", "")
        year  = info.get("year", "")
        att   = info.get("total_attendance", 0)
        pct   = min(att / 30.0, 1.0)   # assume 30-session semester

        dpg.set_value("card_name",  name)
        dpg.set_value("card_meta",  f"ID: {sid}   ·   {major}   Year {year}")
        dpg.set_value("card_count", str(att))
        dpg.set_value("card_bar",   pct)
        dpg.set_value("card_pct",   f"{int(pct * 100)}%")

        if info.get("already_marked"):
            dpg.set_value("card_badge", "⏱  Already marked")
            dpg.configure_item("card_badge", color=C_AMBER)
        else:
            dpg.set_value("card_badge", "✓  PRESENT")
            dpg.configure_item("card_badge", color=C_GREEN)

    # ─────────────────────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        dpg.create_context()
        self._build_ui()

        dpg.create_viewport(
            title="Face Attendance Kiosk",
            width=WIN_W, height=WIN_H,
            resizable=False,
        )
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_win", True)
        dpg.show_viewport()

        # Launch camera thread
        threading.Thread(target=self._camera_loop, daemon=True).start()

        # Render loop (main thread)
        while dpg.is_dearpygui_running():
            self._update()
            dpg.render_dearpygui_frame()

        self._cap.release()
        dpg.destroy_context()


def main():
    app = KioskApp()
    app.run()


if __name__ == "__main__":
    main()
