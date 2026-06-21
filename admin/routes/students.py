"""
admin/routes/students.py
────────────────────────
Student CRUD + encoding-sync endpoints.

Changes from v1:
  • Firebase Storage  → Cloudinary (shared/storage.py)
  • face_recognition  → shared/encoder.py (InsightFace)
  • _run_sync iterates Realtime DB student IDs, re-downloads from Cloudinary
"""
from __future__ import annotations

import os
import tempfile
import threading

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, flash, jsonify,
)

from shared.firebase_config import get_firebase
from shared.encoder import generate_and_store_encoding

students_bp = Blueprint("students", __name__)

_ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

def _ext_ok(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in _ALLOWED_EXT


def _all_students() -> list[dict]:
    db, _, _ = get_firebase()
    raw = db.reference("Students").get() or {}
    out = []
    for sid, info in raw.items():
        att = info.get("total_attendance", 0)
        pct = min(int(att / 30 * 100), 100)
        out.append({
            "id":    sid,
            "name":  info.get("name",  ""),
            "major": info.get("major", ""),
            "year":  info.get("year",  ""),
            "att":   att,
            "pct":   pct,
            "email": info.get("email", ""),
        })
    return sorted(out, key=lambda s: s["name"].lower())


# ── Sync-encodings state machine ──────────────────────────────────────────────
_sync_lock  = threading.Lock()
_sync_state: dict = {"running": False, "done": 0, "total": 0, "ok": [], "errors": []}


def _run_sync():
    """
    Background worker: for every student in Realtime DB, download their photo
    from Cloudinary and re-generate the InsightFace encoding in Firestore.
    """
    global _sync_state

    db, _, _ = get_firebase()
    students  = db.reference("Students").get() or {}
    ids       = list(students.keys())

    with _sync_lock:
        _sync_state.update({"total": len(ids), "done": 0, "ok": [], "errors": []})

    for sid in ids:
        tmp_path = None
        try:
            from shared.storage import download_student_photo_to_file
            tmp_path = download_student_photo_to_file(sid)
            if tmp_path is None:
                raise RuntimeError("Photo not found in Cloudinary")
            generate_and_store_encoding(sid, tmp_path)
            with _sync_lock:
                _sync_state["done"] += 1
                _sync_state["ok"].append(sid)
        except Exception as exc:
            with _sync_lock:
                _sync_state["done"] += 1
                _sync_state["errors"].append({"id": sid, "error": str(exc)})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    with _sync_lock:
        _sync_state["running"] = False


@students_bp.route("/admin/sync-encodings", methods=["POST"])
def sync_encodings():
    global _sync_state
    with _sync_lock:
        if _sync_state.get("running"):
            return jsonify({"error": "Sync already in progress."}), 409
        _sync_state = {"running": True, "done": 0, "total": 0, "ok": [], "errors": []}
    threading.Thread(target=_run_sync, daemon=True).start()
    return jsonify({"started": True}), 202


@students_bp.route("/admin/sync-status")
def sync_status():
    with _sync_lock:
        state = dict(_sync_state)
    return jsonify(state)


# ── List ──────────────────────────────────────────────────────────────────────

@students_bp.route("/students")
def students_list():
    return render_template("students.html", students=_all_students())


# ── Add ───────────────────────────────────────────────────────────────────────

@students_bp.route("/students/add", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        sid          = request.form.get("student_id",    "").strip()
        name         = request.form.get("name",           "").strip()
        major        = request.form.get("major",          "").strip()
        year         = int(request.form.get("year",        1))
        starting_yr  = int(request.form.get("starting_year", 2024))
        email        = request.form.get("email",          "").strip()
        parent_email = request.form.get("parent_email",  "").strip()
        photo        = request.files.get("photo")

        if not sid or not name:
            flash("Student ID and Name are required.", "error")
            return redirect(url_for("students.add_student"))

        db, _, _ = get_firebase()
        db.reference(f"Students/{sid}").set({
            "name":                 name,
            "major":                major,
            "year":                 year,
            "starting year":        starting_yr,
            "total_attendance":     0,
            "standing":             "G",
            "last_attendance_time": "2000-01-01 00:00:00",
            "email":                email,
            "parent_email":         parent_email,
        })

        if photo and photo.filename and _ext_ok(photo.filename):
            _upload_and_encode(sid, photo)

        flash(f"Student '{name}' added successfully.", "success")
        return redirect(url_for("students.students_list"))

    return render_template("add_student.html", student=None)


# ── Edit ──────────────────────────────────────────────────────────────────────

@students_bp.route("/students/edit/<sid>", methods=["GET", "POST"])
def edit_student(sid: str):
    db, _, _ = get_firebase()

    if request.method == "POST":
        name         = request.form.get("name",          "").strip()
        major        = request.form.get("major",         "").strip()
        year         = int(request.form.get("year",       1))
        starting_yr  = int(request.form.get("starting_year", 2024))
        email        = request.form.get("email",         "").strip()
        parent_email = request.form.get("parent_email", "").strip()
        photo        = request.files.get("photo")

        db.reference(f"Students/{sid}").update({
            "name":          name,
            "major":         major,
            "year":          year,
            "starting year": starting_yr,
            "email":         email,
            "parent_email":  parent_email,
        })

        if photo and photo.filename and _ext_ok(photo.filename):
            _upload_and_encode(sid, photo)

        flash("Student updated.", "success")
        return redirect(url_for("students.students_list"))

    info    = db.reference(f"Students/{sid}").get() or {}
    student = {"id": sid, **info}
    return render_template("add_student.html", student=student)


# ── Delete ────────────────────────────────────────────────────────────────────

@students_bp.route("/students/delete/<sid>", methods=["POST"])
def delete_student(sid: str):
    db, _, fs = get_firebase()

    db.reference(f"Students/{sid}").delete()

    try:
        fs.collection("students").document(sid).delete()
    except Exception:
        pass

    # Delete photo from Cloudinary
    try:
        from shared.storage import delete_student_photo
        delete_student_photo(sid)
    except Exception:
        pass

    flash("Student deleted.", "success")
    return redirect(url_for("students.students_list"))


# ── Upload helper (Cloudinary) ────────────────────────────────────────────────

def _upload_and_encode(sid: str, file_storage) -> None:
    """Save uploaded file → Cloudinary → generate InsightFace encoding."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        file_storage.save(tmp_path)
        from shared.storage import upload_student_photo
        upload_student_photo(sid, tmp_path)
        generate_and_store_encoding(sid, tmp_path)
    except Exception as exc:
        flash(f"Photo upload/encoding failed: {exc}", "error")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
