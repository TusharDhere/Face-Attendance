"""
admin/auth.py
─────────────
Single-admin authentication via flask-login.

Credentials are read from .env:
    ADMIN_USER=admin
    ADMIN_PASS=your-password

Session lifetime is set to 8 hours in create_app().
The /login route handles both GET (show form) and POST (verify + redirect).
"""
from __future__ import annotations

import os

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, session,
)
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user,
    login_required, current_user,
)

auth_bp   = Blueprint("auth", __name__)
login_mgr = LoginManager()


# ── Singleton admin user ──────────────────────────────────────────────────────

class _AdminUser(UserMixin):
    """There is exactly one admin account; its id is always '1'."""
    id = "1"

    @property
    def username(self) -> str:
        return os.getenv("ADMIN_USER", "admin")


_ADMIN = _AdminUser()


@login_mgr.user_loader
def _load_user(user_id: str) -> _AdminUser | None:
    return _ADMIN if user_id == "1" else None


@login_mgr.unauthorized_handler
def _on_unauthorized():
    """Redirect unauthenticated requests to /login, preserving the target URL."""
    return redirect(url_for("auth.login", next=request.path))


# ── Routes ────────────────────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("attendance.dashboard"))

    error: str | None = None

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password =  request.form.get("password") or ""

        expected_user = os.getenv("ADMIN_USER", "admin")
        expected_pass = os.getenv("ADMIN_PASS", "")

        if not expected_pass:
            error = "ADMIN_PASS is not set in .env — cannot log in."
        elif username == expected_user and password == expected_pass:
            session.permanent = True          # honour PERMANENT_SESSION_LIFETIME (8 h)
            login_user(_ADMIN, remember=False)
            target = request.args.get("next") or url_for("attendance.dashboard")
            # Safety: only allow relative redirects
            if not target.startswith("/"):
                target = url_for("attendance.dashboard")
            return redirect(target)
        else:
            error = "Incorrect username or password."

    return render_template("login.html", error=error)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
