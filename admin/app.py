"""
admin/app.py
────────────
Flask admin panel — create_app() factory.

New in v2:
  • flask-login with single admin account (ADMIN_USER / ADMIN_PASS from .env)
  • 8-hour permanent session
  • before_request guard: every route except /login + /static requires auth

Run from the project root:
    python -m admin.app
    flask --app admin.app run --debug
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, redirect, url_for, request
from dotenv import load_dotenv

load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.secret_key                 = os.getenv("FLASK_SECRET", "change-this-in-production")
    app.permanent_session_lifetime = timedelta(hours=8)

    # ── Flask-Login ───────────────────────────────────────────────────────
    from admin.auth import auth_bp, login_mgr
    login_mgr.init_app(app)
    login_mgr.login_view    = "auth.login"   # type: ignore[assignment]
    login_mgr.login_message = ""

    # ── Blueprints ────────────────────────────────────────────────────────
    from admin.routes.students   import students_bp
    from admin.routes.attendance import attendance_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(students_bp)
    app.register_blueprint(attendance_bp)

    # ── Explicit context processor so current_user is always in templates ─
    # (flask-login's init_app registers its own, but this guarantees it even
    #  when flask-login has unexpected initialisation order issues)
    @app.context_processor
    def _inject_user():
        try:
            from flask_login import current_user as _cu
            return dict(current_user=_cu)
        except Exception:
            return dict(current_user=type("Anon", (), {
                "is_authenticated": False, "username": ""
            })())

    # ── Auth guard: every route except auth.* and static requires login ───
    _PUBLIC = {"auth.login", "auth.logout", "static"}

    @app.before_request
    def _require_login():
        endpoint = request.endpoint or ""
        # Allow public endpoints and let Flask-Login handle the rest
        if endpoint in _PUBLIC:
            return None
        try:
            from flask_login import current_user as _cu
            if not _cu.is_authenticated:
                return redirect(url_for("auth.login", next=request.path))
        except Exception:
            return redirect(url_for("auth.login"))
        return None

    # ── Root redirect ─────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return redirect(url_for("attendance.dashboard"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000, host="0.0.0.0")