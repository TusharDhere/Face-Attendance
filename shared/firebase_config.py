"""
shared/firebase_config.py
─────────────────────────
Single Firebase initialisation point for the entire project.

Note: Firebase Storage is NOT used by this project — photos are stored on
Cloudinary (see shared/storage.py) to avoid Firebase's Blaze billing plan
requirement. FIREBASE_BUCKET is therefore optional; only set it if you
specifically need the storage_admin module for something else.

A NOTE ON FIRESTORE DATABASE MODE
──────────────────────────────────
If your project's `(default)` Firestore database was ever provisioned in
"Datastore Mode" (common on older/App-Engine-touched projects), the modern
Firestore API used here cannot read it — you'll see:

    FailedPrecondition: 400 The Cloud Firestore API is not available for
    Firestore in Datastore Mode database projects/.../databases/(default).

This can't be fixed in code or switched in place. The fix is to create a
*named* database in Native mode (Firebase console → Firestore Database →
Add database → Firestore in Native mode → give it an ID, e.g.
"face-attendance") and set FIRESTORE_DATABASE_ID in .env to that ID.
Leave FIRESTORE_DATABASE_ID unset if your (default) database is already
in Native mode — most new projects are fine and need no change here.

Usage
-----
    from shared.firebase_config import get_firebase
    db, storage, fs = get_firebase()
    # db      → firebase_admin.db   (Realtime Database)
    # storage → firebase_admin.storage  (unused by this project — kept for API parity)
    # fs      → google.cloud.firestore.Client
"""
from __future__ import annotations

import os

import firebase_admin
from firebase_admin import credentials, db, storage, firestore
from dotenv import load_dotenv

load_dotenv()

_app: firebase_admin.App | None = None
_fs_client = None


def get_firebase():
    """
    Lazily initialises Firebase on first call; all subsequent calls are
    instant no-ops that return the same cached objects.

    Returns
    -------
    db       : firebase_admin.db module  (Realtime Database helpers)
    storage  : firebase_admin.storage module (unused — Cloudinary handles photos)
    fs       : firestore.Client
    """
    global _app, _fs_client

    if _app is None:
        key_path = os.getenv("FIREBASE_KEY_PATH", "serviceAccountKey.json")
        db_url   = os.getenv("FIREBASE_DB_URL")
        bucket   = os.getenv("FIREBASE_BUCKET")            # optional — Cloudinary handles photos
        fs_db_id = os.getenv("FIRESTORE_DATABASE_ID")       # optional — see module docstring

        if not db_url:
            raise EnvironmentError(
                "FIREBASE_DB_URL is not set. Add it to .env "
                "(e.g. https://YOUR-PROJECT-default-rtdb.firebaseio.com/)"
            )

        cred = credentials.Certificate(key_path)
        options = {"databaseURL": db_url}
        if bucket:
            options["storageBucket"] = bucket   # only included if explicitly set

        _app = firebase_admin.initialize_app(cred, options)

        if fs_db_id:
            _fs_client = firestore.client(_app, database_id=fs_db_id)
        else:
            _fs_client = firestore.client(_app)

    return db, storage, _fs_client
