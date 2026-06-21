# Face Attendance System v2

A complete Python face-recognition attendance system with:

- **Kiosk** — Dear PyGui desktop app; live camera, multi-face recognition,
  blink + head-yaw liveness, per-student attendance logging.
- **Registration** — Dear PyGui sub-window; captures 10 frames, averages
  encodings, uploads photo, writes the DB record — no terminal needed.
- **Admin panel** — Flask web app; dashboard analytics, student CRUD,
  attendance log, CSV/Excel/PDF export, login-protected, email notifications.

---

## Project structure

```
FaceAttendance/
├── kiosk/
│   ├── main.py           ← Dear PyGui kiosk (run this on the attendance terminal)
│   ├── recognizer.py     ← InsightFace detection/matching + MediaPipe liveness
│   └── registration.py   ← GUI student-registration window
│
├── admin/
│   ├── app.py             ← Flask admin panel (login-protected)
│   ├── auth.py            ← single-admin login via flask-login
│   ├── routes/
│   │   ├── students.py    ← CRUD + Cloudinary upload + sync-encodings
│   │   └── attendance.py  ← dashboard, log, CSV/Excel/PDF export
│   └── templates/
│       ├── base.html
│       ├── login.html
│       ├── dashboard.html
│       ├── students.html
│       ├── add_student.html
│       ├── attendance.html
│       └── export_pdf.html
│
├── shared/
│   ├── firebase_config.py  ← single Firebase init (Realtime DB + Firestore only)
│   ├── storage.py          ← Cloudinary photo upload/download (replaces Storage)
│   └── encoder.py          ← InsightFace encoding generate/load via Firestore
│
├── AddDataToDatabase.py    ← one-time seed script
├── .env                    ← your credentials (never commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 1. Prerequisites

- Python 3.10, 3.11, 3.12, 3.13, or **3.14** (InsightFace has no dlib dependency)
- A webcam
- A **free** Firebase project (Realtime Database + Firestore only — Storage
  is not required, so you stay on the free Spark plan)
- A **free** Cloudinary account for photo storage

---

## 2. Installation

```bash
cd FaceAttendance

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

On first run, InsightFace automatically downloads its detection +
recognition model (`buffalo_sc`, ~85 MB) to `~/.insightface/models/`.
This happens once and requires an internet connection the first time only.

---

## 3. Firebase setup (Realtime DB + Firestore — no billing required)

1. Go to [Firebase console](https://console.firebase.google.com) → create a project.
2. **Project settings → Service accounts → Generate new private key**
   → save as `serviceAccountKey.json` in the project root.
3. In the console, enable **Realtime Database** and **Firestore Database**
   (leave both in test mode for development, lock down rules before
   production). You do **not** need to enable Storage.
4. Copy the values into `.env`:

```dotenv
FIREBASE_KEY_PATH=serviceAccountKey.json
FIREBASE_DB_URL=https://YOUR-PROJECT-default-rtdb.firebaseio.com/
```

`FIREBASE_BUCKET` is optional and can stay commented out — this project
never touches Firebase Storage.

---

## 4. Cloudinary setup (free photo storage)

1. Sign up free at [cloudinary.com/users/register/free](https://cloudinary.com/users/register/free)
   — no credit card needed.
2. On your Cloudinary dashboard, copy three values: **Cloud name**,
   **API Key**, **API Secret**.
3. Add them to `.env`:

```dotenv
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

Free tier limits: 25 GB storage, 25 GB bandwidth/month, 25 credits/month
for transformations (face-crop avatar generation uses a tiny fraction of
this for a typical class size).

---

## 5. Seed the database (optional)

```bash
python AddDataToDatabase.py
```

Adds one sample student to Realtime DB. Use the admin panel for real
student onboarding (it also uploads the photo and generates the encoding
in one step).

---

## 6. Run the kiosk

```bash
python -m kiosk.main
```

The window opens at 1280×720. Left panel = live camera feed with bounding
boxes. Right panel = student info card (shown once a recognised, live
face passes liveness checks).

**Liveness checks (anti-spoofing):**
- The student must **blink** at least once within 3 seconds.
- Then must **look left or look right**, as randomly prompted.
- Only after both pass is attendance recorded.
- A 30-second cooldown prevents duplicate entries per student.

**Recognition:** InsightFace's SCRFD detector finds all faces in the
frame; the ArcFace embedding (512-d) is matched against stored encodings
using cosine distance (threshold 0.40 — same person typically scores
< 0.4, different people > 0.55).

---

## 7. Register a new student (GUI)

```bash
python -m kiosk.registration
```

Fill in the form, click **Start Capture**. The window auto-captures 10
frames, averages their ArcFace embeddings (re-normalised), uploads the
best frame to Cloudinary, and writes the Realtime DB record — no
terminal or script editing needed.

**Restart the kiosk** afterward so it reloads encodings from Firestore
(or extend `kiosk/main.py` to call `recognizer.update_encodings(*load_all_encodings())`
on a timer for hot-reload).

---

## 8. Run the admin panel

```bash
python -m admin.app
# or
flask --app admin.app run --debug
```

Open [http://localhost:5000](http://localhost:5000) — you'll be redirected
to `/login`. Sign in with the `ADMIN_USER` / `ADMIN_PASS` you set in `.env`.
Sessions last 8 hours.

| Page | URL |
|---|---|
| Dashboard | `/dashboard` |
| Students | `/students` |
| Add student | `/students/add` |
| Attendance log | `/attendance?month=2026-06` |
| Export CSV | `/export/csv?month=2026-06` |
| Export Excel | `/export/excel` |
| Export PDF | `/export/pdf?month=2026-06` |

**Sync Encodings button** (Students page): re-downloads every student's
photo from Cloudinary and regenerates their InsightFace encoding in
Firestore. Use this if you ever migrate encodings (e.g. switching
InsightFace model sizes) or suspect drift.

---

## 9. Admin login

Set in `.env`:

```dotenv
FLASK_SECRET=replace-with-a-long-random-string
ADMIN_USER=admin
ADMIN_PASS=a-strong-password
```

Single hardcoded account, protected by `flask-login`, 8-hour session
lifetime, all routes guarded except `/login`.

---

## 10. Email notifications (optional)

Add SMTP credentials to `.env`:

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASS=your_16_char_app_password
```

> For Gmail, generate an **App Password** (not your account password) at
> <https://myaccount.google.com/apppasswords>.

Notifications are sent only when a student's record has a non-empty
`email` or `parent_email` field. Leave SMTP_* blank to disable entirely.


## 11. Architecture diagram

```
Camera feed
    │
    ▼
kiosk/recognizer.py
    ├── InsightFace (SCRFD detect + ArcFace 512-d encode, full BGR frame)
    ├── MediaPipe Face Mesh (blink EAR + head-yaw, full frame)
    │
    ├── unknown face     → red bbox, no action
    ├── not live         → amber bbox, "Please blink / Look left" overlay
    └── known + live     → green bbox
            │
            ├── in cooldown  → show card with "Already marked" badge
            └── new mark     → write Realtime DB + Firestore attendance log
                               → fetch avatar from Cloudinary
                               → send email (background thread)
                               → show student info card (6 s)

shared/firebase_config.py    ← Realtime DB + Firestore init (no Storage)
shared/storage.py            ← Cloudinary upload / fetch / delete
shared/encoder.py            ← InsightFace encode + Firestore store/load

admin/app.py (Flask, login-protected)
    ├── /login           ← single admin account, 8h session
    ├── /dashboard       ← Chart.js analytics, low-attendance alerts
    ├── /students        ← CRUD + Cloudinary upload + Sync Encodings
    ├── /attendance      ← monthly log
    └── /export/csv|excel|pdf ← download
```

---

## 12. Switching InsightFace model size

`buffalo_sc` (~85 MB) is used by default for fast CPU inference. For
higher accuracy on a GPU machine, change the model name in
`shared/encoder.py`, `kiosk/recognizer.py`, and `kiosk/registration.py`:

```python
FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
```

`buffalo_l` (~700 MB) trades download size and inference speed for
improved accuracy on difficult angles/lighting. If you switch models,
run **Sync Encodings** from the admin panel to regenerate all stored
embeddings, since different models produce incompatible vector spaces.

---

## Licence

MIT — free to use, modify, and redistribute.
