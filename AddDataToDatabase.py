"""
AddDataToDatabase.py  (corrected: was AddDadaToDatabase.py)
────────────────────
One-time seed script — adds sample student records to Firebase Realtime DB.

Run from the project root:
    python AddDataToDatabase.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.firebase_config import get_firebase

db, _, _ = get_firebase()

ref = db.reference("Students")

data = {
    "254525": {
        "name":                 "Aarav Shah",
        "major":                "Computer Science",
        "starting year":        2021,
        "total_attendance":     8,
        "standing":             "G",
        "year":                 3,
        "last_attendance_time": "2024-01-09 00:30:22",
        "email":                "",
        "parent_email":         "",
    },
}

for student_id, record in data.items():
    ref.child(student_id).set(record)
    print(f"  ✓  Wrote record for {record['name']} ({student_id})")

print("\nDone. You can add more students via the admin panel or this script.")
