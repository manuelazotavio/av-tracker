"""
Migrate existing .npy embeddings, metrics JSON, and validation JSON files
into the SQLite database.

Usage:
    python scripts/migrate_to_db.py
"""
import os
import sys
import re
import json
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.database import get_db

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VOICE_DIR = os.path.join(BASE_DIR, "data", "embeddings")
FACE_DIR = os.path.join(BASE_DIR, "data", "face_embeddings")
SESSIONS_DIR = os.path.join(BASE_DIR, "realtime_sessions")

# Known genders (from the codebase name lists)
KNOWN_FEMALE = {"manuela", "isabel", "luisa", "luísa", "sabrina", "adriana"}
KNOWN_MALE = {"joao", "joão", "gustavo", "heitor", "kauan", "cauã", "arthur",
              "gabriel", "lucas", "pedro", "rafael", "bruno"}


def _guess_gender(name: str) -> str | None:
    key = name.lower().split()[0]
    if key in KNOWN_FEMALE:
        return "female"
    if key in KNOWN_MALE:
        return "male"
    return None


def _parse_emb_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    if base.endswith("_auto"):
        base = base[:-5]
    parts = base.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return base


def migrate_voice_embeddings(db):
    if not os.path.exists(VOICE_DIR):
        print("  No voice embeddings directory found.")
        return 0
    count = 0
    for f in sorted(os.listdir(VOICE_DIR)):
        if not f.endswith(".npy"):
            continue
        name = _parse_emb_name(f)
        if re.match(r'^Person_\d+$', name):
            continue  # skip orphan generic embeddings
        emb = np.load(os.path.join(VOICE_DIR, f))
        gender = _guess_gender(name)
        speaker_id = db.add_speaker(name, gender)
        db.save_voice_embedding(speaker_id, emb, source_file=f)
        count += 1
    return count


def migrate_face_embeddings(db):
    if not os.path.exists(FACE_DIR):
        print("  No face embeddings directory found.")
        return 0
    count = 0
    for f in sorted(os.listdir(FACE_DIR)):
        if not f.endswith(".npy"):
            continue
        name = _parse_emb_name(f)
        if re.match(r'^Person_\d+$', name):
            continue
        emb = np.load(os.path.join(FACE_DIR, f))
        gender = _guess_gender(name)
        speaker_id = db.add_speaker(name, gender)
        db.save_face_embedding(speaker_id, emb, source_file=f)
        count += 1
    return count


def migrate_sessions(db):
    metrics_files = sorted(glob.glob(os.path.join(SESSIONS_DIR, "metrics_*.json")))
    count = 0
    for mf in metrics_files:
        with open(mf, encoding="utf-8") as f:
            data = json.load(f)
        session_id = data.get("session_id", "")
        if not session_id:
            continue

        # Load transcript if available
        transcript = None
        tf = os.path.join(SESSIONS_DIR, f"transcript_{session_id}.txt")
        if os.path.exists(tf):
            with open(tf, encoding="utf-8") as f:
                transcript = f.read()

        audio_file = None
        af = os.path.join(SESSIONS_DIR, f"audio_{session_id}.wav")
        if os.path.exists(af):
            audio_file = af

        session_pk = db.save_session(
            session_id=session_id,
            started_at=data.get("session_start"),
            config=data.get("config"),
            summary=data.get("summary"),
            transcript=transcript,
            audio_file=audio_file,
        )

        # Save segments
        segments = data.get("segments", [])
        if segments:
            db.save_segments(session_pk, segments)

        # Load validation if available
        vf = os.path.join(SESSIONS_DIR, f"validation_{session_id}.json")
        if os.path.exists(vf):
            with open(vf, encoding="utf-8") as f:
                val = json.load(f)
            lines = []
            # Reconstruct lines from confusion or per-line data if available
            db.save_validation(
                session_pk=session_pk,
                accuracy=val.get("accuracy", 0),
                named_accuracy=val.get("named_accuracy", 0),
                total_lines=val.get("validated_lines", 0),
                correct=val.get("correct", 0),
                incorrect=val.get("incorrect", 0),
                details=val,
            )

        count += 1
    return count


def main():
    db = get_db()
    print("Migrating to SQLite database...")
    print(f"  DB path: {db.db_path}\n")

    n = migrate_voice_embeddings(db)
    print(f"  Voice embeddings: {n} imported")

    n = migrate_face_embeddings(db)
    print(f"  Face embeddings:  {n} imported")

    n = migrate_sessions(db)
    print(f"  Sessions:         {n} imported")

    # Summary
    speakers = db.list_speakers()
    sessions = db.list_sessions()
    print(f"\nDatabase summary:")
    print(f"  Speakers: {len(speakers)}")
    for s in speakers:
        print(f"    {s['name']:20s}  voice={s['voice_count']}  face={s['face_count']}")
    print(f"  Sessions: {len(sessions)}")

    print(f"\nDone! Database: {db.db_path}")
    db.close()


if __name__ == "__main__":
    main()
