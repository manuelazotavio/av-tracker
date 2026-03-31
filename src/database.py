"""
AV-Tracker SQLite database — single-file storage for speakers, embeddings,
sessions, segments, and validations.

Usage:
    from src.database import get_db
    db = get_db()                       # opens/creates av_tracker.db
    db.add_speaker("Manuela", "female")
    db.save_voice_embedding(speaker_id, embedding_np, source="meeting1.mp4")
"""
import os
import re
import sqlite3
import json
import numpy as np
from datetime import datetime
from contextlib import contextmanager

_DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "av_tracker.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    gender      TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS voice_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id  INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    embedding   BLOB    NOT NULL,
    source_file TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id  INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    embedding   BLOB    NOT NULL,
    source_file TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    UNIQUE NOT NULL,
    started_at  TEXT,
    config      TEXT,
    summary     TEXT,
    transcript  TEXT,
    audio_file  TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_pk      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seg_index       INTEGER NOT NULL,
    timestamp       TEXT,
    speaker_name    TEXT,
    text            TEXT,
    decision        TEXT,
    decision_type   TEXT,
    raw_best_name   TEXT,
    raw_conf        REAL,
    final_name      TEXT,
    final_speaker   TEXT,
    is_identified   INTEGER,
    is_fn           INTEGER,
    fp_risk         TEXT,
    audio_gender    TEXT,
    whisper_ms      REAL,
    verifier_ms     REAL,
    segment_ms      REAL,
    metrics_json    TEXT
);

CREATE TABLE IF NOT EXISTS validations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_pk      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    validated_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    accuracy        REAL,
    named_accuracy  REAL,
    total_lines     INTEGER,
    correct         INTEGER,
    incorrect       INTEGER,
    details_json    TEXT
);

CREATE TABLE IF NOT EXISTS validation_lines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    validation_id   INTEGER NOT NULL REFERENCES validations(id) ON DELETE CASCADE,
    seg_index       INTEGER NOT NULL,
    predicted       TEXT,
    corrected       TEXT
);

CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_pk);
CREATE INDEX IF NOT EXISTS idx_voice_emb_speaker ON voice_embeddings(speaker_id);
CREATE INDEX IF NOT EXISTS idx_face_emb_speaker ON face_embeddings(speaker_id);
CREATE INDEX IF NOT EXISTS idx_validation_lines_val ON validation_lines(validation_id);
"""


class TrackerDB:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or _DEFAULT_DB
        self._conn = None

    # -- Connection ----------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
        return self._conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Speakers ------------------------------------------------------------
    def add_speaker(self, name: str, gender: str = None) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO speakers (name, gender) VALUES (?, ?)",
                (name, gender),
            )
            cur.execute("SELECT id FROM speakers WHERE name = ?", (name,))
            return cur.fetchone()["id"]

    def get_speaker(self, name: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM speakers WHERE name = ?", (name,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_speakers(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("""
                SELECT s.*,
                       COUNT(DISTINCT ve.id) as voice_count,
                       COUNT(DISTINCT fe.id) as face_count
                FROM speakers s
                LEFT JOIN voice_embeddings ve ON ve.speaker_id = s.id AND ve.is_active = 1
                LEFT JOIN face_embeddings fe ON fe.speaker_id = s.id AND fe.is_active = 1
                GROUP BY s.id
                ORDER BY s.name
            """)
            return [dict(r) for r in cur.fetchall()]

    def rename_speaker(self, old_name: str, new_name: str):
        with self._cursor() as cur:
            cur.execute("UPDATE speakers SET name = ? WHERE name = ?", (new_name, old_name))

    def delete_speaker(self, name: str):
        with self._cursor() as cur:
            cur.execute("DELETE FROM speakers WHERE name = ?", (name,))

    # -- Voice Embeddings ----------------------------------------------------
    def save_voice_embedding(self, speaker_id: int, embedding: np.ndarray,
                             source_file: str = None) -> int:
        blob = embedding.astype(np.float32).tobytes()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO voice_embeddings (speaker_id, embedding, source_file) VALUES (?, ?, ?)",
                (speaker_id, blob, source_file),
            )
            return cur.lastrowid

    def get_voice_embeddings(self, speaker_id: int, active_only: bool = True) -> list[dict]:
        with self._cursor() as cur:
            sql = "SELECT * FROM voice_embeddings WHERE speaker_id = ?"
            if active_only:
                sql += " AND is_active = 1"
            cur.execute(sql, (speaker_id,))
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
                results.append(d)
            return results

    def get_all_voice_embeddings(self, active_only: bool = True) -> dict[str, np.ndarray]:
        """Returns {speaker_name: averaged_embedding} for the verifier."""
        with self._cursor() as cur:
            sql = """
                SELECT s.name, ve.embedding
                FROM voice_embeddings ve
                JOIN speakers s ON s.id = ve.speaker_id
            """
            if active_only:
                sql += " WHERE ve.is_active = 1"
            cur.execute(sql)
            from collections import defaultdict
            raw = defaultdict(list)
            for row in cur.fetchall():
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                raw[row["name"]].append(emb)
            result = {}
            for name, embs in raw.items():
                avg = np.mean(embs, axis=0)
                norm = np.linalg.norm(avg)
                result[name] = avg / norm if norm > 0 else avg
            return result

    def deactivate_voice_embedding(self, emb_id: int):
        with self._cursor() as cur:
            cur.execute("UPDATE voice_embeddings SET is_active = 0 WHERE id = ?", (emb_id,))

    def delete_voice_embedding(self, emb_id: int):
        with self._cursor() as cur:
            cur.execute("DELETE FROM voice_embeddings WHERE id = ?", (emb_id,))

    def count_voice_embeddings(self, speaker_id: int) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as n FROM voice_embeddings WHERE speaker_id = ? AND is_active = 1",
                        (speaker_id,))
            return cur.fetchone()["n"]

    # -- Face Embeddings -----------------------------------------------------
    def save_face_embedding(self, speaker_id: int, embedding: np.ndarray,
                            source_file: str = None) -> int:
        blob = embedding.astype(np.float32).tobytes()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO face_embeddings (speaker_id, embedding, source_file) VALUES (?, ?, ?)",
                (speaker_id, blob, source_file),
            )
            return cur.lastrowid

    def get_face_embeddings(self, speaker_id: int, active_only: bool = True) -> list[dict]:
        with self._cursor() as cur:
            sql = "SELECT * FROM face_embeddings WHERE speaker_id = ?"
            if active_only:
                sql += " AND is_active = 1"
            cur.execute(sql, (speaker_id,))
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
                results.append(d)
            return results

    def get_all_face_embeddings(self, active_only: bool = True) -> dict[str, np.ndarray]:
        """Returns {speaker_name: averaged_embedding} for the face tracker."""
        with self._cursor() as cur:
            sql = """
                SELECT s.name, fe.embedding
                FROM face_embeddings fe
                JOIN speakers s ON s.id = fe.speaker_id
            """
            if active_only:
                sql += " WHERE fe.is_active = 1"
            cur.execute(sql)
            from collections import defaultdict
            raw = defaultdict(list)
            for row in cur.fetchall():
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                raw[row["name"]].append(emb)
            result = {}
            for name, embs in raw.items():
                avg = np.mean(embs, axis=0)
                norm = np.linalg.norm(avg)
                result[name] = avg / norm if norm > 0 else avg
            return result

    def deactivate_face_embedding(self, emb_id: int):
        with self._cursor() as cur:
            cur.execute("UPDATE face_embeddings SET is_active = 0 WHERE id = ?", (emb_id,))

    def delete_face_embedding(self, emb_id: int):
        with self._cursor() as cur:
            cur.execute("DELETE FROM face_embeddings WHERE id = ?", (emb_id,))

    # -- Sessions ------------------------------------------------------------
    def save_session(self, session_id: str, started_at: str = None,
                     config: dict = None, summary: dict = None,
                     transcript: str = None, audio_file: str = None) -> int:
        with self._cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO sessions
                    (session_id, started_at, config, summary, transcript, audio_file)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                started_at,
                json.dumps(config, ensure_ascii=False) if config else None,
                json.dumps(summary, ensure_ascii=False) if summary else None,
                transcript,
                audio_file,
            ))
            cur.execute("SELECT id FROM sessions WHERE session_id = ?", (session_id,))
            return cur.fetchone()["id"]

    def get_session(self, session_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["config"] = json.loads(d["config"]) if d["config"] else {}
            d["summary"] = json.loads(d["summary"]) if d["summary"] else {}
            return d

    def list_sessions(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("""
                SELECT s.*,
                       COUNT(seg.id) as segment_count,
                       v.accuracy as validated_accuracy
                FROM sessions s
                LEFT JOIN segments seg ON seg.session_pk = s.id
                LEFT JOIN validations v ON v.session_pk = s.id
                GROUP BY s.id
                ORDER BY s.session_id DESC
            """)
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["config"] = json.loads(d["config"]) if d["config"] else {}
                d["summary"] = json.loads(d["summary"]) if d["summary"] else {}
                results.append(d)
            return results

    # -- Segments ------------------------------------------------------------
    def save_segments(self, session_pk: int, segments: list[dict]):
        with self._cursor() as cur:
            for i, seg in enumerate(segments):
                cur.execute("""
                    INSERT INTO segments
                        (session_pk, seg_index, timestamp, speaker_name, text,
                         decision, decision_type, raw_best_name, raw_conf,
                         final_name, final_speaker, is_identified, is_fn,
                         fp_risk, audio_gender, whisper_ms, verifier_ms,
                         segment_ms, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_pk, i,
                    seg.get("timestamp"),
                    seg.get("final_name") or seg.get("final_speaker"),
                    seg.get("text", ""),
                    seg.get("decision"),
                    seg.get("decision_type"),
                    seg.get("raw_best_name"),
                    seg.get("raw_conf", 0),
                    seg.get("final_name"),
                    seg.get("final_speaker"),
                    1 if seg.get("is_identified") else 0,
                    1 if seg.get("is_fn") else 0,
                    seg.get("fp_risk", "none"),
                    seg.get("audio_gender"),
                    seg.get("whisper_ms", 0),
                    seg.get("verifier_ms", 0),
                    seg.get("segment_ms", 0),
                    json.dumps(seg, ensure_ascii=False),
                ))

    def get_segments(self, session_pk: int) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM segments WHERE session_pk = ? ORDER BY seg_index",
                (session_pk,),
            )
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["metrics"] = json.loads(d["metrics_json"]) if d["metrics_json"] else {}
                results.append(d)
            return results

    # -- Validations ---------------------------------------------------------
    def save_validation(self, session_pk: int, accuracy: float,
                        named_accuracy: float, total_lines: int,
                        correct: int, incorrect: int,
                        details: dict = None,
                        lines: list[tuple] = None) -> int:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO validations
                    (session_pk, accuracy, named_accuracy, total_lines,
                     correct, incorrect, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_pk, accuracy, named_accuracy, total_lines,
                correct, incorrect,
                json.dumps(details, ensure_ascii=False) if details else None,
            ))
            val_id = cur.lastrowid
            if lines:
                for seg_idx, predicted, corrected in lines:
                    cur.execute("""
                        INSERT INTO validation_lines
                            (validation_id, seg_index, predicted, corrected)
                        VALUES (?, ?, ?, ?)
                    """, (val_id, seg_idx, predicted, corrected))
            return val_id

    def get_validation(self, session_pk: int) -> dict | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM validations WHERE session_pk = ? ORDER BY id DESC LIMIT 1",
                (session_pk,),
            )
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["details"] = json.loads(d["details_json"]) if d["details_json"] else {}
            return d

    # -- Validation feedback ---------------------------------------------------
    def apply_validation_feedback(self, session_id: str, corrections: dict[int, tuple[str, str]]):
        """Apply validation corrections to improve future sessions.

        Args:
            session_id: the session being validated
            corrections: {seg_index: (predicted_speaker, corrected_speaker)}
                         only includes lines where predicted != corrected

        Actions:
            1. DELETE all .npy voice/face files for wrongly-attributed names
               (they are contaminated with the wrong person's biometrics)
            2. Deactivate corresponding embeddings in the DB
            3. Register corrected speaker names in the DB
        """
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        voice_dir = os.path.join(base_dir, "data", "embeddings")
        face_dir = os.path.join(base_dir, "data", "face_embeddings")

        # Build mapping: wrong_name → correct_name (from corrections)
        renames = {}  # wrong -> correct
        for seg_idx, (predicted, corrected) in corrections.items():
            if predicted != corrected and predicted and corrected:
                if not re.match(r'^(spk_\d+|Person_\d+|Desconhecido_\d+|Unknown)$', predicted):
                    renames[predicted] = corrected

        if not renames:
            return []

        actions = []

        for wrong_name, correct_name in renames.items():
            # 1) Deactivate ALL voice embeddings under the wrong name in DB
            with self._cursor() as cur:
                cur.execute("""
                    SELECT ve.id FROM voice_embeddings ve
                    JOIN speakers s ON s.id = ve.speaker_id
                    WHERE s.name = ? AND ve.is_active = 1
                """, (wrong_name,))
                for row in cur.fetchall():
                    self.deactivate_voice_embedding(row["id"])
                    actions.append(f"DB: deactivated voice emb {row['id']} ({wrong_name})")

            # 2) Deactivate ALL face embeddings under the wrong name in DB
            with self._cursor() as cur:
                cur.execute("""
                    SELECT fe.id FROM face_embeddings fe
                    JOIN speakers s ON s.id = fe.speaker_id
                    WHERE s.name = ? AND fe.is_active = 1
                """, (wrong_name,))
                for row in cur.fetchall():
                    self.deactivate_face_embedding(row["id"])
                    actions.append(f"DB: deactivated face emb {row['id']} ({wrong_name})")

            # 3) DELETE all .npy files on disk for the wrong name
            #    These are what the verifier/tracker actually load — DB alone isn't enough
            for emb_dir, emb_type in [(voice_dir, "voice"), (face_dir, "face")]:
                if not os.path.exists(emb_dir):
                    continue
                for f in list(os.listdir(emb_dir)):
                    if not f.endswith(".npy"):
                        continue
                    # Match: "Arthur_20260331.npy", "Arthur_auto.npy", "Arthur.npy"
                    base = os.path.splitext(f)[0]
                    if base.endswith("_auto"):
                        base = base[:-5]
                    parts = base.split("_")
                    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                        file_name = "_".join(parts[:-2])
                    else:
                        file_name = base
                    if file_name == wrong_name:
                        try:
                            os.remove(os.path.join(emb_dir, f))
                            actions.append(f"Deleted {emb_type}: {f} (was '{wrong_name}')")
                        except OSError:
                            pass

            # 4) Ensure corrected speaker exists in DB
            self.add_speaker(correct_name)

        return actions

    # -- Utility queries -----------------------------------------------------
    def get_session_accuracy_trend(self) -> list[dict]:
        """Returns chronological list of sessions with accuracy (for graphs)."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT s.session_id, s.summary,
                       v.accuracy, v.named_accuracy
                FROM sessions s
                LEFT JOIN validations v ON v.session_pk = s.id
                ORDER BY s.session_id
            """)
            results = []
            for row in cur.fetchall():
                d = {"session_id": row["session_id"]}
                summary = json.loads(row["summary"]) if row["summary"] else {}
                d.update(summary)
                d["validated_accuracy"] = row["accuracy"]
                d["validated_named_accuracy"] = row["named_accuracy"]
                results.append(d)
            return results


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: TrackerDB | None = None

def get_db(db_path: str = None) -> TrackerDB:
    global _instance
    if _instance is None or (db_path and _instance.db_path != db_path):
        _instance = TrackerDB(db_path)
    return _instance
