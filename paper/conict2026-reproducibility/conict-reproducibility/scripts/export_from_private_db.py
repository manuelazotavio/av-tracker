#!/usr/bin/env python3
"""Exporta dados derivados e anônimos do banco privado do AV-Tracker.

O banco SQLite original não é copiado. Nomes, transcrições, timestamps absolutos,
áudios, imagens e embeddings são excluídos do artefato público.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VOICE_ONLY = {"VERIFIER_HIGH", "VERIFIER_MOD"}
VOICE_CORROBORATED = {"VERIFIER_WEAK"}
ASD = {"ASD", "ASD_GUESS"}
FACE_GENDER = {
    "SESSION_FACE",
    "GENDER_FACE",
    "SINGLE_FACE",
    "GENDER_REROUTE",
    "FACE_ONLY",
    "GENDER_OVERRIDE",
}
SESSION = {"SESSION_HINT", "SESSION_TRACKER"}


def evidence_group(decision: str) -> str:
    if decision in VOICE_ONLY:
        return "voice_only"
    if decision in VOICE_CORROBORATED:
        return "voice_corroborated"
    if decision in ASD:
        return "active_speaker"
    if decision in FACE_GENDER:
        return "face_gender"
    if decision in SESSION:
        return "session_context"
    raise ValueError(f"Tipo de decisão não documentado: {decision}")


def normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def is_generic_label(value: Any) -> bool:
    label = normalized(value)
    if not label:
        return True
    patterns = (
        r"unknown(?: speaker)?",
        r"desconhecid[oa]",
        r"speaker(?:[_\s-]*\d+)?",
        r"spk(?:[_\s-]*\d+)?",
        r"person(?:[_\s-]*\d+)?",
        r"none",
        r"n/?a",
    )
    return any(re.fullmatch(pattern, label) for pattern in patterns)


def read_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def as_number(value: Any, digits: int = 6) -> Any:
    if value is None or value == "":
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


def as_bool(value: Any) -> int:
    return int(bool(value))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connect_read_only(database: Path) -> sqlite3.Connection:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="Banco SQLite privado")
    parser.add_argument("--output", required=True, type=Path, help="Diretório data/ do artefato")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    connection = connect_read_only(args.db)

    session_rows = connection.execute(
        """
        SELECT s.id, s.started_at, s.config, COUNT(g.id) AS segment_count
        FROM sessions AS s
        JOIN segments AS g ON g.session_pk = s.id
        GROUP BY s.id
        ORDER BY s.started_at, s.id
        """
    ).fetchall()
    session_map = {row["id"]: f"S{index:03d}" for index, row in enumerate(session_rows, 1)}

    exported_sessions: list[dict[str, Any]] = []
    for order, row in enumerate(session_rows, 1):
        config = read_json(row["config"])
        exported_sessions.append(
            {
                "anonymous_session_id": session_map[row["id"]],
                "session_order": order,
                "expected_participants": config.get("num_speakers", ""),
                "segment_count": row["segment_count"],
                "whisper_model_size": config.get("whisper_size", ""),
                "sample_rate_hz": config.get("sample_rate", ""),
                "verifier_threshold": config.get("verifier_threshold", ""),
                "verifier_confidence_min": config.get("verifier_confidence_min", ""),
                "face_match_threshold": config.get("face_match_threshold", ""),
                "face_merge_threshold": config.get("face_merge_threshold", ""),
            }
        )

    segment_rows = connection.execute(
        """
        SELECT id, session_pk, seg_index, decision_type, raw_conf, is_identified,
               is_fn, fp_risk, whisper_ms, verifier_ms, segment_ms, metrics_json
        FROM segments
        WHERE session_pk IN (SELECT DISTINCT session_pk FROM segments)
        ORDER BY session_pk, seg_index, id
        """
    ).fetchall()

    exported_segments: list[dict[str, Any]] = []
    segment_risk: dict[tuple[int, int], str] = {}
    for row in segment_rows:
        if row["session_pk"] not in session_map:
            continue
        metrics = read_json(row["metrics_json"])
        decision = str(row["decision_type"] or metrics.get("decision_type") or "")
        duration = as_number(metrics.get("segment_duration_s"))
        total_ms = as_number(row["segment_ms"] if row["segment_ms"] is not None else metrics.get("segment_ms"))
        rtf = ""
        if duration not in ("", 0) and total_ms != "":
            rtf = round(float(total_ms) / (float(duration) * 1000.0), 6)
        risk = normalized(row["fp_risk"] or metrics.get("fp_risk"))
        risk_group = "flagged" if risk in {"medium", "high"} else "unflagged"
        segment_risk[(row["session_pk"], row["seg_index"])] = risk_group
        exported_segments.append(
            {
                "anonymous_session_id": session_map[row["session_pk"]],
                "segment_index": row["seg_index"],
                "decision_type": decision,
                "evidence_group": evidence_group(decision),
                "raw_confidence": as_number(row["raw_conf"]),
                "risk_group": risk_group,
                "identified": as_bool(row["is_identified"]),
                "possible_false_negative": as_bool(row["is_fn"]),
                "active_speaker_detected": as_bool(metrics.get("asd_active")),
                "detected_faces": metrics.get("n_faces", ""),
                "segment_duration_s": duration,
                "transcription_ms": as_number(
                    row["whisper_ms"] if row["whisper_ms"] is not None else metrics.get("whisper_ms")
                ),
                "voice_verification_ms": as_number(
                    row["verifier_ms"] if row["verifier_ms"] is not None else metrics.get("verifier_ms")
                ),
                "total_processing_ms": total_ms,
                "real_time_factor": rtf,
            }
        )

    latest_validations = connection.execute(
        """
        SELECT v.id, v.session_pk
        FROM validations AS v
        JOIN (
            SELECT session_pk, MAX(id) AS latest_id
            FROM validations
            GROUP BY session_pk
        ) AS latest ON latest.latest_id = v.id
        ORDER BY v.session_pk
        """
    ).fetchall()

    exported_validation: list[dict[str, Any]] = []
    for validation in latest_validations:
        session_pk = validation["session_pk"]
        if session_pk not in session_map:
            continue
        lines = connection.execute(
            """
            SELECT seg_index, predicted, corrected
            FROM validation_lines
            WHERE validation_id = ?
            ORDER BY seg_index, id
            """,
            (validation["id"],),
        ).fetchall()
        for line in lines:
            predicted_named = not is_generic_label(line["predicted"])
            exact = normalized(line["predicted"]) == normalized(line["corrected"])
            exported_validation.append(
                {
                    "anonymous_session_id": session_map[session_pk],
                    "segment_index": line["seg_index"],
                    "predicted_named": as_bool(predicted_named),
                    "exact_match": as_bool(exact),
                    "risk_group": segment_risk.get((session_pk, line["seg_index"]), "unflagged"),
                }
            )

    # Asserções funcionam como um teste contra exportações silenciosamente incompletas.
    named = [row for row in exported_validation if row["predicted_named"]]
    flagged = [row for row in named if row["risk_group"] == "flagged"]
    unflagged = [row for row in named if row["risk_group"] == "unflagged"]
    observed = {
        "sessions": len(exported_sessions),
        "segments": len(exported_segments),
        "validation_lines": len(exported_validation),
        "exact_matches": sum(row["exact_match"] for row in exported_validation),
        "named_predictions": len(named),
        "named_exact_matches": sum(row["exact_match"] for row in named),
        "flagged_named_predictions": len(flagged),
        "flagged_named_exact_matches": sum(row["exact_match"] for row in flagged),
        "unflagged_named_predictions": len(unflagged),
        "unflagged_named_exact_matches": sum(row["exact_match"] for row in unflagged),
    }
    expected = {
        "sessions": 92,
        "segments": 1660,
        "validation_lines": 330,
        "exact_matches": 37,
        "named_predictions": 124,
        "named_exact_matches": 37,
        "flagged_named_predictions": 113,
        "flagged_named_exact_matches": 28,
        "unflagged_named_predictions": 11,
        "unflagged_named_exact_matches": 9,
    }
    if observed != expected:
        raise RuntimeError(f"Falha nas invariantes da exportação: observado={observed}; esperado={expected}")

    sessions_path = output / "sessions.csv"
    segments_path = output / "segments.csv"
    validation_path = output / "validation.csv"
    write_csv(sessions_path, list(exported_sessions[0]), exported_sessions)
    write_csv(segments_path, list(exported_segments[0]), exported_segments)
    write_csv(validation_path, list(exported_validation[0]), exported_validation)

    privacy_report = {
        "status": "passed",
        "included_files": ["sessions.csv", "segments.csv", "validation.csv"],
        "excluded_categories": [
            "personal names and speaker labels",
            "transcriptions and utterance text",
            "absolute timestamps and database identifiers",
            "audio, video and face crops",
            "voice and face embeddings",
            "candidate identities and demographic attributes",
        ],
        "row_counts": observed,
        "decision_counts": dict(sorted(Counter(row["decision_type"] for row in exported_segments).items())),
        "sha256": {
            path.name: sha256(path) for path in (sessions_path, segments_path, validation_path)
        },
    }
    (output / "privacy_report.json").write_text(
        json.dumps(privacy_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    connection.close()
    print(json.dumps(observed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
