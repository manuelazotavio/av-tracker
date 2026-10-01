#!/usr/bin/env python3
"""Reproduz as tabelas e os testes estatísticos do artigo usando apenas stdlib."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: str) -> float:
    return float(value)


def i(value: str) -> int:
    return int(value)


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return 100.0 * (center - margin), 100.0 * (center + margin)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    row_1 = a + b
    row_2 = c + d
    col_1 = a + c
    total = row_1 + row_2

    def probability(x: int) -> float:
        return math.comb(col_1, x) * math.comb(total - col_1, row_1 - x) / math.comb(total, row_1)

    lower = max(0, row_1 - (total - col_1))
    upper = min(row_1, col_1)
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lower, upper + 1) if probability(x) <= observed + 1e-15))


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "maximum": round(max(values), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sessions = read_csv(args.data / "sessions.csv")
    segments = read_csv(args.data / "segments.csv")
    validation = read_csv(args.data / "validation.csv")
    total = len(segments)

    decision_counts = Counter(row["decision_type"] for row in segments)
    decision_rows = [
        {"decision_type": key, "segments": count, "percentage": pct(count, total)}
        for key, count in sorted(decision_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_csv(args.output / "table_decisions.csv", list(decision_rows[0]), decision_rows)

    group_counts = Counter(row["evidence_group"] for row in segments)
    group_order = ["voice_only", "voice_corroborated", "active_speaker", "face_gender", "session_context"]
    evidence_rows = [
        {"evidence_group": group, "segments": group_counts[group], "percentage": pct(group_counts[group], total)}
        for group in group_order
    ]
    write_csv(args.output / "table_evidence_groups.csv", list(evidence_rows[0]), evidence_rows)

    participants = {row["anonymous_session_id"]: i(row["expected_participants"]) for row in sessions}
    by_size: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in segments:
        by_size[participants[row["anonymous_session_id"]]].append(row)
    size_rows: list[dict[str, Any]] = []
    for size in sorted(by_size):
        rows = by_size[size]
        size_total = len(rows)
        voice_only = sum(row["evidence_group"] == "voice_only" for row in rows)
        voice_cor = sum(row["evidence_group"] == "voice_corroborated" for row in rows)
        other = size_total - voice_only - voice_cor
        size_rows.append(
            {
                "expected_participants": size,
                "segments": size_total,
                "voice_only": voice_only,
                "voice_only_percentage": pct(voice_only, size_total),
                "voice_corroborated": voice_cor,
                "voice_corroborated_percentage": pct(voice_cor, size_total),
                "multimodal_or_context": other,
                "multimodal_or_context_percentage": pct(other, size_total),
            }
        )
    write_csv(args.output / "table_session_groups.csv", list(size_rows[0]), size_rows)

    durations = [f(row["segment_duration_s"]) for row in segments]
    transcription_ms = [f(row["transcription_ms"]) for row in segments]
    verifier_ms = [f(row["voice_verification_ms"]) for row in segments]
    total_ms = [f(row["total_processing_ms"]) for row in segments]
    rtfs = [f(row["real_time_factor"]) for row in segments]

    exact = sum(i(row["exact_match"]) for row in validation)
    named = [row for row in validation if i(row["predicted_named"]) == 1]
    named_exact = sum(i(row["exact_match"]) for row in named)
    flagged = [row for row in named if row["risk_group"] == "flagged"]
    unflagged = [row for row in named if row["risk_group"] == "unflagged"]
    flagged_exact = sum(i(row["exact_match"]) for row in flagged)
    unflagged_exact = sum(i(row["exact_match"]) for row in unflagged)
    fisher_p = fisher_exact_two_sided(unflagged_exact, len(unflagged) - unflagged_exact, flagged_exact, len(flagged) - flagged_exact)

    validation_rows: list[dict[str, Any]] = []
    for label, successes, sample in (
        ("all_validated_segments", exact, len(validation)),
        ("named_predictions", named_exact, len(named)),
        ("unflagged_named_predictions", unflagged_exact, len(unflagged)),
        ("flagged_named_predictions", flagged_exact, len(flagged)),
    ):
        low, high = wilson(successes, sample)
        validation_rows.append(
            {
                "subset": label,
                "correct": successes,
                "total": sample,
                "accuracy_percentage": pct(successes, sample),
                "wilson_95_low_percentage": round(low, 3),
                "wilson_95_high_percentage": round(high, 3),
            }
        )
    write_csv(args.output / "validation_statistics.csv", list(validation_rows[0]), validation_rows)

    summary = {
        "dataset": {
            "sessions_with_segments": len(sessions),
            "segments": total,
            "validated_sessions": len({row["anonymous_session_id"] for row in validation}),
            "validated_segments": len(validation),
            "expected_participant_distribution": dict(
                sorted(Counter(i(row["expected_participants"]) for row in sessions).items())
            ),
        },
        "latency": {
            "segment_duration_seconds": describe(durations),
            "transcription_milliseconds": describe(transcription_ms),
            "voice_verification_milliseconds": describe(verifier_ms),
            "total_processing_milliseconds": describe(total_ms),
            "real_time_factor": describe(rtfs),
            "segments_below_real_time": sum(value < 1.0 for value in rtfs),
            "segments_below_real_time_percentage": pct(sum(value < 1.0 for value in rtfs), len(rtfs)),
        },
        "validation": {
            "statistics": validation_rows,
            "fisher_exact_two_sided_p": round(fisher_p, 10),
            "contingency_table": {
                "unflagged": [unflagged_exact, len(unflagged) - unflagged_exact],
                "flagged": [flagged_exact, len(flagged) - flagged_exact],
            },
        },
        "decision_counts": dict(sorted(decision_counts.items())),
        "evidence_group_counts": {group: group_counts[group] for group in group_order},
    }

    expected_core = {
        "sessions_with_segments": 92,
        "segments": 1660,
        "validated_sessions": 24,
        "validated_segments": 330,
        "named_predictions": 124,
        "named_exact": 37,
        "unflagged": [9, 2],
        "flagged": [28, 85],
    }
    observed_core = {
        "sessions_with_segments": len(sessions),
        "segments": total,
        "validated_sessions": len({row["anonymous_session_id"] for row in validation}),
        "validated_segments": len(validation),
        "named_predictions": len(named),
        "named_exact": named_exact,
        "unflagged": [unflagged_exact, len(unflagged) - unflagged_exact],
        "flagged": [flagged_exact, len(flagged) - flagged_exact],
    }
    if observed_core != expected_core:
        raise RuntimeError(f"Falha nas invariantes da análise: {observed_core} != {expected_core}")

    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
