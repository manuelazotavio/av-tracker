#!/usr/bin/env python3
"""Analyze the unbiased ground-truth validation sample selected by
sample_ground_truth.py, once you've actually validated those sessions
(via app/pages/validate.py or scripts/validate_session.py).

Unlike the pre-existing 24-session validation set (which was hand-picked
because those sessions were suspected of containing errors, and so cannot
support an accuracy claim), the sessions in ground_truth_sample/manifest.csv
were drawn with a fixed random seed from the never-before-validated
population, stratified by expected group size. This script computes accuracy
and Wilson 95% CIs per group size from exactly that sample, plus a
(now unbiased) check of whether the self-diagnostic risk flag correlates
with correctness, mirroring the biased-sample version already reported in
wvc2026.tex's Limitations section.

Run with no arguments after validating everything in the manifest; it will
tell you what's still missing if the sample is incomplete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return 100.0 * (center - margin), 100.0 * (center + margin)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    row_1, row_2 = a + b, c + d
    col_1 = a + c
    total = row_1 + row_2

    def probability(x: int) -> float:
        return math.comb(col_1, x) * math.comb(total - col_1, row_1 - x) / math.comb(total, row_1)

    lower = max(0, row_1 - (total - col_1))
    upper = min(row_1, col_1)
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lower, upper + 1) if probability(x) <= observed + 1e-12))


GENERIC_PATTERNS = ("spk_", "person_", "desconhecido_", "unknown", "speaker_")


def is_generic(name: str) -> bool:
    n = (name or "").strip().lower()
    return n == "" or any(n.startswith(p) for p in GENERIC_PATTERNS) or n.split("_")[0] in ("spk", "speaker")


def normalized(value) -> str:
    import re
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[3] / "av_tracker.db")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent.parent / "ground_truth_sample" / "manifest.csv")
    args = parser.parse_args()

    manifest_rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if not manifest_rows:
        raise SystemExit(f"No sessions in {args.manifest} -- run sample_ground_truth.py first.")

    manifest_pks = {int(row["session_pk"]) for row in manifest_rows}
    size_by_pk = {int(row["session_pk"]): int(row["expected_participants"]) for row in manifest_rows}

    con = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    validated_pks = {row["session_pk"] for row in con.execute("SELECT DISTINCT session_pk FROM validations")}
    missing = sorted(manifest_pks - validated_pks)
    if missing:
        missing_ids = [row["session_id"] for row in manifest_rows if int(row["session_pk"]) in missing]
        print(f"{len(missing)}/{len(manifest_pks)} sampled sessions are not validated yet:")
        for sid in missing_ids:
            print(f"  - realtime_sessions/transcript_{sid}.txt")
        print("\nValidate these (Streamlit 'Validate' page or scripts/validate_session.py) and re-run.")
        raise SystemExit(1)

    latest_validation_id = {}
    for row in con.execute(
        """
        SELECT session_pk, MAX(id) AS latest_id FROM validations
        WHERE session_pk IN ({})
        GROUP BY session_pk
        """.format(",".join(str(pk) for pk in manifest_pks))
    ):
        latest_validation_id[row["session_pk"]] = row["latest_id"]

    segment_meta = {}
    for row in con.execute(
        """
        SELECT session_pk, seg_index, decision_type, fp_risk, metrics_json
        FROM segments
        WHERE session_pk IN ({})
        """.format(",".join(str(pk) for pk in manifest_pks))
    ):
        metrics = {}
        try:
            metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
        except (TypeError, json.JSONDecodeError):
            pass
        risk = normalized(row["fp_risk"] or metrics.get("fp_risk"))
        segment_meta[(row["session_pk"], row["seg_index"])] = {
            "decision_type": row["decision_type"] or metrics.get("decision_type"),
            "risk_group": "flagged" if risk in {"medium", "high"} else "unflagged",
        }

    lines = []
    for pk, validation_id in latest_validation_id.items():
        for row in con.execute(
            "SELECT seg_index, predicted, corrected FROM validation_lines WHERE validation_id = ?",
            (validation_id,),
        ):
            meta = segment_meta.get((pk, row["seg_index"]), {})
            lines.append(
                {
                    "session_pk": pk,
                    "expected_participants": size_by_pk[pk],
                    "predicted": row["predicted"],
                    "corrected": row["corrected"],
                    "predicted_named": not is_generic(row["predicted"]),
                    "exact_match": normalized(row["predicted"]) == normalized(row["corrected"]),
                    "risk_group": meta.get("risk_group", "unflagged"),
                    "decision_type": meta.get("decision_type"),
                }
            )

    print(f"=== Ground-truth sample: {len(lines)} validated segments across {len(manifest_pks)} sessions ===\n")

    all_correct = sum(l["exact_match"] for l in lines)
    lo, hi = wilson(all_correct, len(lines))
    print(f"Overall (all validated segments): {all_correct}/{len(lines)} = {100*all_correct/len(lines):.1f}% "
          f"(95% CI [{lo:.1f}, {hi:.1f}])")

    named = [l for l in lines if l["predicted_named"]]
    named_correct = sum(l["exact_match"] for l in named)
    if named:
        lo, hi = wilson(named_correct, len(named))
        print(f"Named predictions only: {named_correct}/{len(named)} = {100*named_correct/len(named):.1f}% "
              f"(95% CI [{lo:.1f}, {hi:.1f}])")

    print("\n=== Accuracy by expected group size (all validated segments) ===")
    by_size = defaultdict(list)
    for l in lines:
        by_size[l["expected_participants"]].append(l)
    for size in sorted(by_size):
        rows = by_size[size]
        correct = sum(r["exact_match"] for r in rows)
        lo, hi = wilson(correct, len(rows))
        print(f"  {size} speakers: {correct}/{len(rows)} = {100*correct/len(rows):.1f}% (95% CI [{lo:.1f}, {hi:.1f}])")

    print("\n=== Accuracy by expected group size (named predictions only) ===")
    by_size_named = defaultdict(list)
    for l in named:
        by_size_named[l["expected_participants"]].append(l)
    for size in sorted(by_size_named):
        rows = by_size_named[size]
        correct = sum(r["exact_match"] for r in rows)
        lo, hi = wilson(correct, len(rows))
        print(f"  {size} speakers: {correct}/{len(rows)} = {100*correct/len(rows):.1f}% (95% CI [{lo:.1f}, {hi:.1f}]) [n={len(rows)}]")

    flagged = [l for l in named if l["risk_group"] == "flagged"]
    unflagged = [l for l in named if l["risk_group"] == "unflagged"]
    if flagged and unflagged:
        f_correct = sum(l["exact_match"] for l in flagged)
        u_correct = sum(l["exact_match"] for l in unflagged)
        p = fisher_exact_two_sided(u_correct, len(unflagged) - u_correct, f_correct, len(flagged) - f_correct)
        print("\n=== Risk-flag correlation with correctness (unbiased sample) ===")
        lo, hi = wilson(u_correct, len(unflagged))
        print(f"  Unflagged: {u_correct}/{len(unflagged)} = {100*u_correct/len(unflagged):.1f}% (95% CI [{lo:.1f}, {hi:.1f}])")
        lo, hi = wilson(f_correct, len(flagged))
        print(f"  Flagged:   {f_correct}/{len(flagged)} = {100*f_correct/len(flagged):.1f}% (95% CI [{lo:.1f}, {hi:.1f}])")
        print(f"  Fisher exact two-sided p = {p:.4g}")
    else:
        print("\n(Not enough flagged/unflagged named predictions in this sample for the risk-flag check.)")


if __name__ == "__main__":
    main()
