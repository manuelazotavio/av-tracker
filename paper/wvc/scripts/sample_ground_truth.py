#!/usr/bin/env python3
"""Draw an unbiased, stratified sample of sessions for manual ground-truth
validation, addressing the WVC 2026 review's request for a properly sampled
(not error-selected) validated subset of ~100-200 segments.

The existing validation set (24 sessions / 330 segments, see
paper/conict-reproducibility) was built by hand-picking sessions already
*suspected* of containing errors during development, so it cannot support a
claim about overall accuracy. This script samples from the sessions that have
NEVER been validated, with a fixed random seed, stratified by the session's
configured number of expected speakers, so the resulting sample is not
conditioned on anyone's belief about where the errors are.

Sampling policy per group size (tune via --target-segments-* flags; default
budget totals ~180 segments across ~15 sessions):
  - The 3-speaker stratum is always sampled to its FULL never-validated
    population (103 segments / 4 sessions) and is not configurable, since
    that is the specific cell the review flagged as statistically fragile
    (Table II's 68.9% risk spike, previously resting on only 4 sessions with
    no ground-truth check at all) and its total population is already small.
  - The 1-, 2-, and 4-speaker strata are sampled down to a segment budget
    each, since their populations are larger and full coverage would impose
    disproportionate manual-labeling effort for comparatively little gain in
    interval width.

Output: a manifest (CSV + human-readable Markdown) listing which transcript
files to run through app/pages/validate.py (Streamlit) or
scripts/validate_session.py (CLI). This script only *selects* the sample; it
does not and cannot perform the actual validation, which requires a human
listening/watching to determine the true speaker per segment.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[3] / "av_tracker.db")
    parser.add_argument("--seed", type=int, default=20260805, help="Fixed seed for reproducible sampling")
    parser.add_argument("--target-segments-1speaker", type=int, default=20,
                         help="Segment budget for the 1-speaker stratum (never-validated population: 69)")
    parser.add_argument("--target-segments-2speaker", type=int, default=35,
                         help="Segment budget for the 2-speaker stratum (never-validated population: 118)")
    parser.add_argument("--target-segments-4speaker", type=int, default=22,
                         help="Segment budget for the 4-speaker stratum (never-validated population: 1000+)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "ground_truth_sample")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    sessions = con.execute(
        """
        SELECT s.id, s.session_id, s.started_at, s.config, COUNT(g.id) AS segment_count
        FROM sessions s JOIN segments g ON g.session_pk = s.id
        GROUP BY s.id
        """
    ).fetchall()
    already_validated_pks = {row["session_pk"] for row in con.execute("SELECT DISTINCT session_pk FROM validations")}

    by_size: dict[int, list[sqlite3.Row]] = {}
    for row in sessions:
        cfg = json.loads(row["config"]) if row["config"] else {}
        n = cfg.get("num_speakers")
        if row["id"] in already_validated_pks:
            continue  # excluded: these were selected because suspected of errors
        by_size.setdefault(n, []).append(row)

    rng = random.Random(args.seed)
    manifest: list[dict] = []

    for size, pool in sorted(by_size.items(), key=lambda kv: (kv[0] is None, kv[0])):
        pool = sorted(pool, key=lambda r: r["session_id"])  # stable order before shuffling
        rng.shuffle(pool)

        budget = {
            1: args.target_segments_1speaker,
            2: args.target_segments_2speaker,
            4: args.target_segments_4speaker,
        }.get(size)

        if budget is None:
            # 3-speaker (and any other un-budgeted stratum): take the whole
            # never-validated population. Not configurable for group size 3 --
            # see module docstring for why.
            chosen = pool
        else:
            # Keep at least a few distinct sessions per budgeted stratum even
            # if that means overshooting the segment budget slightly -- a
            # single oversized session would leave that group size with the
            # same one-session-dominates risk flagged for the 3-speaker cell.
            min_sessions = min(3, len(pool))
            chosen, running = [], 0
            for row in pool:
                if running >= budget and len(chosen) >= min_sessions:
                    break
                chosen.append(row)
                running += row["segment_count"]

        for row in chosen:
            manifest.append(
                {
                    "expected_participants": size,
                    "session_id": row["session_id"],
                    "session_pk": row["id"],
                    "started_at": row["started_at"],
                    "segment_count": row["segment_count"],
                    "transcript_file": f"realtime_sessions/transcript_{row['session_id']}.txt",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        import csv
        fields = ["expected_participants", "session_id", "session_pk", "started_at", "segment_count", "transcript_file"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    totals: dict[int, dict[str, int]] = {}
    for row in manifest:
        t = totals.setdefault(row["expected_participants"], {"sessions": 0, "segments": 0})
        t["sessions"] += 1
        t["segments"] += row["segment_count"]

    md_path = args.output_dir / "MANIFEST.md"
    lines = [
        "# Ground-truth validation sample (unbiased, stratified)",
        "",
        f"Seed: `{args.seed}` (rerun this script with the same seed to reproduce this exact list).",
        "Sessions already in the biased development-period validation set are excluded.",
        "",
        "| Expected speakers | Sessions | Segments |",
        "|---|---:|---:|",
    ]
    grand_sessions = grand_segments = 0
    for size in sorted(totals):
        t = totals[size]
        lines.append(f"| {size} | {t['sessions']} | {t['segments']} |")
        grand_sessions += t["sessions"]
        grand_segments += t["segments"]
    lines.append(f"| **Total** | **{grand_sessions}** | **{grand_segments}** |")
    lines += [
        "",
        "## How to validate each session",
        "",
        "Either:",
        "1. `streamlit run app/app.py` -> Validate page -> pick the transcript from the list below, correct each line, Save; or",
        "2. `python scripts/validate_session.py realtime_sessions/transcript_<session_id>.txt` (faster, terminal-only).",
        "",
        "Both write to the `validations` / `validation_lines` tables in `av_tracker.db`, which "
        "`paper/wvc/scripts/analyze_ground_truth.py` reads once you're done.",
        "",
        "## Sessions to validate",
        "",
        "| Speakers | Session ID | Segments | Transcript |",
        "|---|---|---:|---|",
    ]
    for row in sorted(manifest, key=lambda r: (r["expected_participants"], r["session_id"])):
        lines.append(
            f"| {row['expected_participants']} | {row['session_id']} | {row['segment_count']} | `{row['transcript_file']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print()
    for size in sorted(totals):
        t = totals[size]
        print(f"  {size} speakers: {t['sessions']} sessions, {t['segments']} segments")
    print(f"  TOTAL: {grand_sessions} sessions, {grand_segments} segments")


if __name__ == "__main__":
    main()
