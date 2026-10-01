# Ground-truth validation sample (unbiased, stratified)

Seed: `20260805` (rerun this script with the same seed to reproduce this exact list).
Sessions already in the biased development-period validation set are excluded.

| Expected speakers | Sessions | Segments |
|---|---:|---:|
| 1 | 6 | 21 |
| 2 | 3 | 74 |
| 3 | 4 | 103 |
| 4 | 3 | 25 |
| **Total** | **16** | **223** |

## How to validate each session

Either:
1. `streamlit run app/app.py` -> Validate page -> pick the transcript from the list below, correct each line, Save; or
2. `python scripts/validate_session.py realtime_sessions/transcript_<session_id>.txt` (faster, terminal-only).

Both write to the `validations` / `validation_lines` tables in `av_tracker.db`, which `paper/wvc/scripts/analyze_ground_truth.py` reads once you're done.

## Sessions to validate

| Speakers | Session ID | Segments | Transcript |
|---|---|---:|---|
| 1 | 20260316_145449 | 3 | `realtime_sessions/transcript_20260316_145449.txt` |
| 1 | 20260316_145744 | 2 | `realtime_sessions/transcript_20260316_145744.txt` |
| 1 | 20260316_150453 | 3 | `realtime_sessions/transcript_20260316_150453.txt` |
| 1 | 20260513_141719 | 8 | `realtime_sessions/transcript_20260513_141719.txt` |
| 1 | 20260518_122957 | 3 | `realtime_sessions/transcript_20260518_122957.txt` |
| 1 | 20260521_092938 | 2 | `realtime_sessions/transcript_20260521_092938.txt` |
| 2 | 20260325_094600 | 3 | `realtime_sessions/transcript_20260325_094600.txt` |
| 2 | 20260325_095359 | 36 | `realtime_sessions/transcript_20260325_095359.txt` |
| 2 | 20260325_102521 | 35 | `realtime_sessions/transcript_20260325_102521.txt` |
| 3 | 20260331_100102 | 33 | `realtime_sessions/transcript_20260331_100102.txt` |
| 3 | 20260331_101732 | 25 | `realtime_sessions/transcript_20260331_101732.txt` |
| 3 | 20260521_092211 | 22 | `realtime_sessions/transcript_20260521_092211.txt` |
| 3 | 20260521_092223 | 23 | `realtime_sessions/transcript_20260521_092223.txt` |
| 4 | 20260406_201001 | 3 | `realtime_sessions/transcript_20260406_201001.txt` |
| 4 | 20260406_203808 | 3 | `realtime_sessions/transcript_20260406_203808.txt` |
| 4 | 20260410_130527 | 19 | `realtime_sessions/transcript_20260410_130527.txt` |
