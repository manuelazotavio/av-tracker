"""
Post-session validation: manually correct speaker attributions and compute accuracy.

Usage:
    python scripts/validate_session.py                              # latest session
    python scripts/validate_session.py realtime_sessions/transcript_20260331_135527.txt

Workflow:
    1. Shows each line of the transcript
    2. Press Enter to accept, or type the correct speaker name to fix
    3. Saves a validated transcript and accuracy metrics (JSON)

The validation JSON can be loaded by plot_metrics.py for cross-session accuracy tracking.
"""
import os
import sys
import json
import re
from datetime import datetime
from collections import defaultdict


def load_transcript(path: str) -> list[dict]:
    """Parse transcript lines into structured entries."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^\[(\d{2}:\d{2}:\d{2})\]\s+(.+?):\s+(.+)$', line)
            if m:
                entries.append({
                    "timestamp": m.group(1),
                    "speaker": m.group(2),
                    "text": m.group(3),
                    "corrected": None,  # filled during validation
                })
    return entries


def is_generic(name: str) -> bool:
    return bool(re.match(r'^(spk_\d+|Person_\d+|Desconhecido_\d+|Unknown)$', name))


def validate_interactive(entries: list[dict]) -> list[dict]:
    """Interactively validate each line. Enter=accept, name=correct, s=skip, q=quit."""
    print("\n" + "="*70)
    print("  POST-SESSION VALIDATION")
    print("  Enter=accept | type correct name | s=skip | q=quit early")
    print("="*70 + "\n")

    known_corrections = {}  # original -> corrected (reuse across lines)

    for i, entry in enumerate(entries):
        speaker = entry["speaker"]
        text = entry["text"][:80]
        ts = entry["timestamp"]

        # Auto-apply previous corrections for the same speaker
        if speaker in known_corrections:
            entry["corrected"] = known_corrections[speaker]
            tag = f" -> {entry['corrected']}" if entry["corrected"] != speaker else ""
            print(f"  [{ts}] {speaker}{tag}: {text}")
            continue

        color_speaker = speaker if not is_generic(speaker) else f"*{speaker}*"
        prompt = f"  [{ts}] {color_speaker}: {text}\n    Correct speaker [{speaker}]: "
        resp = input(prompt).strip()

        if resp.lower() == 'q':
            # Mark remaining as not validated
            for j in range(i, len(entries)):
                entries[j]["corrected"] = None
            break
        elif resp.lower() == 's':
            entry["corrected"] = None
            continue
        elif resp == '':
            entry["corrected"] = speaker  # accepted as-is
            known_corrections[speaker] = speaker
        else:
            entry["corrected"] = resp
            known_corrections[speaker] = resp
            print(f"    -> Corrected to: {resp}")

    return entries


def compute_metrics(entries: list[dict]) -> dict:
    """Compute accuracy metrics from validated entries."""
    validated = [e for e in entries if e["corrected"] is not None]
    if not validated:
        return {"total": 0, "validated": 0}

    correct = sum(1 for e in validated if e["speaker"] == e["corrected"])
    incorrect = sum(1 for e in validated if e["speaker"] != e["corrected"])
    generic_correct = sum(1 for e in validated if is_generic(e["speaker"]) and e["speaker"] == e["corrected"])
    generic_incorrect = sum(1 for e in validated if is_generic(e["speaker"]) and e["speaker"] != e["corrected"])
    named_correct = sum(1 for e in validated if not is_generic(e["speaker"]) and e["speaker"] == e["corrected"])
    named_incorrect = sum(1 for e in validated if not is_generic(e["speaker"]) and e["speaker"] != e["corrected"])

    # Per-speaker breakdown
    speaker_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))  # predicted -> actual -> count
    for e in validated:
        predicted = e["speaker"]
        actual = e["corrected"]
        speaker_stats[actual]["total"] += 1
        if predicted == actual:
            speaker_stats[actual]["correct"] += 1
        else:
            speaker_stats[actual]["incorrect"] += 1
        confusion[predicted][actual] += 1

    total = len(validated)
    return {
        "total_lines": len(entries),
        "validated_lines": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(correct / total, 4) if total else 0,
        "error_rate": round(incorrect / total, 4) if total else 0,
        "generic_correct": generic_correct,
        "generic_incorrect": generic_incorrect,
        "named_correct": named_correct,
        "named_incorrect": named_incorrect,
        "named_accuracy": round(named_correct / (named_correct + named_incorrect), 4) if (named_correct + named_incorrect) else 0,
        "per_speaker": {
            name: {
                "correct": s["correct"],
                "incorrect": s["incorrect"],
                "total": s["total"],
                "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0,
            }
            for name, s in sorted(speaker_stats.items())
        },
        "confusion_matrix": {pred: dict(actuals) for pred, actuals in confusion.items()},
    }


def save_results(entries: list[dict], metrics: dict, transcript_path: str):
    """Save validated transcript and metrics."""
    base = os.path.splitext(transcript_path)[0]
    ts_match = re.search(r'(\d{8}_\d{6})', base)
    session_id = ts_match.group(1) if ts_match else datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.dirname(transcript_path) or "realtime_sessions"

    # Validated transcript
    val_file = os.path.join(output_dir, f"validated_{session_id}.txt")
    with open(val_file, "w", encoding="utf-8") as f:
        for e in entries:
            actual = e["corrected"] or e["speaker"]
            marker = ""
            if e["corrected"] is not None and e["corrected"] != e["speaker"]:
                marker = f"  [was: {e['speaker']}]"
            f.write(f"[{e['timestamp']}] {actual}: {e['text']}{marker}\n")
    print(f"\nSaved validated transcript: {val_file}")

    # Metrics JSON
    metrics["session_id"] = session_id
    metrics["validated_at"] = datetime.now().isoformat()
    metrics_file = os.path.join(output_dir, f"validation_{session_id}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved validation metrics: {metrics_file}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Total lines:      {metrics['total_lines']}")
    print(f"  Validated:         {metrics['validated_lines']}")
    print(f"  Correct:           {metrics['correct']}")
    print(f"  Incorrect:         {metrics['incorrect']}")
    print(f"  Accuracy:          {metrics['accuracy']:.1%}")
    print(f"  Named accuracy:    {metrics['named_accuracy']:.1%}")
    print(f"\n  Per speaker:")
    for name, s in metrics["per_speaker"].items():
        print(f"    {name:20s}  {s['correct']}/{s['total']}  ({s['accuracy']:.0%})")
    if metrics.get("confusion_matrix"):
        print(f"\n  Confusion (predicted -> actual):")
        for pred, actuals in sorted(metrics["confusion_matrix"].items()):
            for actual, count in sorted(actuals.items()):
                if pred != actual:
                    print(f"    {pred:20s} -> {actual:20s}  x{count}")
    print()


def find_latest_transcript(sessions_dir="realtime_sessions"):
    import glob
    files = sorted(glob.glob(os.path.join(sessions_dir, "transcript_*.txt")))
    return files[-1] if files else None


def main():
    args = sys.argv[1:]
    if args:
        path = args[0]
    else:
        path = find_latest_transcript()
        if not path:
            print("No transcript found in realtime_sessions/")
            sys.exit(1)
        print(f"Using latest: {path}")

    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    entries = load_transcript(path)
    if not entries:
        print("No entries found in transcript.")
        sys.exit(1)

    print(f"Loaded {len(entries)} lines from {path}")
    entries = validate_interactive(entries)
    metrics = compute_metrics(entries)
    save_results(entries, metrics, path)


if __name__ == "__main__":
    main()
