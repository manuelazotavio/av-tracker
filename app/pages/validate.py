"""Validation page: interactively correct speaker attributions, save to DB."""
import os
import sys
import re
import json
import glob
import streamlit as st
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database import get_db

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SESSIONS_DIR = os.path.join(BASE_DIR, "realtime_sessions")


def _find_transcripts():
    return sorted(glob.glob(os.path.join(SESSIONS_DIR, "transcript_*.txt")), reverse=True)


def _parse_transcript(path):
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
                })
    return entries


def _is_generic(name):
    return bool(re.match(r'^(spk_\d+|Person_\d+|Desconhecido_\d+|Unknown)$', name))


def _compute_metrics(entries, corrections):
    validated = [(e, corrections.get(i, e["speaker"])) for i, e in enumerate(entries) if i in corrections]
    if not validated:
        return {}

    correct = sum(1 for e, c in validated if e["speaker"] == c)
    incorrect = sum(1 for e, c in validated if e["speaker"] != c)
    total = len(validated)

    named_pairs = [(e, c) for e, c in validated if not _is_generic(e["speaker"])]
    named_correct = sum(1 for e, c in named_pairs if e["speaker"] == c)

    speaker_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    for e, actual in validated:
        predicted = e["speaker"]
        speaker_stats[actual]["total"] += 1
        if predicted == actual:
            speaker_stats[actual]["correct"] += 1
        else:
            speaker_stats[actual]["incorrect"] += 1
        confusion[predicted][actual] += 1

    return {
        "total_lines": len(entries),
        "validated_lines": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(correct / total, 4) if total else 0,
        "error_rate": round(incorrect / total, 4) if total else 0,
        "named_correct": named_correct,
        "named_accuracy": round(named_correct / len(named_pairs), 4) if named_pairs else 0,
        "per_speaker": {
            name: {**s, "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0}
            for name, s in sorted(speaker_stats.items())
        },
        "confusion_matrix": {pred: dict(acts) for pred, acts in confusion.items()},
    }


def render():
    st.title("✅ Validate Session")

    transcripts = _find_transcripts()
    if not transcripts:
        st.info("No transcripts found.")
        return

    selected = st.selectbox(
        "Select transcript",
        transcripts,
        format_func=lambda x: os.path.basename(x),
    )

    entries = _parse_transcript(selected)
    if not entries:
        st.warning("No entries in transcript.")
        return

    st.write(f"**{len(entries)} lines** — correct speakers below, then click Save.")

    all_speakers = sorted(set(e["speaker"] for e in entries))

    key = f"corrections_{os.path.basename(selected)}"
    if key not in st.session_state:
        st.session_state[key] = {}
    corrections = st.session_state[key]

    # Bulk rename
    with st.expander("Bulk rename"):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            bulk_from = st.selectbox("Replace", all_speakers, key="bulk_from")
        with col2:
            bulk_to = st.text_input("With", key="bulk_to")
        with col3:
            st.write("")
            st.write("")
            if st.button("Apply") and bulk_to:
                for i, e in enumerate(entries):
                    if e["speaker"] == bulk_from:
                        corrections[i] = bulk_to
                st.rerun()

    # Lines
    for i, entry in enumerate(entries):
        original = entry["speaker"]
        current = corrections.get(i, original)
        is_corrected = current != original

        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            icon = "🟢" if not is_corrected else "🔴"
            st.write(f"{icon} `{entry['timestamp']}`")
        with col2:
            st.write(entry["text"][:100])
        with col3:
            new_val = st.text_input("Speaker", value=current, key=f"spk_{i}",
                                    label_visibility="collapsed")
            if new_val != corrections.get(i, original):
                corrections[i] = new_val

    st.session_state[key] = corrections
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept all unchanged", use_container_width=True):
            for i, e in enumerate(entries):
                if i not in corrections:
                    corrections[i] = e["speaker"]
            st.session_state[key] = corrections
            st.rerun()

    with col2:
        if st.button("💾 Save Validation", type="primary", use_container_width=True):
            for i, e in enumerate(entries):
                if i not in corrections:
                    corrections[i] = e["speaker"]

            metrics = _compute_metrics(entries, corrections)

            ts_match = re.search(r'(\d{8}_\d{6})', os.path.basename(selected))
            session_id = ts_match.group(1) if ts_match else datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save validated transcript file
            val_file = os.path.join(SESSIONS_DIR, f"validated_{session_id}.txt")
            with open(val_file, "w", encoding="utf-8") as f:
                for i, e in enumerate(entries):
                    actual = corrections.get(i, e["speaker"])
                    marker = f"  [was: {e['speaker']}]" if actual != e["speaker"] else ""
                    f.write(f"[{e['timestamp']}] {actual}: {e['text']}{marker}\n")

            # Save validation JSON file (backward compat)
            metrics["session_id"] = session_id
            metrics["validated_at"] = datetime.now().isoformat()
            metrics_file = os.path.join(SESSIONS_DIR, f"validation_{session_id}.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            # Save to database
            db = get_db()
            session_row = db.get_session(session_id)
            if session_row:
                lines = [(i, entries[i]["speaker"], corrections[i])
                         for i in sorted(corrections.keys())]
                db.save_validation(
                    session_pk=session_row["id"],
                    accuracy=metrics["accuracy"],
                    named_accuracy=metrics.get("named_accuracy", 0),
                    total_lines=metrics["validated_lines"],
                    correct=metrics["correct"],
                    incorrect=metrics["incorrect"],
                    details=metrics,
                    lines=lines,
                )

            # Apply feedback: deactivate wrong embeddings, rename files
            wrong_corrections = {
                i: (entries[i]["speaker"], corrections[i])
                for i in corrections
                if entries[i]["speaker"] != corrections[i]
            }
            if wrong_corrections:
                feedback_actions = db.apply_validation_feedback(session_id, wrong_corrections)
                if feedback_actions:
                    with st.expander(f"🔄 Feedback applied ({len(feedback_actions)} actions)"):
                        for a in feedback_actions:
                            st.text(a)
                    st.info("Contaminated embeddings were deactivated. Next session will use corrected data.")

            st.success(f"Saved to DB + files!")

            # Summary
            st.subheader("Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['accuracy']:.0%}")
            col2.metric("Correct", metrics["correct"])
            col3.metric("Incorrect", metrics["incorrect"])

            if metrics.get("per_speaker"):
                st.write("**Per Speaker:**")
                ps_data = [
                    {"Speaker": n, "Correct": s["correct"], "Incorrect": s["incorrect"],
                     "Total": s["total"], "Accuracy": f"{s['accuracy']:.0%}"}
                    for n, s in metrics["per_speaker"].items()
                ]
                st.dataframe(ps_data, use_container_width=True)

            if metrics.get("confusion_matrix"):
                errors = []
                for pred, acts in metrics["confusion_matrix"].items():
                    for actual, count in acts.items():
                        if pred != actual:
                            errors.append({"Predicted": pred, "Actual": actual, "Count": count})
                if errors:
                    st.write("**Errors:**")
                    st.dataframe(errors, use_container_width=True)
