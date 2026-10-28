"""
Generates graphs from AV-Tracker session metrics JSON files.

Usage:
    python scripts/plot_metrics.py                          # latest session
    python scripts/plot_metrics.py realtime_sessions/metrics_20260316_143022.json
    python scripts/plot_metrics.py realtime_sessions/metrics_*.json   # compare sessions
"""
import json
import sys
import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_latest_metrics(sessions_dir: str = "realtime_sessions") -> str | None:
    files = sorted(glob.glob(os.path.join(sessions_dir, "metrics_*.json")))
    return files[-1] if files else None


DECISION_ORDER = [
    "VERIFIER_HIGH", "VERIFIER_MOD", "VERIFIER_WEAK", "ASD", "FACE_ONLY",
    "SINGLE_FACE", "GENDER_FACE", "GENDER_REROUTE",
    "GENDER_VERIFIER_HINT", "GENDER_OVERRIDE",
    "SESSION_HINT", "SESSION_TRACKER",
]

DECISION_COLORS = {
    "VERIFIER_HIGH": "#2ecc71",
    "VERIFIER_MOD": "#27ae60",
    "VERIFIER_WEAK": "#a3d977",
    "ASD": "#3498db",
    "FACE_ONLY": "#9b59b6",
    "SINGLE_FACE": "#e67e22",
    "GENDER_FACE": "#1abc9c",
    "GENDER_REROUTE": "#16a085",
    "GENDER_VERIFIER_HINT": "#f1c40f",
    "GENDER_OVERRIDE": "#e74c3c",
    "SESSION_HINT": "#95a5a6",
    "SESSION_TRACKER": "#7f8c8d",
}


def _color(dt: str) -> str:
    return DECISION_COLORS.get(dt, "#bdc3c7")


# ── Single-session plots ────────────────────────────────────────────────────

def plot_single_session(data: dict, out_dir: str | None = None):
    segs = data["segments"]
    summary = data["summary"]
    config = data["config"]
    session_id = data["session_id"]

    if not segs:
        print("No segments to plot.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"Session {session_id}  |  {summary['total_segments']} segments  |  "
        f"ID rate: {summary['identification_rate']:.0%}  |  "
        f"FN rate: {summary['fn_rate']:.0%}",
        fontsize=13, fontweight="bold",
    )

    # 1) Decision distribution (bar)
    ax = axes[0, 0]
    counts = summary["decision_counts"]
    labels = [d for d in DECISION_ORDER if d in counts]
    labels += sorted(set(counts) - set(DECISION_ORDER))
    vals = [counts.get(l, 0) for l in labels]
    colors = [_color(l) for l in labels]
    bars = ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Segments")
    ax.set_title("Decision Distribution")
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9)

    # 2) Verifier confidence histogram
    ax = axes[0, 1]
    raw_confs = [s["raw_conf"] for s in segs if s["raw_conf"] > 0]
    if raw_confs:
        ax.hist(raw_confs, bins=30, range=(0, 1), color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(config["verifier_threshold"], color="#e74c3c", ls="--", lw=1.5,
                   label=f"threshold={config['verifier_threshold']}")
        ax.axvline(config["verifier_confidence_min"], color="#e67e22", ls="--", lw=1.5,
                   label=f"conf_min={config['verifier_confidence_min']}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Raw Verifier Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Verifier Confidence Distribution")

    # 3) Per-speaker verifier score boxplot
    ax = axes[1, 0]
    speaker_scores = defaultdict(list)
    for s in segs:
        name = s.get("raw_best_name")
        if name and s["raw_conf"] > 0:
            speaker_scores[name].append(s["raw_conf"])
    if speaker_scores:
        sorted_speakers = sorted(speaker_scores.keys())
        box_data = [speaker_scores[sp] for sp in sorted_speakers]
        bp = ax.boxplot(box_data, labels=sorted_speakers, vert=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.6)
        ax.axhline(config["verifier_threshold"], color="#e74c3c", ls="--", lw=1,
                   label=f"threshold={config['verifier_threshold']}")
        ax.axhline(config["verifier_confidence_min"], color="#e67e22", ls="--", lw=1,
                   label=f"conf_min={config['verifier_confidence_min']}")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Raw Confidence")
    ax.set_title("Per-Speaker Verifier Scores")

    # 4) Margin distribution
    ax = axes[1, 1]
    margins = [s["margin"] for s in segs if s["margin"] is not None]
    if margins:
        ax.hist(margins, bins=25, range=(0, max(margins) + 0.02),
                color="#9b59b6", edgecolor="white", alpha=0.85)
        ax.axvline(0.04, color="#e74c3c", ls="--", lw=1.5, label="margin threshold=0.04")
        ax.legend(fontsize=8)
    ax.set_xlabel("Margin (1st - 2nd candidate)")
    ax.set_ylabel("Count")
    ax.set_title("Candidate Margin Distribution")

    # 5) Rolling identification, FN & FP rate
    ax = axes[2, 0]
    window = max(5, len(segs) // 10)
    if len(segs) >= window:
        id_flags = np.array([1.0 if s["is_identified"] else 0.0 for s in segs])
        fn_flags = np.array([1.0 if s["is_fn"] else 0.0 for s in segs])
        fp_flags = np.array([1.0 if s.get("fp_risk") in ("high", "medium") else 0.0 for s in segs])
        kernel = np.ones(window) / window
        id_rolling = np.convolve(id_flags, kernel, mode="valid")
        fn_rolling = np.convolve(fn_flags, kernel, mode="valid")
        fp_rolling = np.convolve(fp_flags, kernel, mode="valid")
        x = np.arange(len(id_rolling))
        ax.plot(x, id_rolling, color="#2ecc71", lw=2, label="Identification rate")
        ax.plot(x, fn_rolling, color="#e74c3c", lw=1.5, label="FN rate")
        ax.plot(x, fp_rolling, color="#e67e22", lw=1.5, ls="--", label="FP risk rate")
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=8)
    ax.set_xlabel(f"Segment index (rolling window={window})")
    ax.set_ylabel("Rate")
    ax.set_title("Identification, FN & FP Risk Over Time")

    # 6) Latency breakdown
    ax = axes[2, 1]
    seg_indices = np.arange(len(segs))
    whisper_ms = [s["whisper_ms"] for s in segs]
    verifier_ms = [s["verifier_ms"] for s in segs]
    ax.bar(seg_indices, whisper_ms, color="#3498db", alpha=0.7, label="Whisper")
    ax.bar(seg_indices, verifier_ms, bottom=whisper_ms, color="#e67e22", alpha=0.7, label="Verifier")
    ax.set_xlabel("Segment index")
    ax.set_ylabel("ms")
    ax.set_title("Latency per Segment")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"session_{session_id}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    # --- Face recognition metrics (separate figure) ---
    face = data.get("face")
    if face:
        plot_face_session(face, session_id, out_dir=out_dir)

    plt.show()


def plot_face_session(face: dict, session_id: str, out_dir: str | None = None):
    """Face recognition metrics for a single session."""
    stats = face["match_score_stats"]
    samples = face.get("match_scores_sample", [])
    fn_events = face.get("face_fn_events", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Face Recognition — Session {session_id}  |  "
        f"Tracks: {face['total_tracks']}  |  "
        f"Identified: {face['tracks_identified']}  |  "
        f"Face FN: {face['face_fn_count']}",
        fontsize=13, fontweight="bold",
    )

    # 1) Face match score histogram
    ax = axes[0, 0]
    if samples:
        scores = [s["best_score"] for s in samples]
        ax.hist(scores, bins=30, range=(0, 1), color="#3498db", edgecolor="white", alpha=0.85)
        thr = face.get("match_threshold", 0.45)
        ax.axvline(thr, color="#e74c3c", ls="--", lw=1.5, label=f"threshold={thr}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Best Cosine Similarity")
    ax.set_ylabel("Count (sampled)")
    ax.set_title("Face Match Score Distribution")

    # 2) Per-person face score boxplot
    ax = axes[0, 1]
    person_scores = defaultdict(list)
    for s in samples:
        person_scores[s["best_name"]].append(s["best_score"])
    if person_scores:
        # Filter out "Unknown" for cleaner visualization
        names = sorted(n for n in person_scores if n != "Unknown")
        if names:
            box_data = [person_scores[n] for n in names]
            bp = ax.boxplot(box_data, labels=names, vert=True, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#9b59b6")
                patch.set_alpha(0.6)
            thr = face.get("match_threshold", 0.45)
            ax.axhline(thr, color="#e74c3c", ls="--", lw=1, label=f"threshold={thr}")
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Per-Person Face Scores")

    # 3) Rolling face match rate over time
    ax = axes[1, 0]
    if len(samples) >= 10:
        matched_flags = np.array([1.0 if s["matched"] else 0.0 for s in samples])
        window = max(5, len(samples) // 10)
        kernel = np.ones(window) / window
        rolling = np.convolve(matched_flags, kernel, mode="valid")
        ax.plot(np.arange(len(rolling)), rolling, color="#2ecc71", lw=2)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel(f"Sample index (1 sample = 30 frames)")
    ax.set_ylabel("Match Rate")
    ax.set_title("Face Match Rate Over Time")

    # 4) Summary stats table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["Total tracks", str(face["total_tracks"])],
        ["Identified (real name)", str(face["tracks_identified"])],
        ["Generic (Person_N)", str(face["tracks_generic"])],
        ["Face FNs (voice rescued)", str(face["face_fn_count"])],
        ["Auto-enrollments", str(face["enrollments"])],
        ["Known embeddings", str(face["known_embeddings_count"])],
        ["Total frames", str(face["total_frames"])],
        ["Match threshold", str(face.get("match_threshold", "?"))],
        ["Mean score", f"{stats['mean']:.4f}"],
        ["Median score", f"{stats['median']:.4f}"],
        ["% matched", f"{stats['pct_matched']:.1%}"],
        ["Mean (matched)", f"{stats['mean_matched']:.4f}"],
        ["Mean (unmatched)", f"{stats['mean_unmatched']:.4f}"],
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title("Face Recognition Summary", fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"face_{session_id}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")


# ── Multi-session comparison ────────────────────────────────────────────────

def plot_compare_sessions(paths: list[str], out_dir: str | None = None):
    sessions = []
    for p in paths:
        d = load_metrics(p)
        if d["segments"]:
            sessions.append(d)
    if not sessions:
        print("No sessions with data.")
        return

    sessions.sort(key=lambda d: d["session_id"])
    labels = [s["session_id"] for s in sessions]
    has_face = any("face" in s for s in sessions)
    n_rows = 3 if has_face else 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    fig.suptitle(f"Session Comparison ({len(sessions)} sessions)", fontsize=13, fontweight="bold")

    # Load validation data if available
    val_data = {}
    for s in sessions:
        sid = s["session_id"]
        val_path = os.path.join("realtime_sessions", f"validation_{sid}.json")
        if os.path.exists(val_path):
            with open(val_path, encoding="utf-8") as f:
                val_data[sid] = json.load(f)

    # 1) Identification rate, FN rate & validated accuracy
    ax = axes[0, 0]
    id_rates = [s["summary"]["identification_rate"] for s in sessions]
    fn_rates = [s["summary"]["fn_rate"] for s in sessions]
    x = np.arange(len(labels))
    fp_rates = [s["summary"].get("fp_rate", 0) for s in sessions]
    bw = 0.2
    ax.bar(x - 1.5*bw, id_rates, bw, color="#2ecc71", label="Voice ID rate")
    ax.bar(x - 0.5*bw, fn_rates, bw, color="#e74c3c", label="Voice FN rate")
    ax.bar(x + 0.5*bw, fp_rates, bw, color="#e67e22", label="FP risk rate")
    # Overlay validated accuracy as line if available
    val_acc = [val_data.get(s["session_id"], {}).get("accuracy") for s in sessions]
    if any(v is not None for v in val_acc):
        _vx = [i for i, v in enumerate(val_acc) if v is not None]
        _vy = [val_acc[i] for i in _vx]
        ax.plot(_vx, _vy, "D-", color="#8e44ad", lw=2, ms=8, label="Validated accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Voice Identification & FN Rate per Session")
    ax.legend(fontsize=8)

    # 2) Decision distribution stacked bar
    ax = axes[0, 1]
    all_types = set()
    for s in sessions:
        all_types.update(s["summary"]["decision_counts"].keys())
    ordered = [d for d in DECISION_ORDER if d in all_types]
    ordered += sorted(all_types - set(DECISION_ORDER))
    bottoms = np.zeros(len(sessions))
    for dt in ordered:
        vals = []
        for s in sessions:
            total = s["summary"]["total_segments"]
            count = s["summary"]["decision_counts"].get(dt, 0)
            vals.append(count / total if total else 0)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, color=_color(dt), label=dt, width=0.6)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Decision Distribution (%)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # 3) Average verifier confidence trend
    ax = axes[1, 0]
    avg_confs = [s["summary"]["avg_raw_conf"] for s in sessions]
    med_confs = [s["summary"]["median_raw_conf"] for s in sessions]
    ax.plot(x, avg_confs, "o-", color="#3498db", lw=2, label="Mean")
    ax.plot(x, med_confs, "s--", color="#9b59b6", lw=1.5, label="Median")
    last_cfg = sessions[-1]["config"]
    ax.axhline(last_cfg["verifier_threshold"], color="#e74c3c", ls=":", lw=1,
               label=f"threshold={last_cfg['verifier_threshold']}")
    ax.axhline(last_cfg["verifier_confidence_min"], color="#e67e22", ls=":", lw=1,
               label=f"conf_min={last_cfg['verifier_confidence_min']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Confidence")
    ax.set_title("Verifier Confidence Trend")
    ax.legend(fontsize=8)

    # 4) Latency trend
    ax = axes[1, 1]
    avg_seg_ms = [s["summary"]["avg_segment_ms"] for s in sessions]
    avg_w_ms = [s["summary"]["avg_whisper_ms"] for s in sessions]
    avg_v_ms = [s["summary"]["avg_verifier_ms"] for s in sessions]
    ax.plot(x, avg_seg_ms, "o-", color="#2c3e50", lw=2, label="Total segment")
    ax.plot(x, avg_w_ms, "s--", color="#3498db", lw=1.5, label="Whisper")
    ax.plot(x, avg_v_ms, "^--", color="#e67e22", lw=1.5, label="Verifier")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ms")
    ax.set_title("Average Latency per Session")
    ax.legend(fontsize=8)

    # 5-6) Face recognition metrics (if available)
    if has_face:
        # 5) Face match rate + face FN trend
        ax = axes[2, 0]
        face_match_rates = []
        face_fn_counts = []
        for s in sessions:
            f = s.get("face", {})
            ms = f.get("match_score_stats", {})
            face_match_rates.append(ms.get("pct_matched", 0))
            face_fn_counts.append(f.get("face_fn_count", 0))
        ax.plot(x, face_match_rates, "o-", color="#2ecc71", lw=2, label="Face match rate")
        ax2 = ax.twinx()
        ax2.bar(x, face_fn_counts, color="#e74c3c", alpha=0.3, width=0.4, label="Face FN count")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_ylabel("Match Rate")
        ax2.set_ylabel("Face FN Count")
        ax.set_title("Face Match Rate & FN Trend")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        # 6) Face score trend (mean matched vs unmatched)
        ax = axes[2, 1]
        mean_matched = []
        mean_unmatched = []
        for s in sessions:
            ms = s.get("face", {}).get("match_score_stats", {})
            mean_matched.append(ms.get("mean_matched", 0))
            mean_unmatched.append(ms.get("mean_unmatched", 0))
        ax.plot(x, mean_matched, "o-", color="#2ecc71", lw=2, label="Mean (matched)")
        ax.plot(x, mean_unmatched, "s--", color="#e74c3c", lw=1.5, label="Mean (unmatched)")
        # Show face threshold from last session
        face_thr = sessions[-1].get("config", {}).get("face_match_threshold")
        if face_thr:
            ax.axhline(face_thr, color="#e67e22", ls=":", lw=1, label=f"threshold={face_thr}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Face Score Trend")
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "session_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    out_dir = "realtime_sessions/plots"

    if not args:
        latest = find_latest_metrics()
        if not latest:
            print("No metrics files found in realtime_sessions/. Run a session first.")
            sys.exit(1)
        print(f"Loading latest: {latest}")
        args = [latest]

    # Expand globs on Windows (shell doesn't do it)
    expanded = []
    for a in args:
        if "*" in a or "?" in a:
            expanded.extend(sorted(glob.glob(a)))
        else:
            expanded.append(a)

    if not expanded:
        print("No matching files found.")
        sys.exit(1)

    if len(expanded) == 1:
        data = load_metrics(expanded[0])
        plot_single_session(data, out_dir=out_dir)
    else:
        print(f"Comparing {len(expanded)} sessions...")
        plot_compare_sessions(expanded, out_dir=out_dir)


if __name__ == "__main__":
    main()
