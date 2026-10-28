"""Metrics dashboard — visual, at-a-glance session analysis."""
import os
import sys
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database import get_db

# ── Palette & helpers ──────────────────────────────────────────────────────────
PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e", "#16a085", "#8e44ad"]

GENERIC_RE = __import__("re").compile(r"^(spk_\d+|Person_\d+|Desconhecido_\d+|Unknown(_\d+)?)$")


def _is_generic(name: str) -> bool:
    return not name or bool(GENERIC_RE.match(name))


def _simplify_decision(d: str) -> str:
    """Map raw decision strings to human-readable categories."""
    if not d:
        return "Unknown"
    d_up = d.upper()
    if d_up.startswith("VERIFIER"):
        return "🎤 Voice match"
    if d_up.startswith("ASD"):
        return "👄 Lip movement"
    if d_up.startswith(("FACE_ONLY", "SINGLE_FACE", "GENDER_FACE", "GENDER_REROUTE")):
        return "📷 Camera / gender"
    if d_up.startswith("SESSION_FACE"):
        return "🔗 Face-session link"
    if d_up.startswith("SESSION_HINT"):
        return "📊 Diarization"
    if d_up.startswith("SESSION_TRACKER"):
        return "🔄 Voice tracking"
    return "❓ Other"


def _display_name(seg) -> str:
    return seg.get("final_name") or seg.get("final_speaker") or "?"


def _health_badge(id_rate: float) -> str:
    if id_rate >= 0.80:
        return "🟢 Good"
    if id_rate >= 0.50:
        return "🟡 Partial"
    return "🔴 Low"


def _speaker_color(speakers: list, name: str) -> str:
    try:
        return PALETTE[speakers.index(name) % len(PALETTE)]
    except ValueError:
        return "#95a5a6"


# ── Main render ───────────────────────────────────────────────────────────────
def render():
    st.title("📊 Metrics")

    db = get_db()
    sessions = db.list_sessions()

    if not sessions:
        st.info("No sessions found. Run a session first.")
        return

    # ── Session selector ──
    session_ids = [s["session_id"] for s in sessions]
    selected_id = st.selectbox("Session", session_ids, index=0,
                               label_visibility="collapsed")
    data = next(s for s in sessions if s["session_id"] == selected_id)
    summary = data["summary"]
    config  = data["config"]

    session_row = db.get_session(selected_id)
    session_pk  = session_row["id"] if session_row else None

    segs = db.get_segments(session_pk) if session_pk else []
    val  = db.get_validation(session_pk) if session_pk else None

    # ── KPI bar ──
    total_segs = summary.get("total_segments", 0)
    id_rate    = summary.get("identification_rate", 0)
    avg_conf   = summary.get("avg_raw_conf", 0)
    n_speakers = summary.get("unique_speakers", 0)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Transcribed segments", total_segs)
    k2.metric("Distinct speakers", n_speakers)
    k3.metric("Identified", f"{id_rate:.0%}")
    k4.metric("Avg confidence", f"{avg_conf:.2f}")
    k5.metric("Quality", _health_badge(id_rate))

    if val:
        acc_str  = f"**{val['accuracy']:.0%}**"
        nacc_str = f"  |  named only: **{val['named_accuracy']:.0%}**" if val.get("named_accuracy") else ""
        st.success(f"✅ Validation available — real accuracy: {acc_str}{nacc_str}")

    st.divider()

    # ── Speaker cards ──
    if segs:
        sp_stats: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "conf_sum": 0.0, "conf_n": 0,
            "decisions": defaultdict(int),
        })
        for seg in segs:
            name = _display_name(seg)
            sp_stats[name]["count"] += 1
            if seg.get("raw_conf"):
                sp_stats[name]["conf_sum"] += seg["raw_conf"]
                sp_stats[name]["conf_n"]   += 1
            dec = _simplify_decision(seg.get("decision", ""))
            sp_stats[name]["decisions"][dec] += 1

        all_speakers = sorted(sp_stats.keys(),
                              key=lambda n: (int(_is_generic(n)), -sp_stats[n]["count"]))

        st.subheader("👥 Speakers found")
        cols = st.columns(min(len(all_speakers), 4))
        for i, sp_name in enumerate(all_speakers):
            sd     = sp_stats[sp_name]
            avg_c  = sd["conf_sum"] / sd["conf_n"] if sd["conf_n"] else None
            main_dec = max(sd["decisions"], key=sd["decisions"].get) if sd["decisions"] else "—"
            color  = _speaker_color(all_speakers, sp_name)
            badge  = "✅ identified" if not _is_generic(sp_name) else "❓ unnamed"

            with cols[i % len(cols)]:
                st.markdown(
                    f'<div style="border-left:4px solid {color};padding:8px 12px;'
                    f'border-radius:4px;background:#1e1e1e;margin-bottom:8px">'
                    f'<b style="font-size:15px">{sp_name}</b><br>'
                    f'<span style="font-size:11px;color:#aaa">{badge}</span></div>',
                    unsafe_allow_html=True,
                )
                st.metric("Segments", sd["count"])
                if avg_c is not None:
                    st.metric("Avg conf", f"{avg_c:.2f}")
                st.caption(f"Main method: {main_dec}")

    st.divider()

    # ── Timeline ──
    if segs:
        st.subheader("🕐 Who spoke when")
        st.caption("Each square = one segment. Hover to see the transcribed text.")

        fig = go.Figure()
        for sp_name in all_speakers:
            color = _speaker_color(all_speakers, sp_name)
            xs, hovers = [], []
            for i, seg in enumerate(segs):
                if _display_name(seg) == sp_name:
                    xs.append(i)
                    txt  = (seg.get("text") or "")[:80]
                    conf = seg.get("raw_conf")
                    dec  = _simplify_decision(seg.get("decision", ""))
                    hovers.append(
                        f"{txt}<br>{'conf: '+f'{conf:.2f}' if conf else 'conf: —'} | {dec}"
                    )

            fig.add_trace(go.Scatter(
                x=xs,
                y=[sp_name] * len(xs),
                mode="markers",
                marker=dict(symbol="square", size=14, color=color,
                            line=dict(width=1, color="rgba(0,0,0,0.3)")),
                name=sp_name,
                text=hovers,
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
            ))

        fig.update_layout(
            height=max(180, len(all_speakers) * 60 + 80),
            xaxis_title="Segment #",
            yaxis=dict(categoryorder="array",
                       categoryarray=list(reversed(all_speakers))),
            margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(orientation="h", y=-0.25),
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font_color="#eee",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Decision breakdown ──
    if segs:
        st.subheader("📋 How each segment was identified")

        dec_counts: dict[str, int] = defaultdict(int)
        for seg in segs:
            dec_counts[_simplify_decision(seg.get("decision", ""))] += 1

        col_chart, col_legend = st.columns([1, 1])

        with col_chart:
            labels = list(dec_counts.keys())
            values = list(dec_counts.values())
            colors = PALETTE[:len(labels)]
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.45,
                marker_colors=colors,
                textinfo="percent+label",
                hovertemplate="%{label}: %{value} segments (%{percent})<extra></extra>",
            ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                paper_bgcolor="#111", font_color="#eee",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_legend:
            st.markdown("""
**What each method means:**

🎤 **Voice match** — ECAPA embedding matched a saved speaker

👄 **Lip movement** — ASD detected which face was moving its mouth

📷 **Camera / gender** — single visible face, or gender-compatible name

🔗 **Face-session link** — face was linked to this speaker in a prior turn

📊 **Diarization** — pyannote split the turn; no visual confirmation

🔄 **Voice tracking** — voice similarity across chunks; no saved reference
""")

    # ── Cross-session trend ──
    with st.expander("📈 Trend across sessions"):
        sessions_chrono = list(reversed(sessions))
        labels    = [s["session_id"] for s in sessions_chrono]
        id_rates  = [s["summary"].get("identification_rate", 0) for s in sessions_chrono]
        val_accs  = [s.get("validated_accuracy") for s in sessions_chrono]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels, y=id_rates,
            name="Identified % (system)",
            mode="lines+markers",
            line=dict(color="#3498db", width=2),
            marker=dict(size=8),
        ))
        _vx = [labels[i] for i, v in enumerate(val_accs) if v is not None]
        _vy = [v for v in val_accs if v is not None]
        if _vx:
            fig.add_trace(go.Scatter(
                x=_vx, y=_vy,
                name="Validated accuracy",
                mode="lines+markers",
                line=dict(color="#2ecc71", width=2, dash="dot"),
                marker=dict(size=10, symbol="diamond"),
            ))
        fig.add_hline(y=0.95, line_dash="dash", line_color="#e74c3c",
                      annotation_text="target 95%")
        fig.update_layout(
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1.05],
            height=280,
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#eee",
            legend=dict(orientation="h", y=-0.35),
            margin=dict(l=10, r=10, t=10, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

        rows = []
        for s in sessions_chrono:
            sm = s["summary"]
            rows.append({
                "Session":    s["session_id"],
                "Segments":   sm.get("total_segments", 0),
                "Speakers":   sm.get("unique_speakers", 0),
                "Identified": f"{sm.get('identification_rate', 0):.0%}",
                "Avg conf":   f"{sm.get('avg_raw_conf', 0):.2f}",
                "Validated":  f"{s['validated_accuracy']:.0%}" if s.get("validated_accuracy") is not None else "—",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Transcript ──
    if segs:
        with st.expander("📝 Session transcript"):
            prev_sp = None
            for seg in segs:
                name = _display_name(seg)
                text = seg.get("text", "")
                ts   = (seg.get("timestamp") or "")[:19]
                if name != prev_sp:
                    color = _speaker_color(all_speakers, name)
                    st.markdown(
                        f'<span style="color:{color};font-weight:bold">{name}</span>',
                        unsafe_allow_html=True,
                    )
                    prev_sp = name
                st.markdown(
                    f'<span style="color:#888;font-size:11px">{ts}</span> {text}',
                    unsafe_allow_html=True,
                )
