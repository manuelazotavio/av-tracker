"""Metrics dashboard: visualize session metrics from database."""
import os
import sys
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database import get_db


def render():
    st.title("📊 Metrics Dashboard")

    db = get_db()
    sessions = db.list_sessions()

    if not sessions:
        st.info("No sessions in database. Run a session or migrate existing data first.")
        st.code("python scripts/migrate_to_db.py", language="bash")
        return

    tab_compare, tab_single = st.tabs(["Cross-Session Comparison", "Single Session"])

    # --- Cross-session ---
    with tab_compare:
        st.subheader("Session Trends")

        labels = [s["session_id"] for s in sessions]
        labels.reverse()
        sessions_chrono = list(reversed(sessions))

        id_rates = [s["summary"].get("identification_rate", 0) for s in sessions_chrono]
        fn_rates = [s["summary"].get("fn_rate", 0) for s in sessions_chrono]
        fp_rates = [s["summary"].get("fp_rate", 0) for s in sessions_chrono]
        val_acc = [s.get("validated_accuracy") for s in sessions_chrono]

        # 1) Rates chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=id_rates, name="ID Rate", marker_color="#2ecc71"))
        fig.add_trace(go.Bar(x=labels, y=fn_rates, name="FN Rate", marker_color="#e74c3c"))
        fig.add_trace(go.Bar(x=labels, y=fp_rates, name="FP Risk", marker_color="#e67e22"))
        if any(v is not None for v in val_acc):
            _vx = [labels[i] for i, v in enumerate(val_acc) if v is not None]
            _vy = [v for v in val_acc if v is not None]
            fig.add_trace(go.Scatter(x=_vx, y=_vy, name="Validated Accuracy",
                                     mode="lines+markers", marker=dict(size=10, color="#8e44ad"),
                                     line=dict(width=3)))
        fig.update_layout(title="Identification & Error Rates", barmode="group",
                          yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # 2) Decision distribution
        all_types = set()
        for s in sessions_chrono:
            all_types.update(s["summary"].get("decision_counts", {}).keys())
        ordered = sorted(all_types)

        fig2 = go.Figure()
        for dt in ordered:
            vals = []
            for s in sessions_chrono:
                total = s["summary"].get("total_segments", 1)
                count = s["summary"].get("decision_counts", {}).get(dt, 0)
                vals.append(count / total if total else 0)
            fig2.add_trace(go.Bar(x=labels, y=vals, name=dt))
        fig2.update_layout(title="Decision Distribution (%)", barmode="stack",
                           yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # 3) Confidence & latency
        col1, col2 = st.columns(2)
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=labels,
                y=[s["summary"].get("avg_raw_conf", 0) for s in sessions_chrono],
                name="Mean Conf", mode="lines+markers"))
            fig3.add_trace(go.Scatter(
                x=labels,
                y=[s["summary"].get("median_raw_conf", 0) for s in sessions_chrono],
                name="Median Conf", mode="lines+markers"))
            fig3.update_layout(title="Verifier Confidence Trend", height=350)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=labels,
                y=[s["summary"].get("avg_segment_ms", 0) for s in sessions_chrono],
                name="Total", mode="lines+markers"))
            fig4.add_trace(go.Scatter(
                x=labels,
                y=[s["summary"].get("avg_whisper_ms", 0) for s in sessions_chrono],
                name="Whisper", mode="lines+markers"))
            fig4.update_layout(title="Average Latency (ms)", height=350)
            st.plotly_chart(fig4, use_container_width=True)

        # 4) Summary table
        st.subheader("Session Summary")
        table_data = []
        for s in sessions_chrono:
            sm = s["summary"]
            table_data.append({
                "Session": s["session_id"],
                "Segments": sm.get("total_segments", 0),
                "ID Rate": f"{sm.get('identification_rate', 0):.0%}",
                "FN Rate": f"{sm.get('fn_rate', 0):.0%}",
                "FP Rate": f"{sm.get('fp_rate', 0):.0%}",
                "Speakers": sm.get("unique_speakers", 0),
                "Avg Conf": f"{sm.get('avg_raw_conf', 0):.2f}",
                "Validated": f"{s['validated_accuracy']:.0%}" if s.get("validated_accuracy") is not None else "-",
            })
        st.dataframe(table_data, use_container_width=True)

    # --- Single session ---
    with tab_single:
        session_ids = [s["session_id"] for s in sessions]
        selected_id = st.selectbox("Select session", session_ids, index=0)
        data = next(s for s in sessions if s["session_id"] == selected_id)
        summary = data["summary"]
        config = data["config"]

        # Get session PK for segment query
        session_row = db.get_session(selected_id)
        session_pk = session_row["id"] if session_row else None

        st.subheader(f"Session {selected_id}")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Segments", summary.get("total_segments", 0))
        col2.metric("ID Rate", f"{summary.get('identification_rate', 0):.0%}")
        col3.metric("FN Rate", f"{summary.get('fn_rate', 0):.0%}")
        col4.metric("FP Rate", f"{summary.get('fp_rate', 0):.0%}")

        with st.expander("Configuration"):
            st.json(config)

        # Load segments from DB
        if session_pk:
            segs = db.get_segments(session_pk)
            if segs:
                # Per-segment confidence
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=[s["raw_conf"] for s in segs],
                    mode="markers+lines",
                    marker=dict(
                        color=["#2ecc71" if s["is_identified"] else "#e74c3c" for s in segs],
                        size=6,
                    ),
                    text=[f"{s.get('final_name') or s['final_speaker']}" for s in segs],
                    hovertemplate="%{text}<br>Conf: %{y:.2f}<extra></extra>",
                ))
                thr = config.get("verifier_threshold", 0.8)
                fig.add_hline(y=thr, line_dash="dash", line_color="red",
                              annotation_text=f"threshold={thr}")
                fig.update_layout(title="Per-Segment Verifier Confidence", height=350,
                                  xaxis_title="Segment", yaxis_title="Raw Confidence")
                st.plotly_chart(fig, use_container_width=True)

                # Decision pie
                col1, col2 = st.columns(2)
                with col1:
                    dc = summary.get("decision_counts", {})
                    if dc:
                        fig = px.pie(names=list(dc.keys()), values=list(dc.values()),
                                     title="Decision Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fn_segs = [s for s in segs if s.get("is_fn")]
                    fp_segs = [s for s in segs if s.get("fp_risk") in ("high", "medium")]
                    st.metric("False Negatives", len(fn_segs))
                    st.metric("False Positive Risks", len(fp_segs))

                # Transcript with highlighting
                with st.expander("Transcript"):
                    for s in segs:
                        name = s.get("final_name") or s.get("final_speaker", "?")
                        ts = (s.get("timestamp") or "")[:19]
                        text = s.get("text", "")
                        st.text(f"[{ts}] {name}: {text}")

        # Validation
        if session_pk:
            val = db.get_validation(session_pk)
            if val:
                st.subheader("Validation Results")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{val['accuracy']:.0%}")
                col2.metric("Named Accuracy", f"{val['named_accuracy']:.0%}")

                details = val.get("details", {})
                if details.get("per_speaker"):
                    ps_data = [
                        {"Speaker": n, "Correct": s["correct"], "Incorrect": s["incorrect"],
                         "Total": s["total"], "Accuracy": f"{s['accuracy']:.0%}"}
                        for n, s in details["per_speaker"].items()
                    ]
                    st.dataframe(ps_data, use_container_width=True)
