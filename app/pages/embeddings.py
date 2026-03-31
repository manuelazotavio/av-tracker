"""Embeddings manager: view, delete, and enroll — reads from disk (.npy) as source of truth."""
import os
import sys
import re
import glob
import subprocess
import streamlit as st
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database import get_db

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VOICE_DIR = os.path.join(BASE_DIR, "data", "embeddings")
FACE_DIR = os.path.join(BASE_DIR, "data", "face_embeddings")
SESSIONS_DIR = os.path.join(BASE_DIR, "realtime_sessions")


def _parse_name(filename):
    base = os.path.splitext(filename)[0]
    if base.endswith("_auto"):
        base = base[:-5]
    parts = base.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return base


def _list_disk_embeddings(emb_dir):
    if not os.path.exists(emb_dir):
        return {}
    result = defaultdict(list)
    for f in sorted(os.listdir(emb_dir)):
        if not f.endswith(".npy"):
            continue
        name = _parse_name(f)
        stat = os.stat(os.path.join(emb_dir, f))
        result[name].append({
            "file": f,
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return dict(result)


def render():
    st.title("🧠 Embeddings Manager")

    tab_voice, tab_face, tab_enroll = st.tabs(["Voice Embeddings", "Face Embeddings", "Enroll Speaker"])

    # --- Voice ---
    with tab_voice:
        st.subheader("Voice Embeddings (disk)")
        voice = _list_disk_embeddings(VOICE_DIR)
        if not voice:
            st.info("No voice embeddings found.")
        else:
            total = sum(len(v) for v in voice.values())
            st.caption(f"{len(voice)} speakers, {total} files")
            for name in sorted(voice):
                files = voice[name]
                with st.expander(f"🎤 {name} ({len(files)} files)"):
                    for f in files:
                        col1, col2, col3 = st.columns([3, 2, 1])
                        col1.text(f["file"])
                        col2.text(f["date"])
                        if col3.button("🗑️", key=f"dv_{f['file']}"):
                            os.remove(os.path.join(VOICE_DIR, f["file"]))
                            st.success(f"Deleted {f['file']}")
                            st.rerun()

            st.divider()
            if st.button("🗑️ Delete ALL Person_N voice embeddings"):
                count = 0
                for f in os.listdir(VOICE_DIR):
                    if re.match(r'^Person_\d+', f) and f.endswith(".npy"):
                        os.remove(os.path.join(VOICE_DIR, f))
                        count += 1
                if count:
                    st.success(f"Deleted {count} files.")
                    st.rerun()

    # --- Face ---
    with tab_face:
        st.subheader("Face Embeddings (disk)")
        face = _list_disk_embeddings(FACE_DIR)
        if not face:
            st.info("No face embeddings found.")
        else:
            total = sum(len(v) for v in face.values())
            st.caption(f"{len(face)} speakers, {total} files")
            for name in sorted(face):
                files = face[name]
                with st.expander(f"👤 {name} ({len(files)} files)"):
                    for f in files:
                        col1, col2, col3 = st.columns([3, 2, 1])
                        col1.text(f["file"])
                        col2.text(f["date"])
                        if col3.button("🗑️", key=f"df_{f['file']}"):
                            os.remove(os.path.join(FACE_DIR, f["file"]))
                            st.success(f"Deleted {f['file']}")
                            st.rerun()

            st.divider()
            if st.button("🗑️ Delete ALL Person_N face embeddings"):
                count = 0
                for f in os.listdir(FACE_DIR):
                    if re.match(r'^Person_\d+', f) and f.endswith(".npy"):
                        os.remove(os.path.join(FACE_DIR, f))
                        count += 1
                if count:
                    st.success(f"Deleted {count} files.")
                    st.rerun()

    # --- Enroll ---
    with tab_enroll:
        st.subheader("Enroll Speaker from Video/Audio")

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Speaker name", placeholder="Manuela")
        with col2:
            exts = ("*.mp4", "*.mkv", "*.avi", "*.MOV", "*.wav", "*.m4a")
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(BASE_DIR, ext)))
            files.extend(glob.glob(os.path.join(SESSIONS_DIR, "audio_*.wav")))
            files = sorted(files, key=os.path.getmtime, reverse=True)
            selected_file = st.selectbox("Audio/Video file", files, format_func=os.path.basename)

        col1, col2 = st.columns(2)
        with col1:
            start = st.number_input("Start (seconds)", min_value=0.0, value=0.0, step=1.0)
        with col2:
            end = st.number_input("End (seconds, 0=full)", min_value=0.0, value=0.0, step=1.0)

        if st.button("📥 Enroll", type="primary") and name and selected_file:
            cmd = [sys.executable, os.path.join(BASE_DIR, "enroll_speaker.py"),
                   name, selected_file, "--start", str(start)]
            if end > 0:
                cmd.extend(["--end", str(end)])

            with st.spinner(f"Enrolling '{name}'..."):
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR, timeout=120)
                st.code(str(result.stdout or "") + str(result.stderr or ""), language="text")

            if result.returncode == 0:
                st.success(f"Enrolled '{name}'!")
            else:
                st.error("Enrollment failed.")
