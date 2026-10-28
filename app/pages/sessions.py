"""Sessions page: run tracker on video files or webcam."""
import os
import sys
import glob
import subprocess
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SESSIONS_DIR = os.path.join(BASE_DIR, "realtime_sessions")


def _find_videos():
    exts = ("*.mp4", "*.mkv", "*.avi", "*.MOV", "*.mov", "*.webm")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(BASE_DIR, ext)))
    return sorted(videos, key=os.path.getmtime, reverse=True)


def _find_transcripts():
    pattern = os.path.join(SESSIONS_DIR, "transcript_*.txt")
    return sorted(glob.glob(pattern), reverse=True)


def render():
    st.title("🎬 Sessions")

    tab_run, tab_history = st.tabs(["Run Tracker", "History"])

    # --- Run Tracker ---
    with tab_run:
        st.subheader("Start a new session")

        mode = st.radio("Mode", ["Video file", "Webcam + Microphone", "Meeting (screen capture)"], horizontal=True)

        video_file = None
        if mode == "Video file":
            videos = _find_videos()
            if videos:
                video_file = st.selectbox("Select video", videos, format_func=os.path.basename)
            else:
                st.warning("No video files found in project directory.")

        num_speakers = st.number_input("Number of speakers (0 = no limit)", min_value=0, max_value=20, value=4)

        if mode == "Meeting (screen capture)":
            monitor = st.number_input("Monitor index", min_value=0, max_value=5, value=1)

        if st.button("▶️ Start Session", type="primary", use_container_width=True):
            # Build CLI command (no interactive prompts needed)
            cmd_parts = [
                f'cd /d "{BASE_DIR}"',
                f'call "%USERPROFILE%\\miniconda3\\condabin\\conda.bat" activate tracker',
            ]

            run_args = "python run_multimodal_tracker.py --headless"
            if mode == "Video file" and video_file:
                run_args += f' --video "{video_file}"'
            elif mode == "Meeting (screen capture)":
                run_args += f' --meeting --monitor {monitor}'

            if num_speakers and num_speakers > 0:
                run_args += f' --speakers {num_speakers}'

            cmd_parts.append(run_args)
            full_cmd = " && ".join(cmd_parts)

            # Launch minimized — CMD stays in taskbar, OpenCV window appears normally
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 6  # SW_MINIMIZE
            subprocess.Popen(
                f'cmd /c "{full_cmd}"',
                startupinfo=startupinfo,
                cwd=BASE_DIR,
            )

            st.success("Session launched!")
            st.info("Press **q** on the video preview window to stop. Refresh this page after to see results.")

    # --- History ---
    with tab_history:
        st.subheader("Past sessions")
        transcripts = _find_transcripts()
        if not transcripts:
            st.info("No sessions yet. Run a session first.")
            return

        for t in transcripts[:20]:
            fname = os.path.basename(t)
            with st.expander(fname):
                with open(t, encoding="utf-8") as f:
                    content = f.read()
                st.text(content[:5000])
                if len(content) > 5000:
                    st.caption(f"... ({len(content)} chars total)")
