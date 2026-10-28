"""
Sistema multimodal — painel Streamlit

Launch: streamlit run app/app.py
Or use the local launcher script.
"""
import streamlit as st

st.set_page_config(
    page_title="Sistema multimodal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar navigation ---
page = st.sidebar.radio(
    "Navigation",
    ["Sessions", "Metrics", "Validate", "Embeddings"],
    index=0,
)

if page == "Sessions":
    from pages import sessions
    sessions.render()
elif page == "Metrics":
    from pages import metrics
    metrics.render()
elif page == "Validate":
    from pages import validate
    validate.render()
elif page == "Embeddings":
    from pages import embeddings
    embeddings.render()
