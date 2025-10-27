import streamlit as st
import runpy

st.set_page_config(page_title="Credit Risk Dashboard", page_icon="💳", layout="wide")

st.sidebar.title("Dashboard")
page = st.sidebar.radio(
    "Go to",
    [
        "View & Filter Data",
        "SQL Query",
        "Visualization",
        "Prediction"
    ]
)

if page == "View & Filter Data":
    runpy.run_path("view.py")

elif page == "SQL Query":
    runpy.run_path("sqlq.py")

elif page == "Visualization":
    runpy.run_path("graph.py")

elif page == "Prediction":
    runpy.run_path("predict.py")
