import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="View & Filter Data", page_icon="📄", layout="wide")
st.title("View and Filter Dataset")
st.write("Explore and filter the customer credit risk dataset interactively.")

@st.cache_data
def get_data():
    f = Path("cleaned_good_bad_customers.csv")
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)

df = get_data()
if df.empty:
    st.stop()

st.subheader("Filter Options")

min_inc = int(df["income"].min())
max_inc = int(df["income"].max())
inc = st.slider("Income", min_value=min_inc, max_value=max_inc, value=(min_inc, max_inc))
edu = st.selectbox("Education", ["All"] + sorted(df["education"].dropna().unique()))
reg = st.selectbox("Region", ["All"] + sorted(df["region"].dropna().unique()))

out = df[df["income"].between(inc[0], inc[1])]
if edu != "All":
    out = out[out["education"] == edu]
if reg != "All":
    out = out[out["region"] == reg]

st.subheader("Filtered Results")
st.write(f"Rows: {out.shape[0]}")
st.dataframe(out, use_container_width=True) 