import streamlit as st
import pandas as pd

st.set_page_config(page_title="Visualization")
st.title("Data Visualization")

df=pd.read_csv("cleaned_good_bad_customers.csv")
if df["bad_client_target"].dtype == object:
    df["bad_client_target"] = df["bad_client_target"].map({"Yes": 1, "No": 0})

st.subheader("Income Distribution")
income_counts=df["income"].value_counts().sort_index()
st.bar_chart(income_counts)

st.subheader("Bad Clients by Region")
by_reg=df.groupby("region")["bad_client_target"].mean() * 100
st.bar_chart(by_reg)

st.subheader("Bad Clients by Education")
by_edu=df.groupby("education")["bad_client_target"].mean() * 100
st.bar_chart(by_edu)
