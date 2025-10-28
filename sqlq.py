import streamlit as st
import pandas as pd
from pandasql import sqldf

st.set_page_config(page_title="SQL Query")

st.title("SQL Query Page")
st.write("Run simple SQL queries on the customer dataset using pandasql.")


data=pd.read_csv("cleaned_good_bad_customers.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

st.subheader("Write your SQL query")
query=st.text_area("", "")

if st.button("Run"):
    try:
        result = sqldf(query, locals())
        st.dataframe(result)
    except Exception as e:
        st.error(f"Error: {e}")
