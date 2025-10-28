import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

st.set_page_config(page_title="Prediction")

st.title("Credit Risk Prediction Page")
st.write("Use the trained KNN model to predict if a client is **Good (0)** or **Bad (1)**.")

# Load model + scaler
try:
    model=load("knn_classifier.joblib")
    scaler=load("scaler.joblib")
except:
    st.stop()

# Load dataset (to get column names)
data=pd.read_csv("cleaned_good_bad_customers.csv")
feature_cols=[c for c in data.columns if c != "bad_client_target"]

st.subheader("Enter Customer Details")

user={}
user['month']=st.number_input('month', int(data['month'].min()), int(data['month'].max()), int(data['month'].mean()), step=1)
user['credit_amount']=st.number_input('credit_amount', int(data['credit_amount'].min()), int(data['credit_amount'].max()), int(data['credit_amount'].mean()), step=1)
user['credit_term']=st.number_input('credit_term', int(data['credit_term'].min()), int(data['credit_term'].max()), int(data['credit_term'].mean()), step=1)
user['age']=st.number_input('age', int(data['age'].min()), int(data['age'].max()), int(data['age'].mean()), step=1)
user['sex']=st.selectbox('sex', ['female', 'male'])
user['education']=st.selectbox('education', list(data['education'].unique()))
user['product_type']=st.selectbox('product_type', list(data['product_type'].unique()))
user['having_children_flg']=st.number_input('having_children_flg', int(data['having_children_flg'].min()), int(data['having_children_flg'].max()), int(data['having_children_flg'].mean()), step=1)
user['region']=st.number_input('region', int(data['region'].min()), int(data['region'].max()), int(data['region'].mean()), step=1)
user['income']=st.number_input('income', int(data['income'].min()), int(data['income'].max()), int(data['income'].mean()), step=1)
user['family_status']=st.selectbox('family_status', list(data['family_status'].unique()))
user['phone_operator']=int(st.number_input('phone_operator', value=0, step=1))

user['is_client']=st.number_input('is_client', 0, 1, 1, step=1)

input_df=pd.DataFrame([user])
from sklearn.preprocessing import LabelEncoder
for c in input_df.select_dtypes(exclude=['int64', 'float64']).columns:
    le=LabelEncoder()
    le.fit(data[c].astype(str))
    input_df[c]=le.transform(input_df[c].astype(str))

input_scaled=scaler.transform(input_df)
if st.button("Predict"):
    pred=model.predict(input_scaled)[0]
    if pred==1:
        st.write("Prediction: Bad Client (1)")
    else:
        st.write("Prediction: Good Client (0)")
