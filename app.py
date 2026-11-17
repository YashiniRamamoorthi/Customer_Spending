#streamlit

import pandas as pd
import streamlit as st
import pickle

#page configuration
st.set_page_config(page_title="Customer spending prediction",page_icon="customer_logo.jpg",layout="wide")

with st.sidebar:
    st.title("Customer spending prediction")
    st.image("customer_logo.jpg")

#load the pipeline

with open("best_model_pipeline.pkl","rb") as file:
    best_pipeline=pickle.load(file)

#load the cleaned dataframe
df=pd.read_csv("cleaned_data.csv")

#user input
with st.container(border=True):
    col1,col2=st.columns(2)

    with col1:
        Age=st.number_input("Age",min_value=1,max_value=100)
        Annual_Income=st.number_input("Annual_Income",min_value=150000)
        Website_Visits=st.number_input("Website_Visits",min_value=1)
        Previous_Purchases=st.number_input("Previous_Purchase",min_value=0)

    with col2:

        Membership_Type=st.selectbox("Membership",options=df["Membership_Type"].unique())
        Preferred_Category=st.selectbox("Preferred category",options=df["Preferred_Category"].unique())
        City_Tier=st.selectbox("City_Tier",options=df["City_Tier"].unique())
        Discount_Used=st.selectbox("Discount_Used",options=df["Discount_Used"].unique())

new_data={"Age":Age,
"Annual_Income":Annual_Income,"Website_Visits":Website_Visits,"Previous_Purchases":Previous_Purchases,
"Membership_Type":Membership_Type,"Preferred_Category":Preferred_Category,"City_Tier":City_Tier,"Discount_Used":Discount_Used}

new_data=pd.DataFrame(new_data,index=[0])

#predictions
if st.button("Predict Spend"):

    prediction=best_pipeline.predict(new_data)[0]
    predictions=round(prediction,ndigits=2)
    st.header(f"predicted spend :{predictions} Rs.")