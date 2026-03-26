import streamlit as st
import pandas as pd 
import joblib 
 
model= joblib.load("logistic_regression_heart.pkl")
scaler= joblib.load('scaler.pkl')
expected_columns= joblib.load('columns.pkl')

st.title("Heart stroke prediction by Manasvi")
st.markdown('Provide the following details:')

age= st.slider("Age",18,100,40)
sex= st.selectbox("SEX",['M','F'])
chest_pain= st.selectbox("Chest pain type",["ATA","NAP","TA","ASY"])
resting_BP= st.number_input("Resting blood pressure(mm,Hg)",80,200,120)
cholesterol= st.number_input("cholesterol(mg/dl)",100,600,200)
fasting_BS= st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
resting_ECG= st.selectbox("Resting ECG ",["Normal","ST","LVH"])
max_HR= st.slider("Max Heart Rate",60,220,150)
exercise_angina= st.selectbox("Exercise angina",["Y","N"])
old_peak= st.slider("Oldpeak (ST depression)",0.0,6.0,1.0)
st_slope= st.selectbox("ST Slope",["Up","Flat","Down"])

if st.button("Predict"):
    raw_input= {
        "Age" : age,
        "SEX"+ sex : 1,
        "Chest pain type"+chest_pain:1,
        "Resting blood pressure" :resting_BP,
        "cholesterol" : cholesterol,
        "Fasting Blood Sugar": fasting_BS,
        "Resting ECG "+ resting_ECG: 1,
        "Max Heart Rate": max_HR,
        "Exercise angina"+ exercise_angina:1,
        "Oldpeak (ST depression)": old_peak,
        "ST Slope"+ st_slope:1,
    }

    input_df= pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]= 0

    input_df = input_df[expected_columns]
    scaled_input= scaler.transform(input_df)
    prediction= model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("HIGH RISK OF HEART DISEASE!!!!")

    else:
        st.success("LOW RISK OF HEART DISEASE")
