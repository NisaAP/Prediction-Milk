import numpy as np
from web_function import predict
from web_function import load_data
import streamlit as st


def app(df, X, y):
    st.title("Prediksi Kualitas Susu")
    st.write('Silahkan Masukan Input Memprediksi Kualitas Susu :')

    Taste_dict = {'Good': 1, 'Bad': 0}
    Odor_dict= {'Good': 1, 'Bad': 0}
    Fat_dict = {'High': 1, 'Low': 0}
    Turbidity_dict = {'High': 1, 'Low': 0}


    col1, col2 = st.columns(2)

    with col1:
       pH = st.number_input('Input pH', min_value=3.0, max_value=9.5)
    with col2:
        Temprature = st.number_input('Input Temperatur', min_value=34, max_value=90)
    with col1:
        Taste = st.selectbox('Input Taste', ('Bad', 'Good'))
    with col2:
        Odor = st.selectbox('Input Odor', ('Bad', 'Good'))
    with col1:
        Fat = st.selectbox('Input Fat', ('Low', 'High'))
    with col2:
        Turbidity = st.selectbox('Input Turbidity', ('Low', 'High'))
    with col1:
        Colour = st.number_input('Input Color', min_value=240, max_value=255)

    prediction = None 

    if st.button('Milk Prediction'):
        df, X, y = load_data()
        Taste_val = Taste_dict[Taste]
        Odor_val = Odor_dict[Odor]
        Fat_val = Fat_dict[Fat]
        Turbidity_val = Turbidity_dict[Turbidity]

        st.info("Prediksi Sukses...")

    
        features = [pH, Temprature, Taste_val, Odor_val,Fat_val,Turbidity_val, Colour]

        prediction,score =predict(X, y, features)
    

    if prediction is not None:
        if prediction == 2:
            milk_prediction_text = 'Kualitas Susu Bagus'
        elif prediction == 1:
            milk_prediction_text = 'Kualitas Susu Sedang'
        else:
            milk_prediction_text = 'Kualitas Susu Buruk'

        st.success(milk_prediction_text)
        
        st.write("Tingkat Akurasi Model Yang Digunakan",(score*100),"%")
