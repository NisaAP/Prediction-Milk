import streamlit as st

def app():
    #judul halaman
    st.markdown(
        """
        <style>
        .css-2trqyj {
            font-family: 'Times New Roman', Times, serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Aplikasi Milk Quality Prediction")
    st.subheader("Selamat Datang Di Aplikasi Prediksi Kualitas Susu!")
    st.image("cover.jpg")
    st.write("Aplikasi prediksi kualitas susu adalah aplikasi untuk mempreadiksi kualitas susu tersebut Bagus, Sedang atau Buruk. Untuk mengetahui kualitas aplikasi menggunakan beberapa faktor seperti PH, Suhu, Rasa, Bau, Kandungan lemak, kekeruhan dan warna pada susu tersebut  ")
 