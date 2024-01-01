import streamlit as st
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from web_function import train_model,load_data 


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='coolwarm', xticklabels=['Low(0)', 'Medium(1)', 'High(2)'], yticklabels= ['Low(0)', 'Medium(1)', 'High(2)'], fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    fig = plt.gcf()  # Simpan referensi ke gambar
    plt.close()  # Tutup plot
    st.pyplot(fig)  # Tampilkan gambar yang disimpan

def knn_visualization(k, X, y_test, y_pred):
    plt.figure(figsize=(10, 8))
    correct_pred = (y_pred == y_test)
    wrong_pred = (y_pred != y_test)
    plt.scatter(X[y_test == 2]['pH'], X[y_test == 2]['Colour'], color='red', label=' Grade = 2 (High)', alpha=0.7, s=50)
    plt.scatter(X[y_test == 1]['pH'], X[y_test == 1]['Colour'], color='green', label='Grade = 1 (Medium)', alpha=0.7, s=50)
    plt.scatter(X[y_test == 0]['pH'], X[y_test == 0]['Colour'], color='blue', label='Grade = 0 (Low)', alpha=0.7, s=50)
    plt.xlabel('pH')
    plt.ylabel('Colour')
    plt.title(f'KNN Scatter Plot (K = {k})')
    plt.legend()
    plt.scatter(x=X.iloc[0]['pH'], y=X.iloc[0]['Colour'], color='black', s=300, marker='*')

    fig = plt.gcf()  # Simpan referensi ke gambar
    plt.close()  # Tutup plot
    st.pyplot(fig)  # Tampilkan gambar yang disimpan


    
def app(df, X, y):
    warnings.filterwarnings('ignore') 
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Visualisasi Prediksi Kualitas Susu")  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model, score = train_model(X_train, y_train)  # Melatih model menggunakan data latih
    
    if st.checkbox("Plot Confusion Matrix"):
        y_pred = model.predict(X_test)  # Menghasilkan prediksi dari model menggunakan data uji
        plot_confusion_matrix(y_test, y_pred)  # Menampilkan confusion matrix menggunakan data uji

    if st.checkbox("Plot K-Neighboors"): 
        k_value = 2 # Ganti sesuai kebutuhan
        knn_visualization(k_value, X_test, y_test, model.predict(X_test))  # Menampilkan plot KNN menggunakan data uji
