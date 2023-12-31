import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

@st.cache_data
def load_data():
    # load dataset
    df =pd.read_csv('milk-quality.csv')

    X = df[["pH","Temprature","Taste","Odor","Fat","Turbidity","Colour"]]
    y = df["Grade"]  

    return df, X, y 

@st.cache_data
def train_model(X, y):
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(X, y)

    score = model.score(X, y)

    return model, score  

def predict(X, y, features):
    model, score = train_model(X, y)

    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score
  