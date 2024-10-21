# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score
from data_preparation import load_and_prepare_data

@st.cache_resource
def load_resources():
    file_path = 'data/Consulta_JMA.xlsx'
    X, _, tfidf, label_encoders = load_and_prepare_data(file_path)
    model = load_model('model.h5')
    return X, tfidf, label_encoders, model

def predict_category(subject, tfidf, model, label_encoders):
    """
    Predice las categorías basadas en el texto del asunto utilizando un modelo de red neuronal.

    Args:
        subject (str): Texto del asunto.
        tfidf (TfidfVectorizer): Objeto TfidfVectorizer para transformar el texto del asunto.
        model (keras.Model): Modelo de Keras para realizar la predicción.
        label_encoders (dict): Diccionario de codificadores de etiquetas para decodificar las predicciones.

    Returns:
        dict: Diccionario con las categorías predichas ajustadas para 'eml_categoria', 'eml_dueño', 'tar_tiempo' y
        'tar_impacto'.
    """
    X_text = tfidf.transform([subject]).toarray()
    predictions = model.predict(X_text)

    predicted_categories = {}
    for i, col in enumerate(['Categoria']):
        predicted_categories[col] = label_encoders[col].inverse_transform([np.argmax(predictions[i])])[0]

    categoria = predicted_categories['Categoria']
    #idioma = predicted_categories['Idioma']


    return predicted_categories

X, tfidf, label_encoders, model = load_resources()

st.title("Tarea Classification Predictor")

subject = st.text_input("Subject / Conversation", placeholder="Type the subject here")

if st.button("Predict"):
    predicted_categories = predict_category(subject, tfidf, model, label_encoders)
    st.write("Predicted Categories:")
    for col, category in predicted_categories.items():
        st.write(f"{col}: {category}")
