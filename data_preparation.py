# data_preparation.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_and_prepare_data(file_path):
    """
    Carga y prepara los datos de un archivo Excel para ser utilizados en un modelo de aprendizaje automático.

    Args:
        file_path (str): La ruta del archivo Excel que contiene los datos.

    Returns:
        X (numpy.ndarray): Matriz de características obtenida mediante la vectorización TF-IDF de la columna 'Cuerpo'.
        y (numpy.ndarray): Matriz de etiquetas con la columna codificada 'Categoria'.
        label_encoders (dict): Diccionario de objetos LabelEncoder ajustados para la columna 'Categoria'.
    """
    # Cargar los datos
    df = pd.read_excel(file_path)

    # Completar los valores faltantes en cada columna
    df['Categoria'] = df['Categoria'].fillna("sin categoría")
    df['Cuerpo'] = df['Cuerpo'].fillna("sin cuerpo")
    
    def cat_simplification(df):
        df['Categoria'] = df['Categoria'].replace({'PREP': 'P', 'PMOD': 'P', 'OMOD' : 'O', 'OREP' : 'O', 'SAT' : 'I'})
    
    # Agrupar por CodigoTarea y unir los correos electrónicos en una sola entrada
    df = df.groupby(by='CodigoTarea').agg({
        'IDEmail': 'first',
        'Cuerpo': 'first',
        'FechaCreacion': 'first',
        'Categoria': 'first'
    }).reset_index()

    # Preprocesamiento de texto
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('spanish'))

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    df['Cuerpo'] = df['Cuerpo'].apply(preprocess_text)

    # Vectorización TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['Cuerpo']).toarray()

    # Codificación de etiquetas
    label_encoders = {}
    label_encoders['Categoria'] = LabelEncoder()
    y = label_encoders['Categoria'].fit_transform(df['Categoria'])

    return X, y, label_encoders

if __name__ == "__main__":
    file_path = 'data/Consulta_JMA.xlsx'
    X, y, label_encoders = load_and_prepare_data(file_path)
    print(f"Características (X): {X.shape}")
    print(f"Etiquetas (y): {y.shape}")
    print(f"Label Encoders: {label_encoders}")
    print(f"Data Preparation Successful!!")