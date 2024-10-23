# data_preparation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import openpyxl
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_and_prepare_data(file_path):
    """
    Esta función carga y prepara los datos de un archivo Excel para ser utilizados en un modelo de aprendizaje
    automático.

    Args:
        file_path (str): La ruta del archivo Excel que contiene los datos.

    Returns:
        X (numpy.ndarray): Matriz de características obtenida mediante la vectorización TF-IDF de la columna
        'tar_subject'.
        y (numpy.ndarray): Matriz de etiquetas con las columnas codificadas 'eml_categoria', 'eml_dueño', 'tar_tiempo' y
        'tar_impacto'.
        tfidf (TfidfVectorizer): Objeto TfidfVectorizer ajustado, usado para transformar la columna 'tar_subject'.
        label_encoders (dict): Diccionario de objetos LabelEncoder ajustados para las columnas 'eml_categoria',
        'eml_dueño', 'tar_tiempo' y 'tar_impacto'.
    """
    # Cargar los datos

    df = pd.read_excel(file_path)
    # Completamos los valores faltantes en cada columna
    df['Categoria'] = df['Categoria'].fillna("sin categoría")
    df['Cuerpo'] = df['Cuerpo'].fillna("sin cuerpo")
    
    # Agrupar por CodigoTarea y unir los correos electrónicos en una sola entrada
    df = df.groupby(by='CodigoTarea').agg({
        'IDEmail': 'first',
        'Cuerpo': 'first',
        'FechaCreacion': 'first',
        'Categoria': 'first',
        'Idioma': 'first'
    }).reset_index()
    
    # df['Categoria'] = df['Categoria'].replace({'PREP': 'P', 'PMOD': 'P', 'OMOD' : 'O', 'OREP' : 'O', 'SAT' : 'I'})

    # Asegúrate de descargar las stopwords
    nltk.download('stopwords')

    # Función para limpiar el texto
    def clean_text(text):
        # Eliminar 'xd'
        text = re.sub(r'\bxd\b', '', text)

        # Eliminar caracteres no deseados (como 'x000d')
        text = re.sub(r'\bx000d\b', '', text)

        # Eliminar URLs y correos electrónicos
        text = re.sub(r'http\S+|www\S+|https\S+|mailto:\S+', '', text)

        # Convertir a minúsculas
        text = text.lower()

        # Eliminar caracteres no alfabéticos y espacios extra
        text = re.sub(r'[^a-záéíóúñü\s]', '', text)

        # Dividir el texto en palabras
        words = text.split()
        
        words = word_tokenize(text)

        # Filtrar stopwords
        spanish_stopwords = set(stopwords.words('spanish'))
        filtered_words = [word for word in words if word not in spanish_stopwords]
        

        return ' '.join(filtered_words)

    # Supongamos que df_grouped ya está definido
    # Aplica la función a cada fila de la columna 'Cuerpo'
    df['Cuerpo'] = df['Cuerpo'].apply(clean_text)

    
    # Selección de las columnas relevantes
    df = df[['Cuerpo', 'Categoria']]

    # Codificación de etiquetas
    label_encoders = {}
    for column in ['Categoria']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Procesamiento del texto
    tfidf = TfidfVectorizer(max_features=5000)
    X_text = tfidf.fit_transform(df['Cuerpo']).toarray()

    # Asignar valores del modelo
    X = X_text
    y = df[['Categoria']].values

    return X, y, tfidf, label_encoders


if __name__ == "__main__":
    path = 'data/Consulta_JMA.xlsx'
    X, y, tfidf, label_encoders = load_and_prepare_data(path)

    # Guardar los objetos procesados si es necesario
    print("Data preparation completed successfully.")
