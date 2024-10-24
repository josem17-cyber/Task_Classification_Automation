# data_preparation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import openpyxl
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

def load_and_prepare_data(file_path, word2vec_model):
    """
    Esta función carga y prepara los datos de un archivo Excel para ser utilizados en un modelo de aprendizaje
    automático.

    Args:
        file_path (str): La ruta del archivo Excel que contiene los datos.
        word2vec_model: Modelo Word2Vec entrenado para generar vectores de palabras.

    Returns:
        X (numpy.ndarray): Matriz de características obtenida mediante el WordEmbedding de la columna 'Cuerpo'.
        y (numpy.ndarray): Matriz de etiquetas con las columnas codificadas 'Categoria'.
        label_encoders (dict): Diccionario de objetos LabelEncoder ajustados para las columnas 'Categoria'.
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
        words = word_tokenize(text)

        # Filtrar stopwords
        spanish_stopwords = set(stopwords.words('spanish'))
        filtered_words = [word for word in words if word not in spanish_stopwords]
        
        return filtered_words  # Devolvemos la lista de palabras

    # Aplica la función a cada fila de la columna 'Cuerpo'
    df['Cuerpo'] = df['Cuerpo'].apply(clean_text)

    # Selección de las columnas relevantes
    df = df[['Cuerpo', 'Categoria']]
    
    df['Categoria'] = df['Categoria'].replace({'PREP': 'P', 'PMOD': 'P', 'OMOD' : 'O', 'OREP' : 'O', 'SAT' : 'I'})

    # Codificación de etiquetas
    label_encoders = {}
    for column in ['Categoria']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Procesamiento del texto usando Word2Vec
    def get_word_vector(words):
        # Genera un vector promedio para las palabras en la lista
        vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)  # Vector de ceros si no hay palabras

    # Aplicar la función para convertir el texto en vectores
    X = np.array(df['Cuerpo'].apply(get_word_vector).tolist())

    y = df[['Categoria']].values

    return X, y, label_encoders


if __name__ == "__main__":
    path = 'data/Consulta_JMA.xlsx'
    
    # Carga tu modelo de Word2Vec
    word2vec_model = Word2Vec.load('model.bin')  # Asegúrate de que el modelo esté guardado

    X, y, label_encoders = load_and_prepare_data(path, word2vec_model)

    # Guardar los objetos procesados si es necesario
    print("Data preparation completed successfully.")

