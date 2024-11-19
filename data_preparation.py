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
    def cat_simplification(df):
        df['Categoria'] = df['Categoria'].replace({'PREP': 'P', 'PMOD': 'P', 'OMOD' : 'O', 'OREP' : 'O', 'SAT' : 'I', 'C': 'OTHER', 'D': 'OTHER', 'DOC': 'OTHER'})
    
    # Cargar los datos
    df = pd.read_excel(file_path)
    
    df['Cuerpo'] = df['Cuerpo'].fillna("sin cuerpo")
    df['De'] = df['De'].fillna("sin de")

    df_filtred = df[df['CodigoTarea'].notna()]
    
    df_grouped = df_filtred.groupby(by='CodigoTarea').agg({
        'IDEmail': 'first',
        'De': 'first',
        'Cuerpo': 'first',
        'Categoria': 'first'
    }).reset_index()
    
    df_nan = df[df['CodigoTarea'].isna()]
    
    df = pd.concat([df_grouped, df_nan], ignore_index=True)
    
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
    df['Cuerpo'] = df['Cuerpo'] + df['De']
    
    cat_simplification(df)  
    
    print(df['Categoria'].value_counts())
    
    print(df.head())
    
    # Vectorización TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['Cuerpo']).toarray()

    # Codificación de etiquetas
    label_encoders = {}
    label_encoders['Categoria'] = LabelEncoder()
    y = label_encoders['Categoria'].fit_transform(df['Categoria'])

    return X, y, label_encoders

if __name__ == "__main__":
    file_path = 'data/final_data/CAT_SPAM_2023_E.xlsx'
    X, y, label_encoders = load_and_prepare_data(file_path)
    print(f"Características (X): {X.shape}")
    print(f"Etiquetas (y): {y.shape}")
    print(f"Label Encoders: {label_encoders}")
    print(f"Data Preparation Successful!!")