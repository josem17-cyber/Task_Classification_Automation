# data_preparation.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import openpyxl

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
    df['Tiempo'] = df['Tiempo'].fillna("sin tiempo")
    df['Impacto'] = df['Impacto'].fillna("sin impacto")

    # Agrupar por CodigoTarea y unir los correos electrónicos en una sola entrada
    df = df.groupby('CodigoTarea').agg({
        'IDEmail': 'first',  # Suponemos que el email no cambia, tomamos el primero
        'Cuerpo': '.'.join,  # Unificamos los cuerpos en una sola cadena
        'Dueño': 'first',  # Suponemos que el dueño no cambia, tomamos el primero
        'FechaTerminado': 'first',  # Suponemos que la fecha no cambia, tomamos la primera
        'Categoria': 'first',  # Suponemos que la categoría no cambia, tomamos la primera
        'Idioma': 'first',  # Suponemos que el idioma no cambia, tomamos el primero
        'Tiempo': 'first',  # Sumamos el tiempo
        'Impacto': 'first'  # Promediamos el impacto
    }).reset_index()

    # df = df[df['eml_dueño'].isin(allowed_owners)]
    # allowed_owners = ["LC", "LG", "OM", "MLM", "FD", "AB", "DC", "JG"]

    # Selección de las columnas relevantes
    df = df[['Cuerpo', 'Categoria']]
    
    categorias_a_mantener = ['SEG', 'OMOD', 'PREP', 'OREP', 'PMOD', 'SAT', 'I']
    
    df = df[df['Categoria'].isin(categorias_a_mantener)]
    
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
    path = 'data/Automatizacion_Clasificacion_EDA_HIDRAL.xlsx'
    X, y, tfidf, label_encoders = load_and_prepare_data(path)

    # Guardar los objetos procesados si es necesario
    print("Data preparation completed successfully.")
