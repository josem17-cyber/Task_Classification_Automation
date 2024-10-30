import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from data_preparation import load_and_prepare_data

data_spam = pd.read_excel('data_spam/SPAM_JMA.xlsx')
data_main = pd.read_excel('data_spam/2024-TODOS.xlsx')

# Asegúrate de descargar las stopwords
nltk.download('stopwords')

# Función para limpiar el texto
def clean_text(text):
    text = str(text)
    # Eliminar caracteres no deseados (como xd)
    text = re.sub(r'\bxd\b', '', text)
    
    # Eliminar caracteres no deseados (como x000d)
    text = re.sub(r'\bx000d\b', '', text)
    
    # Eliminar URLs y correos electrónicos
    text = re.sub(r'http\S+|www\S+|https\S+|mailto:\S+', '', text)
    
    # Mantener solo palabras en español (puedes ajustar esto según tus necesidades)
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos y espacios extra
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    
    # Dividir el texto en palabras
    words = text.split()
    
    # Filtrar stopwords
    spanish_stopwords = set(stopwords.words('spanish'))
    filtered_words = [word for word in words if word not in spanish_stopwords]
    
    return ' '.join(filtered_words)


data_spam['eml_body'] = data_spam['eml_body'].apply(clean_text) 
data_spam['Categoria'] = 'SPAM'
data_spam['Cuerpo'] = data_spam['eml_body'] + data_spam['eml_from']
data_spam = data_spam[['Cuerpo', 'Categoria']]


data_main = data_main.groupby(by='CodigoTarea').agg({
        'IDEmail': 'first',
        'De': 'first',
        'Cuerpo': 'first',
        'FechaCreacion': 'first',
        'Categoria': 'first',
        'Idioma': 'first'
    }).reset_index()
data_main['Cuerpo'] = data_main['Cuerpo'].apply(clean_text)
data_main['Cuerpo'] = data_main['Cuerpo'] + data_main['De']
data_main = data_main[['Cuerpo', 'Categoria']]


data = pd.concat([data_spam, data_main])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data['Cuerpo'] = data['Cuerpo'].fillna("sin cuerpo")

data_m = data.copy()

data_m['SPAM'] = data_m['Categoria'].apply(lambda x: 1 if x == 'SPAM' else 0)

X_train, X_test, y_train, y_test = train_test_split(data_m.Cuerpo, data_m.SPAM, test_size=0.25)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

# Reporte de clasificación
print(classification_report(y_test, y_pred))
