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
import joblib

# data_spam = pd.read_excel('data_spam/SPAM_JMA.xlsx')
# data_main = pd.read_excel('data_spam/2024-TODOS.xlsx')
train_df = pd.read_excel('data/final_data/CAT_SPAM_2023_E.xlsx')
test_df = pd.read_excel('data/final_data/CAT_SPAM_2024_E.xlsx')

# Agrupamos por código de tarea

# Filtrar filas donde 'Categoria' no es NaN
train_df_filtered = train_df[train_df['CodigoTarea'].notna()]
test_df_filtered = test_df[test_df['CodigoTarea'].notna()]

# Agrupar solo las filas donde 'Categoria' no es NaN
train_df_grouped = train_df_filtered.groupby(by='CodigoTarea').agg({
    'IDEmail': 'first',
    'De': 'first',
    'Cuerpo': 'first',
    'Categoria': 'first'
}).reset_index()

test_df_grouped = test_df_filtered.groupby(by='CodigoTarea').agg({
    'IDEmail': 'first',
    'De': 'first',
    'Cuerpo': 'first',
    'Categoria': 'first'
}).reset_index()

# Concatenar las filas agrupadas con las filas donde 'Categoria' es NaN
train_df_nan = train_df[train_df['CodigoTarea'].isna()]
test_df_nan = test_df[test_df['CodigoTarea'].isna()]

train_df = pd.concat([train_df_grouped, train_df_nan], ignore_index=True)
test_df = pd.concat([test_df_grouped, test_df_nan], ignore_index=True)

# Descargamos los stopwords en español  
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

# Preparamos train_df
train_df['Cuerpo'] = train_df['Cuerpo'].apply(clean_text) 
train_df['Cuerpo'] = train_df['Cuerpo'] + train_df['De']
train_df = train_df[['Cuerpo', 'Categoria']]

# Preparamos test_df
test_df['Cuerpo'] = test_df['Cuerpo'] + test_df['De']
test_df = test_df[['Cuerpo', 'Categoria']]

# Rellenar valores nulos    

train_df['Cuerpo'] = train_df['Cuerpo'].fillna("sin cuerpo")
test_df['Cuerpo'] = test_df['Cuerpo'].fillna("sin cuerpo")  


train_df_m = train_df.copy()
test_df_m = test_df.copy()  



train_df_m['SPAM'] = train_df_m['Categoria'].apply(lambda x: 1 if x == 'SPAM' else 0)
test_df_m['SPAM'] = test_df_m['Categoria'].apply(lambda x: 1 if x == 'SPAM' else 0)

# X_train, X_test, y_train, y_test = train_test_split(train_df_m.Cuerpo, train_df_m.SPAM, test_size=0.25)

X_train = train_df_m.Cuerpo 
y_train = train_df_m.SPAM  
X_test = test_df_m.Cuerpo
y_test = test_df_m.SPAM  

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

joblib.dump(clf, 'spam_model.pkl')  

print('Modelo entrenado y guardado en spam_model.pkl')  

# Reporte de clasificación
print(classification_report(y_test, y_pred, target_names=['NO SPAM', 'SPAM']))    