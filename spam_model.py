import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from data_preparation import load_and_prepare_data

data = pd.read_csv('data/spam.csv')

data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

# Predicciones
y_pred = clf.predict(X_test)

# Reporte de clasificación
print(classification_report(y_test, y_pred, target_names=data['Category'].unique()))
