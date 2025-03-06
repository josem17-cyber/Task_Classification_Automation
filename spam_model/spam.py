import joblib

# Cargar el modelo
model = joblib.load('/home/jose-manuel/Escritorio/Task_Classification_Automation/spam_model/spam_model.pkl')

# Leer el contenido del archivo de entrada
with open('/home/jose-manuel/Escritorio/Task_Classification_Automation/spam_model/in.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()

# Realizar la predicción para todo el texto
prediction = model.predict([input_text])

# Escribir la salida
with open('/home/jose-manuel/Escritorio/Task_Classification_Automation/spam_model/out.txt', 'w', encoding='utf-8') as file:
    file.write(f'{"SPAM" if prediction[0] == 1 else "NOT SPAM"}\n')

# Vaciar el archivo de entrada
open('/home/jose-manuel/Escritorio/Task_Classification_Automation/spam_model/in.txt', 'w').close()
