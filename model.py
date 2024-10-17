# model.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from data_preparation import load_and_prepare_data
from keras.callbacks import EarlyStopping # type: ignore
import os
import warnings
import tensorflow as tf
from sklearn.metrics import classification_report

# Suprimir mensajes de advertencia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir logs de TensorFlow
tf.get_logger().setLevel('ERROR')  # Suprimir logs de TensorFlow
warnings.filterwarnings('ignore')  # Suprimir advertencias de otras librerías



def create_model(input_dim, num_classes_categoria):
    """
    Esta función crea un modelo de red neuronal para predecir múltiples salidas categóricas.

    Args:
        input_dim (int): Dimensión de la entrada (número de características).
        num_classes_categoria (int): Número de clases para la salida 'Categoria'.
        num_classes_dueno (int): Número de clases para la salida 'Dueño'.

    Returns:
        model (keras.Model): Modelo de Keras compilado listo para el entrenamiento.
    """
    input_layer = Input(shape=(input_dim,))
    dense_layer = Dense(512, activation='relu')(input_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(256, activation='relu')(dropout_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(128, activation='relu')(dropout_layer)
    dropout_layer = Dropout(0.5)(dense_layer)

    output1 = Dense(num_classes_categoria, activation='softmax', name='categoria_output')(dropout_layer)

    model = Model(inputs=input_layer, outputs=output1)
    model.compile(optimizer=Adam(),
                loss={'categoria_output': 'sparse_categorical_crossentropy'},
                metrics={'categoria_output': 'accuracy'})
    return model

if __name__ == "__main__":
    file_path = 'data/Automatizacion_Clasificacion_EDA_HIDRAL.xlsx'
    X, y, _, label_encoders = load_and_prepare_data(file_path)
    y_categoria = y.astype(int)

    num_classes_categoria = len(label_encoders['Categoria'].classes_)

    # Imprimir el número de clases para verificar
    print("Número de clases:")
    print(f"Categoría: {num_classes_categoria}")

    assert y_categoria.max() < num_classes_categoria, "Etiquetas de categoría fuera de rango"

    model = create_model(X.shape[1], num_classes_categoria)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X, {'categoria_output': y_categoria},
            epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    model.save('model.h5')
    print("Model Saved Succesfully!!")

# Hacer predicciones
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Obtener las clases predichas

    # Imprimir el classification report
    print(classification_report(y_categoria, y_pred_classes, target_names=label_encoders['Categoria'].classes_))