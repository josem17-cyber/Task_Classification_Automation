import os
import numpy as np
from keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import classification_report
from data_preparation import load_and_prepare_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_model(input_shape, num_classes):
    """
    Esta función crea un modelo de red neuronal para predecir múltiples salidas categóricas.

    Args:
        input_dim (int): Dimensión de la entrada (número de características).
        num_classes(int): Número de clases para la salida 'Categoria'.

    Returns:
        model (keras.Model): Modelo de Keras compilado listo para el entrenamiento.
    """
    input_layer = Input(shape=(input_shape,))
    dense_layer = Dense(512, activation='relu')(input_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(256, activation='relu')(dropout_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(128, activation='relu')(dropout_layer)
    dropout_layer = Dropout(0.5)(dense_layer)

    output1 = Dense(num_classes, activation='softmax', name='categoria_output')(dropout_layer)
    
    model = Model(inputs=input_layer, outputs=output1)
    model.compile(optimizer=Adam(),
                loss={'categoria_output': 'sparse_categorical_crossentropy'},
                metrics={'categoria_output': 'accuracy'})
    return model

def load_data(file_path):
    X, y, label_encoders = load_and_prepare_data(file_path)
    y_categoria = y.astype(int)
    return X, y_categoria, label_encoders

def train_model(X, y_categoria, num_classes_categoria):
    model = create_model(X.shape[1], num_classes_categoria)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X, {'categoria_output': y_categoria},
            epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    return model

def save_model(model, file_path):
    model.save(file_path)
    print("Model Saved Successfully!!")

def evaluate_model(model, X, y_categoria, label_encoders):
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_categoria, y_pred_classes, target_names=label_encoders['Categoria'].classes_))

if __name__ == "__main__":
    try:
        file_path = '/home/jose-manuel/Escritorio/Task_Classification_Automation/classification_model/data/CAT_SPAM_2023_E.xlsx'
        X, y_categoria, label_encoders = load_data(file_path)

        num_classes_categoria = len(label_encoders['Categoria'].classes_)
        print(f"Número de clases: Categoría: {num_classes_categoria}")

        assert y_categoria.max() < num_classes_categoria, "Etiquetas de categoría fuera de rango"

        model = train_model(X, y_categoria, num_classes_categoria)
        save_model(model, '/home/jose-manuel/Escritorio/Task_Classification_Automation/classification_model/model.h5')
        evaluate_model(model, X, y_categoria, label_encoders)
    except Exception as e:
        print(f"An error occurred: {e}")