import tensorflow as tf
import numpy as np

def load_model(model_path):
    """Memuat model TensorFlow dari path yang diberikan."""
    return tf.keras.models.load_model(model_path)

def predict(data, model_path='app/models/model.h5'):
    """Melakukan prediksi berdasarkan data yang diberikan dan model yang dimuat."""
    model = load_model(model_path)
    prediction = model.predict(data)
    return prediction
