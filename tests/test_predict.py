import unittest
import numpy as np
import tensorflow as tf
from app.models.predict import predict, load_model

class TestPredict(unittest.TestCase):
    def setUp(self):
        """Setup yang dijalankan sebelum setiap tes."""
        # Path ke model yang akan digunakan (ubah jika perlu)
        self.model_path = 'app/models/model.h5'
        
        # Dummy data untuk pengujian prediksi
        self.data = np.array([[0.5, 1.5]])  # Contoh data yang sesuai dengan input model
    
    def test_load_model(self):
        """Test untuk memeriksa apakah model dapat dimuat dengan benar."""
        model = load_model(self.model_path)
        self.assertIsInstance(model, tf.keras.Model)  # Memastikan model adalah instance dari tf.keras.Model
    
    def test_predict(self):
        """Test untuk memeriksa apakah fungsi prediksi memberikan output yang valid."""
        model = load_model(self.model_path)
        prediction = predict(self.data, self.model_path)
        
        # Memastikan prediksi adalah array
        self.assertIsInstance(prediction, np.ndarray)
        
        # Memastikan prediksi tidak kosong
        self.assertGreater(prediction.size, 0)
    
    def test_invalid_model_path(self):
        """Test untuk menangani jalur model yang tidak valid."""
        with self.assertRaises(OSError):
            predict(self.data, 'app/model/model.h5')

if __name__ == '__main__':
    unittest.main()
