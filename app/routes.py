from flask import Flask, request, jsonify, render_template
from app import app
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import io
import cv2
import base64

# Load model
MODEL_PATH = os.path.join(os.getcwd(), 'app', 'models', 'model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Labels
LABELS = ["cukup nutrisi", "kurang nutrisi", "rusak"]

# Utility Functions
def preprocess_image(image_bytes):
    """Normalize and preprocess the input image for model prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def remove_shadows(image):
    """Remove shadows from an image using morphological operations."""
    rgb_planes = cv2.split(image)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    return cv2.merge(result_planes)

def detect_green_hue(image):
    """Detect green hues in an image using HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([25, 30, 30])
    upper_bound = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return cv2.bitwise_and(image, image, mask=mask)

def detect_red_hue(image):
    """Detect red hues in an image using HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 200, 200])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return cv2.bitwise_and(image, image, mask=mask)

def enhanced_grayscale(image):
    """Enhance grayscale contrast using CLAHE."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def process_image(image):
    """Process the image for various features."""
    image_resized = cv2.resize(image, (128, 128))
    result_green = cv2.cvtColor(detect_green_hue(image_resized), cv2.COLOR_BGR2RGB)
    result_red = cv2.cvtColor(detect_red_hue(image_resized), cv2.COLOR_BGR2RGB)
    shadow_removed = cv2.cvtColor(remove_shadows(image_resized), cv2.COLOR_BGR2RGB)
    enhanced_gray = enhanced_grayscale(image_resized)
    combined_image = cv2.addWeighted(shadow_removed, 0.5,
                                     cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR), 0.5, 0)
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    return result_green, result_red, shadow_removed, combined_image_rgb, enhanced_gray

def convert_to_base64(image):
    """Convert an image to a base64-encoded string."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image_processing')
def image_processing():
    return render_template('image_processing.html')

@app.route('/class', methods=['GET', 'POST'])
def class_view():
    prediction = None
    error_message = None
    uploaded_image_rgb = None
    if request.method == 'POST':
        try:
            # Membaca file gambar yang diunggah
            image_bytes = request.files['image'].read()
            
            # Membuka gambar dengan OpenCV dan konversi ke RGB
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            uploaded_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess gambar
            processed_image = preprocess_image(image_bytes)
            
            # Melakukan prediksi
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class]
            
            # Membuat dictionary untuk menyimpan hasil prediksi
            prediction = {
                'label': LABELS[predicted_class],
                'confidence': float(confidence)
            }
        except Exception as e:
            # Menangani error jika ada masalah saat memproses gambar
            error_message = f"Error processing image: {str(e)}"

    # Konversi gambar RGB ke base64 agar dapat dirender di template
    uploaded_image_url = convert_to_base64(uploaded_image_rgb) if uploaded_image_rgb is not None else None

    # Mengembalikan hasil prediksi atau error ke template
    return render_template(
        'class.html',
        prediction=prediction,
        error=error_message,
        uploaded_image_url=uploaded_image_url
    )


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/process_image', methods=['POST'])
def process_uploaded_image():
    """Handle image processing requests."""
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"error": "No image file found or selected"}), 400
    try:
        # Baca gambar asli dari upload
        image_bytes = request.files['image'].read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Pastikan gambar asli diubah ke RGB
        img_array = np.array(img)  # Konversi ke array NumPy
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # OpenCV bekerja dalam BGR, pastikan kita gunakan format ini

        # Konversi gambar asli ke base64 untuk ditampilkan kembali
        original_image_url = convert_to_base64(img_array)  # Tidak perlu mengonversi lagi jika sudah RGB

        # Proses gambar untuk fitur lain
        result_green, result_red, shadow_removed, combined_image, enhanced_gray = process_image(img_rgb)

        # Konversi semua hasil ke format RGB untuk konsistensi
        result_green_rgb = result_green  # Sudah dalam RGB di fungsi process_image
        result_red_rgb = result_red      # Sudah dalam RGB di fungsi process_image
        shadow_removed_rgb = shadow_removed  # Sudah dalam RGB di fungsi process_image
        combined_image_rgb = combined_image  # Sudah dalam RGB di fungsi process_image

        # Konversi ke base64 untuk ditampilkan di template
        return render_template(
            'image_processing.html',
            original_image_url=original_image_url,  # Gambar asli
            result_green_url=convert_to_base64(result_green_rgb),  # Deteksi hijau
            result_red_url=convert_to_base64(result_red_rgb),      # Deteksi merah
            shadow_removed_url=convert_to_base64(shadow_removed_rgb),  # Penghapusan bayangan
            combined_image_url=convert_to_base64(combined_image_rgb)   # Gambar gabungan
        )
    except Exception as e:
        return render_template('image_processing.html', error=f"Error processing image: {str(e)}")

