from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)

# Daftar kelas tipe kulit
class_names = ['dry', 'normal', 'oily', 'sensitive']

# Load model
model = tf.keras.models.load_model("FaceDetectionModel.keras")

# Fungsi untuk mempersiapkan gambar
def prepare_image(file, target_size=(224, 224)):
    # Memuat gambar dari file stream
    img = Image.open(file)
    
    # Konversi ke RGB (jika gambar memiliki saluran alpha atau grayscale)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Ubah ukuran gambar
    img = img.resize(target_size)
    
    # Konversi ke array dan normalisasi
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

# Fungsi untuk prediksi tipe kulit
def predict_skin_type(image_data):
    image_array = prepare_image(image_data)
    predictions = model.predict(image_array)
    probabilities = predictions[0]

    class_probabilities = {class_names[i]: prob * 100 for i, prob in enumerate(probabilities)}
    predicted_class = class_names[np.argmax(probabilities)]
    predicted_probability = np.max(probabilities) * 100

    return class_probabilities, predicted_class, predicted_probability

# Fungsi untuk memberikan rekomendasi produk
def recommend_products(skin_types):
    # Membaca file CSV untuk produk skincare
    df = pd.read_csv("Skincare.csv")

    # Filter produk yang sesuai dengan tipe kulit
    recommended_products = df[df['skintype'].apply(lambda x: all(st in x for st in skin_types))]

    # Batasi kolom yang akan ditampilkan
    display_columns = ['product_name', 'brand', 'notable_effects', 'price', 'product_href']
    recommendations = recommended_products[display_columns].sample(n=5).to_dict(orient='records')
    
    return recommendations

# Endpoint untuk prediksi tipe kulit dan rekomendasi produk
@app.route('/predict', methods=['POST'])
def predict_skin_type():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No image file provided."}), 400
    
    try:
        # Proses gambar langsung dari stream
        img_array = prepare_image(file)

        # Prediksi jenis kulit
        predictions = model.predict(img_array)
        class_labels = ['Dry', 'Normal', 'Oily', 'Sensitive']
        predicted_probabilities = predictions[0]

        prediction_dict = {label: float(prob) for label, prob in zip(class_labels, predicted_probabilities)}

        # Rekomendasi produk
        skin_types = [label for label, prob in prediction_dict.items() if prob > 0.25]  # Atur threshold jika diperlukan
        df = pd.read_csv("Skincare.csv")
        recommended_products = df[df['skintype'].apply(lambda x: all(st in x for st in skin_types))]
        
        # Batasi ke kolom yang relevan
        display_columns = ['product_name', 'brand', 'notable_effects', 'price', 'product_href']
        recommendations = recommended_products[display_columns].sample(n=5).to_dict(orient='records')

        return jsonify({
            "predictions": prediction_dict,
            "skin_types": skin_types,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Mengambil port dari environment variable
    app.run(host='0.0.0.0', port=port)