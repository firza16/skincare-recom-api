from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from PIL import Image


app = Flask(__name__)

# Daftar kelas tipe kulit
class_names = ['Dry', 'Normal', 'Oily', 'Sensitive']

# Load model
model = tf.keras.models.load_model("FaceDetectionModel.keras")

# Fungsi untuk mempersiapkan gambar
def prepare_image(file, target_size=(224, 224)):
    img = Image.open(BytesIO(file.read()))  # Konversi FileStorage ke BytesIO
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Fungsi untuk memberikan rekomendasi produk
def recommend_products(skin_types, csv_file='Skincare.csv'):
    df = pd.read_csv(csv_file)
    recommended_products = df[df['skintype'].apply(lambda x: all(st in x for st in skin_types))]
    
    display_columns = ['product_name', 'brand', 'notable_effects', 'price', 'product_href', 'picture_src']
    recommended_products = recommended_products[display_columns]
    
    recommendations = recommended_products.sample(n=5).to_dict(orient='records')
    return recommendations

# Fungsi untuk prediksi tipe kulit
def predict_skin_type(image_data):
    image_array = prepare_image(image_data)
    predictions = model.predict(image_array)
    
    predicted_probabilities = predictions[0]
    prediction_dict = {class_names[i]: prob * 100 for i, prob in enumerate(predicted_probabilities)}
    
    sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)
    
    skin_types = [label for label, prob in sorted_predictions if prob > 25]  # Atur threshold jika diperlukan
    return skin_types, prediction_dict

# Endpoint untuk prediksi tipe kulit dan rekomendasi produk
@app.route('/predict', methods=['POST'])
def predict_and_recommend():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No image file provided."}), 400
    
    try:
        skin_types, prediction_dict = predict_skin_type(file)
        recommendations = recommend_products(skin_types)
        
        return jsonify({
            "predictions": prediction_dict,
            "skin_types": skin_types,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8080, debug=True)
