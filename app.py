import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)

MODEL_PATH = r"C:\Users\ferha\Masaüstü\Tumor\brain_tumor_model.h5"
model = load_model(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Dosya yüklenmedi"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400

    upload_folder = "static/uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    img = load_img(file_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "No Tumor" if prediction[0][0] > 0.5 else "Tumor"

    return jsonify({"prediction": result, "image_path": file_path})

if __name__ == "__main__":
    app.run(debug=True)
