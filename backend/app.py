from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import cv2  # Add OpenCV for video processing
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model once at startup
MODEL_PATH = "movie_recognition_model.h5"
model = load_model(MODEL_PATH)

# Define class names (make sure it matches your dataset)
DATASET_DIR = "dataset/"
class_names = sorted(os.listdir(DATASET_DIR))

@app.route("/", methods=["GET"])
def home():
    return "Server is running!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Use POST to send an image.", 405  # Return a user-friendly message

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_path = "temp.jpg"
    file.save(img_path)

    # Preprocess the image
    IMG_SIZE = (224, 224)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    return jsonify({"movie": predicted_class, "confidence": confidence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

#see you later spotra