import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os

# Define constants
IMG_SIZE = (224, 224)

# Load trained model
model = load_model("movie_recognition_model.h5")

# Get class names from dataset directory
class_names = sorted([d for d in os.listdir("dataset/") if os.path.isdir(os.path.join("dataset/", d))])

def predict_movie(img_path):
    """
    Loads an image, preprocesses it, and predicts the movie.
    """
    # Load and preprocess the image
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Get confidence score
    confidence = np.max(predictions) * 100

    return predicted_class, confidence

# Test the model
test_image_path = "dataset/interstellar/frame_120.jpg"  # Make sure this file exists

if os.path.exists(test_image_path):
    predicted_movie, confidence = predict_movie(test_image_path)
    print(f"Predicted movie: {predicted_movie}")
    print(f"Confidence: {confidence:.2f}%")
else:
    print(f"Error: Test image '{test_image_path}' not found.")
