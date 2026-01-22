import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/arecanut_ripeness_model_YCgCr.h5')

IMG_SIZE = 100

def preprocess_image_YCgCr(image_path):
    """
    Convert image to YCgCr, resize, and normalize.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")

    img_YCgCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCgCr
    img_resized = cv2.resize(img_YCgCr, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    
    return img_normalized

def predict_image_YCgCr(image_path):
    """
    Predict whether an arecanut bunch is ripe or unripe.
    """
    try:
        img = preprocess_image_YCgCr(image_path)
        img_input = np.expand_dims(img, axis=0)  # Add batch dimension
        
        prediction = model.predict(img_input)
        result = 'Ripe' if prediction[0][0] >= 0.5 else 'Unripe'
        print(f"Prediction: {result} (Confidence: {prediction[0][0]:.2f})")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    image_path = r"C:\Users\tarun\Downloads\images (3).jpg" # Replace with the test image path
    predict_image_YCgCr(image_path)
