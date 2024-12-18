from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('imageclassifier.h5')

# Map predictions to labels
labels = {0: "Happy", 1: "Sad"}

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded file
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    try:
        # Preprocess the image
        img = Image.open(filepath).convert('RGB')  # Ensure the image is in RGB format
        img = img.resize((256, 256))  # Resize the image to match the model's input size
        img_array = np.array(img) / 255.0  # Scale pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_label = labels[int(np.round(prediction[0][0]))]  # Map prediction to label

        return render_template('result.html', label=predicted_label)

    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

# Run the app
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')  # Create uploads directory if it doesn't exist
    app.run(debug=True)
