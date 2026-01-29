"""
Plant Disease Detection AI Service
Flask API that loads a Keras model and provides predictions for plant disease detection.
"""

import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tf_keras as keras  # Use tf_keras for Keras 2 compatibility

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'best_plant_disease_model.keras')
IMAGE_SIZE = (224, 224)

# Class labels (39 plant diseases) - matches the model training
CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy",
    "Background",
    "Blueberry___Healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___Healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___Healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___Healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy",
    "Raspberry___Healthy",
    "Soybean___Bacterial_pustule",
    "Soybean___Healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___Healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Healthy"
]

# Load model at startup
print(f"Loading model from: {MODEL_DIR}")
model = None

def load_model():
    """Load the plant disease model from Keras SavedModel format"""
    global model
    
    model_weights = os.path.join(MODEL_DIR, 'model.weights.h5')
    model_config = os.path.join(MODEL_DIR, 'config.json')
    
    if os.path.exists(model_config) and os.path.exists(model_weights):
        print(f"Found Keras format model files")
        try:
            # Load the config
            with open(model_config, 'r') as f:
                config_str = json.dumps(json.load(f))
            
            print(f"Building model from config...")
            # Create model from config JSON
            model = keras.models.model_from_json(config_str)
            
            # Load weights
            print(f"Loading weights from {model_weights}...")
            model.load_weights(model_weights)
            
            print("✅ Model loaded successfully!")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"❌ Model files not found in {MODEL_DIR}")
        return None

# Load model when app starts
model = load_model()


def format_disease_name(class_label):
    """
    Convert class label to human-readable disease name.
    Example: "Tomato___Early_blight" -> "Tomato - Early Blight"
    """
    parts = class_label.split("___")
    if len(parts) == 2:
        crop = parts[0].replace("_", " ").replace(",", "")
        disease = parts[1].replace("_", " ").title()
        return f"{crop} - {disease}"
    return class_label.replace("_", " ")


def is_healthy(class_label):
    """Check if the prediction indicates a healthy plant."""
    return "healthy" in class_label.lower()


def preprocess_image(image_bytes):
    """
    Preprocess image for model prediction.
    - Resize to 224x224
    - Normalize pixel values to [0, 1]
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to expected input size
    image = image.resize(IMAGE_SIZE)
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': len(CLASS_LABELS)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict plant disease from uploaded image.
    
    Expects multipart form data with 'image' field.
    Returns JSON with prediction results.
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'The AI model failed to load. Please check the model path.'
        }), 500
    
    # Check if image was provided
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided',
            'message': 'Please upload an image file with the key "image"'
        }), 400
    
    try:
        # Read image file
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        if len(image_bytes) == 0:
            return jsonify({
                'error': 'Empty image',
                'message': 'The uploaded image appears to be empty'
            }), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get prediction probabilities
        probabilities = predictions[0]
        
        # Get top prediction
        top_index = np.argmax(probabilities)
        top_confidence = float(probabilities[top_index]) * 100
        top_class = CLASS_LABELS[top_index] if top_index < len(CLASS_LABELS) else f'Class {top_index}'
        
        # Get alternative predictions (top 3)
        sorted_indices = np.argsort(probabilities)[::-1]
        alternatives = []
        for i in sorted_indices[1:4]:  # Skip the top one, get next 3
            if i < len(CLASS_LABELS):
                alternatives.append({
                    'diseaseName': format_disease_name(CLASS_LABELS[i]),
                    'confidence': float(probabilities[i]) * 100
                })
        
        # Build response
        response = {
            'diseaseName': format_disease_name(top_class),
            'rawClassName': top_class,
            'confidence': round(top_confidence, 2),
            'isHealthy': is_healthy(top_class),
            'alternativePredictions': alternatives
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """Return all supported disease classes."""
    return jsonify({
        'classes': [
            {
                'index': i,
                'rawName': label,
                'displayName': format_disease_name(label),
                'isHealthy': is_healthy(label)
            }
            for i, label in enumerate(CLASS_LABELS)
        ]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"\n🚀 Starting Plant Disease Detection API on port {port}")
    print(f"📍 Endpoints:")
    print(f"   POST /predict - Upload image for disease detection")
    print(f"   GET  /health  - Health check")
    print(f"   GET  /classes - List all disease classes\n")
    app.run(host='0.0.0.0', port=port, debug=True)
