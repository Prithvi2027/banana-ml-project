"""
Flask API for banana ripeness prediction
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torch
import io
import os

import config
from model import BananaRipenessModel
from dataset import get_transforms

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
CORS(app)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
transform = get_transforms(train=False)


def load_model():
    """Load the trained model"""
    global model
    if model is None:
        model = BananaRipenessModel(
            num_classes=config.NUM_CLASSES,
            model_name=config.MODEL_NAME,
            pretrained=False
        )
        
        # Try to load best model, fallback to final model
        if os.path.exists(config.BEST_MODEL_PATH):
            checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {config.BEST_MODEL_PATH}")
        elif os.path.exists(config.FINAL_MODEL_PATH):
            model.load_state_dict(torch.load(config.FINAL_MODEL_PATH, map_location=device))
            print(f"Loaded final model from {config.FINAL_MODEL_PATH}")
        else:
            raise FileNotFoundError("No trained model found. Please train the model first.")
        
        model = model.to(device)
        model.eval()
        print(f"Model loaded on {device}")


def predict_image(image):
    """Predict banana ripeness from image"""
    # Preprocess image
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    # Get class name and days
    class_idx = predicted.item()
    class_name = config.CLASS_NAMES[class_idx]
    days_left = config.CLASS_TO_DAYS[class_name]
    confidence_score = confidence.item() * 100
    
    # Get all class probabilities
    all_probs = {
        config.CLASS_NAMES[i]: float(probabilities[0][i] * 100) 
        for i in range(len(config.CLASS_NAMES))
    }
    
    return {
        'class': class_name,
        'days_left': days_left,
        'confidence': round(confidence_score, 2),
        'probabilities': all_probs
    }


def get_recommendation(class_name, days_left):
    """Get eating recommendation based on ripeness"""
    recommendations = {
        'unripe': f"Still too green. Wait {days_left} days before eating.",
        'ripe': f"Perfect for eating! Consume within {days_left} days.",
        'overripe': f"Very ripe with spots. Eat today or use for smoothies/baking!",
        'rotten': "Too late to eat. Discard this banana."
    }
    return recommendations.get(class_name, "Unknown ripeness stage")


def get_stage_emoji(class_name):
    """Get emoji for ripeness stage"""
    emojis = {
        'unripe': '🟢',
        'ripe': '🟡',
        'overripe': '🟠',
        'rotten': '🔴'
    }
    return emojis.get(class_name, '❓')


@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and predict
        image = Image.open(io.BytesIO(file.read()))
        result = predict_image(image)
        
        # Add additional info
        result['recommendation'] = get_recommendation(result['class'], result['days_left'])
        result['emoji'] = get_stage_emoji(result['class'])
        result['display_days'] = 'Today!' if result['days_left'] == 0 else f"{result['days_left']} day{'s' if result['days_left'] > 1 else ''}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run app
    print("\n" + "="*50)
    print("Starting Flask API server...")
    print("Visit: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
