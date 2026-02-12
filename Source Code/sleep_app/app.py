from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pickle
import logging
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer, Dropout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model and scaler as None
model = None
scaler = None

def validate_input_data(data, feature_names):
    """Validate input data ranges"""
    ranges = {
        'Age': (1, 120),
        'Sleep Duration': (0, 24),
        'Quality of Sleep': (1, 10),
        'Physical Activity Level': (0, 1440),
        'Stress Level': (1, 10),
        'Heart Rate': (40, 200),
        'Daily Steps': (0, 100000),
        'Systolic': (70, 200),
        'Diastolic': (40, 130)
    }
    
    for feature in feature_names:
        value = float(data[feature])
        min_val, max_val = ranges[feature]
        if not min_val <= value <= max_val:
            raise ValueError(f"{feature} must be between {min_val} and {max_val}")

def create_model():
    """Create model with the correct architecture"""
    model = Sequential([
        InputLayer(input_shape=(9,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_ml_components():
    """Load and verify ML components"""
    global model, scaler
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model1.h5')
        scaler_path = os.path.join(current_dir, 'scaler1.pkl')
        
        # Verify files exist
        if not all(os.path.exists(p) for p in [model_path, scaler_path]):
            raise FileNotFoundError("Model or scaler file not found")
            
        # Load model
        model = create_model()
        model.load_weights(model_path)
        
        # Load and test scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Verify scaler works
        test_data = np.array([[30, 7.0, 7, 50, 5, 78, 7500, 128, 84]])
        _ = scaler.transform(test_data)
        
        logger.info("ML components loaded and verified successfully")
        return None
        
    except Exception as e:
        error_msg = f"Error loading ML components: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Initialize components
error_message = load_ml_components()

# Define features
feature_names = [
    'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/procedure')
def procedure():
    return render_template('procedure.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded. ' + (error_message or '')})
        
    try:
        # Get and validate data
        data = request.json
        if not all(f in data for f in feature_names):
            missing = [f for f in feature_names if f not in data]
            return jsonify({'error': f'Missing features: {", ".join(missing)}'})
            
        validate_input_data(data, feature_names)
        
        # Prepare and transform input
        input_data = np.array([float(data[f]) for f in feature_names]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        pred_class = np.argmax(prediction, axis=1)[0]
        
        # Format result
        classes = {0: 'No Disorder', 1: 'Sleep Apnea', 2: 'Insomnia'}
        result = classes.get(pred_class, "Unknown")
        confidence = float(prediction[0][pred_class])
        
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%"
        })
        
    except ValueError as ve:
        return jsonify({'error': f'Validation error: {str(ve)}'})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'})

if __name__ == '__main__':
    app.run(debug=True)