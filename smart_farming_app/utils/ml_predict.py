import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
from PIL import Image
import random 

# --- MODEL CREATION SCRIPT (For First Run) ---
def create_dummy_models():
    """Creates and saves simple dummy ML models for the Flask app to load."""
    
    # Ensures the 'models' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 1. Crop Recommendation Model Data
    data_crop = {
        'N': [90, 85, 60, 50, 10], 'P': [42, 58, 55, 30, 10], 'K': [43, 41, 44, 20, 10],
        'temperature': [20.8, 21.7, 23.0, 30.5, 15.0], 'humidity': [82, 80, 82, 60, 70],
        'ph': [6.5, 7.0, 7.8, 6.2, 5.5], 'rainfall': [202, 226, 263, 100, 50],
        'label': ['rice', 'rice', 'maize', 'cotton', 'wheat']
    }
    df_crop = pd.DataFrame(data_crop)
    X_crop = df_crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_crop = df_crop['label']
    crop_model = RandomForestClassifier(random_state=42)
    crop_model.fit(X_crop, y_crop)
    joblib.dump(crop_model, 'models/crop_recommendation_model.pkl')

    # 2. Fertilizer Recommendation Model (Simplified for demo)
    DUMMY_FERTILIZER_MAPPING = {
        'rice': 'Urea and Zinc', 
        'maize': 'DAP and Potash', 
        'cotton': 'NPK 14-35-14', 
        'wheat': 'NPK 10-26-26',
        'default': 'NPK general purpose'
    }
    joblib.dump(DUMMY_FERTILIZER_MAPPING, 'models/fertilizer_recommendation_model.pkl')

# ----------------- Prediction Functions (Used by app.py) -----------------

CROP_MODEL_PATH = 'models/crop_recommendation_model.pkl'
FERT_MODEL_PATH = 'models/fertilizer_recommendation_model.pkl'

# Load trained models
try:
    CROP_MODEL = joblib.load(CROP_MODEL_PATH)
    DUMMY_FERTILIZER_MAPPING = joblib.load(FERT_MODEL_PATH)
except:
    CROP_MODEL = None
    DUMMY_FERTILIZER_MAPPING = {}

def get_crop_recommendation(N, P, K, pH, temp, humidity, rainfall):
    """Suggests the best crop and provides prediction probabilities."""
    if CROP_MODEL is None:
        return "Model Error", []
    
    features = np.array([[N, P, K, temp, humidity, pH, rainfall]])
    
    prediction = CROP_MODEL.predict(features)[0]
    probabilities = CROP_MODEL.predict_proba(features)[0]
    
    class_labels = CROP_MODEL.classes_
    prob_list = sorted(zip(class_labels, probabilities), key=lambda x: x[1], reverse=True)
    
    # Format top 3 recommendations
    top_recommendations = [
        {'crop': item[0].upper(), 'confidence': f"{item[1]*100:.2f}%"} 
        for item in prob_list[:3]
    ]

    return prediction, top_recommendations

def get_fertilizer_recommendation(crop_name):
    """Recommends fertilizer based on the suggested crop (simplified)."""
    return DUMMY_FERTILIZER_MAPPING.get(crop_name, DUMMY_FERTILIZER_MAPPING['default'])

def analyze_soil_health(N, P, K, pH):
    """Simple rule-based soil health analysis."""
    if N > 80 and P > 40 and K > 150 and pH >= 6.0 and pH <= 7.5:
        return "Good 🟢"
    elif N < 30 or P < 15 or K < 50 or pH < 5.0 or pH > 8.0:
        return "Poor 🔴"
    else:
        return "Moderate 🟡"

def get_price_forecast(crop_name):
    """Placeholder for Time Series Price Forecasting."""
    forecast = {
        'rice': "₹2200/quintal (Expected increase: +5%)",
        'maize': "₹1800/quintal (Expected decrease: -2%)",
        'cotton': "₹7000/quintal (Stable)",
        'wheat': "₹2400/quintal (Expected increase: +10%)",
        'default': "Price data unavailable."
    }
    return forecast.get(crop_name, forecast['default'])

def predict_leaf_disease(image_file_stream):
    """
    Simulates a CNN prediction for a leaf disease, returning pesticide quantity.
    """
    
    # --- Simulation Step: Image Processing (Placeholder) ---
    try:
        # Referenced technologies: OpenCV for image preprocessing, TensorFlow/PyTorch for model
        img = Image.open(image_file_stream)
        img.verify()
    except Exception:
        return {
            'disease': "Invalid File or Format",
            'confidence': "0.00%",
            'remedy': "Requires valid JPEG or PNG. System uses OpenCV for resizing.",
            'severity': 'Critical',
            'pesticide_qty': '0.0 L/Ha',
            'model_used': 'Input Error'
        }

    # --- Simulation Step: Model Prediction ---
    
    disease_options = [
        {'name': "Tomato Mosaic Virus", 'severity': 'High', 'pesticide': '1.5 L/Ha (Antiviral)', 'model': 'CNN-MobileNetV3'},
        {'name': "Apple Scab", 'severity': 'Moderate', 'pesticide': '0.75 Kg/Ha (Fungicide)', 'model': 'YOLOv5'},
        {'name': "Wheat Rust (Puccinia)", 'severity': 'High', 'pesticide': '1.0 L/Ha (Triazole)', 'model': 'ResNet'},
        {'name': "Healthy Leaf", 'severity': 'Low', 'pesticide': '0.0 L/Ha (None)', 'model': 'MobileNet'}
    ]
    
    result = random.choice(disease_options)
    
    return {
        'disease': result['name'],
        'confidence': f"{random.uniform(90.0, 99.9):.2f}%",
        'remedy': "Consult local expert for application timing.",
        'severity': result['severity'],
        'pesticide_qty': result['pesticide'],
        'model_used': result['model'] 
    }

def get_mock_crop_data():
    """Generates mock data for the Crop Management page."""
    mock_data = [
        {'name': 'Wheat (Field 1)', 'yield': '5.2 T/Ha', 'confidence': '88%'},
        {'name': 'Rice (Field 2)', 'yield': '3.9 T/Ha', 'confidence': '95%'},
        {'name': 'Maize (Field 3)', 'yield': '7.1 T/Ha', 'confidence': '75%'}
    ]
    return mock_data

def get_irrigation_alert(soil_ph, current_temp, current_rainfall, humidity):
    """Determines smart irrigation alert status based on combined data."""
    alert = {'message': "System Operational - No immediate action required.", 'type': 'low'}

    # High Temperature Warning (Requires attention)
    if current_temp > 35 and humidity < 40:
        alert = {'message': "HIGH TEMPERATURE WARNING: Severe heat and low humidity. Check soil moisture immediately.", 'type': 'high'}
    
    # Skip Irrigation (Rain expected)
    elif current_rainfall > 1.0: # If rainfall > 1.0 mm/hr (mock)
        alert = {'message': "SKIP IRRIGATION: Rain detected or forecasted. Water scheduling paused.", 'type': 'medium'}
        
    # Water Now (Low humidity and high temp, or low pH which increases nutrient need)
    elif current_temp > 30 and humidity < 60:
        alert = {'message': "WATER NOW: High ambient temperature detected. Initiate irrigation cycle.", 'type': 'high'}
        
    # pH Warning
    elif soil_ph < 5.5 or soil_ph > 7.8:
        alert = {'message': "pH ALERT: Soil acidity imbalance detected. Adjust water input.", 'type': 'medium'}
        
    return alert

def generate_chatbot_response(query):
    """Simulates an AI chatbot response trained on Indian agriculture data."""
    query = query.lower()
    
    if "crop" in query or "soil" in query:
        response = "Based on regional data, **Maize** and **Pulses** thrive well in balanced loamy soils with good drainage. For high potash soil, try **Potato** or **Sugarcane**."
    elif "seed" in query or "best variety" in query:
        response = "The recommended **hybrid seed** for your region's current climate is 'Pusa Basmati 1121' for paddy, offering high yield and disease resistance."
    elif "scheme" in query or "subsidy" in query:
        response = "**Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)** provides income support. Also, check the **Soil Health Card Scheme** for free testing and fertilizer recommendations."
    elif "pesticide" in query or "disease" in query:
        response = "If you are seeing early blight, use a copper-based fungicide like Copper Oxychloride. Use **1.5 kg per hectare** mixed with 500L of water. Always wear protective gear."
    else:
        response = "I am trained on topics related to Indian agriculture, including crop schedules, seed types, and schemes. Please ask a specific farming question!"
        
    return response