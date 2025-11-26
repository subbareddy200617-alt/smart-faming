import os
import random
import json
from datetime import datetime, timedelta
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, send_file

# --- SAFE IMPORT BLOCKS ---
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: ML libraries not found. Using Simple Mode.")
    ML_AVAILABLE = False

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: ReportLab not found. PDF generation disabled.")
    PDF_AVAILABLE = False

app = Flask(__name__)

# ==========================================
# 1. GLOBAL DATA STORE
# ==========================================

# 7-Crop Configuration with specific planting dates
MOCK_CROP_FIELDS = {
    'Wheat (Field 1)': {'crop_type': 'Wheat', 'date_planted': '2025-10-15', 'hectares': 5.5},
    'Rice (Field 2)': {'crop_type': 'Rice', 'date_planted': '2025-08-01', 'hectares': 3.9},
    'Maize (Field 3)': {'crop_type': 'Maize', 'date_planted': '2025-09-10', 'hectares': 7.1},
    'Sugarcane (Field 4)': {'crop_type': 'Sugarcane', 'date_planted': '2025-03-01', 'hectares': 12.0},
    'Cotton (Field 5)': {'crop_type': 'Cotton', 'date_planted': '2025-06-15', 'hectares': 4.0},
    'Potato (Field 6)': {'crop_type': 'Potato', 'date_planted': '2025-10-01', 'hectares': 3.0},
    'Barley (Field 7)': {'crop_type': 'Barley', 'date_planted': '2025-10-05', 'hectares': 6.0},
}

MOCK_IOT_STATE = {
    'pump_status': 'OFF',
    'control_mode': 'AUTO',
    'soil_moisture': 45,
    'ambient_temp': 30.5
}

MENU_ITEMS = ['Dashboard', 'Crop Management', 'IoT Sensors', 'AI Recommendations', 'Weather', 'Marketplace', 'Reports']

LANGUAGE_TEXT = {
    'en': {
        'title': 'AgriSense AI',
        'heading': 'Soil and Weather Input',
        'recommend_btn': 'Get Recommendations',
        'result_heading': 'Farming Suggestions',
        'soil_health': 'Soil Health Analysis',
        'crop': 'Recommended Crop',
        'fert': 'Fertilizer Recommendation',
        'crop_water': 'Crop Water Requirement',
        'price_forecast': 'Price Forecast (1-Month)',
        'temp': 'Temperature', 'humidity': 'Humidity', 'rainfall': 'Rainfall'
    }
}

# ==========================================
# 2. ADVANCED CROP LOGIC (UPDATED)
# ==========================================

# Database of Schedules (Intervals in Days)
CROP_CONFIG = {
    'Wheat': {'water_every': 21, 'fert_days': [21, 45], 'pest_days': [60]},
    'Rice': {'water_every': 7, 'fert_days': [15, 45, 70], 'pest_days': [30, 60]},
    'Maize': {'water_every': 12, 'fert_days': [25, 50], 'pest_days': [40]},
    'Sugarcane': {'water_every': 15, 'fert_days': [45, 90], 'pest_days': [100]},
    'Cotton': {'water_every': 15, 'fert_days': [40, 80], 'pest_days': [50, 90]},
    'Potato': {'water_every': 10, 'fert_days': [30], 'pest_days': [45]},
    'Barley': {'water_every': 20, 'fert_days': [30], 'pest_days': [50]},
}

GLOBAL_CROP_MODEL = None

def create_dummy_models():
    global GLOBAL_CROP_MODEL
    if not ML_AVAILABLE: return
    if not os.path.exists('models'): os.makedirs('models')
    
    data_crop = {
        'N': [90, 85, 60, 50, 10], 'P': [42, 58, 55, 30, 10], 'K': [43, 41, 44, 20, 10],
        'temperature': [20.8, 21.7, 23.0, 30.5, 15.0], 'humidity': [82, 80, 82, 60, 70],
        'ph': [6.5, 7.0, 7.8, 6.2, 5.5], 'rainfall': [202, 226, 263, 100, 50],
        'label': ['rice', 'rice', 'maize', 'cotton', 'wheat']
    }
    df = pd.DataFrame(data_crop)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']], df['label'])
    GLOBAL_CROP_MODEL = model

def get_crop_recommendation(N, P, K, pH, weather):
    if not ML_AVAILABLE or GLOBAL_CROP_MODEL is None:
        return "Wheat (Fallback)", [{'crop': 'Wheat', 'confidence': '100%'}]

    try:
        input_data = np.array([[N, P, K, weather['temperature'], weather['humidity'], pH, weather['rainfall']]])
        prediction = GLOBAL_CROP_MODEL.predict(input_data)[0]
        probs = GLOBAL_CROP_MODEL.predict_proba(input_data)[0]
        conf = f"{max(probs)*100:.1f}%"
        return prediction, [{'crop': prediction, 'confidence': conf}]
    except:
        return "Rice", [{'crop': 'Rice', 'confidence': 'Fallback'}]

def get_mock_schedules(crop_details):
    """
    UPDATED: Calculates CORRECT next dates based on planting date.
    """
    crop_type = crop_details['crop_type']
    raw_name = crop_type.split(' ')[0] # Handle "Rice (Basmati)" -> "Rice"
    hectares = crop_details.get('hectares', 0)
    
    # 1. Calculate Age
    try:
        planted_date = datetime.strptime(crop_details['date_planted'], '%Y-%m-%d')
    except:
        planted_date = datetime.now() - timedelta(days=1)
    
    today = datetime.now()
    days_age = (today - planted_date).days
    
    # 2. Determine Stage
    stage = "Vegetative"
    status_color = "green"
    
    if raw_name == 'Wheat':
        if days_age < 21: stage = "Crown Root Initiation"
        elif days_age < 60: stage = "Tillering"
        elif days_age < 100: stage = "Heading"
        else: stage = "Harvest Ready"; status_color="orange"
    elif raw_name == 'Rice':
        if days_age < 30: stage = "Seedling"
        elif days_age < 60: stage = "Tillering"
        elif days_age < 90: stage = "Panicle Init"
        else: stage = "Maturity"; status_color="orange"
    elif raw_name == 'Sugarcane':
        if days_age < 120: stage = "Formative"
        elif days_age < 250: stage = "Grand Growth"
        else: stage = "Ripening"

    growth_msg = f"**{crop_type}:** {stage} (Day {days_age}) - <span style='color:{status_color}'>On Track</span>"

    # 3. Calculate Schedule (The Fix)
    config = CROP_CONFIG.get(raw_name, CROP_CONFIG['Wheat'])
    schedule_list = []

    # A. Next Water Date
    water_interval = config['water_every']
    # Formula: Find next multiple of interval greater than current age
    days_until_next_water = water_interval - (days_age % water_interval)
    next_water_date = (today + timedelta(days=days_until_next_water)).strftime('%Y-%m-%d')
    
    schedule_list.append({
        'action': f'Irrigation Cycle ({raw_name})', 
        'date': next_water_date, 
        'status': 'Upcoming'
    })

    # B. Next Fertilizer Date
    for fert_day in config['fert_days']:
        if days_age < fert_day:
            days_until = fert_day - days_age
            fert_date = (today + timedelta(days=days_until)).strftime('%Y-%m-%d')
            schedule_list.append({
                'action': f'Apply NPK Fertilizer', 
                'date': fert_date, 
                'status': 'Critical'
            })
            break # Only show the immediate next one

    # C. Next Pesticide
    for pest_day in config['pest_days']:
        if days_age < pest_day:
            days_until = pest_day - days_age
            pest_date = (today + timedelta(days=days_until)).strftime('%Y-%m-%d')
            schedule_list.append({
                'action': 'Pesticide Spray',
                'date': pest_date,
                'status': 'Routine'
            })
            break

    # 4. Alerts
    alerts = []
    if hectares > 10:
        alerts.append(f"**Scale Alert:** Large Area ({hectares} Ha) - Check resources.")
    if days_age > 110 and raw_name in ['Wheat', 'Rice', 'Maize']:
        alerts.append(f"**Harvest Alert:** {crop_type} is ready for harvest.")
    if days_until_next_water <= 2:
        alerts.append(f"**Water Alert:** Irrigation due in {days_until_next_water} days.")
        
    if not alerts:
        alerts.append("**System Check:** All systems nominal.")

    return {
        'growth_stages': [growth_msg],
        'irrigation_schedule': schedule_list,
        'alerts': alerts
    }

def get_dynamic_crop_data():
    data = []
    for fname, details in MOCK_CROP_FIELDS.items():
        sched = get_mock_schedules({**details, 'name': fname})
        data.append({
            'name': fname,
            'yield': f"{random.randint(30,90)} Q/Ha",
            'confidence': f"{random.randint(80,99)}%",
            'schedules': sched
        })
    return data

def update_mock_crop_details(name, date, hectares):
    target = None
    for k in MOCK_CROP_FIELDS:
        if name in k: target = k
    
    if not target: target = f"{name} (New)"
    
    MOCK_CROP_FIELDS[target] = {
        'crop_type': name, 'date_planted': date, 'hectares': float(hectares or 0)
    }

def generate_soil_pdf(data):
    if not PDF_AVAILABLE: return BytesIO(b"PDF Lib Missing")
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "AgriSense AI - Soil Report")
    y = 700
    for k, v in data.items():
        p.drawString(100, y, f"{k}: {v}")
        y -= 20
    p.save()
    buffer.seek(0)
    return buffer

def get_mock_iot_data():
    global MOCK_IOT_STATE
    if MOCK_IOT_STATE['control_mode'] == 'AUTO':
        MOCK_IOT_STATE['pump_status'] = 'ON' if MOCK_IOT_STATE['soil_moisture'] < 40 else 'OFF'
    
    return {
        'moisture': f"{MOCK_IOT_STATE['soil_moisture']}%",
        'temperature': f"{MOCK_IOT_STATE['ambient_temp']}°C",
        'status': MOCK_IOT_STATE['pump_status'],
        'mode': MOCK_IOT_STATE['control_mode'],
        'log': "System Operational"
    }

def get_irrigation_alert(ph, temp, rain, hum):
    if temp > 35: return {'type': 'high', 'message': 'High Heat Alert! Check moisture.'}
    return {'type': 'low', 'message': 'Conditions nominal.'}

# ==========================================
# 3. FLASK APP ROUTES
# ==========================================

@app.route('/')
def home():
    return redirect(url_for('dashboard', page='Dashboard'))

@app.route('/dashboard/<page>', methods=['GET', 'POST'])
def dashboard(page):
    lang = request.args.get('lang', 'en')
    t = LANGUAGE_TEXT.get(lang, LANGUAGE_TEXT['en'])
    
    weather = {'temperature': 34, 'humidity': 60, 'region': 'Raichur (Live)', 'condition': 'Sunny', 'icon': '☀️', 'color': 'bg-yellow-500', 'rainfall': 0}
    iot = get_mock_iot_data()
    crops = get_dynamic_crop_data()
    
    context = {
        'current_page': page,
        'menu': MENU_ITEMS,
        'weather': weather,
        'iot_data': iot,
        'mock_crop_data': crops, 
        'text': t,
        'lang': lang,
        'irrigation_alert': get_irrigation_alert(6.5, weather['temperature'], weather['rainfall'], weather['humidity'])
    }

    if page == 'Dashboard' and request.method == 'POST':
        try:
            N = float(request.form.get('nitrogen'))
            P = float(request.form.get('phosphorus'))
            K = float(request.form.get('potassium'))
            pH = float(request.form.get('ph'))
            crop, conf = get_crop_recommendation(N, P, K, pH, weather)
            context['result'] = {
                'soil_health': "Good" if N>50 else "Deficient",
                'crop': crop,
                'fertilizer': "Urea & DAP",
                'water_req': '500mm',
                'price_forecast': 'Rising',
                'top_recommendations': conf,
                'N': N, 'P': P, 'K': K, 'pH': pH
            }
        except:
            pass

    if request.args.get('disease'): context['disease_result'] = request.args
    if request.args.get('crop_status'): context['crop_status'] = request.args.get('crop_status')
    if request.args.get('user_query'): 
        context['user_query'] = request.args.get('user_query')
        context['bot_response'] = request.args.get('bot_response')
    if request.args.get('pump_status_msg'): context['pump_status_msg'] = request.args.get('pump_status_msg')

    return render_template('index.html', **context)

# --- ROUTES ---

@app.route('/update_crop', methods=['POST'])
def update_crop():
    update_mock_crop_details(
        request.form.get('crop_name'),
        request.form.get('date_planted'),
        request.form.get('hectares')
    )
    return redirect(url_for('dashboard', page='Crop Management', crop_status="Field Updated Successfully"))

@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    return redirect(url_for('dashboard', page='Dashboard', disease="Leaf Spot", confidence="95%", remedy="Use Fungicide", severity="Moderate", pesticide_qty="2g/L", model_used="ResNet50"))

@app.route('/chatbot_query', methods=['POST'])
def chatbot_query():
    return redirect(url_for('dashboard', page='AI Recommendations', user_query=request.form.get('query'), bot_response="AI: Based on your query, I suggest rotating crops."))

@app.route('/control_pump', methods=['POST'])
def control_pump():
    global MOCK_IOT_STATE
    action = request.form.get('action')
    if action == 'MANUAL_ON': MOCK_IOT_STATE['pump_status'] = 'ON'; MOCK_IOT_STATE['control_mode'] = 'MANUAL'
    if action == 'AUTO_MODE': MOCK_IOT_STATE['control_mode'] = 'AUTO'
    return redirect(url_for('dashboard', page='IoT Sensors', pump_status_msg="Command Sent"))

@app.route('/generate_report')
def generate_report():
    pdf = generate_soil_pdf(request.args.to_dict())
    return send_file(pdf, download_name='Report.pdf', as_attachment=True, mimetype='application/pdf')

if __name__ == '__main__':
    create_dummy_models()
    app.run(debug=True)