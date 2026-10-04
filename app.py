import pandas as pd
import os
import json
import numpy as np
import joblib
import sqlite3
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ── Load ML models ─────────────────────────────────────────
crop_model    = joblib.load('saved_models/crop_model.pkl')
crop_labels   = joblib.load('saved_models/crop_labels.pkl')
yield_model   = joblib.load('saved_models/yield_model.pkl')
yield_le_area = joblib.load('saved_models/yield_le_area.pkl')
yield_le_item = joblib.load('saved_models/yield_le_item.pkl')
yield_crops   = joblib.load('saved_models/yield_crops.pkl')
yield_areas   = joblib.load('saved_models/yield_areas.pkl')

# ── Load disease model lazily (loaded on first use) ────────
disease_session  = None
class_labels   = None

def load_disease_model():
    global disease_session, class_labels
    if disease_session is None:
        import onnxruntime as ort
        disease_session = ort.InferenceSession('saved_models/disease_model.onnx')
        with open('saved_models/class_labels.json') as f:
            class_labels = json.load(f)
    return disease_session, class_labels

# ── Disease treatment recommendations ──────────────────────
DISEASE_INFO = {
    'Cashew anthracnose': {
        'cause': 'Fungal infection (Colletotrichum gloeosporioides)',
        'symptoms': 'Dark brown/black lesions on leaves, fruits, and twigs',
        'treatment': 'Apply copper-based fungicides. Remove and burn infected parts. Improve air circulation.',
        'prevention': 'Use disease-resistant varieties. Avoid overhead irrigation. Apply preventive fungicide at flowering.'
    },
    'Cashew gumosis': {
        'cause': 'Fungal/bacterial infection or physical damage',
        'symptoms': 'Gum exudate from bark, cankers on stem',
        'treatment': 'Scrape infected bark, apply Bordeaux paste. Use systemic fungicide.',
        'prevention': 'Avoid stem injuries. Ensure proper drainage. Prune infected branches.'
    },
    'Cashew healthy': {
        'cause': 'N/A',
        'symptoms': 'No disease detected',
        'treatment': 'Your crop appears healthy. Maintain current practices.',
        'prevention': 'Continue regular monitoring, proper fertilisation, and irrigation management.'
    },
    'Cashew leaf miner': {
        'cause': 'Insect pest (Acrocercops syngramma)',
        'symptoms': 'Serpentine mines/tunnels visible on leaves, leaf curling',
        'treatment': 'Apply systemic insecticide (imidacloprid or lambda-cyhalothrin). Remove heavily infested leaves.',
        'prevention': 'Regular field monitoring. Use neem-based biopesticides as preventive measure.'
    },
    'Cashew red rust': {
        'cause': 'Algal infection (Cephaleuros virescens)',
        'symptoms': 'Orange-red powdery spots on upper leaf surface',
        'treatment': 'Apply copper oxychloride fungicide. Remove infected leaves.',
        'prevention': 'Improve canopy ventilation. Avoid excessive nitrogen fertilisation.'
    },
    'Cassava bacterial blight': {
        'cause': 'Bacterial infection (Xanthomonas axonopodis)',
        'symptoms': 'Angular leaf spots, wilting, stem dieback, gummosis',
        'treatment': 'Remove and destroy infected plants. Apply copper-based bactericide. Use clean planting material.',
        'prevention': 'Plant certified disease-free cuttings. Practice crop rotation. Avoid working in fields when wet.'
    },
    'Cassava brown spot': {
        'cause': 'Fungal infection (Cercosporidium henningsii)',
        'symptoms': 'Brown angular spots with yellow halo on leaves',
        'treatment': 'Apply mancozeb or chlorothalonil fungicide. Remove severely infected leaves.',
        'prevention': 'Use resistant varieties. Ensure adequate potassium nutrition. Avoid dense planting.'
    },
    'Cassava green mite': {
        'cause': 'Mite infestation (Mononychellus tanajoa)',
        'symptoms': 'Leaf distortion, chlorotic spots, stunted growth',
        'treatment': 'Apply acaricide (abamectin or sulphur-based). Introduce predatory mites as biocontrol.',
        'prevention': 'Plant early in the season. Use tolerant varieties. Avoid water stress.'
    },
    'Cassava healthy': {
        'cause': 'N/A',
        'symptoms': 'No disease detected',
        'treatment': 'Your crop appears healthy. Maintain current practices.',
        'prevention': 'Continue regular monitoring, weeding, and appropriate fertilisation.'
    },
    'Cassava mosaic': {
        'cause': 'Viral infection (Cassava Mosaic Virus) transmitted by whiteflies',
        'symptoms': 'Mosaic yellowing/greening of leaves, distorted leaflets, stunted growth',
        'treatment': 'No cure for infected plants. Remove and destroy infected plants immediately. Control whitefly vectors with imidacloprid.',
        'prevention': 'Plant CMD-resistant varieties (e.g. TMS 98/0505). Use virus-free planting material. Control whitefly populations.'
    },
    'Maize fall armyworm': {
        'cause': 'Insect pest (Spodoptera frugiperda)',
        'symptoms': 'Ragged holes in leaves, sawdust-like frass in whorl, damaged tassels',
        'treatment': 'Apply insecticide (emamectin benzoate or chlorpyrifos) directly into whorl. Use Bt-based biopesticide.',
        'prevention': 'Early planting. Regular scouting. Use pheromone traps. Encourage natural enemies (parasitic wasps).'
    },
    'Maize grasshoper': {
        'cause': 'Insect pest (various Acrididae species)',
        'symptoms': 'Irregular leaf margins, skeletonised leaves, defoliation',
        'treatment': 'Apply malathion or lambda-cyhalothrin during early morning. Spot-treat affected areas.',
        'prevention': 'Early planting to avoid peak grasshopper season. Encourage bird predators.'
    },
    'Maize healthy': {
        'cause': 'N/A',
        'symptoms': 'No disease detected',
        'treatment': 'Your crop appears healthy. Maintain current practices.',
        'prevention': 'Continue regular fertilisation, irrigation, and pest monitoring.'
    },
    'Maize leaf beetle': {
        'cause': 'Insect pest (Diabrotica species)',
        'symptoms': 'Skeletonised leaves with translucent streaks between veins',
        'treatment': 'Apply carbaryl or lambda-cyhalothrin. For severe infestation use systemic insecticide.',
        'prevention': 'Crop rotation to break beetle life cycle. Avoid late planting.'
    },
    'Maize leaf blight': {
        'cause': 'Fungal infection (Helminthosporium turcicum)',
        'symptoms': 'Long elliptical grey-green to tan lesions on leaves',
        'treatment': 'Apply mancozeb or propiconazole fungicide. Remove infected leaves.',
        'prevention': 'Use resistant varieties. Ensure good drainage. Practice crop rotation.'
    },
    'Maize leaf spot': {
        'cause': 'Fungal infection (Cercospora zeae-maydis)',
        'symptoms': 'Small rectangular grey lesions with tan borders running parallel to leaf veins',
        'treatment': 'Apply strobilurin or triazole fungicide. Remove infected lower leaves.',
        'prevention': 'Use tolerant hybrids. Ensure adequate spacing for air circulation. Rotate with non-host crops.'
    },
    'Maize lethal necrosis': {
    'cause': 'Co-infection of Maize Chlorotic Mottle Virus (MCMV) and a potyvirus, transmitted by thrips, beetles, and aphids',
    'symptoms': 'Yellowing starting from leaf edges, necrosis spreading to whole plant, premature death, poor cob formation',
    'treatment': 'No cure once infected. Remove and destroy infected plants immediately. Apply systemic insecticide (imidacloprid) to control insect vectors.',
    'prevention': 'Use MLN-tolerant varieties. Control vector insects. Avoid planting near infected fields. Practice crop rotation.'
},
    'Maize streak virus': {
        'cause': 'Viral infection (Maize Streak Virus) transmitted by leafhoppers',
        'symptoms': 'Pale yellow streaks running length of leaves, stunted plants',
        'treatment': 'No cure. Remove and destroy infected plants. Control leafhopper vectors with systemic insecticide.',
        'prevention': 'Plant MSV-resistant varieties. Early planting. Control leafhopper populations with imidacloprid seed treatment.'
    },
    'Tomato healthy': {
        'cause': 'N/A',
        'symptoms': 'No disease detected',
        'treatment': 'Your crop appears healthy. Maintain current practices.',
        'prevention': 'Continue staking, pruning, regular fertilisation and disease monitoring.'
    },
    'Tomato leaf blight': {
        'cause': 'Fungal infection (Alternaria solani / Phytophthora infestans)',
        'symptoms': 'Brown/black lesions with concentric rings, yellowing, rapid defoliation',
        'treatment': 'Apply mancozeb or chlorothalonil fungicide every 7-10 days. Remove infected leaves immediately.',
        'prevention': 'Avoid overhead irrigation. Mulch to prevent soil splash. Use disease-free transplants.'
    },
    'Tomato leaf curl': {
        'cause': 'Viral infection (Tomato Yellow Leaf Curl Virus) transmitted by whiteflies',
        'symptoms': 'Upward leaf curling, yellowing of leaf margins, stunted growth',
        'treatment': 'No cure. Remove infected plants. Control whitefly with imidacloprid or thiamethoxam.',
        'prevention': 'Use TYLCV-resistant varieties. Install insect-proof nets in nursery. Apply reflective mulches to deter whiteflies.'
    },
    'Tomato septoria leaf spot': {
        'cause': 'Fungal infection (Septoria lycopersici)',
        'symptoms': 'Small circular spots with dark borders and light centres on lower leaves',
        'treatment': 'Apply copper-based or mancozeb fungicide. Remove and destroy infected lower leaves.',
        'prevention': 'Avoid wetting foliage. Practice crop rotation. Remove plant debris after harvest.'
    },
    'Tomato verticulium wilt': {
        'cause': 'Fungal infection (Verticillium dahliae) — soil-borne',
        'symptoms': 'V-shaped yellow lesions on leaf margins, wilting, brown vascular tissue',
        'treatment': 'No effective chemical cure. Remove infected plants. Solarise soil before next planting.',
        'prevention': 'Use resistant varieties. Practice 3-year crop rotation. Avoid over-irrigation.'
    }
}
# ── Database setup ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('agrosense.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            module    TEXT NOT NULL,
            input_data TEXT NOT NULL,
            result    TEXT NOT NULL,
            confidence REAL,
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(module, input_data, result, confidence=None):
    try:
        conn = sqlite3.connect('agrosense.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_log 
            (module, input_data, result, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (module, json.dumps(input_data), result, confidence,
              datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Logging error: {e}")

# ── Helper functions ───────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    return np.expand_dims(img_array, axis=0)
    
# ── Routes ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/yield')
def yield_page():
    return render_template('yield.html',
                           crops=yield_crops,
                           areas=yield_areas)

# ── Disease detection ──────────────────────────────────────
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file. Use JPG or PNG.'})

    try:
        image = Image.open(file.stream)
        image.verify()
        if image.format.lower() not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Invalid image format. Use JPG or PNG.'})
        file.stream.seek(0)
    except (OSError, Image.UnidentifiedImageError):
        return jsonify({'error': 'The uploaded file is not a valid image.'})

    extension = file.filename.rsplit('.', 1)[1].lower()
    filename = f'{uuid.uuid4().hex}.{extension}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        session, labels = load_disease_model()
        img_array = preprocess_image(filepath).astype(np.float32)

        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: img_array})[0]

        probs = predictions[0]
        top3_idx = np.argsort(probs)[-3:][::-1]

        class_idx    = str(top3_idx[0])
        confidence   = float(probs[top3_idx[0]]) * 100
        disease_name = labels[class_idx]

        top3 = [{'disease': labels[str(idx)],
                 'confidence': round(float(probs[idx]) * 100, 2)}
                for idx in top3_idx]

        healthy_classes = {
            'Maize healthy', 'Cassava healthy',
            'Tomato healthy', 'Cashew healthy'
        }
        top3_names         = {labels[str(idx)] for idx in top3_idx}
        is_healthy         = disease_name in healthy_classes
        any_healthy_top3   = bool(top3_names & healthy_classes)
        is_uncertain       = confidence < 60 and not any_healthy_top3
        low_confidence     = confidence < 70

        info = DISEASE_INFO.get(disease_name, {})

        log_prediction(
            module='disease',
            input_data={'filename': filename},
            result=disease_name,
            confidence=round(confidence, 2)
        )

        return jsonify({
            'disease':        disease_name,
            'confidence':     round(confidence, 2),
            'is_healthy':     is_healthy,
            'low_confidence': low_confidence,
            'is_uncertain':   is_uncertain,
            'top3':           top3,
            'cause':          info.get('cause', 'N/A'),
            'symptoms':       info.get('symptoms', 'N/A'),
            'treatment':      info.get('treatment', 'Consult an agricultural extension officer.'),
            'prevention':     info.get('prevention', 'Practice good crop husbandry.'),
            'image_path':     f'static/uploads/{filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ── Crop recommendation ────────────────────────────────────
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()
        features = pd.DataFrame([[
    float(data['N']),
    float(data['P']),
    float(data['K']),
    float(data['temperature']),
    float(data['humidity']),
    float(data['ph']),
    float(data['rainfall'])
]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        prediction = crop_model.predict(features)[0]
        probabilities = crop_model.predict_proba(features)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)

        # Top 3 crops
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3 = [{'crop': crop_labels[i],
                 'confidence': round(float(probabilities[i]) * 100, 2)}
                for i in top3_idx]

        log_prediction(
            module='recommendation',
            input_data={
                'N': data['N'], 'P': data['P'], 'K': data['K'],
                'temperature': data['temperature'],
                'humidity': data['humidity'],
                'ph': data['ph'],
                'rainfall': data['rainfall']
            },
            result=prediction,
            confidence=confidence
        )
        return jsonify({
            'recommended_crop': prediction,
            'confidence': confidence,
            'top3': top3
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ── Yield prediction ───────────────────────────────────────
@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        area = data['area']
        item = data['item']
        year = float(data['year'])
        rainfall = float(data['rainfall'])
        pesticides = float(data['pesticides'])
        temp = float(data['temperature'])

        # Handle unseen labels gracefully
        if area not in yield_le_area.classes_:
            area = 'Niger'
        if item not in yield_le_item.classes_:
            return jsonify({'error': f'Crop "{item}" not in training data.'})

        area_enc = yield_le_area.transform([area])[0]
        item_enc = yield_le_item.transform([item])[0]

        features = pd.DataFrame([[area_enc, item_enc, year,
                          rainfall, pesticides, temp]],
                        columns=['Area_enc', 'Item_enc', 'Year',
                                 'average_rain_fall_mm_per_year',
                                 'pesticides_tonnes', 'avg_temp'])
        prediction = yield_model.predict(features)[0]
        prediction = round(float(prediction), 2)

        log_prediction(
            module='yield',
            input_data={
                'area': area, 'item': item, 'year': year,
                'rainfall': rainfall, 'pesticides': pesticides,
                'temperature': temp
            },
            result=str(prediction),
            confidence=None
        )
        return jsonify({
            'yield_ton_per_ha': prediction,
            'crop': item,
            'area': area,
            'interpretation': interpret_yield(item, prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def interpret_yield(crop, value):
    benchmarks = {
        'Maize': 3.0, 'Rice, paddy': 3.5, 'Cassava': 12.0,
        'Yams': 12.0, 'Sorghum': 1.5, 'Millet': 1.2,
        'Tomatoes': 15.0, 'Groundnuts, with shell': 1.2
    }
    bench = benchmarks.get(crop, 2.0)
    if value >= bench * 1.2:
        return 'Excellent — significantly above average yield for this crop.'
    elif value >= bench * 0.9:
        return 'Good — around average expected yield.'
    elif value >= bench * 0.6:
        return 'Below average — consider improving soil nutrition or irrigation.'
    else:
        return 'Poor — significant intervention recommended. Review soil health and inputs.'

    # ── Admin prediction log viewer ────────────────────────────
@app.route('/logs')
def logs():
    conn = sqlite3.connect('agrosense.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, module, input_data, result, confidence, timestamp
        FROM prediction_log
        ORDER BY id DESC
        LIMIT 100
    ''')
    records = cursor.fetchall()
    conn.close()
    return render_template('logs.html', records=records)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    init_db()
    app.run(debug=True)