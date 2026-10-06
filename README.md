# AgroSense — Smart Agriculture System

A machine learning-based web application for crop disease detection, crop recommendation, and yield prediction, calibrated to Nigerian and West African agricultural conditions.

Built as an undergraduate thesis project at the **Department of Computer Science, Modibbo Adama University (MAU), Yola, Adamawa State, Nigeria**.

---

## Features

- **Crop Disease Detection** — Upload a leaf photo and the system identifies the disease across 23 classes covering Cashew, Cassava, Maize, and Tomato crops, trained on real West African field images
- **Crop Recommendation** — Enter soil (N, P, K, pH) and climate (temperature, humidity, rainfall) data to get the most suitable crop recommendation from 10 Nigerian staple crops
- **Yield Prediction** — Predict expected crop yield in tons/hectare based on environmental and soil parameters
- **Prediction Logging** — Every query is logged to a SQLite database, viewable at `/logs`
- **Uncertainty Detection** — Low-confidence disease predictions trigger a warning rather than a misleading result

---

## Model Performance

| Module | Algorithm | Performance |
|---|---|---|
| Disease Detection | EfficientNetB0 (Transfer Learning) | 73.18% validation accuracy |
| Crop Recommendation | Random Forest (300 estimators) | 97.25% accuracy |
| Yield Prediction | Random Forest Regressor | R² = 0.9738, MAE = 0.5257 t/ha |

---

## Tech Stack

- **Backend:** Python 3.x, Flask 2.3
- **ML/DL Training:** TensorFlow 2.x, Keras, scikit-learn
- **Inference:** ONNX Runtime (no GPU or TensorFlow required to run)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Database:** SQLite3

---

## Installation

### Prerequisites

- Python 3.8 or higher (tested on 3.12 and 3.14)
- pip
- A terminal / command prompt

### Steps

**1. Clone or unzip the project folder**

```bash
cd smart_agri
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Check the model files**

The trained model files are included in `saved_models/`. Keep the directory and filenames unchanged; the app loads these files when it starts.

**5. Run the application**

```bash
python app.py
```

Open your browser and go to: `http://127.0.0.1:5000`

---

## Usage

### Disease Detection
1. Click **Disease Detection** in the navigation bar
2. Upload a clear, well-lit photograph of a crop leaf (JPG or PNG, max 16MB)
3. Click **Analyze Image**
4. The system returns the predicted disease, confidence score, cause, symptoms, treatment, and prevention advice

> **Tip:** Use close-up photos in natural lighting. Blurry or distant images may trigger the uncertainty warning.

### Crop Recommendation
1. Click **Crop Recommendation**
2. Enter your soil nutrient levels (N, P, K), soil pH, temperature, humidity, and monthly rainfall
3. Click **Get Recommendation**
4. The system returns the top 3 recommended crops with confidence scores

### Yield Prediction
1. Click **Yield Prediction**
2. Select your country/region and crop type
3. Enter year, temperature, annual rainfall, pesticide usage, farm size, soil quality, and soil type
4. Click **Predict Yield**
5. The system returns predicted yield in tons/hectare with a qualitative interpretation

The Random Forest provides a baseline yield per hectare from the FAO features. Farm size is used to calculate projected total production; it does not apply a blanket farm-size penalty to yield per hectare. Soil-quality and soil-type factors are provisional scenario multipliers, not coefficients learned from the FAO data or calibrated against local field observations.

This distinction matters because farm-size/productivity relationships are not universally monotonic. Omotilewa et al. (2021) report a U-shaped relationship across Nigerian farm sizes, which does not justify the previous rule that yield per hectare always falls as farm size increases: [A revisit of farm size and productivity](https://doi.org/10.1016/j.worlddev.2021.105592). A Nigerian field study of soil texture and crop response likewise reports crop- and location-specific results rather than general soil-class multipliers: [Stephen and Fagbola (2022)](https://doi.org/10.62773/jcocs.v3i1.149).

### Prediction Logs
Navigate to `http://127.0.0.1:5000/logs` to view the last 100 predictions across all three modules.

---

## Project Structure

```
smart_agri/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── agrosense.db              # SQLite prediction log (auto-created on first run)
├── data/                     # Training datasets (CSV files)
│   ├── Crop_recommendation.csv
│   ├── nigerian_crop_data.csv
│   └── yield_df.csv
├── models/                   # Model training scripts
│   ├── train_crop.py
│   └── train_yield.py
├── saved_models/             # Trained model artefacts
│   ├── disease_model.onnx
│   ├── class_labels.json
│   ├── crop_model.pkl
│   └── ...
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/              # Uploaded leaf images (auto-cleared)
└── templates/
    ├── base.html
    ├── index.html
    ├── disease.html
    ├── recommend.html
    ├── yield.html
    └── logs.html
```

---

## Datasets

| Dataset | Source | Use |
|---|---|---|
| CCMT Crop Pest and Disease Detection | Mensah et al. (2023), *Data in Brief* | Disease detection training |
| Tanzania Maize Disease Dataset | Tanzania Agricultural Research Institute (2023) | Disease detection training |
| FAO Crop Yield Dataset | Patel (2020), Kaggle | Yield prediction training |
| Nigerian Crop Agronomic Parameters | Generated from FAO/IITA guidelines | Crop recommendation training |

### Citations

Mensah, G. A., Asare-Bediako, E., & Acheampong, A. (2023). CCMT: Dataset for crop pest and disease detection. *Data in Brief*, 49, 109306. https://doi.org/10.1016/j.dib.2023.109306

---

## Supported Disease Classes (23)

| Crop | Classes |
|---|---|
| Cashew | Anthracnose, Gumosis, Healthy, Leaf Miner, Red Rust |
| Cassava | Bacterial Blight, Brown Spot, Green Mite, Healthy, Mosaic |
| Maize | Fall Armyworm, Grasshopper, Healthy, Leaf Beetle, Leaf Blight, Leaf Spot, Lethal Necrosis, Streak Virus |
| Tomato | Healthy, Leaf Blight, Leaf Curl, Septoria Leaf Spot, Verticillium Wilt |

---

## Supported Crops (Recommendation)

Maize, Rice, Cassava, Yam, Sorghum, Millet, Cowpea, Groundnut, Soybean, Plantain

---

## Known Limitations

- Disease detection accuracy (73.18%) reflects real-world field-condition images. Laboratory-condition images (plain background, controlled lighting) may yield different results
- Fall armyworm detection performs best when larval presence or frass residue is visible in the image
- Yield prediction model is trained on FAO data up to 2013; predictions beyond this year are extrapolations
- The system requires internet connectivity; no offline mode is currently supported

---

## License

This project was developed for academic purposes at Modibbo Adama University (MAU), Yola. All rights reserved by the author.