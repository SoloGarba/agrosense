import joblib
import json
import onnxruntime as ort
import numpy as np

print('Loading crop model...')
crop_model = joblib.load('saved_models/crop_model.pkl')
crop_labels = joblib.load('saved_models/crop_labels.pkl')
print(f'✅ Crop labels: {crop_labels}')

print('\nLoading yield model...')
yield_model = joblib.load('saved_models/yield_model.pkl')
yield_crops = joblib.load('saved_models/yield_crops.pkl')
print(f'✅ Yield crops: {yield_crops}')

print('\nLoading disease model (ONNX)...')
session = ort.InferenceSession('saved_models/disease_model.onnx')
with open('saved_models/class_labels.json') as f:
    labels = json.load(f)
print(f'✅ Disease classes: {list(labels.values())}')

print('\n✅ All models loaded successfully!')