import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv('data/yield_df.csv')
df = df.drop(columns=['Unnamed: 0'])

# ── Convert yield: hg/ha → tons/ha ────────────────────────
df['yield_ton_per_ha'] = df['hg/ha_yield'] / 10000
df = df.drop(columns=['hg/ha_yield'])

# ── Filter African countries for regional relevance ────────
african_countries = [
    'Nigeria', 'Ghana', 'Niger', 'Mali', 'Senegal', 'Guinea',
    'Cameroon', 'Togo', 'Benin', 'Burkina Faso', 'Côte d\'Ivoire',
    'Sierra Leone', 'Liberia', 'Gambia', 'Chad', 'Sudan',
    'Ethiopia', 'Kenya', 'Uganda', 'Tanzania', 'Rwanda',
    'South Africa', 'Zambia', 'Zimbabwe', 'Mozambique', 'Angola',
    'Congo', 'Egypt', 'Morocco', 'Algeria', 'Tunisia'
]
df_africa = df[df['Area'].isin(african_countries)]

# Fall back to full dataset if African subset is too small
if len(df_africa) < 500:
    print("African subset too small, using full dataset")
    df_filtered = df.copy()
else:
    print(f"Using African subset: {len(df_africa)} records")
    df_filtered = df_africa.copy()

print(f"Countries in dataset: {df_filtered['Area'].nunique()}")
print(f"Crops in dataset: {df_filtered['Item'].nunique()}")
print(f"Crops: {sorted(df_filtered['Item'].unique())}")

# ── Drop rows with missing values ──────────────────────────
df_filtered = df_filtered.dropna()
print(f"Clean records: {len(df_filtered)}")

# ── Encode categorical columns ─────────────────────────────
le_area = LabelEncoder()
le_item = LabelEncoder()

df_filtered['Area_enc'] = le_area.fit_transform(df_filtered['Area'])
df_filtered['Item_enc'] = le_item.fit_transform(df_filtered['Item'])

# ── Features & target ──────────────────────────────────────
features = ['Area_enc', 'Item_enc', 'Year',
            'average_rain_fall_mm_per_year',
            'pesticides_tonnes', 'avg_temp']

X = df_filtered[features]
y = df_filtered['yield_ton_per_ha']

# ── Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train Random Forest Regressor ──────────────────────────
print("\nTraining yield prediction model...")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print(f"\nMAE  : {mae:.4f} tons/ha")
print(f"RMSE : {rmse:.4f} tons/ha")
print(f"R²   : {r2:.4f}")

# ── Save model and encoders ────────────────────────────────
os.makedirs('saved_models', exist_ok=True)
joblib.dump(model,   'saved_models/yield_model.pkl')
joblib.dump(le_area, 'saved_models/yield_le_area.pkl')
joblib.dump(le_item, 'saved_models/yield_le_item.pkl')

# Save crop and country lists for Flask dropdowns
joblib.dump(sorted(df_filtered['Item'].unique().tolist()),
            'saved_models/yield_crops.pkl')
joblib.dump(sorted(df_filtered['Area'].unique().tolist()),
            'saved_models/yield_areas.pkl')

print("\n✅ Yield model saved to saved_models/yield_model.pkl")
print(f"✅ Encoders and label lists saved")