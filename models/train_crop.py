import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

np.random.seed(42)

def generate_crop(name, N, P, K, temp, humidity, ph, rainfall, n=200):
    return pd.DataFrame({
        'N':           np.random.randint(N[0],        N[1]+1,        n),
        'P':           np.random.randint(P[0],        P[1]+1,        n),
        'K':           np.random.randint(K[0],        K[1]+1,        n),
        'temperature': np.round(np.random.uniform(temp[0],     temp[1],     n), 2),
        'humidity':    np.round(np.random.uniform(humidity[0], humidity[1], n), 2),
        'ph':          np.round(np.random.uniform(ph[0],       ph[1],       n), 2),
        'rainfall':    np.round(np.random.uniform(rainfall[0], rainfall[1], n), 2),
        'label':       name
    })

# ── Nigerian/West African crops only ──────────────────────
# Sources: FAO Crop Production Guidelines, IITA Nigeria,
# NAERLS Advisory Bulletins, Adamawa State ADP Reports
df = pd.concat([

    generate_crop('maize',      N=(60,120), P=(30,60),  K=(30,60),
                  temp=(20,30), humidity=(55,75), ph=(5.5,7.0),
                  rainfall=(60,200)),

    generate_crop('rice',       N=(80,100), P=(40,60),  K=(40,50),
                  temp=(20,27), humidity=(80,90), ph=(6.0,7.0),
                  rainfall=(200,300)),

    generate_crop('cassava',    N=(20,40),  P=(20,40),  K=(40,80),
                  temp=(25,35), humidity=(60,80), ph=(4.5,6.5),
                  rainfall=(83,150)),

    generate_crop('yam',        N=(25,50),  P=(40,80),  K=(150,250),
                  temp=(25,35), humidity=(70,90), ph=(5.0,6.5),
                  rainfall=(83,150)),

    generate_crop('sorghum',    N=(20,40),  P=(10,20),  K=(10,25),
                  temp=(28,35), humidity=(40,60), ph=(6.0,7.5),
                  rainfall=(33,67)),

    generate_crop('millet',     N=(15,30),  P=(8,15),   K=(8,20),
                  temp=(28,38), humidity=(35,55), ph=(5.5,7.0),
                  rainfall=(21,50)),

    generate_crop('cowpea',     N=(20,40),  P=(50,80),  K=(15,30),
                  temp=(25,35), humidity=(70,90), ph=(5.5,7.0),
                  rainfall=(50,100)),

    generate_crop('groundnut',  N=(15,30),  P=(50,80),  K=(20,40),
                  temp=(25,35), humidity=(50,75), ph=(5.5,7.0),
                  rainfall=(50,83)),

    generate_crop('soybean',    N=(15,25),  P=(60,90),  K=(30,50),
                  temp=(22,32), humidity=(55,75), ph=(6.0,7.0),
                  rainfall=(50,100)),

    generate_crop('plantain',   N=(80,120), P=(40,60),  K=(200,300),
                  temp=(24,30), humidity=(75,95), ph=(5.5,7.0),
                  rainfall=(150,208)),

], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/nigerian_crop_data.csv', index=False)

print(f"Dataset: {df.shape}")
print(f"Crops ({df['label'].nunique()}): {sorted(df['label'].unique())}")
print(f"\nSamples per crop:")
print(df['label'].value_counts().sort_index())

# ── Train ──────────────────────────────────────────────────
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Nigerian crop recommendation model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Feature importance ─────────────────────────────────────
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
importances = model.feature_importances_
print("\nFeature Importances:")
for f, i in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"  {f}: {i:.4f}")

# ── Save ───────────────────────────────────────────────────
os.makedirs('saved_models', exist_ok=True)
joblib.dump(model, 'saved_models/crop_model.pkl')
joblib.dump(sorted(df['label'].unique().tolist()),
            'saved_models/crop_labels.pkl')
print("\n✅ Nigerian crop model saved!")