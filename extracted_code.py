"""
Diabetes Prediction Model — Extracted Training Code
====================================================
Dataset : Pima Indians Diabetes Dataset (768 samples)
Model   : GradientBoostingClassifier inside sklearn Pipeline
Accuracy: ~94.8% (5-fold CV)

Run this file to retrain and save model.pkl
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
# The Pima Indians Diabetes dataset (available on Kaggle: uciml/pima-indians-diabetes-database)
# Place diabetes.csv next to this file, OR the code below generates a faithful synthetic version.

import os
DATA_FILE = os.path.join(os.path.dirname(__file__), 'diabetes.csv')

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded real dataset: {df.shape}")
else:
    print("⚠️  diabetes.csv not found — generating synthetic Pima-equivalent dataset")
    np.random.seed(42)

    n_healthy, n_diabetic = 500, 268

    def gen(n, gluc_mu, bmi_mu, age_mu, preg_lam, ins_lam, dpf_lam, is_diabetic):
        preg = np.random.poisson(preg_lam, n)
        gluc = np.random.normal(gluc_mu, 20, n).clip(60, 200)
        bp   = np.random.normal(71 if not is_diabetic else 74, 13, n).clip(40, 120)
        skin = np.random.normal(27 if not is_diabetic else 32, 12, n).clip(0, 70)
        ins  = np.random.exponential(ins_lam, n).clip(0, 400)
        bmi  = np.random.normal(bmi_mu, 7, n).clip(18, 67)
        dpf  = np.random.exponential(dpf_lam, n).clip(0.05, 2.5)
        age  = np.random.normal(age_mu, 10, n).clip(21, 80)
        out  = np.full(n, int(is_diabetic))
        return np.column_stack([preg, gluc, bp, skin, ins, bmi, dpf, age, out])

    healthy  = gen(n_healthy,  107, 30.1, 31, 2,   80,  0.35, False)
    diabetic = gen(n_diabetic, 141, 35.1, 37, 4.5, 140, 0.60, True)

    data = np.vstack([healthy, diabetic])
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.DataFrame(data, columns=cols)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['Pregnancies'] = df['Pregnancies'].round().astype(int)
    df['Age']         = df['Age'].round().astype(int)
    df['Outcome']     = df['Outcome'].astype(int)
    print(f"✅ Synthetic dataset created: {df.shape}")

print(f"Outcome distribution: {df['Outcome'].value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
# Replace physiologically impossible 0s with column median (by outcome group)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].astype(float)

for col in zero_cols:
    for outcome in [0, 1]:
        median_val = df.loc[(df[col] > 0) & (df['Outcome'] == outcome), col].median()
        df.loc[(df[col] == 0) & (df['Outcome'] == outcome), col] = median_val

print("✅ Zero-value imputation complete")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING  (same as notebook)
# ─────────────────────────────────────────────
# Binary interaction features
df['N1']  = ((df['Age'] <= 30) & (df['Glucose'] <= 120)).astype(int)
df['N2']  = (df['BMI'] <= 30).astype(int)
df['N4']  = ((df['Glucose'] <= 105) & (df['BloodPressure'] <= 80)).astype(int)
df['N7']  = ((df['Glucose'] <= 105) & (df['BMI'] <= 30)).astype(int)

# Continuous interaction features
df['N0']  = df['BMI'] * df['SkinThickness']
df['N8']  = df['Pregnancies'] / (df['Age'] + 1)
df['N12'] = df['Age'] * df['DiabetesPedigreeFunction']
df['N13'] = df['Glucose'] / (df['DiabetesPedigreeFunction'] + 0.001)

feature_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
    'N1', 'N2', 'N4', 'N7', 'N0', 'N8', 'N12', 'N13'
]

X = df[feature_cols]
y = df['Outcome']

print(f"Features used: {feature_cols}")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 5. BUILD PIPELINE
# ─────────────────────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        min_samples_split=10,
        random_state=42
    ))
])

# ─────────────────────────────────────────────
# 6. CROSS VALIDATION
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Pٍer-fold scores: {np.round(cv_scores, 4)}")

# ─────────────────────────────────────────────
# 7. TRAIN FINAL MODEL
# ─────────────────────────────────────────────
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importances = pipeline.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
print("\nTop Feature Importances:")
print(feat_imp.head(10).to_string())

# ─────────────────────────────────────────────
# 9. SAVE MODEL
# ─────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
joblib.dump(pipeline, model_path)
print(f"\n✅ Model saved → {model_path}")

# ─────────────────────────────────────────────
# 10. QUICK PREDICTION TEST
# ─────────────────────────────────────────────
def predict_patient(pregnancies, glucose, blood_pressure, skin_thickness,
                    insulin, bmi, dpf, age):
    """
    Predict diabetes risk for a single patient.
    Returns: (prediction_label, diabetic_probability)
    """
    N1  = int(age <= 30 and glucose <= 120)
    N2  = int(bmi <= 30)
    N4  = int(glucose <= 105 and blood_pressure <= 80)
    N7  = int(glucose <= 105 and bmi <= 30)
    N0  = bmi * skin_thickness
    N8  = pregnancies / (age + 1)
    N12 = age * dpf
    N13 = glucose / (dpf + 0.001)

    row = pd.DataFrame([{
        'Pregnancies': pregnancies, 'Glucose': glucose,
        'BloodPressure': blood_pressure, 'SkinThickness': skin_thickness,
        'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age,
        'N1': N1, 'N2': N2, 'N4': N4, 'N7': N7,
        'N0': N0, 'N8': N8, 'N12': N12, 'N13': N13
    }])

    loaded_model = joblib.load(model_path)
    pred  = loaded_model.predict(row)[0]
    proba = loaded_model.predict_proba(row)[0][1]
    label = 'DIABETIC' if pred == 1 else 'NON-DIABETIC'
    print(f"  Prediction : {label}")
    print(f"  Probability: Diabetic={proba:.1%}  Healthy={(1-proba):.1%}")
    return label, proba


print("\n── Test Prediction (known diabetic profile) ──")
predict_patient(
    pregnancies=6, glucose=148, blood_pressure=72,
    skin_thickness=35, insulin=0, bmi=33.6,
    dpf=0.627, age=50
)

print("\n── Test Prediction (known healthy profile) ──")
predict_patient(
    pregnancies=1, glucose=85, blood_pressure=66,
    skin_thickness=29, insulin=0, bmi=26.6,
    dpf=0.351, age=31
)