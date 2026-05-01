"""
ISL Model Training Script
==========================
1. Generates synthetic dataset (if not already present)
2. Trains a scikit-learn MLPClassifier on the 63-dim landmark features
3. Saves model, label encoder, and scaler to model/

Run:  python model/train_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV  = os.path.join(ROOT, "data", "isl_keypoints.csv")
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "isl_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")


def ensure_dataset():
    if not os.path.exists(DATA_CSV):
        print("Dataset not found - generating ...")
        sys.path.insert(0, ROOT)
        from data.generate_dataset import generate_dataset
        df = generate_dataset(samples_per_class=800)
        df.to_csv(DATA_CSV, index=False)
        print(f"Dataset saved: {DATA_CSV}")
    else:
        print(f"Dataset found: {DATA_CSV}")


def train():
    ensure_dataset()

    df = pd.read_csv(DATA_CSV)
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    print(f"\nTraining MLP on {len(X_train)} samples …")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[OK] Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artefacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,   MODEL_PATH)
    joblib.dump(le,      ENCODER_PATH)
    joblib.dump(scaler,  SCALER_PATH)
    print(f"\nModel saved   -> {MODEL_PATH}")
    print(f"Encoder saved -> {ENCODER_PATH}")
    print(f"Scaler saved  -> {SCALER_PATH}")

    return acc


if __name__ == "__main__":
    train()
