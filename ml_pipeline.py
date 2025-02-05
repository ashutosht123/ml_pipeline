import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def load_data():
    """Fetches dataset (Can be replaced with API call or DB fetch)"""
    X, y = make_classification(n_samples=1000, n_features=20)
    return X, y

def train_pipeline():
    """Train ML model and save it"""
    print("📌 Loading dataset...")
    X, y = load_data()

    print("📌 Splitting dataset...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📌 Creating pipeline...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        ("classifier", LogisticRegression())
    ])

    print("📌 Training model...")
    pipeline.fit(X_train, y_train)

    print("📌 Saving model...")
    joblib.dump(pipeline, "model.pkl")

    print("✅ Model training complete & saved as 'model.pkl'")

if __name__ == "__main__":
    train_pipeline()
