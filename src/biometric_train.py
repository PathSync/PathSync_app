import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_loader import load_biometric_data, preprocess_biometric_data

def train_biometric_model(data_path):
    """Train biometric verification model"""
    # Load and preprocess data

    df = load_biometric_data(data_path)
    df, province_map, citizenship_map = preprocess_biometric_data(df)

    # Features and target
    features = ['age', 'gender', 'province_encoded', 'biometric_score']
    X = df[features]
    y = df['citizenship_encoded']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Biometric Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred,
                                target_names=['SA Citizen', 'Non-Citizen', 'Review']))

    # Save model and preprocessing info
    joblib.dump(model, '../models/biometric_model.pkl')
    preprocessing_info = {
        'province_map': province_map,
        'citizenship_map': citizenship_map
    }
    joblib.dump(preprocessing_info, '../models/biometric_preprocessing.pkl')
    return model, preprocessing_info

if __name__ == "__main__":
    train_biometric_model('../data/sample_Biometric.csv')