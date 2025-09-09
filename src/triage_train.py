import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_loader import load_triage_data, preprocess_triage_data


def train_triage_model(data_path):
    """Train triage priority model"""

    # Load and preprocess data
    df = load_triage_data(data_path)
    df, priority_map = preprocess_triage_data(df)

    # Features and target
    features = ['age', 'gender', 'hr_bpm', 'temp_c', 'resp_rate',
                'systolic_bp', 'diastolic_bp', 'o2_sat', 'pain_score']
    X = df[features]
    y = df['priority_encoded']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Triage Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test,
                                y_pred,labels=[0,1,2],
                                target_names=['Red', 'Yellow', 'Green']))

    # Save model and preprocessing info
    joblib.dump(model, '../models/triage_model.pkl')
    joblib.dump(priority_map, '../models/triage_preprocessing.pkl')

    return model, priority_map


if __name__ == "__main__":
    train_triage_model('../data/sample_triage.csv')