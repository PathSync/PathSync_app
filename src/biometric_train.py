# src/biometric_train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import LabelEncoder


class BiometricModelTrainer:
    """Train biometric verification model"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.preprocessor = {}

    def preprocess_data(self, df):
        """Preprocess biometric data"""
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_citizenship = LabelEncoder()

        df_processed = df.copy()
        df_processed['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df_processed['citizenship_encoded'] = le_citizenship.fit_transform(df['citizenship_status'])

        # Store preprocessors
        self.preprocessor['gender_encoder'] = le_gender
        self.preprocessor['citizenship_encoder'] = le_citizenship

        return df_processed

    def train(self, df):
        """Train the biometric verification model"""
        # Preprocess data
        df_processed = self.preprocess_data(df)

        # Features and target
        features = ['age', 'gender_encoded', 'facial_match_score',
                    'fingerprint_match_score', 'has_medical_aid', 'citizenship_encoded']
        X = df_processed[features]
        y = df_processed['id_verified']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Biometric Model Performance:")
        print(classification_report(y_test, y_pred))

        return self.model

    def save_model(self, filepath):
        """Save model and preprocessors"""
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor
        }, filepath)