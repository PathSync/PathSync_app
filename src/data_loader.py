# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class HealthcareDataLoader:
    """Simplified data loader for model development"""

    @staticmethod
    def create_sample_biometric_data():
        """Create mock biometric data for development"""
        np.random.seed(42)
        n_samples = 1000

        data = {
            'patient_id': range(n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'id_verified': np.random.choice([True, False], n_samples, p=[0.8, 0.2]),
            'facial_match_score': np.random.uniform(0.7, 1.0, n_samples),
            'fingerprint_match_score': np.random.uniform(0.6, 1.0, n_samples),
            'citizenship_status': np.random.choice(['SA_Citizen', 'Non_Citizen', 'Unknown'], n_samples,
                                                   p=[0.7, 0.2, 0.1]),
            'has_medical_aid': np.random.choice([True, False], n_samples, p=[0.4, 0.6])
        }

        return pd.DataFrame(data)

    @staticmethod
    def create_sample_triage_data():
        """Create mock clinical triage data"""
        np.random.seed(42)
        n_samples = 1000

        data = {
            'patient_id': range(n_samples),
            'heart_rate': np.random.randint(60, 140, n_samples),
            'temp_c': np.random.uniform(36.0, 39.5, n_samples),
            'resp_rate': np.random.randint(12, 30, n_samples),
            'o2_sat': np.random.randint(85, 100, n_samples),
            'systolic_bp': np.random.randint(100, 180, n_samples),
            'pain_score': np.random.randint(0, 10, n_samples),
            'arrival_mode': np.random.choice(['Walk-in', 'Ambulance', 'Private'], n_samples),
            'triage_priority': np.random.choice(['Red', 'Yellow', 'Green'], n_samples, p=[0.2, 0.3, 0.5])
        }

        return pd.DataFrame(data)