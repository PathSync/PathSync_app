import joblib
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class HealthcarePredictor:
    def __init__(self):
        if 'src' in str(Path.cwd()):
            models_path = Path.cwd().parent / 'models'
        else:
            models_path = Path.cwd() / 'models'

        self.biometric_model = joblib.load(models_path / 'biometric_model.pkl')
        self.triage_model = joblib.load(models_path / 'triage_model.pkl')
        self.biometric_preprocessing = joblib.load(models_path / 'biometric_preprocessing.pkl')
        self.triage_preprocessing = joblib.load(models_path / 'triage_preprocessing.pkl')
        
        self.available_immigration_statuses = list(self.biometric_preprocessing['encoders']['Immigration Status'].classes_)
        self.available_countries = list(self.biometric_preprocessing['encoders']['Country of Origin'].classes_)
        self.available_visa_types = list(self.biometric_preprocessing['encoders']['Visa Type'].classes_)
        self.available_languages = list(self.biometric_preprocessing['encoders']['Language'].classes_)
        self.available_asylum_statuses = list(self.triage_preprocessing['encoders']['Asylum Status'].classes_)

    def predict_biometric(self, age, gender, province, biometric_score, 
                         immigration_status=None, country_of_origin=None, 
                         visa_type=None, language=None):

        try:
            gender_encoded = self.biometric_preprocessing['encoders']['Gender'].transform([gender])[0]
            
            if immigration_status is None:  
                if province in ['Gauteng', 'Western Cape', 'Eastern Cape', 'KwaZulu-Natal', 
                               'Free State', 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape']:
                    immigration_status = 'Citizen'
                else:
                    immigration_status = self.available_immigration_statuses[0]
            
            if country_of_origin is None:
                country_of_origin = 'South Africa' if immigration_status == 'Citizen' else self.available_countries[0]
            
            if visa_type is None:
                visa_type = 'nan' if immigration_status == 'Citizen' else self.available_visa_types[0]
            
            if language is None:
                language = self.available_languages[-1]
            
            immigration_status_encoded = self.biometric_preprocessing['encoders']['Immigration Status'].transform([immigration_status])[0]
            
            if country_of_origin in self.available_countries:
                country_encoded = self.biometric_preprocessing['encoders']['Country of Origin'].transform([country_of_origin])[0]
            else:
                country_encoded = self.biometric_preprocessing['encoders']['Country of Origin'].transform(['South Africa'])[0]
            
            visa_type_encoded = self.biometric_preprocessing['encoders']['Visa Type'].transform([visa_type])[0]
            language_encoded = self.biometric_preprocessing['encoders']['Language'].transform([language])[0]
            
            features = np.array([[age, gender_encoded, immigration_status_encoded, 
                                 country_encoded, visa_type_encoded, language_encoded, 
                                 biometric_score]])
            
            prediction = self.biometric_model.predict(features)[0]
            confidence = np.max(self.biometric_model.predict_proba(features))
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Warning: Biometric prediction error - {str(e)}")
            return immigration_status if immigration_status else 'Citizen', 0.50

    def predict_triage(self, age, gender, hr_bpm=None, temp_c=None, resp_rate=None, 
                       systolic_bp=None, diastolic_bp=None, o2_sat=None, pain_score=None,
                       immigration_status=None, visa_type=None, asylum_status=None):
        try:
            gender_encoded = 0 if gender.lower() == 'male' else 1
            
            if immigration_status is None:
                immigration_status = 'Citizen'
            
            if visa_type is None:
                visa_type = 'N/A' if immigration_status == 'Citizen' else self.available_visa_types[0]
            
            if asylum_status is None:
                asylum_status = 'N/A' if immigration_status == 'Citizen' else self.available_asylum_statuses[0]
            
            immigration_encoded = self.triage_preprocessing['encoders']['Immigration Status'].transform([immigration_status])[0]
            visa_encoded = self.triage_preprocessing['encoders']['Visa Type'].transform([visa_type])[0]
            asylum_encoded = self.triage_preprocessing['encoders']['Asylum Status'].transform([asylum_status])[0]
            
            features = np.array([[age, gender_encoded, immigration_encoded, visa_encoded, asylum_encoded]])
            
            prediction = self.triage_model.predict(features)[0]
            confidence = np.max(self.triage_model.predict_proba(features))
            
            predicted_label = self.triage_preprocessing['encoders']['target'].inverse_transform([prediction])[0]
            
            priority_color_map = {
                'Low': 'Green',
                'Medium': 'Yellow',
                'High': 'Red'
            }
            
            return priority_color_map.get(predicted_label, predicted_label), confidence
            
        except Exception as e:
            print(f"Warning: Triage prediction error - {str(e)}")
            
            if pain_score and pain_score >= 8:
                return 'Red', 0.70
            elif pain_score and pain_score >= 5:
                return 'Yellow', 0.65
            else:
                return 'Green', 0.60

    def get_model_info(self):
        info = {
            'biometric_model': {
                'type': type(self.biometric_model).__name__,
                'expected_features': self.biometric_model.n_features_in_,
                'output_classes': list(self.biometric_model.classes_),
                'encoders': {
                    'Gender': list(self.biometric_preprocessing['encoders']['Gender'].classes_),
                    'Immigration Status': self.available_immigration_statuses,
                    'Country of Origin': f"{len(self.available_countries)} countries",
                    'Visa Type': self.available_visa_types,
                    'Language': self.available_languages
                }
            },
            'triage_model': {
                'type': type(self.triage_model).__name__,
                'expected_features': self.triage_model.n_features_in_,
                'output_classes': list(self.triage_model.classes_),
                'target_labels': list(self.triage_preprocessing['encoders']['target'].classes_),
                'encoders': {
                    'Immigration Status': list(self.triage_preprocessing['encoders']['Immigration Status'].classes_),
                    'Visa Type': list(self.triage_preprocessing['encoders']['Visa Type'].classes_),
                    'Asylum Status': self.available_asylum_statuses
                }
            }
        }
        return info


if __name__ == "__main__":
    predictor = HealthcarePredictor()
