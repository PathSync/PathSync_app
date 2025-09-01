# app/services/data/data_loader.py
import pandas as pd
import os
from app.core.config import settings


class DataLoader:
    """Service to load and manage healthcare datasets"""

    def __init__(self):
        self.raw_data_path = os.path.join(settings.DATA_DIR, "raw")
        self.processed_data_path = os.path.join(settings.DATA_DIR, "processed")

    def load_triage_data(self, filename="healthcare_hospital_triage.csv"):
        """
        Load the triage dataset from the raw data directory
        """
        file_path = os.path.join(self.raw_data_path, filename)

        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded dataset with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            # Return demo data if real data isn't available
            return self._create_demo_data()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self._create_demo_data()

    def _create_demo_data(self):
        """Create demo data if the real dataset isn't available"""
        print("Creating demo data for development purposes...")

        data = {
            'age': [25, 67, 45, 32, 19, 55, 28, 61, 39, 72, 35, 48, 22, 59, 41],
            'gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
                       'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'province': ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Gauteng', 'Eastern Cape',
                         'Limpopo', 'Free State', 'North West', 'Mpumalanga', 'Western Cape',
                         'Gauteng', 'KwaZulu-Natal', 'Eastern Cape', 'Free State', 'Mpumalanga'],
            'has_medical_aid': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            'medical_scheme': ['Discovery', 'Bonitas', 'None', 'Momentum', 'None',
                               'Discovery', 'None', 'Bonitas', 'None', 'Discovery',
                               'Momentum', 'None', 'None', 'Discovery', 'None'],
            'facility_type': ['Private', 'Private', 'Public', 'Private', 'Public',
                              'Private', 'Public', 'Private', 'Public', 'Private',
                              'Private', 'Public', 'Public', 'Private', 'Public'],
            'visit_type': ['Emergency', 'Outpatient', 'Emergency', 'Outpatient', 'Emergency',
                           'Outpatient', 'Emergency', 'Outpatient', 'Emergency', 'Outpatient',
                           'Emergency', 'Outpatient', 'Emergency', 'Follow-up', 'Outpatient'],
            'arrival_via_ambulance': [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            'hr_bpm': [72, 85, 110, 68, 95, 78, 105, 82, 115, 90, 88, 92, 101, 76, 84],
            'temp_c': [36.8, 37.2, 38.5, 36.5, 37.8, 36.9, 38.2, 37.1, 39.1, 37.5, 37.0, 38.0, 37.6, 36.7, 37.3],
            'resp_rate': [16, 18, 24, 15, 22, 17, 26, 19, 28, 20, 18, 21, 25, 16, 19],
            'systolic_bp': [120, 145, 130, 118, 125, 138, 135, 142, 140, 150, 132, 128, 139, 126, 134],
            'diastolic_bp': [80, 90, 85, 78, 82, 88, 87, 92, 90, 95, 84, 82, 89, 81, 86],
            'o2_sat': [98, 96, 92, 99, 94, 97, 91, 95, 89, 93, 96, 93, 90, 98, 95],
            'icd10_code': ['J06', 'I10', 'R05', 'Z00', 'S83', 'E11', 'J45', 'I25', 'A09', 'I50',
                           'M54', 'J02', 'S06', 'Z01', 'J20'],
            'pain_score': [2, 5, 7, 1, 6, 3, 8, 4, 9, 7, 5, 6, 8, 2, 4],
            'triage_priority': ['Green', 'Yellow', 'Red', 'Green', 'Yellow',
                                'Green', 'Red', 'Yellow', 'Red', 'Red',
                                'Yellow', 'Yellow', 'Red', 'Green', 'Yellow']
        }

        return pd.DataFrame(data)