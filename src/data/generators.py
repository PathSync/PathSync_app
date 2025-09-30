"""
Data generators for creating realistic healthcare datasets.
Implements Factory Pattern and Strategy Pattern for different data generation strategies.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Generator, Protocol
import numpy as np
from .base_models import BiometricData, TriageData, BiometricDataModel, TriageDataModel


class DataGenerationStrategy(Protocol):
    """Strategy interface for data generation following Strategy Pattern."""
    
    def generate_batch(self, count: int) -> List[dict]:
        """Generate a batch of data records."""
        ...


class BiometricDataGenerator:
    """
    Generator for realistic biometric verification data.
    Follows Single Responsibility Principle - only generates biometric data.
    """
    
    def __init__(self, seed: int = None):
        """Initialize with optional seed for reproducibility."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.provinces = [
            'Gauteng', 'Western Cape', 'Eastern Cape', 'KwaZulu-Natal',
            'Free State', 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape'
        ]
        
        self.genders = ['Male', 'Female', 'Other']
        self.citizenship_statuses = ['SA', 'Non-SA', 'Review']
        
        self.province_weights = [0.25, 0.20, 0.12, 0.20, 0.06, 0.10, 0.05, 0.02, 0.00]
        self.citizenship_weights = [0.85, 0.12, 0.03]
        self.gender_weights = [0.48, 0.50, 0.02]
    
    def _generate_national_id(self, age: int, gender: str) -> str:
        """Generate realistic South African ID number format."""
        current_year = datetime.now().year
        birth_year = current_year - age
        year_digits = str(birth_year)[-2:]
        
        month = f"{random.randint(1, 12):02d}"
        day = f"{random.randint(1, 28):02d}"
        
        if gender == 'Male':
            sequence = f"{random.randint(5000, 9999)}"
        else:
            sequence = f"{random.randint(0, 4999):04d}"
        
        citizenship_digit = "0"
        
        check_digit = str(random.randint(0, 9))
        
        return f"{year_digits}{month}{day}{sequence}{citizenship_digit}{check_digit}"
    
    def _calculate_biometric_score(self, citizenship: str, age: int) -> float:
        """Calculate realistic biometric scores based on citizenship and age."""
        if citizenship == 'SA':
            base_score = np.random.normal(0.85, 0.1)
        elif citizenship == 'Non-SA':
            base_score = np.random.normal(0.45, 0.15)
        else:
            base_score = np.random.normal(0.65, 0.2)
        
        age_factor = 1.0 if 18 <= age <= 65 else 0.95
        
        score = np.clip(base_score * age_factor, 0.0, 1.0)
        return round(score, 3)
    
    def generate_single(self) -> BiometricDataModel:
        """Generate a single biometric data record."""
        age = np.random.randint(18, 85)
        gender = np.random.choice(self.genders, p=self.gender_weights)
        province = np.random.choice(self.provinces, p=self.province_weights)
        citizenship = np.random.choice(self.citizenship_statuses, p=self.citizenship_weights)
        
        patient_id = str(uuid.uuid4())[:8]
        national_id = self._generate_national_id(age, gender)
        biometric_score = self._calculate_biometric_score(citizenship, age)
        
        days_ago = random.randint(0, 365)
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        data = BiometricData(
            patient_id=patient_id,
            age=age,
            gender=gender,
            province=province,
            national_id=national_id,
            biometric_score=biometric_score,
            citizenship_status=citizenship,
            timestamp=timestamp
        )
        
        return BiometricDataModel(data)
    
    def generate_batch(self, count: int) -> List[BiometricDataModel]:
        """Generate a batch of biometric data records."""
        return [self.generate_single() for _ in range(count)]


class TriageDataGenerator:
    """
    Generator for realistic medical triage data.
    Follows Single Responsibility Principle - only generates triage data.
    """
    
    def __init__(self, seed: int = None):
        """Initialize with optional seed for reproducibility."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.genders = ['Male', 'Female', 'Other']
        self.priorities = ['Red', 'Yellow', 'Green']
        self.priority_weights = [0.15, 0.35, 0.50]
        
        self.symptoms_by_priority = {
            'Red': ['chest_pain', 'difficulty_breathing', 'unconscious', 'severe_bleeding', 
                   'stroke_symptoms', 'heart_attack', 'severe_trauma'],
            'Yellow': ['moderate_pain', 'fever', 'vomiting', 'dehydration', 'infection',
                      'mild_breathing_issues', 'moderate_bleeding'],
            'Green': ['minor_cuts', 'cold_symptoms', 'headache', 'minor_pain',
                     'routine_checkup', 'medication_refill', 'minor_infection']
        }
    
    def _generate_vitals_for_priority(self, priority: str, age: int) -> dict:
        """Generate realistic vital signs based on triage priority and age."""
        if age < 30:
            base_hr = (60, 90)
            base_temp = (36.0, 37.0)
            base_resp = (12, 18)
            base_systolic = (110, 130)
            base_diastolic = (70, 85)
            base_o2 = (98, 100)
        elif age < 60:
            base_hr = (65, 95)
            base_temp = (36.2, 37.2)
            base_resp = (14, 20)
            base_systolic = (120, 140)
            base_diastolic = (75, 90)
            base_o2 = (96, 99)
        else:
            base_hr = (70, 100)
            base_temp = (36.0, 37.3)
            base_resp = (16, 22)
            base_systolic = (130, 150)
            base_diastolic = (80, 95)
            base_o2 = (94, 98)
        
        if priority == 'Red':
            hr_bpm = random.randint(max(30, base_hr[0] - 20), min(200, base_hr[1] + 40))
            temp_c = round(random.uniform(max(30.0, base_temp[0] - 2), 
                                        min(42.0, base_temp[1] + 3)), 1)
            resp_rate = random.randint(max(5, base_resp[0] - 5), min(40, base_resp[1] + 15))
            systolic_bp = random.randint(max(60, base_systolic[0] - 30), 
                                       min(220, base_systolic[1] + 50))
            diastolic_bp = random.randint(max(30, base_diastolic[0] - 20), 
                                        min(130, base_diastolic[1] + 30))
            o2_sat = random.randint(max(60, base_o2[0] - 15), base_o2[1] - 5)
            pain_score = random.randint(7, 10)
            
        elif priority == 'Yellow':
            hr_bpm = random.randint(max(40, base_hr[0] - 10), min(150, base_hr[1] + 20))
            temp_c = round(random.uniform(max(35.0, base_temp[0] - 1), 
                                        min(39.5, base_temp[1] + 2)), 1)
            resp_rate = random.randint(max(8, base_resp[0] - 2), min(30, base_resp[1] + 8))
            systolic_bp = random.randint(max(80, base_systolic[0] - 15), 
                                       min(180, base_systolic[1] + 25))
            diastolic_bp = random.randint(max(50, base_diastolic[0] - 10), 
                                        min(110, base_diastolic[1] + 15))
            o2_sat = random.randint(max(85, base_o2[0] - 8), base_o2[1])
            pain_score = random.randint(4, 7)
            
        else:
            hr_bpm = random.randint(base_hr[0], base_hr[1])
            temp_c = round(random.uniform(base_temp[0], base_temp[1] + 0.5), 1)
            resp_rate = random.randint(base_resp[0], base_resp[1])
            systolic_bp = random.randint(base_systolic[0], base_systolic[1] + 10)
            diastolic_bp = random.randint(base_diastolic[0], base_diastolic[1] + 5)
            o2_sat = random.randint(base_o2[0], base_o2[1])
            pain_score = random.randint(0, 4)
        
        return {
            'hr_bpm': hr_bpm,
            'temp_c': temp_c,
            'resp_rate': resp_rate,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'o2_sat': o2_sat,
            'pain_score': pain_score
        }
    
    def generate_single(self) -> TriageDataModel:
        """Generate a single triage data record."""
        age = np.random.randint(18, 85)
        gender = np.random.choice(self.genders)
        priority = np.random.choice(self.priorities, p=self.priority_weights)
        
        vitals = self._generate_vitals_for_priority(priority, age)
        
        symptom_pool = self.symptoms_by_priority[priority]
        num_symptoms = random.randint(1, min(3, len(symptom_pool)))
        symptoms = random.sample(symptom_pool, num_symptoms)
        
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
        
        patient_id = str(uuid.uuid4())[:8]
        
        data = TriageData(
            patient_id=patient_id,
            age=age,
            gender=gender,
            triage_priority=priority,
            symptoms=symptoms,
            timestamp=timestamp,
            **vitals
        )
        
        return TriageDataModel(data)
    
    def generate_batch(self, count: int) -> List[TriageDataModel]:
        """Generate a batch of triage data records."""
        return [self.generate_single() for _ in range(count)]


class DataGeneratorFactory:
    """
    Factory class for creating different types of data generators.
    Implements Factory Pattern for better extensibility.
    """
    
    @staticmethod
    def create_biometric_generator(seed: int = None) -> BiometricDataGenerator:
        """Create a biometric data generator."""
        return BiometricDataGenerator(seed)
    
    @staticmethod
    def create_triage_generator(seed: int = None) -> TriageDataGenerator:
        """Create a triage data generator."""
        return TriageDataGenerator(seed)
    
    @staticmethod
    def get_available_generators() -> List[str]:
        """Get list of available generator types."""
        return ['biometric', 'triage']
