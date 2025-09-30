"""
Base data models for the healthcare AI system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd


@dataclass
class PatientData:
    patient_id: str
    age: int
    gender: str
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not isinstance(self.age, int) or self.age < 0 or self.age > 150:
            raise ValueError(f"Invalid age: {self.age}. Must be between 0 and 150.")
        
        if self.gender not in ['Male', 'Female', 'Other']:
            raise ValueError(f"Invalid gender: {self.gender}. Must be Male, Female, or Other.")
        
        if not self.patient_id or not isinstance(self.patient_id, str):
            raise ValueError("Patient ID must be a non-empty string.")


@dataclass
class BiometricData(PatientData):
    """Biometric verification data model."""
    province: str
    national_id: str
    biometric_score: float
    citizenship_status: str
    timestamp: datetime = None
    
    def __post_init__(self):
        """Extended validation for biometric data."""
        super().__post_init__()
        if self.timestamp is None:
            self.timestamp = datetime.now()
        self._validate_biometric()
    
    def _validate_biometric(self):
        """Validate biometric-specific data."""
        if not 0.0 <= self.biometric_score <= 1.0:
            raise ValueError(f"Biometric score must be between 0.0 and 1.0, got {self.biometric_score}")
        
        if self.citizenship_status not in ['SA', 'Non-SA', 'Review']:
            raise ValueError(f"Invalid citizenship status: {self.citizenship_status}")
        
        valid_provinces = [
            'Gauteng', 'Western Cape', 'Eastern Cape', 'KwaZulu-Natal',
            'Free State', 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape'
        ]
        if self.province not in valid_provinces:
            raise ValueError(f"Invalid province: {self.province}")


@dataclass
class TriageData(PatientData):
    """Medical triage data model."""
    hr_bpm: int
    temp_c: float
    resp_rate: int
    systolic_bp: int
    diastolic_bp: int
    o2_sat: int
    pain_score: int
    triage_priority: str
    timestamp: datetime = None
    symptoms: Optional[List[str]] = None
    
    def __post_init__(self):
        """Extended validation for triage data."""
        super().__post_init__()
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.symptoms is None:
            self.symptoms = []
        self._validate_triage()
    
    def _validate_triage(self):
        """Validate triage-specific vital signs."""
        vital_ranges = {
            'hr_bpm': (30, 200),
            'temp_c': (30.0, 45.0),
            'resp_rate': (5, 50),
            'systolic_bp': (60, 250),
            'diastolic_bp': (30, 150),
            'o2_sat': (60, 100),
            'pain_score': (0, 10)
        }
        
        for field, (min_val, max_val) in vital_ranges.items():
            value = getattr(self, field)
            if not min_val <= value <= max_val:
                raise ValueError(f"{field} value {value} outside valid range {min_val}-{max_val}")
        
        if self.triage_priority not in ['Red', 'Yellow', 'Green']:
            raise ValueError(f"Invalid triage priority: {self.triage_priority}")


class DataModel(ABC):
    """Abstract base class for data models following Interface Segregation Principle."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        pass
    
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Convert model to pandas DataFrame."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate model data."""
        pass


class BiometricDataModel(DataModel):
    """Biometric data model implementing the DataModel interface."""
    
    def __init__(self, data: BiometricData):
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert biometric data to dictionary."""
        return {
            'patient_id': self.data.patient_id,
            'age': self.data.age,
            'gender': self.data.gender,
            'province': self.data.province,
            'national_id': self.data.national_id,
            'biometric_score': self.data.biometric_score,
            'citizenship_status': self.data.citizenship_status,
            'timestamp': self.data.timestamp.isoformat()
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for ML processing."""
        return pd.DataFrame([self.to_dict()])
    
    def validate(self) -> bool:
        """Validate biometric data."""
        try:
            self.data._validate()
            self.data._validate_biometric()
            return True
        except ValueError:
            return False


class TriageDataModel(DataModel):
    """Triage data model implementing the DataModel interface."""
    
    def __init__(self, data: TriageData):
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triage data to dictionary."""
        return {
            'patient_id': self.data.patient_id,
            'age': self.data.age,
            'gender': self.data.gender,
            'hr_bpm': self.data.hr_bpm,
            'temp_c': self.data.temp_c,
            'resp_rate': self.data.resp_rate,
            'systolic_bp': self.data.systolic_bp,
            'diastolic_bp': self.data.diastolic_bp,
            'o2_sat': self.data.o2_sat,
            'pain_score': self.data.pain_score,
            'triage_priority': self.data.triage_priority,
            'timestamp': self.data.timestamp.isoformat(),
            'symptoms': ','.join(self.data.symptoms) if self.data.symptoms else ''
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for ML processing."""
        return pd.DataFrame([self.to_dict()])
    
    def validate(self) -> bool:
        """Validate triage data."""
        try:
            self.data._validate()
            self.data._validate_triage()
            return True
        except ValueError:
            return False
