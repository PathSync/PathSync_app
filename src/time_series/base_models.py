from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class TimeSeriesData:
    timestamp: datetime
    patient_id: str
    
    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Timestamp must be a datetime object")
        if not self.patient_id:
            raise ValueError("Patient ID cannot be empty")

@dataclass
class HealthTimeSeries(TimeSeriesData):
    vital_signs: Dict[str, float]
    symptoms: List[str]
    medications: List[str]
    notes: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.vital_signs:
            raise ValueError("Vital signs cannot be empty")
        valid_ranges = {
            'heart_rate': (30, 200),
            'blood_pressure_systolic': (70, 250),
            'blood_pressure_diastolic': (40, 150),
            'temperature': (32.0, 45.0),
            'oxygen_saturation': (70, 100),
            'respiratory_rate': (8, 50)
        }
        for vital, value in self.vital_signs.items():
            if vital in valid_ranges:
                min_val, max_val = valid_ranges[vital]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{vital} value {value} out of valid range {valid_ranges[vital]}")

@dataclass 
class VitalSignsTimeSeries(TimeSeriesData):
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    temperature: float
    oxygen_saturation: float
    respiratory_rate: float
    pain_score: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if not (30 <= self.heart_rate <= 200):
            raise ValueError(f"Heart rate {self.heart_rate} out of range")
        if not (70 <= self.systolic_bp <= 250):
            raise ValueError(f"Systolic BP {self.systolic_bp} out of range")
        if not (40 <= self.diastolic_bp <= 150):
            raise ValueError(f"Diastolic BP {self.diastolic_bp} out of range")
        if not (32.0 <= self.temperature <= 45.0):
            raise ValueError(f"Temperature {self.temperature} out of range")
        if not (70 <= self.oxygen_saturation <= 100):
            raise ValueError(f"Oxygen saturation {self.oxygen_saturation} out of range")
        if not (8 <= self.respiratory_rate <= 50):
            raise ValueError(f"Respiratory rate {self.respiratory_rate} out of range")
        if not (0 <= self.pain_score <= 10):
            raise ValueError(f"Pain score {self.pain_score} out of range")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'patient_id': self.patient_id,
            'heart_rate': self.heart_rate,
            'systolic_bp': self.systolic_bp,
            'diastolic_bp': self.diastolic_bp,
            'temperature': self.temperature,
            'oxygen_saturation': self.oxygen_saturation,
            'respiratory_rate': self.respiratory_rate,
            'pain_score': self.pain_score
        }

@dataclass
class TimeSeriesResult:
    analysis_type: str
    patient_id: str
    time_range: Tuple[datetime, datetime]
    metrics: Dict[str, Any]
    trends: Dict[str, str]
    anomalies: List[Dict[str, Any]]
    predictions: Optional[Dict[str, Any]] = None
    risk_level: str = "low"
    
    def __post_init__(self):
        valid_risks = ["low", "medium", "high", "critical"]
        if self.risk_level not in valid_risks:
            raise ValueError(f"Risk level must be one of {valid_risks}")

class TimeSeriesAnalyzer(ABC):
    def __init__(self, name: str):
        self.name = name
        self.analysis_history: List[TimeSeriesResult] = []
    
    @abstractmethod
    def analyze(self, data: List[TimeSeriesData]) -> TimeSeriesResult:
        pass
    
    @abstractmethod
    def validate_data(self, data: List[TimeSeriesData]) -> bool:
        pass
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        if not self.analysis_history:
            return {"message": "No analyses performed"}
        total_analyses = len(self.analysis_history)
        risk_distribution = {}
        for result in self.analysis_history:
            risk = result.risk_level
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        return {
            "analyzer_name": self.name,
            "total_analyses": total_analyses,
            "risk_distribution": risk_distribution,
            "latest_analysis": self.analysis_history[-1].analysis_type if self.analysis_history else None
        }

class TimeSeriesProcessor:
    @staticmethod
    def convert_to_dataframe(data: List[TimeSeriesData]) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()
        records = []
        for item in data:
            if isinstance(item, VitalSignsTimeSeries):
                records.append(item.to_dict())
            elif isinstance(item, HealthTimeSeries):
                record = {
                    'timestamp': item.timestamp,
                    'patient_id': item.patient_id,
                    'symptoms_count': len(item.symptoms),
                    'medications_count': len(item.medications)
                }
                record.update(item.vital_signs)
                records.append(record)
            else:
                records.append({
                    'timestamp': item.timestamp,
                    'patient_id': item.patient_id
                })
        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        return df
    
    @staticmethod
    def calculate_basic_stats(series: pd.Series) -> Dict[str, float]:
        if series.empty:
            return {}
        return {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'median': float(series.median()),
            'trend_slope': TimeSeriesProcessor._calculate_trend_slope(series)
        }
    
    @staticmethod
    def _calculate_trend_slope(series: pd.Series) -> float:
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        y = series.values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0
        x_clean = x[mask]
        y_clean = y[mask]
        n = len(x_clean)
        slope = (n * np.sum(x_clean * y_clean) - np.sum(x_clean) * np.sum(y_clean)) / (n * np.sum(x_clean**2) - np.sum(x_clean)**2)
        return float(slope)
    
    @staticmethod
    def detect_outliers(series: pd.Series, threshold: float = 2.0) -> List[int]:
        if series.empty or series.std() == 0:
            return []
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_indices = z_scores[z_scores > threshold].index.tolist()
        return outlier_indices
