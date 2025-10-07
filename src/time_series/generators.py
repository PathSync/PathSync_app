import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .base_models import VitalSignsTimeSeries, HealthTimeSeries


class TimeSeriesGenerator:
    
    def __init__(self, name: str):
        self.name = name
        self.generation_history = []
    
    def add_noise(self, value: float, noise_level: float = 0.05) -> float:
        noise = random.gauss(0, abs(value) * noise_level)
        return value + noise
    
    def apply_trend(self, base_value: float, trend_factor: float, time_step: int) -> float:
        return base_value + (trend_factor * time_step)


class HealthDataGenerator(TimeSeriesGenerator):
    
    def __init__(self):
        super().__init__("HealthDataGenerator")
        self.patient_profiles = {
            'healthy': {
                'heart_rate': {'mean': 75, 'std': 8, 'trend': 0},
                'systolic_bp': {'mean': 120, 'std': 10, 'trend': 0},
                'diastolic_bp': {'mean': 80, 'std': 8, 'trend': 0},
                'temperature': {'mean': 36.8, 'std': 0.3, 'trend': 0},
                'oxygen_saturation': {'mean': 98, 'std': 1, 'trend': 0},
                'respiratory_rate': {'mean': 16, 'std': 2, 'trend': 0},
                'pain_score': {'mean': 1, 'std': 1, 'trend': 0}
            },
            'deteriorating': {
                'heart_rate': {'mean': 85, 'std': 12, 'trend': 0.5},
                'systolic_bp': {'mean': 130, 'std': 15, 'trend': 0.3},
                'diastolic_bp': {'mean': 85, 'std': 10, 'trend': 0.2},
                'temperature': {'mean': 37.2, 'std': 0.8, 'trend': 0.05},
                'oxygen_saturation': {'mean': 96, 'std': 2, 'trend': -0.1},
                'respiratory_rate': {'mean': 20, 'std': 3, 'trend': 0.2},
                'pain_score': {'mean': 4, 'std': 2, 'trend': 0.1}
            },
            'critical': {
                'heart_rate': {'mean': 110, 'std': 20, 'trend': 1.0},
                'systolic_bp': {'mean': 160, 'std': 25, 'trend': 0.8},
                'diastolic_bp': {'mean': 95, 'std': 15, 'trend': 0.3},
                'temperature': {'mean': 38.5, 'std': 1.2, 'trend': 0.1},
                'oxygen_saturation': {'mean': 92, 'std': 3, 'trend': -0.2},
                'respiratory_rate': {'mean': 28, 'std': 5, 'trend': 0.4},
                'pain_score': {'mean': 7, 'std': 2, 'trend': 0.2}
            },
            'recovering': {
                'heart_rate': {'mean': 80, 'std': 10, 'trend': -0.2},
                'systolic_bp': {'mean': 125, 'std': 12, 'trend': -0.1},
                'diastolic_bp': {'mean': 82, 'std': 8, 'trend': -0.05},
                'temperature': {'mean': 37.0, 'std': 0.5, 'trend': -0.02},
                'oxygen_saturation': {'mean': 97, 'std': 1.5, 'trend': 0.05},
                'respiratory_rate': {'mean': 18, 'std': 2, 'trend': -0.1},
                'pain_score': {'mean': 3, 'std': 1.5, 'trend': -0.1}
            }
        }
    
    def generate_patient_timeline(self, patient_id: str, profile: str = 'healthy',
                                duration_hours: int = 24, interval_minutes: int = 60) -> List[VitalSignsTimeSeries]:
        if profile not in self.patient_profiles:
            raise ValueError(f"Unknown profile: {profile}. Available: {list(self.patient_profiles.keys())}")
        
        profile_data = self.patient_profiles[profile]
        timeline = []
        
        start_time = datetime.now() - timedelta(hours=duration_hours)
        num_points = (duration_hours * 60) // interval_minutes
        
        for i in range(num_points):
            timestamp = start_time + timedelta(minutes=i * interval_minutes)
            
            vitals = {}
            for vital, params in profile_data.items():
                base_value = params['mean']
                std_dev = params['std']
                trend = params['trend']
                
                trended_value = self.apply_trend(base_value, trend, i)
                
                hour = timestamp.hour
                daily_factor = self._get_daily_factor(vital, hour)
                adjusted_value = trended_value * daily_factor
                
                final_value = random.gauss(adjusted_value, std_dev)
                
                final_value = self._apply_vital_constraints(vital, final_value)
                
                vitals[vital] = final_value
            
            vital_signs = VitalSignsTimeSeries(
                timestamp=timestamp,
                patient_id=patient_id,
                heart_rate=vitals['heart_rate'],
                systolic_bp=vitals['systolic_bp'],
                diastolic_bp=vitals['diastolic_bp'],
                temperature=vitals['temperature'],
                oxygen_saturation=vitals['oxygen_saturation'],
                respiratory_rate=vitals['respiratory_rate'],
                pain_score=max(0, min(10, int(vitals['pain_score'])))
            )
            
            timeline.append(vital_signs)
        
        self.generation_history.append({
            'patient_id': patient_id,
            'profile': profile,
            'duration_hours': duration_hours,
            'data_points': len(timeline),
            'generated_at': datetime.now()
        })
        
        return timeline
    
    def _get_daily_factor(self, vital: str, hour: int) -> float:
        patterns = {
            'heart_rate': {
                0: 0.92, 3: 0.88, 6: 0.95, 9: 1.05, 12: 1.08, 15: 1.10, 18: 1.05, 21: 0.98
            },
            'systolic_bp': {
                0: 0.95, 3: 0.90, 6: 1.05, 9: 1.10, 12: 1.08, 15: 1.05, 18: 1.02, 21: 0.98
            },
            'temperature': {
                0: 0.995, 3: 0.990, 6: 0.992, 9: 1.000, 12: 1.005, 15: 1.008, 18: 1.010, 21: 1.002
            },
            'oxygen_saturation': {
                0: 0.998, 3: 0.996, 6: 1.000, 9: 1.002, 12: 1.002, 15: 1.001, 18: 1.000, 21: 0.999
            }
        }
        
        if vital not in patterns:
            return 1.0
        
        pattern = patterns[vital]
        
        closest_hour = min(pattern.keys(), key=lambda x: abs(x - hour))
        return pattern[closest_hour]
    
    def _apply_vital_constraints(self, vital: str, value: float) -> float:
        constraints = {
            'heart_rate': (30, 200),
            'systolic_bp': (70, 250),
            'diastolic_bp': (40, 150),
            'temperature': (32.0, 45.0),
            'oxygen_saturation': (70, 100),
            'respiratory_rate': (8, 50),
            'pain_score': (0, 10)
        }
        
        if vital in constraints:
            min_val, max_val = constraints[vital]
            return max(min_val, min(max_val, value))
        
        return value
    
    def generate_anomaly_scenario(self, patient_id: str, base_profile: str = 'healthy',
                                anomaly_type: str = 'fever_spike') -> List[VitalSignsTimeSeries]:
        timeline = self.generate_patient_timeline(patient_id, base_profile, duration_hours=12)
        
        anomaly_start = len(timeline) // 2
        anomaly_duration = 3
        
        anomaly_patterns = {
            'fever_spike': {
                'temperature': lambda x: x + random.uniform(1.5, 3.0),
                'heart_rate': lambda x: x + random.uniform(15, 30)
            },
            'respiratory_distress': {
                'oxygen_saturation': lambda x: x - random.uniform(8, 15),
                'respiratory_rate': lambda x: x + random.uniform(8, 15),
                'heart_rate': lambda x: x + random.uniform(10, 20)
            },
            'hypotension': {
                'systolic_bp': lambda x: x - random.uniform(30, 50),
                'diastolic_bp': lambda x: x - random.uniform(15, 25),
                'heart_rate': lambda x: x + random.uniform(15, 25)
            },
            'pain_crisis': {
                'pain_score': lambda x: min(10, x + random.uniform(4, 6)),
                'heart_rate': lambda x: x + random.uniform(10, 25),
                'systolic_bp': lambda x: x + random.uniform(15, 30)
            }
        }
        
        if anomaly_type not in anomaly_patterns:
            return timeline
        
        pattern = anomaly_patterns[anomaly_type]
        
        for i in range(anomaly_start, min(anomaly_start + anomaly_duration, len(timeline))):
            vital_signs = timeline[i]
            
            modified_vitals = {}
            for attr in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 
                        'oxygen_saturation', 'respiratory_rate', 'pain_score']:
                current_value = getattr(vital_signs, attr)
                
                if attr in pattern:
                    modified_value = pattern[attr](current_value)
                    modified_value = self._apply_vital_constraints(attr, modified_value)
                    modified_vitals[attr] = modified_value
                else:
                    modified_vitals[attr] = current_value
            
            timeline[i] = VitalSignsTimeSeries(
                timestamp=vital_signs.timestamp,
                patient_id=vital_signs.patient_id,
                heart_rate=modified_vitals['heart_rate'],
                systolic_bp=modified_vitals['systolic_bp'],
                diastolic_bp=modified_vitals['diastolic_bp'],
                temperature=modified_vitals['temperature'],
                oxygen_saturation=modified_vitals['oxygen_saturation'],
                respiratory_rate=modified_vitals['respiratory_rate'],
                pain_score=int(modified_vitals['pain_score'])
            )
        
        return timeline


class PatientTimelineGenerator:
    
    def __init__(self):
        self.health_generator = HealthDataGenerator()
        self.generated_patients = []
    
    def generate_multiple_patients(self, num_patients: int = 5, 
                                 scenarios: Optional[List[str]] = None) -> Dict[str, List[VitalSignsTimeSeries]]:
        if scenarios is None:
            scenarios = ['healthy', 'deteriorating', 'critical', 'recovering']
        
        patient_timelines = {}
        
        for i in range(num_patients):
            patient_id = f"PATIENT_{i+1:03d}"
            scenario = random.choice(scenarios)
            
            timeline = self.health_generator.generate_patient_timeline(
                patient_id=patient_id,
                profile=scenario,
                duration_hours=random.randint(12, 48),
                interval_minutes=random.choice([30, 60, 120])
            )
            
            patient_timelines[patient_id] = timeline
            
            self.generated_patients.append({
                'patient_id': patient_id,
                'scenario': scenario,
                'timeline_length': len(timeline),
                'generated_at': datetime.now()
            })
        
        return patient_timelines
    
    def generate_demo_dataset(self) -> Dict[str, Any]:
        demo_data = {}
        
        scenarios = {
            'stable_patient': self.health_generator.generate_patient_timeline('DEMO_001', 'healthy', 24, 60),
            'deteriorating_patient': self.health_generator.generate_patient_timeline('DEMO_002', 'deteriorating', 36, 60),
            'critical_patient': self.health_generator.generate_patient_timeline('DEMO_003', 'critical', 12, 30),
            'recovering_patient': self.health_generator.generate_patient_timeline('DEMO_004', 'recovering', 48, 120),
        }
        
        anomaly_scenarios = {
            'fever_spike_patient': self.health_generator.generate_anomaly_scenario('DEMO_005', 'healthy', 'fever_spike'),
            'respiratory_distress_patient': self.health_generator.generate_anomaly_scenario('DEMO_006', 'healthy', 'respiratory_distress'),
            'hypotension_patient': self.health_generator.generate_anomaly_scenario('DEMO_007', 'healthy', 'hypotension'),
            'pain_crisis_patient': self.health_generator.generate_anomaly_scenario('DEMO_008', 'healthy', 'pain_crisis')
        }
        
        demo_data.update(scenarios)
        demo_data.update(anomaly_scenarios)
        
        summary = {
            'total_patients': len(demo_data),
            'total_data_points': sum(len(timeline) for timeline in demo_data.values()),
            'scenarios_included': list(scenarios.keys()) + list(anomaly_scenarios.keys()),
            'generated_at': datetime.now(),
            'data_description': 'Comprehensive demo dataset with various patient scenarios and anomalies'
        }
        
        return {
            'patient_timelines': demo_data,
            'summary': summary
        }
    
    def export_to_csv(self, patient_timelines: Dict[str, List[VitalSignsTimeSeries]], 
                     output_path: str = 'time_series_data.csv') -> str:
        all_records = []
        
        for patient_id, timeline in patient_timelines.items():
            for vital_signs in timeline:
                record = vital_signs.to_dict()
                all_records.append(record)
        
        df = pd.DataFrame(all_records)
        df.to_csv(output_path, index=False)
        
        return f"Exported {len(all_records)} records to {output_path}"
    
    def get_generation_summary(self) -> Dict[str, Any]:
        if not self.generated_patients:
            return {"message": "No patients generated yet"}
        
        scenarios = {}
        for patient in self.generated_patients:
            scenario = patient['scenario']
            scenarios[scenario] = scenarios.get(scenario, 0) + 1
        
        return {
            'total_patients_generated': len(self.generated_patients),
            'scenario_distribution': scenarios,
            'total_data_points': sum(p['timeline_length'] for p in self.generated_patients),
            'latest_generation': self.generated_patients[-1]['generated_at'] if self.generated_patients else None
        }
