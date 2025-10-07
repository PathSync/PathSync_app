from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_models import (
    TimeSeriesData,
    VitalSignsTimeSeries,
    TimeSeriesResult,
    TimeSeriesProcessor
)


class HealthForecaster:
    
    def __init__(self, name: str):
        self.name = name
        self.forecast_history: List[Dict[str, Any]] = []
    
    def simple_trend_forecast(self, series: pd.Series, periods: int = 5) -> List[float]:
        if len(series) < 2:
            return [float(series.iloc[-1])] * periods if not series.empty else [0.0] * periods
        
        slope = TimeSeriesProcessor._calculate_trend_slope(series)
        last_value = float(series.iloc[-1])
        
        forecasts = []
        for i in range(1, periods + 1):
            forecast_value = last_value + (slope * i)
            forecasts.append(forecast_value)
        
        return forecasts
    
    def moving_average_forecast(self, series: pd.Series, window: int = 3, periods: int = 5) -> List[float]:
        if len(series) < window:
            return [float(series.mean())] * periods if not series.empty else [0.0] * periods
        
        ma_value = float(series.tail(window).mean())
        return [ma_value] * periods


class VitalSignsForecaster(HealthForecaster):
    
    def __init__(self):
        super().__init__("VitalSignsForecaster")
        self.vital_sign_ranges = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'temperature': (36.1, 37.2),
            'oxygen_saturation': (95, 100),
            'respiratory_rate': (12, 20)
        }
    
    def forecast_vital_signs(self, data: List[VitalSignsTimeSeries], 
                           forecast_hours: int = 24) -> Dict[str, Any]:
        if not data:
            return {"error": "No data provided"}
        
        df = TimeSeriesProcessor.convert_to_dataframe(data)
        patient_id = data[0].patient_id
        
        forecasts = {}
        risk_alerts = []
        
        vital_signs = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 
                      'oxygen_saturation', 'respiratory_rate']
        
        for vital in vital_signs:
            if vital in df.columns:
                series = df[vital].dropna()
                
                if len(series) >= 2:
                    periods = max(1, forecast_hours // 4)
                    forecast_values = self.simple_trend_forecast(series, periods)
                    
                    last_timestamp = df['timestamp'].max()
                    forecast_timestamps = [
                        last_timestamp + timedelta(hours=4 * (i + 1)) 
                        for i in range(periods)
                    ]
                    
                    forecasts[vital] = {
                        'values': forecast_values,
                        'timestamps': forecast_timestamps,
                        'confidence': self._calculate_confidence(series)
                    }
                    
                    normal_range = self.vital_sign_ranges.get(vital, (0, 1000))
                    for i, value in enumerate(forecast_values):
                        if not (normal_range[0] <= value <= normal_range[1]):
                            risk_alerts.append({
                                'vital_sign': vital,
                                'predicted_value': value,
                                'predicted_time': forecast_timestamps[i],
                                'normal_range': normal_range,
                                'severity': self._assess_severity(vital, value, normal_range)
                            })
        
        forecast_result = {
            'patient_id': patient_id,
            'forecast_generated': datetime.now(),
            'forecast_horizon_hours': forecast_hours,
            'forecasts': forecasts,
            'risk_alerts': risk_alerts,
            'overall_risk': self._assess_overall_risk(risk_alerts)
        }
        
        self.forecast_history.append(forecast_result)
        return forecast_result
    
    def _calculate_confidence(self, series: pd.Series) -> float:
        if len(series) < 3:
            return 0.3
        
        cv = series.std() / series.mean() if series.mean() != 0 else 1.0
        data_points = len(series)
        stability_score = max(0, 1 - cv)
        data_score = min(1.0, data_points / 20)
        confidence = (stability_score * 0.7) + (data_score * 0.3)
        return round(confidence, 2)
    
    def _assess_severity(self, vital: str, value: float, normal_range: Tuple[float, float]) -> str:
        min_val, max_val = normal_range
        
        if value < min_val:
            deviation = (min_val - value) / min_val
        else:
            deviation = (value - max_val) / max_val
        
        if deviation > 0.3:
            return 'critical'
        elif deviation > 0.15:
            return 'high'
        elif deviation > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def _assess_overall_risk(self, risk_alerts: List[Dict[str, Any]]) -> str:
        if not risk_alerts:
            return 'low'
        
        severity_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        total_score = sum(severity_scores.get(alert['severity'], 0) for alert in risk_alerts)
        
        if total_score >= 10:
            return 'critical'
        elif total_score >= 6:
            return 'high'
        elif total_score >= 3:
            return 'medium'
        else:
            return 'low'


class RiskPredictor:
    
    def __init__(self):
        self.name = "RiskPredictor"
        self.risk_models = {
            'cardiovascular': self._cardiovascular_risk_model,
            'respiratory': self._respiratory_risk_model,
            'infection': self._infection_risk_model
        }
    
    def predict_risks(self, data: List[VitalSignsTimeSeries]) -> Dict[str, Any]:
        if not data:
            return {"error": "No data provided"}
        
        df = TimeSeriesProcessor.convert_to_dataframe(data)
        patient_id = data[0].patient_id
        
        risk_predictions = {}
        
        for risk_type, model_func in self.risk_models.items():
            try:
                risk_score = model_func(df)
                risk_predictions[risk_type] = {
                    'risk_score': risk_score,
                    'risk_level': self._score_to_level(risk_score),
                    'recommendations': self._get_recommendations(risk_type, risk_score)
                }
            except Exception as e:
                risk_predictions[risk_type] = {'error': str(e)}
        
        return {
            'patient_id': patient_id,
            'prediction_time': datetime.now(),
            'risk_predictions': risk_predictions,
            'overall_risk': self._calculate_overall_risk(risk_predictions)
        }
    
    def _cardiovascular_risk_model(self, df: pd.DataFrame) -> float:
        risk_score = 0.0
        
        if 'heart_rate' in df.columns:
            hr_mean = df['heart_rate'].mean()
            if hr_mean > 100:
                risk_score += 0.3
            elif hr_mean < 50:
                risk_score += 0.2
        
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            sys_mean = df['systolic_bp'].mean()
            dia_mean = df['diastolic_bp'].mean()
            
            if sys_mean > 140 or dia_mean > 90:
                risk_score += 0.4
            elif sys_mean < 90:
                risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    def _respiratory_risk_model(self, df: pd.DataFrame) -> float:
        risk_score = 0.0
        
        if 'oxygen_saturation' in df.columns:
            o2_mean = df['oxygen_saturation'].mean()
            if o2_mean < 90:
                risk_score += 0.6
            elif o2_mean < 95:
                risk_score += 0.3
        
        if 'respiratory_rate' in df.columns:
            rr_mean = df['respiratory_rate'].mean()
            if rr_mean > 24 or rr_mean < 8:
                risk_score += 0.3
        
        return min(risk_score, 1.0)
    
    def _infection_risk_model(self, df: pd.DataFrame) -> float:
        risk_score = 0.0
        
        if 'temperature' in df.columns:
            temp_max = df['temperature'].max()
            temp_mean = df['temperature'].mean()
            
            if temp_max > 38.5:
                risk_score += 0.5
            elif temp_mean > 37.5:
                risk_score += 0.3
        
        if 'heart_rate' in df.columns and 'temperature' in df.columns:
            hr_temp_corr = df['heart_rate'].corr(df['temperature'])
            if hr_temp_corr > 0.5:
                risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    def _score_to_level(self, score: float) -> str:
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        elif score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _get_recommendations(self, risk_type: str, score: float) -> List[str]:
        level = self._score_to_level(score)
        
        recommendations = {
            'cardiovascular': {
                'high': ['Immediate cardiology consultation', 'Continuous cardiac monitoring', 'Consider ECG'],
                'medium': ['Monitor blood pressure regularly', 'Consider cardiac evaluation', 'Lifestyle counseling'],
                'low': ['Regular vital sign monitoring', 'Heart-healthy lifestyle advice'],
                'minimal': ['Continue routine monitoring']
            },
            'respiratory': {
                'high': ['Immediate respiratory assessment', 'Consider oxygen therapy', 'Chest X-ray evaluation'],
                'medium': ['Increased respiratory monitoring', 'Pulmonary function assessment'],
                'low': ['Regular oxygen saturation checks', 'Monitor breathing patterns'],
                'minimal': ['Continue routine monitoring']
            },
            'infection': {
                'high': ['Blood cultures and lab work', 'Consider antibiotic therapy', 'Isolation precautions'],
                'medium': ['Monitor temperature closely', 'Complete blood count', 'Assess for infection source'],
                'low': ['Regular temperature monitoring', 'Watch for infection signs'],
                'minimal': ['Continue routine monitoring']
            }
        }
        
        return recommendations.get(risk_type, {}).get(level, ['Continue monitoring'])
    
    def _calculate_overall_risk(self, risk_predictions: Dict[str, Any]) -> str:
        risk_levels = {'minimal': 0, 'low': 1, 'medium': 2, 'high': 3}
        
        total_score = 0
        valid_predictions = 0
        
        for prediction in risk_predictions.values():
            if 'risk_level' in prediction:
                total_score += risk_levels.get(prediction['risk_level'], 0)
                valid_predictions += 1
        
        if valid_predictions == 0:
            return 'unknown'
        
        avg_score = total_score / valid_predictions
        
        if avg_score >= 2.5:
            return 'high'
        elif avg_score >= 1.5:
            return 'medium'
        elif avg_score >= 0.5:
            return 'low'
        else:
            return 'minimal'


class ForecastingEngine:
    
    def __init__(self):
        self.vital_signs_forecaster = VitalSignsForecaster()
        self.risk_predictor = RiskPredictor()
        self.forecasting_sessions = []
    
    def comprehensive_forecast(self, data: List[VitalSignsTimeSeries], 
                             forecast_hours: int = 24) -> Dict[str, Any]:
        if not data:
            return {"error": "No data provided"}
        
        patient_id = data[0].patient_id
        
        vital_forecasts = self.vital_signs_forecaster.forecast_vital_signs(data, forecast_hours)
        risk_predictions = self.risk_predictor.predict_risks(data)
        
        comprehensive_result = {
            'patient_id': patient_id,
            'analysis_timestamp': datetime.now(),
            'forecast_horizon_hours': forecast_hours,
            'vital_signs_forecast': vital_forecasts,
            'risk_predictions': risk_predictions,
            'recommendations': self._generate_comprehensive_recommendations(vital_forecasts, risk_predictions)
        }
        
        self.forecasting_sessions.append(comprehensive_result)
        return comprehensive_result
    
    def _generate_comprehensive_recommendations(self, vital_forecasts: Dict[str, Any], 
                                             risk_predictions: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        if 'risk_alerts' in vital_forecasts:
            critical_alerts = [alert for alert in vital_forecasts['risk_alerts'] 
                             if alert.get('severity') == 'critical']
            if critical_alerts:
                recommendations.append("URGENT: Critical vital sign deviations predicted - immediate medical attention required")
        
        overall_vital_risk = vital_forecasts.get('overall_risk', 'low')
        overall_health_risk = risk_predictions.get('overall_risk', 'minimal')
        
        if overall_vital_risk == 'critical' or overall_health_risk == 'high':
            recommendations.append("High-risk patient: Implement intensive monitoring protocol")
        elif overall_vital_risk == 'high' or overall_health_risk == 'medium':
            recommendations.append("Moderate-risk patient: Increase monitoring frequency")
        
        if 'risk_predictions' in risk_predictions:
            for risk_type, prediction in risk_predictions['risk_predictions'].items():
                if 'recommendations' in prediction:
                    recommendations.extend([f"{risk_type.title()}: {rec}" for rec in prediction['recommendations'][:2]])
        
        if not recommendations:
            recommendations.append("Patient appears stable - continue routine monitoring")
        
        return recommendations
    
    def get_forecasting_summary(self) -> Dict[str, Any]:
        if not self.forecasting_sessions:
            return {"message": "No forecasting sessions performed"}
        
        total_sessions = len(self.forecasting_sessions)
        patients_analyzed = len(set(session['patient_id'] for session in self.forecasting_sessions))
        
        risk_distribution = {}
        for session in self.forecasting_sessions:
            risk = session.get('risk_predictions', {}).get('overall_risk', 'unknown')
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        return {
            'total_forecasting_sessions': total_sessions,
            'unique_patients_analyzed': patients_analyzed,
            'risk_level_distribution': risk_distribution,
            'latest_session_time': self.forecasting_sessions[-1]['analysis_timestamp'] if self.forecasting_sessions else None
        }
