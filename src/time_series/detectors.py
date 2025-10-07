from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_models import (
    TimeSeriesData,
    VitalSignsTimeSeries,
    TimeSeriesProcessor
)

class AnomalyDetector:
    def __init__(self, name: str):
        self.name = name
        self.detection_history: List[Dict[str, Any]] = []
        self.sensitivity = 2.0

    def detect_statistical_anomalies(self, series: pd.Series) -> List[Dict[str, Any]]:
        if len(series) < 3:
            return []
        anomalies = []
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return []
        z_scores = np.abs((series - mean_val) / std_val)
        outlier_mask = z_scores > self.sensitivity
        for idx in series[outlier_mask].index:
            anomalies.append({
                'index': idx,
                'timestamp': series.index[idx] if hasattr(series.index[idx], 'timestamp') else None,
                'value': float(series.iloc[idx]),
                'z_score': float(z_scores.iloc[idx]),
                'deviation_from_mean': float(series.iloc[idx] - mean_val),
                'type': 'statistical_outlier'
            })
        return anomalies

    def detect_trend_anomalies(self, series: pd.Series, window: int = 5) -> List[Dict[str, Any]]:
        if len(series) < window + 2:
            return []
        anomalies = []
        trends = []
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            slope = TimeSeriesProcessor._calculate_trend_slope(window_data)
            trends.append(slope)
        if not trends:
            return []
        trends_series = pd.Series(trends)
        trend_mean = trends_series.mean()
        trend_std = trends_series.std()
        if trend_std == 0:
            return []
        for i, trend in enumerate(trends):
            z_score = abs(trend - trend_mean) / trend_std
            if z_score > self.sensitivity:
                actual_idx = i + window
                anomalies.append({
                    'index': actual_idx,
                    'timestamp': series.index[actual_idx] if hasattr(series.index[actual_idx], 'timestamp') else None,
                    'value': float(series.iloc[actual_idx]),
                    'trend_change': float(trend),
                    'trend_z_score': float(z_score),
                    'type': 'trend_anomaly'
                })
        return anomalies

class HealthAnomalyDetector(AnomalyDetector):
    def __init__(self):
        super().__init__("HealthAnomalyDetector")
        self.vital_sign_thresholds = {
            'heart_rate': {'min': 40, 'max': 180, 'critical_min': 30, 'critical_max': 200},
            'systolic_bp': {'min': 80, 'max': 160, 'critical_min': 70, 'critical_max': 220},
            'diastolic_bp': {'min': 50, 'max': 100, 'critical_min': 40, 'critical_max': 130},
            'temperature': {'min': 35.0, 'max': 38.0, 'critical_min': 32.0, 'critical_max': 42.0},
            'oxygen_saturation': {'min': 92, 'max': 100, 'critical_min': 80, 'critical_max': 100},
            'respiratory_rate': {'min': 10, 'max': 25, 'critical_min': 6, 'critical_max': 40}
        }

    def detect_health_anomalies(self, data: List[VitalSignsTimeSeries]) -> Dict[str, Any]:
        if not data:
            return {"error": "No data provided"}
        df = TimeSeriesProcessor.convert_to_dataframe(data)
        patient_id = data[0].patient_id
        all_anomalies = []
        anomaly_summary = {}
        vital_signs = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature',
                      'oxygen_saturation', 'respiratory_rate']
        for vital in vital_signs:
            if vital in df.columns:
                vital_anomalies = []
                threshold_anomalies = self._detect_threshold_anomalies(df, vital)
                vital_anomalies.extend(threshold_anomalies)
                statistical_anomalies = self.detect_statistical_anomalies(df[vital])
                for anomaly in statistical_anomalies:
                    anomaly['vital_sign'] = vital
                    anomaly['clinical_significance'] = self._assess_clinical_significance(vital, anomaly['value'])
                vital_anomalies.extend(statistical_anomalies)
                trend_anomalies = self.detect_trend_anomalies(df[vital])
                for anomaly in trend_anomalies:
                    anomaly['vital_sign'] = vital
                vital_anomalies.extend(trend_anomalies)
                all_anomalies.extend(vital_anomalies)
                anomaly_summary[vital] = len(vital_anomalies)
        multi_vital_anomalies = self._detect_multi_vital_anomalies(df)
        all_anomalies.extend(multi_vital_anomalies)
        severity_assessment = self._assess_overall_severity(all_anomalies)
        result = {
            'patient_id': patient_id,
            'detection_timestamp': datetime.now(),
            'total_anomalies': len(all_anomalies),
            'anomalies_by_vital': anomaly_summary,
            'all_anomalies': all_anomalies,
            'severity_assessment': severity_assessment,
            'recommendations': self._generate_recommendations(all_anomalies, severity_assessment)
        }
        self.detection_history.append(result)
        return result

    def _detect_threshold_anomalies(self, df: pd.DataFrame, vital: str) -> List[Dict[str, Any]]:
        if vital not in self.vital_sign_thresholds or vital not in df.columns:
            return []
        thresholds = self.vital_sign_thresholds[vital]
        anomalies = []
        for idx, row in df.iterrows():
            value = row[vital]
            timestamp = row.get('timestamp', None)
            severity = None
            if value <= thresholds['critical_min'] or value >= thresholds['critical_max']:
                severity = 'critical'
            elif value <= thresholds['min'] or value >= thresholds['max']:
                severity = 'warning'
            if severity:
                anomalies.append({
                    'index': idx,
                    'timestamp': timestamp,
                    'vital_sign': vital,
                    'value': float(value),
                    'threshold_type': 'clinical',
                    'severity': severity,
                    'normal_range': f"{thresholds['min']}-{thresholds['max']}",
                    'type': 'threshold_violation'
                })
        return anomalies

    def _detect_multi_vital_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        anomalies = []
        if 'heart_rate' in df.columns and 'temperature' in df.columns:
            for idx, row in df.iterrows():
                if row['heart_rate'] > 100 and row['temperature'] > 38.0:
                    anomalies.append({
                        'index': idx,
                        'timestamp': row.get('timestamp', None),
                        'pattern': 'infection_indicator',
                        'heart_rate': float(row['heart_rate']),
                        'temperature': float(row['temperature']),
                        'severity': 'high',
                        'type': 'multi_vital_pattern'
                    })
        if 'oxygen_saturation' in df.columns and 'respiratory_rate' in df.columns:
            for idx, row in df.iterrows():
                if row['oxygen_saturation'] < 94 and row['respiratory_rate'] > 22:
                    anomalies.append({
                        'index': idx,
                        'timestamp': row.get('timestamp', None),
                        'pattern': 'respiratory_distress',
                        'oxygen_saturation': float(row['oxygen_saturation']),
                        'respiratory_rate': float(row['respiratory_rate']),
                        'severity': 'critical',
                        'type': 'multi_vital_pattern'
                    })
        if 'systolic_bp' in df.columns and 'heart_rate' in df.columns:
            for idx, row in df.iterrows():
                if row['systolic_bp'] > 160 and row['heart_rate'] > 100:
                    anomalies.append({
                        'index': idx,
                        'timestamp': row.get('timestamp', None),
                        'pattern': 'cardiovascular_stress',
                        'systolic_bp': float(row['systolic_bp']),
                        'heart_rate': float(row['heart_rate']),
                        'severity': 'high',
                        'type': 'multi_vital_pattern'
                    })
        return anomalies

    def _assess_clinical_significance(self, vital: str, value: float) -> str:
        if vital not in self.vital_sign_thresholds:
            return 'unknown'
        thresholds = self.vital_sign_thresholds[vital]
        if value <= thresholds['critical_min'] or value >= thresholds['critical_max']:
            return 'critical'
        elif value <= thresholds['min'] or value >= thresholds['max']:
            return 'concerning'
        else:
            return 'mild'

    def _assess_overall_severity(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not anomalies:
            return {'level': 'none', 'score': 0, 'description': 'No anomalies detected'}
        severity_scores = {'mild': 1, 'warning': 2, 'concerning': 2, 'high': 3, 'critical': 4}
        total_score = 0
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get('severity', anomaly.get('clinical_significance', 'mild'))
            score = severity_scores.get(severity, 1)
            total_score += score
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        if total_score >= 10:
            level = 'critical'
        elif total_score >= 6:
            level = 'high'
        elif total_score >= 3:
            level = 'medium'
        else:
            level = 'low'
        return {
            'level': level,
            'score': total_score,
            'severity_counts': severity_counts,
            'description': f"Overall severity: {level} (score: {total_score})"
        }

    def _generate_recommendations(self, anomalies: List[Dict[str, Any]],
                                severity: Dict[str, Any]) -> List[str]:
        recommendations = []
        severity_level = severity.get('level', 'low')
        if severity_level == 'critical':
            recommendations.append("IMMEDIATE MEDICAL ATTENTION REQUIRED")
            recommendations.append("Implement emergency protocols")
            recommendations.append("Continuous monitoring essential")
        elif severity_level == 'high':
            recommendations.append("Urgent medical evaluation needed")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider immediate interventions")
        elif severity_level == 'medium':
            recommendations.append("Medical assessment recommended")
            recommendations.append("Enhanced monitoring advised")
        else:
            recommendations.append("Continue routine monitoring")
            recommendations.append("Document anomalies for trend analysis")
        patterns_seen = set()
        for anomaly in anomalies:
            if anomaly.get('type') == 'multi_vital_pattern':
                pattern = anomaly.get('pattern', '')
                if pattern and pattern not in patterns_seen:
                    patterns_seen.add(pattern)
                    if pattern == 'infection_indicator':
                        recommendations.append("Consider infection workup (blood cultures, CBC)")
                    elif pattern == 'respiratory_distress':
                        recommendations.append("Assess respiratory function, consider oxygen therapy")
                    elif pattern == 'cardiovascular_stress':
                        recommendations.append("Cardiac evaluation recommended, monitor for arrhythmias")
        return recommendations

class PatternDetector:
    def __init__(self):
        self.name = "PatternDetector"
        self.detected_patterns = []

    def detect_daily_patterns(self, data: List[VitalSignsTimeSeries]) -> Dict[str, Any]:
        if not data:
            return {"error": "No data provided"}
        df = TimeSeriesProcessor.convert_to_dataframe(data)
        df['hour'] = df['timestamp'].dt.hour
        patterns = {}
        vital_signs = ['heart_rate', 'systolic_bp', 'temperature', 'oxygen_saturation']
        for vital in vital_signs:
            if vital in df.columns:
                hourly_avg = df.groupby('hour')[vital].mean()
                peak_hour = hourly_avg.idxmax()
                trough_hour = hourly_avg.idxmin()
                variation = (hourly_avg.max() - hourly_avg.min()) / hourly_avg.mean() * 100
                patterns[vital] = {
                    'peak_hour': int(peak_hour),
                    'trough_hour': int(trough_hour),
                    'peak_value': float(hourly_avg.max()),
                    'trough_value': float(hourly_avg.min()),
                    'daily_variation_percent': float(variation),
                    'pattern_strength': 'strong' if variation > 15 else 'moderate' if variation > 5 else 'weak'
                }
        return {
            'patient_id': data[0].patient_id,
            'analysis_period': f"{data[0].timestamp} to {data[-1].timestamp}",
            'daily_patterns': patterns,
            'pattern_summary': self._summarize_daily_patterns(patterns)
        }

    def _summarize_daily_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        strong_patterns = []
        concerning_patterns = []
        for vital, pattern in patterns.items():
            if pattern['pattern_strength'] == 'strong':
                strong_patterns.append(vital)
            if vital == 'temperature' and pattern['peak_value'] > 38.0:
                concerning_patterns.append(f"Temperature peaks at {pattern['peak_hour']}:00 ({pattern['peak_value']:.1f}°C)")
            elif vital == 'heart_rate' and pattern['peak_value'] > 120:
                concerning_patterns.append(f"Heart rate peaks at {pattern['peak_hour']}:00 ({pattern['peak_value']:.0f} bpm)")
        return {
            'strong_patterns_detected': len(strong_patterns),
            'strong_pattern_vitals': strong_patterns,
            'concerning_patterns': concerning_patterns,
            'overall_pattern_strength': 'high' if len(strong_patterns) >= 2 else 'moderate' if len(strong_patterns) == 1 else 'low'
        }

class AlertSystem:
    def __init__(self):
        self.name = "HealthAlertSystem"
        self.active_alerts = []
        self.alert_history = []
        self.alert_rules = self._initialize_alert_rules()

    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        return {
            'critical_vitals': {
                'heart_rate': {'min': 30, 'max': 200},
                'systolic_bp': {'min': 70, 'max': 220},
                'temperature': {'min': 32.0, 'max': 42.0},
                'oxygen_saturation': {'min': 80, 'max': 100}
            },
            'deterioration_patterns': {
                'rapid_hr_increase': {'threshold': 30, 'timeframe_minutes': 60},
                'temperature_spike': {'threshold': 2.0, 'timeframe_minutes': 120},
                'oxygen_drop': {'threshold': 5, 'timeframe_minutes': 30}
            }
        }

    def process_realtime_data(self, new_data: VitalSignsTimeSeries) -> List[Dict[str, Any]]:
        alerts = []
        critical_alerts = self._check_critical_thresholds(new_data)
        alerts.extend(critical_alerts)
        if len(self.alert_history) > 0:
            deterioration_alerts = self._check_deterioration_patterns(new_data)
            alerts.extend(deterioration_alerts)
        for alert in alerts:
            alert['alert_id'] = len(self.alert_history) + 1
            alert['generated_at'] = datetime.now()
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
        return alerts

    def _check_critical_thresholds(self, data: VitalSignsTimeSeries) -> List[Dict[str, Any]]:
        alerts = []
        rules = self.alert_rules['critical_vitals']
        vitals_to_check = {
            'heart_rate': data.heart_rate,
            'systolic_bp': data.systolic_bp,
            'temperature': data.temperature,
            'oxygen_saturation': data.oxygen_saturation
        }
        for vital, value in vitals_to_check.items():
            if vital in rules:
                thresholds = rules[vital]
                if value <= thresholds['min'] or value >= thresholds['max']:
                    alerts.append({
                        'type': 'critical_threshold',
                        'patient_id': data.patient_id,
                        'vital_sign': vital,
                        'value': value,
                        'threshold_range': f"{thresholds['min']}-{thresholds['max']}",
                        'severity': 'critical',
                        'message': f"CRITICAL: {vital} = {value} (normal: {thresholds['min']}-{thresholds['max']})",
                        'timestamp': data.timestamp
                    })
        return alerts

    def _check_deterioration_patterns(self, current_data: VitalSignsTimeSeries) -> List[Dict[str, Any]]:
        alerts = []
        recent_cutoff = current_data.timestamp - timedelta(hours=2)
        recent_alerts = [alert for alert in self.alert_history
                        if alert.get('timestamp', datetime.min) > recent_cutoff]
        if not recent_alerts:
            return alerts
        hr_alerts = [alert for alert in recent_alerts
                    if alert.get('vital_sign') == 'heart_rate']
        if hr_alerts:
            last_hr = hr_alerts[-1].get('value', current_data.heart_rate)
            hr_change = current_data.heart_rate - last_hr
            if hr_change > 30:
                alerts.append({
                    'type': 'rapid_deterioration',
                    'patient_id': current_data.patient_id,
                    'pattern': 'rapid_hr_increase',
                    'change': hr_change,
                    'current_value': current_data.heart_rate,
                    'previous_value': last_hr,
                    'severity': 'high',
                    'message': f"Rapid heart rate increase: {hr_change:.0f} bpm in recent period",
                    'timestamp': current_data.timestamp
                })
        return alerts

    def get_active_alerts(self, patient_id: str = None) -> List[Dict[str, Any]]:
        if patient_id:
            return [alert for alert in self.active_alerts
                   if alert.get('patient_id') == patient_id]
        return self.active_alerts.copy()

    def acknowledge_alert(self, alert_id: int) -> bool:
        for i, alert in enumerate(self.active_alerts):
            if alert.get('alert_id') == alert_id:
                alert['acknowledged_at'] = datetime.now()
                self.active_alerts.pop(i)
                return True
        return False

    def get_alert_summary(self) -> Dict[str, Any]:
        return {
            'total_alerts_generated': len(self.alert_history),
            'active_alerts_count': len(self.active_alerts),
            'alert_types': list(set(alert.get('type', 'unknown') for alert in self.alert_history)),
            'patients_with_active_alerts': list(set(alert.get('patient_id') for alert in self.active_alerts)),
            'last_alert_time': self.alert_history[-1].get('generated_at') if self.alert_history else None
        }
