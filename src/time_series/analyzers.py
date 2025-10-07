from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_models import (
    TimeSeriesData, 
    VitalSignsTimeSeries, 
    HealthTimeSeries,
    TimeSeriesAnalyzer, 
    TimeSeriesResult,
    TimeSeriesProcessor
)


class TrendAnalyzer(TimeSeriesAnalyzer):
    
    def __init__(self):
        super().__init__("TrendAnalyzer")
        self.min_data_points = 5
    
    def validate_data(self, data: List[TimeSeriesData]) -> bool:
        if len(data) < self.min_data_points:
            return False
        patient_ids = {item.patient_id for item in data}
        if len(patient_ids) > 1:
            return False
        return True
    
    def analyze(self, data: List[TimeSeriesData]) -> TimeSeriesResult:
        if not self.validate_data(data):
            raise ValueError("Invalid data for trend analysis")
        patient_id = data[0].patient_id
        timestamps = [item.timestamp for item in data]
        time_range = (min(timestamps), max(timestamps))
        df = TimeSeriesProcessor.convert_to_dataframe(data)
        trends = {}
        metrics = {}
        anomalies = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['patient_id']]
        for col in numeric_columns:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) >= 2:
                    stats = TimeSeriesProcessor.calculate_basic_stats(series)
                    metrics[col] = stats
                    slope = stats.get('trend_slope', 0)
                    if abs(slope) < 0.1:
                        trends[col] = 'stable'
                    elif slope > 0:
                        trends[col] = 'increasing'
                    else:
                        trends[col] = 'decreasing'
                    outliers = TimeSeriesProcessor.detect_outliers(series)
                    for idx in outliers:
                        anomalies.append({
                            'timestamp': df.iloc[idx]['timestamp'],
                            'vital_sign': col,
                            'value': float(series.iloc[idx]),
                            'type': 'outlier',
                            'severity': 'medium'
                        })
        risk_level = self._assess_risk_level(trends, metrics)
        result = TimeSeriesResult(
            analysis_type="trend_analysis",
            patient_id=patient_id,
            time_range=time_range,
            metrics=metrics,
            trends=trends,
            anomalies=anomalies,
            risk_level=risk_level
        )
        self.analysis_history.append(result)
        return result
    
    def _assess_risk_level(self, trends: Dict[str, str], metrics: Dict[str, Any]) -> str:
        risk_indicators = 0
        concerning_trends = {
            'heart_rate': 'increasing',
            'systolic_bp': 'increasing', 
            'temperature': 'increasing',
            'pain_score': 'increasing'
        }
        for vital, concerning_trend in concerning_trends.items():
            if vital in trends and trends[vital] == concerning_trend:
                risk_indicators += 1
        for vital, stats in metrics.items():
            if vital == 'temperature' and stats.get('max', 0) > 38.5:
                risk_indicators += 2
            elif vital == 'heart_rate' and stats.get('max', 0) > 120:
                risk_indicators += 1
            elif vital == 'systolic_bp' and stats.get('max', 0) > 160:
                risk_indicators += 2
        if risk_indicators >= 4:
            return 'critical'
        elif risk_indicators >= 2:
            return 'high'
        elif risk_indicators >= 1:
            return 'medium'
        else:
            return 'low'


class SeasonalAnalyzer(TimeSeriesAnalyzer):
    
    def __init__(self):
        super().__init__("SeasonalAnalyzer")
        self.min_data_points = 10
    
    def validate_data(self, data: List[TimeSeriesData]) -> bool:
        if len(data) < self.min_data_points:
            return False
        timestamps = [item.timestamp for item in data]
        time_span = max(timestamps) - min(timestamps)
        if time_span.days < 3:
            return False
        return True
    
    def analyze(self, data: List[TimeSeriesData]) -> TimeSeriesResult:
        if not self.validate_data(data):
            raise ValueError("Invalid data for seasonal analysis")
        patient_id = data[0].patient_id
        timestamps = [item.timestamp for item in data]
        time_range = (min(timestamps), max(timestamps))
        df = TimeSeriesProcessor.convert_to_dataframe(data)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        patterns = {}
        metrics = {}
        anomalies = []
        trends = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['patient_id', 'hour', 'day_of_week']]
        for col in numeric_columns:
            if col in df.columns:
                hourly_avg = df.groupby('hour')[col].mean()
                daily_avg = df.groupby('day_of_week')[col].mean()
                patterns[col] = {
                    'hourly_pattern': hourly_avg.to_dict(),
                    'daily_pattern': daily_avg.to_dict(),
                    'peak_hour': int(hourly_avg.idxmax()) if not hourly_avg.empty else 0,
                    'peak_day': int(daily_avg.idxmax()) if not daily_avg.empty else 0
                }
                metrics[col] = TimeSeriesProcessor.calculate_basic_stats(df[col])
                trends[col] = 'seasonal_pattern_detected'
        result = TimeSeriesResult(
            analysis_type="seasonal_analysis",
            patient_id=patient_id,
            time_range=time_range,
            metrics=metrics,
            trends=trends,
            anomalies=anomalies,
            predictions={'patterns': patterns},
            risk_level='low'
        )
        self.analysis_history.append(result)
        return result


class HealthTrendAnalyzer(TimeSeriesAnalyzer):
    
    def __init__(self):
        super().__init__("HealthTrendAnalyzer")
        self.trend_analyzer = TrendAnalyzer()
        self.seasonal_analyzer = SeasonalAnalyzer()
        self.min_data_points = 5
    
    def validate_data(self, data: List[TimeSeriesData]) -> bool:
        if len(data) < self.min_data_points:
            return False
        patient_ids = {item.patient_id for item in data}
        if len(patient_ids) > 1:
            return False
        has_vital_signs = any(isinstance(item, VitalSignsTimeSeries) for item in data)
        if not has_vital_signs:
            return False
        return True
    
    def analyze(self, data: List[TimeSeriesData]) -> TimeSeriesResult:
        if not self.validate_data(data):
            raise ValueError("Invalid data for health trend analysis")
        patient_id = data[0].patient_id
        timestamps = [item.timestamp for item in data]
        time_range = (min(timestamps), max(timestamps))
        trend_result = self.trend_analyzer.analyze(data)
        df = TimeSeriesProcessor.convert_to_dataframe(data)
        health_metrics = self._calculate_health_metrics(df)
        risk_level = self._comprehensive_risk_assessment(trend_result, health_metrics)
        predictions = self._generate_health_predictions(df, trend_result)
        result = TimeSeriesResult(
            analysis_type="comprehensive_health_analysis",
            patient_id=patient_id,
            time_range=time_range,
            metrics={**trend_result.metrics, **health_metrics},
            trends=trend_result.trends,
            anomalies=trend_result.anomalies,
            predictions=predictions,
            risk_level=risk_level
        )
        self.analysis_history.append(result)
        return result
    
    def _calculate_health_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        metrics = {}
        if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
            df['cv_stress'] = (df['heart_rate'] / 100) + (df['systolic_bp'] / 200)
            metrics['cardiovascular_stress'] = TimeSeriesProcessor.calculate_basic_stats(df['cv_stress'])
        if 'temperature' in df.columns:
            fever_episodes = len(df[df['temperature'] > 38.0])
            metrics['fever_episodes'] = fever_episodes
        if 'pain_score' in df.columns:
            pain_increasing = df['pain_score'].diff().gt(0).sum()
            metrics['pain_trend_increases'] = int(pain_increasing)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['patient_id']]
        stability_scores = []
        for col in numeric_cols:
            if col in df.columns and len(df[col].dropna()) > 1:
                cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                stability_scores.append(cv)
        metrics['stability_score'] = np.mean(stability_scores) if stability_scores else 0
        return metrics
    
    def _comprehensive_risk_assessment(self, trend_result: TimeSeriesResult, health_metrics: Dict[str, Any]) -> str:
        risk_score = 0
        base_risks = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        risk_score += base_risks.get(trend_result.risk_level, 0)
        if health_metrics.get('fever_episodes', 0) > 2:
            risk_score += 2
        if health_metrics.get('stability_score', 0) > 0.3:
            risk_score += 1
        if health_metrics.get('pain_trend_increases', 0) > 3:
            risk_score += 1
        if risk_score >= 5:
            return 'critical'
        elif risk_score >= 3:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_health_predictions(self, df: pd.DataFrame, trend_result: TimeSeriesResult) -> Dict[str, Any]:
        predictions = {}
        for vital, trend in trend_result.trends.items():
            if vital in df.columns and trend != 'stable':
                current_value = df[vital].iloc[-1] if not df[vital].empty else 0
                if trend == 'increasing':
                    predicted_change = "likely to continue increasing"
                    if vital == 'temperature' and current_value > 37.5:
                        predicted_change += " - monitor for fever"
                    elif vital == 'heart_rate' and current_value > 100:
                        predicted_change += " - potential tachycardia risk"
                else:
                    predicted_change = "likely to continue decreasing"
                    if vital == 'oxygen_saturation' and current_value < 95:
                        predicted_change += " - oxygen support may be needed"
                predictions[vital] = predicted_change
        risk_level = trend_result.risk_level
        if risk_level in ['high', 'critical']:
            predictions['overall_health'] = "Patient requires close monitoring and possible intervention"
        elif risk_level == 'medium':
            predictions['overall_health'] = "Patient shows some concerning trends, regular monitoring recommended"
        else:
            predictions['overall_health'] = "Patient appears stable with normal vital sign patterns"
        return predictions


class TimeSeriesAnalyzerFactory:
    
    @staticmethod
    def create_analyzer(analyzer_type: str) -> TimeSeriesAnalyzer:
        analyzers = {
            'trend': TrendAnalyzer,
            'seasonal': SeasonalAnalyzer,
            'health': HealthTrendAnalyzer
        }
        if analyzer_type not in analyzers:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available: {list(analyzers.keys())}")
        return analyzers[analyzer_type]()
    
    @staticmethod
    def get_available_analyzers() -> List[str]:
        return ['trend', 'seasonal', 'health']
