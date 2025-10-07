from .base_models import (
    TimeSeriesData,
    HealthTimeSeries,
    VitalSignsTimeSeries,
    TimeSeriesAnalyzer,
    TimeSeriesResult
)

from .analyzers import (
    TrendAnalyzer,
    SeasonalAnalyzer,
    HealthTrendAnalyzer,
    TimeSeriesAnalyzerFactory
)

from .forecasters import (
    HealthForecaster,
    VitalSignsForecaster,
    RiskPredictor,
    ForecastingEngine
)

from .detectors import (
    AnomalyDetector,
    HealthAnomalyDetector,
    PatternDetector,
    AlertSystem
)

from .generators import (
    TimeSeriesGenerator,
    HealthDataGenerator,
    PatientTimelineGenerator
)

__version__ = "1.0.0"

def create_health_analyzer() -> HealthTrendAnalyzer:
    """Create a pre-configured health trend analyzer."""
    return HealthTrendAnalyzer()

def create_forecasting_engine() -> ForecastingEngine:
    """Create a comprehensive forecasting engine."""
    return ForecastingEngine()

def create_anomaly_detector() -> HealthAnomalyDetector:
    """Create a health-focused anomaly detector."""
    return HealthAnomalyDetector()

__all__ = [
    'TimeSeriesData',
    'HealthTimeSeries', 
    'VitalSignsTimeSeries',
    'TimeSeriesAnalyzer',
    'TimeSeriesResult',
    'TrendAnalyzer',
    'SeasonalAnalyzer',
    'HealthTrendAnalyzer',
    'TimeSeriesAnalyzerFactory',
    'HealthForecaster',
    'VitalSignsForecaster',
    'RiskPredictor',
    'ForecastingEngine',
    'AnomalyDetector',
    'HealthAnomalyDetector',
    'PatternDetector',
    'AlertSystem',
    'TimeSeriesGenerator',
    'HealthDataGenerator',
    'PatientTimelineGenerator',
    'create_health_analyzer',
    'create_forecasting_engine',
    'create_anomaly_detector'
]
