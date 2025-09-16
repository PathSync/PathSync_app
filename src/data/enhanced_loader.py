"""
Implements Repository Pattern and Builder Pattern for flexible data operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import json

from .base_models import BiometricDataModel, TriageDataModel, BiometricData, TriageData
from .validators import DataValidator, ValidationResult
from .generators import DataGeneratorFactory


class DataLoadingConfig:
    
    def __init__(self):
        self.validate_data: bool = True
        self.generate_missing: bool = False
        self.min_records: int = 100
        self.max_invalid_ratio: float = 0.1
        self.encoding: str = 'utf-8'
        self.date_format: str = '%Y-%m-%d %H:%M:%S'
        self.chunk_size: Optional[int] = None
        self.logger: Optional[logging.Logger] = None
    
    def with_validation(self, enabled: bool = True) -> 'DataLoadingConfig':
        """Enable or disable data validation."""
        self.validate_data = enabled
        return self
    
    def with_generation(self, enabled: bool = True, min_records: int = 100) -> 'DataLoadingConfig':
        """Enable automatic data generation for missing records."""
        self.generate_missing = enabled
        self.min_records = min_records
        return self
    
    def with_quality_threshold(self, max_invalid_ratio: float = 0.1) -> 'DataLoadingConfig':
        """Set maximum allowed ratio of invalid records."""
        self.max_invalid_ratio = max_invalid_ratio
        return self
    
    def with_encoding(self, encoding: str = 'utf-8') -> 'DataLoadingConfig':
        """Set file encoding."""
        self.encoding = encoding
        return self
    
    def with_chunking(self, chunk_size: int) -> 'DataLoadingConfig':
        """Enable chunked loading for large files."""
        self.chunk_size = chunk_size
        return self
    
    def with_logger(self, logger: logging.Logger) -> 'DataLoadingConfig':
        """Set custom logger."""
        self.logger = logger
        return self


class DataRepository:
    """
    Repository class for data persistence and retrieval.
    Implements Repository Pattern for data access abstraction.
    """
    
    def __init__(self, base_path: Union[str, Path] = "Data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        (self.base_path / "biometric").mkdir(exist_ok=True)
        (self.base_path / "triage").mkdir(exist_ok=True)
        (self.base_path / "generated").mkdir(exist_ok=True)
        (self.base_path / "validated").mkdir(exist_ok=True)
    
    def save_biometric_data(self, data: List[BiometricDataModel], filename: str = None) -> Path:
        """Save biometric data to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"biometric_data_{timestamp}.csv"
        
        filepath = self.base_path / "biometric" / filename
        
        df_data = [model.to_dict() for model in data]
        df = pd.DataFrame(df_data)
        
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_triage_data(self, data: List[TriageDataModel], filename: str = None) -> Path:
        """Save triage data to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"triage_data_{timestamp}.csv"
        
        filepath = self.base_path / "triage" / filename
        
        df_data = [model.to_dict() for model in data]
        df = pd.DataFrame(df_data)
        
        df.to_csv(filepath, index=False)
        return filepath
    
    def load_csv_to_dataframe(self, filepath: Union[str, Path], config: DataLoadingConfig) -> pd.DataFrame:
        """Load CSV file to DataFrame with configuration options."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        read_kwargs = {
            'encoding': config.encoding,
            'parse_dates': ['timestamp'] if 'timestamp' in pd.read_csv(filepath, nrows=1).columns else False
        }
        
        if config.chunk_size:
            read_kwargs['chunksize'] = config.chunk_size
        
        try:
            if config.chunk_size:
                chunks = []
                for chunk in pd.read_csv(filepath, **read_kwargs):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(filepath, **read_kwargs)
            
            if config.logger:
                config.logger.info(f"Successfully loaded {len(df)} records from {filepath}")
            
            return df
            
        except Exception as e:
            if config.logger:
                config.logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def save_validation_report(self, results: Dict[str, ValidationResult], 
                              report_name: str = None) -> Path:
        """Save validation results to JSON report."""
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"validation_report_{timestamp}.json"
        
        filepath = self.base_path / "validated" / report_name
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(results),
            'results': {key: result.get_summary() for key, result in results.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath


class EnhancedDataLoader:
    """
    Enhanced data loader with comprehensive data management capabilities.
    Implements Facade Pattern to provide simple interface for complex operations.
    """
    
    def __init__(self, repository: DataRepository = None, config: DataLoadingConfig = None):
        self.repository = repository or DataRepository()
        self.config = config or DataLoadingConfig()
        self.validator = DataValidator()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for data operations."""
        logger = logging.getLogger('enhanced_data_loader')
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def load_biometric_data(self, filepath: Union[str, Path] = None) -> List[BiometricDataModel]:
        """Load and process biometric data with validation and generation."""
        if filepath is None:
            filepath = self.repository.base_path / "sample_biometric.csv"
        
        try:
            df = self.repository.load_csv_to_dataframe(filepath, self.config)
            self.logger.info(f"Loaded {len(df)} biometric records from {filepath}")
            
            data_models = self._dataframe_to_biometric_models(df)
            
            if self.config.validate_data:
                data_models = self._validate_and_filter_biometric(data_models)
            
            if self.config.generate_missing and len(data_models) < self.config.min_records:
                additional_count = self.config.min_records - len(data_models)
                self.logger.info(f"Generating {additional_count} additional biometric records")
                
                generator = DataGeneratorFactory.create_biometric_generator()
                generated_data = generator.generate_batch(additional_count)
                data_models.extend(generated_data)
            
            self.logger.info(f"Final dataset contains {len(data_models)} biometric records")
            return data_models
            
        except Exception as e:
            self.logger.error(f"Error loading biometric data: {str(e)}")
            raise
    
    def load_triage_data(self, filepath: Union[str, Path] = None) -> List[TriageDataModel]:
        """Load and process triage data with validation and generation."""
        if filepath is None:
            filepath = self.repository.base_path / "sample_triage.csv"
        
        try:
            df = self.repository.load_csv_to_dataframe(filepath, self.config)
            self.logger.info(f"Loaded {len(df)} triage records from {filepath}")
            
            data_models = self._dataframe_to_triage_models(df)
            
            if self.config.validate_data:
                data_models = self._validate_and_filter_triage(data_models)
            
            if self.config.generate_missing and len(data_models) < self.config.min_records:
                additional_count = self.config.min_records - len(data_models)
                self.logger.info(f"Generating {additional_count} additional triage records")
                
                generator = DataGeneratorFactory.create_triage_generator()
                generated_data = generator.generate_batch(additional_count)
                data_models.extend(generated_data)
            
            self.logger.info(f"Final dataset contains {len(data_models)} triage records")
            return data_models
            
        except Exception as e:
            self.logger.error(f"Error loading triage data: {str(e)}")
            raise
    
    def _dataframe_to_biometric_models(self, df: pd.DataFrame) -> List[BiometricDataModel]:
        """Convert DataFrame to BiometricDataModel objects."""
        models = []
        
        for _, row in df.iterrows():
            try:
                timestamp = None
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    if isinstance(row['timestamp'], str):
                        timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                    else:
                        timestamp = row['timestamp']
                
                data = BiometricData(
                    patient_id=str(row['patient_id']),
                    age=int(row['age']),
                    gender=str(row['gender']),
                    province=str(row['province']),
                    national_id=str(row['national_id']),
                    biometric_score=float(row['biometric_score']),
                    citizenship_status=str(row['citizenship_status']),
                    timestamp=timestamp or datetime.now()
                )
                
                models.append(BiometricDataModel(data))
                
            except Exception as e:
                self.logger.warning(f"Skipping invalid biometric record: {str(e)}")
                continue
        
        return models
    
    def _dataframe_to_triage_models(self, df: pd.DataFrame) -> List[TriageDataModel]:
        """Convert DataFrame to TriageDataModel objects."""
        models = []
        
        for _, row in df.iterrows():
            try:
                timestamp = None
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    if isinstance(row['timestamp'], str):
                        timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                    else:
                        timestamp = row['timestamp']
                
                symptoms = []
                if 'symptoms' in row and pd.notna(row['symptoms']) and row['symptoms']:
                    symptoms = str(row['symptoms']).split(',')
                
                data = TriageData(
                    patient_id=str(row['patient_id']),
                    age=int(row['age']),
                    gender=str(row['gender']),
                    hr_bpm=int(row['hr_bpm']),
                    temp_c=float(row['temp_c']),
                    resp_rate=int(row['resp_rate']),
                    systolic_bp=int(row['systolic_bp']),
                    diastolic_bp=int(row['diastolic_bp']),
                    o2_sat=int(row['o2_sat']),
                    pain_score=int(row['pain_score']),
                    triage_priority=str(row['triage_priority']),
                    timestamp=timestamp or datetime.now(),
                    symptoms=symptoms
                )
                
                models.append(TriageDataModel(data))
                
            except Exception as e:
                self.logger.warning(f"Skipping invalid triage record: {str(e)}")
                continue
        
        return models
    
    def _validate_and_filter_biometric(self, data_models: List[BiometricDataModel]) -> List[BiometricDataModel]:
        """Validate biometric data and filter out invalid records."""
        validation_results = self.validator.validate_batch_biometric(data_models)
        
        valid_models = []
        invalid_count = 0
        
        for i, model in enumerate(data_models):
            result = validation_results[f"record_{i}"]
            if result.is_valid:
                valid_models.append(model)
            else:
                invalid_count += 1
                self.logger.debug(f"Invalid biometric record {i}: {result.errors}")
        
        invalid_ratio = invalid_count / len(data_models) if data_models else 0
        if invalid_ratio > self.config.max_invalid_ratio:
            self.logger.warning(
                f"High invalid data ratio: {invalid_ratio:.2%} "
                f"(threshold: {self.config.max_invalid_ratio:.2%})"
            )
        
        self.repository.save_validation_report(validation_results, "biometric_validation.json")
        
        self.logger.info(f"Validation complete: {len(valid_models)} valid, {invalid_count} invalid")
        return valid_models
    
    def _validate_and_filter_triage(self, data_models: List[TriageDataModel]) -> List[TriageDataModel]:
        """Validate triage data and filter out invalid records."""
        validation_results = self.validator.validate_batch_triage(data_models)
        
        valid_models = []
        invalid_count = 0
        
        for i, model in enumerate(data_models):
            result = validation_results[f"record_{i}"]
            if result.is_valid:
                valid_models.append(model)
            else:
                invalid_count += 1
                self.logger.debug(f"Invalid triage record {i}: {result.errors}")
        
        invalid_ratio = invalid_count / len(data_models) if data_models else 0
        if invalid_ratio > self.config.max_invalid_ratio:
            self.logger.warning(
                f"High invalid data ratio: {invalid_ratio:.2%} "
                f"(threshold: {self.config.max_invalid_ratio:.2%})"
            )
        
        self.repository.save_validation_report(validation_results, "triage_validation.json")
        
        self.logger.info(f"Validation complete: {len(valid_models)} valid, {invalid_count} invalid")
        return valid_models
    
    def generate_and_save_dataset(self, data_type: str, count: int, filename: str = None) -> Path:
        """Generate and save a complete dataset."""
        if data_type == 'biometric':
            generator = DataGeneratorFactory.create_biometric_generator()
            data = generator.generate_batch(count)
            return self.repository.save_biometric_data(data, filename)
        elif data_type == 'triage':
            generator = DataGeneratorFactory.create_triage_generator()
            data = generator.generate_batch(count)
            return self.repository.save_triage_data(data, filename)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def get_data_summary(self, data_type: str, data_models: List) -> Dict[str, Any]:
        """Get summary statistics for the dataset."""
        if not data_models:
            return {'total_records': 0}
        
        summary = {
            'total_records': len(data_models),
            'timestamp_range': {
                'earliest': min(model.data.timestamp for model in data_models).isoformat(),
                'latest': max(model.data.timestamp for model in data_models).isoformat()
            }
        }
        
        if data_type == 'biometric':
            scores = [model.data.biometric_score for model in data_models]
            summary.update({
                'biometric_score_stats': {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                },
                'citizenship_distribution': pd.Series([model.data.citizenship_status for model in data_models]).value_counts().to_dict(),
                'province_distribution': pd.Series([model.data.province for model in data_models]).value_counts().to_dict()
            })
        
        elif data_type == 'triage':
            priorities = [model.data.triage_priority for model in data_models]
            pain_scores = [model.data.pain_score for model in data_models]
            summary.update({
                'priority_distribution': pd.Series(priorities).value_counts().to_dict(),
                'pain_score_stats': {
                    'mean': np.mean(pain_scores),
                    'std': np.std(pain_scores),
                    'min': np.min(pain_scores),
                    'max': np.max(pain_scores)
                }
            })
        
        return summary
