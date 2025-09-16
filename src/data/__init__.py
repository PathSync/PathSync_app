"""
Enhanced Data Management Package for Healthcare AI System.

This package provides a comprehensive data management solution following
software engineering best practices including OOP, SOLID principles,
and design patterns.

Main Components:
- base_models: Core data models with validation
- generators: Synthetic data generation with realistic patterns
- validators: Comprehensive data validation system
- enhanced_loader: Advanced data loading and processing
"""

from .base_models import (
    PatientData,
    BiometricData,
    TriageData,
    BiometricDataModel,
    TriageDataModel,
    DataModel
)

from .generators import (
    BiometricDataGenerator,
    TriageDataGenerator,
    DataGeneratorFactory
)

from .validators import (
    ValidationRule,
    ValidationResult,
    DataValidator,
    BiometricScoreRule,
    NationalIdRule,
    ProvinceRule,
    VitalSignsRule,
    TriagePriorityConsistencyRule,
    DataTimelinessRule
)

from .enhanced_loader import (
    DataLoadingConfig,
    DataRepository,
    EnhancedDataLoader
)

__version__ = "1.0.0"
__author__ = "Healthcare AI Development Team"

# Convenience function for quick setup
def create_data_loader(validate: bool = True, generate_missing: bool = True, 
                      min_records: int = 100000) -> EnhancedDataLoader:
    """
    Create a pre-configured data loader with common settings.
    
    Args:
        validate: Enable data validation
        generate_missing: Generate additional data if below minimum
        min_records: Minimum number of records to maintain
    
    Returns:
        Configured EnhancedDataLoader instance
    """
    config = (DataLoadingConfig()
              .with_validation(validate)
              .with_generation(generate_missing, min_records)
              .with_quality_threshold(0.1))
    
    return EnhancedDataLoader(config=config)


# Export main classes for easy importing
__all__ = [
    # Base Models
    'PatientData',
    'BiometricData',
    'TriageData',
    'BiometricDataModel',
    'TriageDataModel',
    'DataModel',
    
    # Generators
    'BiometricDataGenerator',
    'TriageDataGenerator',
    'DataGeneratorFactory',
    
    # Validators
    'ValidationRule',
    'ValidationResult',
    'DataValidator',
    'BiometricScoreRule',
    'NationalIdRule',
    'ProvinceRule',
    'VitalSignsRule',
    'TriagePriorityConsistencyRule',
    'DataTimelinessRule',
    
    # Enhanced Loader
    'DataLoadingConfig',
    'DataRepository',
    'EnhancedDataLoader',
    
    # Convenience function
    'create_data_loader'
]
