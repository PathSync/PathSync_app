"""
Data validation system for healthcare data.
Implements Validator Pattern and Command Pattern for flexible validation rules.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .base_models import BiometricDataModel, TriageDataModel


class ValidationRule(ABC):
    """Abstract base class for validation rules following Strategy Pattern."""
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        Validate data against the rule.
        Returns: (is_valid, error_message)
        """
        pass
    
    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Name of the validation rule."""
        pass


class ValidationResult:
    """Encapsulates validation results with detailed feedback."""
    
    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed_rules: List[str] = []
        self.failed_rules: List[str] = []
    
    def add_error(self, rule_name: str, message: str):
        """Add a validation error."""
        self.is_valid = False
        self.errors.append(f"{rule_name}: {message}")
        self.failed_rules.append(rule_name)
    
    def add_warning(self, rule_name: str, message: str):
        """Add a validation warning."""
        self.warnings.append(f"{rule_name}: {message}")
    
    def add_passed_rule(self, rule_name: str):
        """Mark a rule as passed."""
        self.passed_rules.append(rule_name)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'is_valid': self.is_valid,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'passed_rules': len(self.passed_rules),
            'failed_rules': len(self.failed_rules),
            'errors': self.errors,
            'warnings': self.warnings
        }


# Biometric Validation Rules

class BiometricScoreRule(ValidationRule):
    """Validates biometric score ranges and patterns."""
    
    @property
    def rule_name(self) -> str:
        return "BiometricScoreValidation"
    
    def validate(self, data: BiometricDataModel) -> Tuple[bool, str]:
        score = data.data.biometric_score
        
        if not 0.0 <= score <= 1.0:
            return False, f"Biometric score {score} outside valid range [0.0, 1.0]"
        
        # Additional business logic validation
        citizenship = data.data.citizenship_status
        if citizenship == 'SA' and score < 0.3:
            return False, f"SA citizen with unusually low biometric score: {score}"
        
        if citizenship == 'Non-SA' and score > 0.9:
            return False, f"Non-SA citizen with unusually high biometric score: {score}"
        
        return True, ""


class NationalIdRule(ValidationRule):
    """Validates South African ID number format and consistency."""
    
    @property
    def rule_name(self) -> str:
        return "NationalIdValidation"
    
    def validate(self, data: BiometricDataModel) -> Tuple[bool, str]:
        national_id = data.data.national_id
        age = data.data.age
        gender = data.data.gender
        
        if len(national_id) != 13:
            return False, f"National ID must be 13 digits, got {len(national_id)}"
        
        if not national_id.isdigit():
            return False, "National ID must contain only digits"
        
        # Extract birth year from ID
        try:
            birth_year_digits = national_id[:2]
            current_year = datetime.now().year
            
            # Determine century (assumption: 00-30 = 2000s, 31-99 = 1900s)
            if int(birth_year_digits) <= 30:
                birth_year = 2000 + int(birth_year_digits)
            else:
                birth_year = 1900 + int(birth_year_digits)
            
            calculated_age = current_year - birth_year
            
            # Allow for some tolerance (birth month not considered)
            if abs(calculated_age - age) > 1:
                return False, f"Age {age} inconsistent with ID birth year {birth_year}"
            
        except ValueError:
            return False, "Invalid birth year in national ID"
        
        # Gender validation (digit 7: 0-4 female, 5-9 male)
        try:
            gender_digit = int(national_id[6])
            if gender == 'Male' and gender_digit < 5:
                return False, "Gender inconsistent with national ID"
            elif gender == 'Female' and gender_digit >= 5:
                return False, "Gender inconsistent with national ID"
        except (ValueError, IndexError):
            return False, "Invalid gender digit in national ID"
        
        return True, ""


class ProvinceRule(ValidationRule):
    """Validates province data."""
    
    @property
    def rule_name(self) -> str:
        return "ProvinceValidation"
    
    def validate(self, data: BiometricDataModel) -> Tuple[bool, str]:
        valid_provinces = [
            'Gauteng', 'Western Cape', 'Eastern Cape', 'KwaZulu-Natal',
            'Free State', 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape'
        ]
        
        if data.data.province not in valid_provinces:
            return False, f"Invalid province: {data.data.province}"
        
        return True, ""


# Triage Validation Rules

class VitalSignsRule(ValidationRule):
    """Validates vital signs against medical ranges."""
    
    @property
    def rule_name(self) -> str:
        return "VitalSignsValidation"
    
    def validate(self, data: TriageDataModel) -> Tuple[bool, str]:
        vitals = data.data
        errors = []
        
        # Critical ranges that would indicate data quality issues
        critical_ranges = {
            'hr_bpm': (20, 250),
            'temp_c': (30.0, 45.0),
            'resp_rate': (5, 60),
            'systolic_bp': (50, 300),
            'diastolic_bp': (20, 200),
            'o2_sat': (50, 100),
            'pain_score': (0, 10)
        }
        
        for field, (min_val, max_val) in critical_ranges.items():
            value = getattr(vitals, field)
            if not min_val <= value <= max_val:
                errors.append(f"{field} value {value} outside critical range [{min_val}, {max_val}]")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, ""


class TriagePriorityConsistencyRule(ValidationRule):
    """Validates triage priority against vital signs."""
    
    @property
    def rule_name(self) -> str:
        return "TriagePriorityConsistency"
    
    def validate(self, data: TriageDataModel) -> Tuple[bool, str]:
        vitals = data.data
        priority = vitals.triage_priority
        
        # Check for obvious inconsistencies
        critical_indicators = 0
        
        # Count critical vital sign indicators
        if vitals.hr_bpm < 50 or vitals.hr_bpm > 120:
            critical_indicators += 1
        if vitals.temp_c > 38.5 or vitals.temp_c < 35.0:
            critical_indicators += 1
        if vitals.resp_rate > 25 or vitals.resp_rate < 10:
            critical_indicators += 1
        if vitals.systolic_bp > 160 or vitals.systolic_bp < 90:
            critical_indicators += 1
        if vitals.o2_sat < 90:
            critical_indicators += 2  # O2 sat is very important
        if vitals.pain_score >= 8:
            critical_indicators += 1
        
        # Validation logic
        if priority == 'Green' and critical_indicators >= 3:
            return False, f"Green priority with {critical_indicators} critical indicators seems inconsistent"
        
        if priority == 'Red' and critical_indicators == 0 and vitals.pain_score < 7:
            return False, "Red priority without critical indicators seems inconsistent"
        
        return True, ""


class DataTimelinessRule(ValidationRule):
    """Validates data freshness and timeline consistency."""
    
    @property
    def rule_name(self) -> str:
        return "DataTimelinessValidation"
    
    def validate(self, data) -> Tuple[bool, str]:
        timestamp = data.data.timestamp
        now = datetime.now()
        
        # Check if timestamp is in the future
        if timestamp > now:
            return False, f"Timestamp {timestamp} is in the future"
        
        # Check if data is too old (more than 2 years)
        if now - timestamp > timedelta(days=730):
            return False, f"Data timestamp {timestamp} is more than 2 years old"
        
        return True, ""


class DataValidator:
    """
    Main validator class that orchestrates validation rules.
    Implements Command Pattern for flexible rule execution.
    """
    
    def __init__(self):
        self.biometric_rules: List[ValidationRule] = [
            BiometricScoreRule(),
            NationalIdRule(),
            ProvinceRule(),
            DataTimelinessRule()
        ]
        
        self.triage_rules: List[ValidationRule] = [
            VitalSignsRule(),
            TriagePriorityConsistencyRule(),
            DataTimelinessRule()
        ]
    
    def validate_biometric_data(self, data: BiometricDataModel) -> ValidationResult:
        """Validate biometric data against all applicable rules."""
        result = ValidationResult()
        
        for rule in self.biometric_rules:
            try:
                is_valid, message = rule.validate(data)
                if is_valid:
                    result.add_passed_rule(rule.rule_name)
                else:
                    result.add_error(rule.rule_name, message)
            except Exception as e:
                result.add_error(rule.rule_name, f"Validation error: {str(e)}")
        
        return result
    
    def validate_triage_data(self, data: TriageDataModel) -> ValidationResult:
        """Validate triage data against all applicable rules."""
        result = ValidationResult()
        
        for rule in self.triage_rules:
            try:
                is_valid, message = rule.validate(data)
                if is_valid:
                    result.add_passed_rule(rule.rule_name)
                else:
                    result.add_error(rule.rule_name, message)
            except Exception as e:
                result.add_error(rule.rule_name, f"Validation error: {str(e)}")
        
        return result
    
    def validate_batch_biometric(self, data_list: List[BiometricDataModel]) -> Dict[str, ValidationResult]:
        """Validate a batch of biometric data."""
        results = {}
        for i, data in enumerate(data_list):
            results[f"record_{i}"] = self.validate_biometric_data(data)
        return results
    
    def validate_batch_triage(self, data_list: List[TriageDataModel]) -> Dict[str, ValidationResult]:
        """Validate a batch of triage data."""
        results = {}
        for i, data in enumerate(data_list):
            results[f"record_{i}"] = self.validate_triage_data(data)
        return results
    
    def get_batch_summary(self, batch_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Get summary statistics for batch validation."""
        total_records = len(batch_results)
        valid_records = sum(1 for result in batch_results.values() if result.is_valid)
        total_errors = sum(len(result.errors) for result in batch_results.values())
        total_warnings = sum(len(result.warnings) for result in batch_results.values())
        
        return {
            'total_records': total_records,
            'valid_records': valid_records,
            'invalid_records': total_records - valid_records,
            'validation_rate': valid_records / total_records if total_records > 0 else 0,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'average_errors_per_record': total_errors / total_records if total_records > 0 else 0
        }
