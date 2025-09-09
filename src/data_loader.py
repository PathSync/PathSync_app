import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_biometric_data(file_path):
    """Load biometric data from CSV file"""
    df = pd.read_csv(file_path)
    return df


def load_triage_data(file_path):
    """Load triage data from CSV file"""
    df = pd.read_csv(file_path)
    return df


def preprocess_biometric_data(df):
    """Preprocess biometric data"""
    # Encode categorical variables
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})

    # Encode province (simplified for demo)
    provinces = df['province'].unique()
    province_map = {province: i for i, province in enumerate(provinces)}
    df['province_encoded'] = df['province'].map(province_map)

    # Encode citizenship status
    citizenship_map = {'SA': 0, 'Non-SA': 1, 'Review': 2}
    df['citizenship_encoded'] = df['citizenship_status'].map(citizenship_map)

    return df, province_map, citizenship_map


def preprocess_triage_data(df):
    """Preprocess triage data"""
    # Encode gender
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})

    # Encode triage priority
    priority_map = {'Red': 0, 'Yellow': 1, 'Green': 2}
    df['priority_encoded'] = df['triage_priority'].map(priority_map)

    return df, priority_map