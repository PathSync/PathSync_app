import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_and_preprocess_data():
    # Load the largest dataset for better model training
    data_path = Path(__file__).parent.parent / 'Data' / 'sample' / 'sa_80000sample_data.csv'
    df = pd.read_csv(data_path)
    
    # Select relevant features for biometric model
    features = [
        'Age', 'Gender', 'Immigration Status', 'Country of Origin',
        'Visa Type', 'Language', 'Years in SA'
    ]
    
    # Create X (features) dataframe
    X = df[features].copy()
    
    # Create y (target) - we'll use Immigration Status as our target
    y = df['Immigration Status']
    
    # Handle missing values
    X['Years in SA'] = X['Years in SA'].fillna(-1)  # -1 for citizens/permanent residents
    
    # Encode categorical variables
    categorical_features = ['Gender', 'Immigration Status', 'Country of Origin', 
                          'Visa Type', 'Language']
    
    encoders = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        X[feature] = encoder.fit_transform(X[feature].astype(str))
        encoders[feature] = encoder
    
    # Scale numerical features
    scaler = StandardScaler()
    X[['Age', 'Years in SA']] = scaler.fit_transform(X[['Age', 'Years in SA']])
    
    return X, y, encoders, scaler

def train_biometric_model():
    print("Loading and preprocessing data...")
    X, y, encoders, scaler = load_and_preprocess_data()
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save the model and preprocessing objects
    models_path = Path(__file__).parent.parent / 'models'
    models_path.mkdir(exist_ok=True)
    
    print("Saving model and preprocessing objects...")
    joblib.dump(model, models_path / 'biometric_model.pkl')
    joblib.dump({
        'encoders': encoders,
        'scaler': scaler
    }, models_path / 'biometric_preprocessing.pkl')
    
    print("Model training and saving completed!")

if __name__ == '__main__':
    train_biometric_model()