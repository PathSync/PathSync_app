import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

def load_and_preprocess_data():
    # Load the largest dataset for better model training
    data_path = Path(__file__).parent.parent / 'Data' / 'sample' / 'sa_80000sample_data.csv'
    df = pd.read_csv(data_path)
    
    # Select relevant features for triage model
    features = [
        'Age',
        'Immigration Status',
        'Visa Type',
        'Asylum Status',
        'Years in SA'
    ]
    
    # Create X (features) dataframe
    X = df[features].copy()
    
    # Create target variable based on multiple factors
    # This is a simplified triage logic - you may want to adjust based on your specific needs
    def determine_priority(row):
        if pd.notna(row['Asylum Status']) and row['Asylum Status'] != 'N/A':
            return 'High'
        if row['Immigration Status'] == 'Temporary Visa':
            return 'Medium'
        if pd.isna(row['Years in SA']) or row['Years in SA'] == 'N/A':
            return 'Low'
        years_in_sa = float(row['Years in SA']) if pd.notna(row['Years in SA']) else 0
        if years_in_sa < 5:
            return 'Medium'
        return 'Low'
    
    y = df.apply(determine_priority, axis=1)
    
    # Handle missing values
    X['Years in SA'] = X['Years in SA'].fillna(-1)
    X['Asylum Status'] = X['Asylum Status'].fillna('N/A')
    X['Visa Type'] = X['Visa Type'].fillna('N/A')
    
    # Encode categorical variables
    categorical_features = ['Immigration Status', 'Visa Type', 'Asylum Status']
    
    encoders = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        X[feature] = encoder.fit_transform(X[feature].astype(str))
        encoders[feature] = encoder
    
    # Scale numerical features
    scaler = StandardScaler()
    X[['Age', 'Years in SA']] = scaler.fit_transform(X[['Age', 'Years in SA']])
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    encoders['target'] = target_encoder
    
    return X, y, encoders, scaler

def train_triage_model():
    print("Loading and preprocessing data...")
    X, y, encoders, scaler = load_and_preprocess_data()
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
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
    joblib.dump(model, models_path / 'triage_model.pkl')
    joblib.dump({
        'encoders': encoders,
        'scaler': scaler
    }, models_path / 'triage_preprocessing.pkl')
    
    print("Model training and saving completed!")

if __name__ == '__main__':
    train_triage_model()