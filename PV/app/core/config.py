# app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    """Application settings"""

    # Directory settings
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    ML_MODELS_DIR = os.path.join(BASE_DIR, "app", "ml", "models")

    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./healthcare.db")

    # ML model paths
    FACIAL_MODEL_PATH = os.getenv("FACIAL_MODEL_PATH", os.path.join(ML_MODELS_DIR, "facial_model.h5"))
    TRIAGE_MODEL_PATH = os.getenv("TRIAGE_MODEL_PATH", os.path.join(ML_MODELS_DIR, "triage_model.pkl"))

    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))


settings = Settings()