import os
from pathlib import Path
from typing import Dict

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'model'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class Config:
    """Base configuration."""
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = LOGS_DIR / 'app.log'
    
    # Data
    DATA_PATH = DATA_DIR / 'data.csv'
    
    # Model
    MODEL_PATH = MODEL_DIR / 'model.pkl'
    MODEL_TYPE = os.getenv('MODEL_TYPE', 'random_forest')
    
    # Pipeline
    TARGET_COLUMN = os.getenv('TARGET_COLUMN', 'charges')
    TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    MODEL_PATH = LOGS_DIR / 'test_model.pkl'


def get_config(env: str = None) -> Config:
    """Get configuration based on environment.
    
    Args:
        env: Environment name (development, production, testing)
        
    Returns:
        Configuration object
    """
    if env is None:
        env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()