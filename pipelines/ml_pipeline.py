import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

from ingestion.excel_parser import ExcelParser
from features.feature_engineer import FeatureEngineer
from models.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class MLPipeline:
    """End-to-end ML pipeline orchestration."""
    
    def __init__(self, config: Dict):
        """Initialize the ML pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data = None
        self.features_df = None
        self.model = None
        self.results = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and parse data from Excel.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Loaded and normalized DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        parser = ExcelParser(file_path)
        self.data = parser.load_excel()
        self.data = parser.normalize()
        
        logger.info(f"Data shape: {self.data.shape}")
        return self.data
    
    def engineer_features(self, target_column: str) -> pd.DataFrame:
        """Engineer features for the model.
        
        Args:
            target_column: Name of target column
            
        Returns:
            DataFrame with engineered features
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Starting feature engineering")
        
        engineer = FeatureEngineer()
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Scale numerical features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            X = engineer.scale_features(X, numeric_cols, method='standard')
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            X = engineer.encode_categorical(X, categorical_cols)
        
        self.features_df = X
        self.target = y
        
        logger.info(f"Features engineered. Shape: {X.shape}")
        return X
    
    def train_model(self, model_type: str = 'random_forest') -> Dict:
        """Train the ML model.
        
        Args:
            model_type: Type of model to train
            
        Returns:
            Dictionary with training results
        """
        if self.features_df is None or self.target is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")
        
        logger.info(f"Training {model_type} model")
        
        self.model = ModelTrainer(model_type=model_type)
        X_array = self.features_df.values
        y_array = self.target.values
        
        self.results['training'] = self.model.train(X_array, y_array)
        
        logger.info(f"Model trained. Test R2: {self.results['training']['test_r2']:.4f}")
        return self.results['training']
    
    def save_model(self, model_path: str):
        """Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            input_data: Input DataFrame for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        predictions = self.model.predict(input_data.values)
        return predictions
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of pipeline execution.
        
        Returns:
            Dictionary with pipeline summary
        """
        return {
            'data_shape': self.data.shape if self.data is not None else None,
            'features_shape': self.features_df.shape if self.features_df is not None else None,
            'model_type': self.model.model_type if self.model is not None else None,
            'metrics': self.results.get('training', {})
        }