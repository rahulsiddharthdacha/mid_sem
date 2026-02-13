import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate ML models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """Initialize model trainer.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'linear_regression')
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.is_trained = False
        self.metrics = {}
    
    def _initialize_model(self):
        """Initialize the model based on type.
        
        Returns:
            Initialized model
        """
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear_regression':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size ratio
            
        Returns:
            Dictionary with training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        self.metrics = self.evaluate(X_train, X_test, y_train, y_test)
        
        logger.info(f"Model {self.model_type} trained successfully")
        return self.metrics
    
    def evaluate(self, X_train: np.ndarray, X_test: np.ndarray, 
                 y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        return metrics
    
    def save_model(self, path: str):
        """Save trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Get feature importance if available.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature importances
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))