import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature extraction and transformation."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: 'standard' or 'minmax'
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled[columns] = scaler.fit_transform(df[columns])
        self.scalers['numerical'] = scaler
        
        logger.info(f"Scaled {len(columns)} numerical features using {method} scaling")
        return df_scaled
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: Input DataFrame
            columns: Categorical columns to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in columns:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df[col].astype(str))
            self.encoders[col] = encoder
        
        logger.info(f"Encoded {len(columns)} categorical features")
        return df_encoded
    
    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Degree of polynomial
            
        Returns:
            DataFrame with additional polynomial features
        """
        df_poly = df.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                df_poly[f'{col}_power_{d}'] = df[col] ** d
        
        logger.info(f"Created polynomial features up to degree {degree}")
        return df_poly
    
    def create_interaction_features(self, df: pd.DataFrame, column_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features.
        
        Args:
            df: Input DataFrame
            column_pairs: List of column pairs for interaction
            
        Returns:
            DataFrame with interaction features
        """
        df_interaction = df.copy()
        
        for col1, col2 in column_pairs:
            df_interaction[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        logger.info(f"Created {len(column_pairs)} interaction features")
        return df_interaction
    
    def get_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray) -> Dict:
        """Get feature importance rankings.
        
        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            
        Returns:
            Dictionary with ranked features
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df.to_dict('records')
