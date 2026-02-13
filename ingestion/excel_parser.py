import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ExcelParser:
    """Parse and normalize Excel files."""
    
    def __init__(self, file_path: str):
        """Initialize the Excel parser.        
        Args:
            file_path: Path to the Excel file
        """
        self.file_path = file_path
        self.raw_data = None
        self.normalized_data = None
    
    def load_excel(self) -> pd.DataFrame:
        """Load Excel file into DataFrame.        
        Returns:
            Loaded DataFrame
        """
        try:
            self.raw_data = pd.read_excel(self.file_path)
            logger.info(f"Successfully loaded Excel file: {self.file_path}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise    
    def normalize(self) -> pd.DataFrame:
        """Normalize the loaded data.        
        Returns:
            Normalized DataFrame
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_excel() first.")
        
        df = self.raw_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Convert data types
        df = self._convert_dtypes(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        self.normalized_data = df
        logger.info("Data normalization completed")
        return self.normalized_data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame.        
        Args:
            df: Input DataFrame            
        Returns:
            DataFrame with missing values handled
        """
        # Fill missing numeric values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        return df
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types appropriately.        
        Args:
            df: Input DataFrame            
        Returns:
            DataFrame with converted data types
        """
        # Attempt to convert object columns to numeric where possible
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
        
        return df
    
    def get_statistics(self) -> Dict:
        """Get statistics of the normalized data.        
        Returns:
            Dictionary containing data statistics
        """
        if self.normalized_data is None:
            raise ValueError("No normalized data available.")
        
        return {
            'shape': self.normalized_data.shape,
            'columns': list(self.normalized_data.columns),
            'dtypes': self.normalized_data.dtypes.to_dict(),
            'missing_values': self.normalized_data.isnull().sum().to_dict(),
            'statistics': self.normalized_data.describe().to_dict()
        }