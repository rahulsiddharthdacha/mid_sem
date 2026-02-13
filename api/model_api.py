from flask import Flask, request, jsonify
from typing import Dict, Any
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelAPI:
    """Model serving API handler."""
    
    def __init__(self, model_path: str = None):
        """Initialize the API.
        
        Args:
            model_path: Path to the trained model
        """
        self.app = Flask(__name__)
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self._load_model()
        
        self._setup_routes()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint."""
            return jsonify({'status': 'healthy', 'model_loaded': self.model is not None})
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint."""
            try:
                data = request.json
                
                # Convert input to DataFrame
                input_df = pd.DataFrame([data])
                
                # Make prediction
                if self.model is None:
                    return jsonify({'error': 'Model not loaded'}), 400
                
                prediction = self.model.predict(input_df.values)
                
                return jsonify({
                    'prediction': float(prediction[0]),
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/batch-predict', methods=['POST'])
        def batch_predict():
            """Batch prediction endpoint."""
            try:
                data = request.json
                
                # Convert input to DataFrame
                input_df = pd.DataFrame(data['records'])
                
                # Make predictions
                if self.model is None:
                    return jsonify({'error': 'Model not loaded'}), 400
                
                predictions = self.model.predict(input_df.values)
                
                return jsonify({
                    'predictions': predictions.tolist(),
                    'count': len(predictions),
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/model-info', methods=['GET'])
        def model_info():
            """Get model information."""
            return jsonify({
                'model_type': type(self.model).__name__ if self.model else None,
                'status': 'loaded' if self.model else 'not_loaded'
            })
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the API server.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        self.app.run(host=host, port=port, debug=debug)