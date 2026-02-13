#!/usr/bin/env python
"""Script to run the Flask API server."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.model_api import ModelAPI
from configs.config import get_config

if __name__ == "__main__":
    config = get_config()
    
    # Initialize and run API
    api = ModelAPI(model_path=str(config.MODEL_PATH))
    print(f"Starting API on http://0.0.0.0:5000")
    api.run(host='0.0.0.0', port=5000, debug=config.DEBUG)