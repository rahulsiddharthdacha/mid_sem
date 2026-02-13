# ML-Based Detection of Financial Tables in Excel Using Metadata Features

## Overview
This project implements a machine learning system for automatically detecting and identifying financial tables in Excel spreadsheets using metadata and structural features. The system analyzes Excel files to identify table regions, headers, and data cells using both structural and semantic features.

## Features
- **Structural Feature Extraction**: Analyzes cell positions, density, formatting, and layout patterns
- **Semantic Feature Extraction**: Uses sentence transformers to understand cell content semantics
- **ML-Based Table Detection**: Trains classification models to identify table components
- **Automated Pipeline**: End-to-end pipeline for feature extraction, model training, and prediction
- **REST API**: FastAPI-based service for table detection in uploaded Excel files
- **Interactive UI**: Streamlit-based web interface for easy interaction
- **Experiment Tracking**: MLflow integration for model comparison and versioning
- **Pipeline Orchestration**: Apache Airflow for automated workflow management

## Project Structure
```
mid_sem/
├── airflow/              # Airflow DAG for pipeline orchestration
│   └── excel_pipline_dag.py
├── data/                 # Sample Excel files
│   ├── sample.xlsx               # Original financial sample
│   ├── sales_report.xlsx         # Sales data sample
│   ├── financial_statement.xlsx  # Financial statement sample
│   └── inventory_report.xlsx     # Inventory data sample
├── features/             # Feature extraction modules
│   ├── feature_extractor.py      # Main feature extraction orchestrator
│   ├── structural_features.py    # Structural metadata extraction
│   ├── semantic_features.py      # Semantic embeddings
│   └── features.csv              # Extracted features (generated)
├── model/                # Model training scripts
│   └── train_model.py
├── serving/              # Model serving API
│   └── app.py
├── ui/                   # Streamlit web interface
│   ├── app.py                    # Main UI application
│   └── requirements_ui.txt       # UI-specific dependencies
├── run_pipline.py        # Main pipeline execution script
├── start_all_services.sh # Start all services (MLflow, Airflow, UI)
├── stop_all_services.sh  # Stop all services
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rahulsiddharthdacha/mid_sem.git
   cd mid_sem
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Note:** The first run of feature extraction will download the `all-MiniLM-L6-v2` sentence transformer model from HuggingFace (~80MB). This requires an internet connection.

## Usage

### Prerequisites - First Time Setup

Before using the table detection API, you need to:

1. **Generate training features** (if not already present):
   ```bash
   python features/feature_extractor.py
   ```
   This extracts features from `data/sample.xlsx` and saves them to `features/features.csv`.

2. **Train the models** (if models don't exist):
   ```bash
   python model/train_model.py
   ```
   This trains multiple ML models and saves them to the `mlruns/` directory.

**Note:** Pre-trained models are already included in the repository, so you can skip steps 1-2 unless you want to retrain with different data.

3. **Start the API server** (required):
   ```bash
   uvicorn serving.app:app --host 0.0.0.0 --port 8000
   ```
   The server will automatically:
   - Load the best pre-trained model
   - Enable the `/detect-tables` endpoint
   - Use rule-based fallback if needed for robust header detection

### Quick Example - Complete Table Detection Cycle

Here's how to use the API for the complete cycle (upload → detect → JSON output):

1. **Start the API server** (if not already running):
   ```bash
   uvicorn serving.app:app --host 0.0.0.0 --port 8000
   ```

2. **Upload an Excel file and detect tables:**
   ```bash
   curl -X POST "http://localhost:8000/detect-tables" \
     -F "file=@data/sales_report.xlsx" | python -m json.tool
   ```

3. **View the structured JSON output:**
   - `summary`: Overall statistics (total cells, headers, data cells, dimensions)
   - `detected_tables.headers`: List of detected header cells with locations
   - `detected_tables.data`: Data rows organized by row number

**Or use the Python example script:**
```bash
python example_detect_tables.py data/sales_report.xlsx
```

The system will automatically:
- Extract structural features (cell position, density, content type)
- Extract semantic features (text embeddings)
- Use the trained ML model to classify cells as headers or data
- Return structured JSON with detected table components

### Quick Start - Launch All Services

The easiest way to get started is to launch all services at once:

```bash
./start_all_services.sh
```

This will start:
- **MLflow UI** on http://localhost:5000
- **Airflow UI** on http://localhost:8080 (login: admin/admin)
- **Streamlit UI** on http://localhost:8501

To stop all services:
```bash
./stop_all_services.sh
```

### Using the Streamlit UI

The Streamlit interface provides an easy way to interact with the system:

```bash
streamlit run ui/app.py
```

Or use the provided script:
```bash
./start_ui.sh
```

**Features:**
- Upload and analyze Excel files
- View sample data
- Extract and visualize features
- Monitor service status
- Access training instructions

### Running the Complete Pipeline

Execute the end-to-end pipeline for feature extraction and model training:

```bash
python run_pipline.py
```

This will:
1. Extract structural and semantic features from Excel files
2. Train and compare multiple ML models
3. Log experiments to MLflow

### Feature Extraction

Extract features from a specific Excel file:

```bash
python features/feature_extractor.py
```

The extracted features will be saved to `features/features.csv`.

### Model Training

Train table detection models:

```bash
python model/train_model.py
```

Models are trained using:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)

Results are logged to MLflow for comparison.

### API Serving

Start the FastAPI server for table detection:

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

#### API Endpoints

##### GET /
Root endpoint with API information and available endpoints.

##### GET /health
Health check endpoint to verify server and model status.

```bash
curl http://localhost:8000/health
```

##### POST /detect-tables (Recommended)
**Complete Table Detection Endpoint** - Upload an Excel file and get structured JSON output with detected table headers and data.

This endpoint performs the complete cycle:
1. Upload Excel file
2. Extract structural and semantic features
3. Use trained ML model to detect table cells
4. Return structured JSON with detected tables

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- File parameter: `file` (Excel file: .xlsx or .xls)

**Response:**
```json
{
  "status": "success",
  "filename": "sales_report.xlsx",
  "summary": {
    "total_cells": 72,
    "header_cells": 9,
    "data_cells": 63,
    "dimensions": {
      "rows": 8,
      "columns": 9
    }
  },
  "detected_tables": {
    "headers": [
      {
        "row": 0,
        "column": 0,
        "value": "Product"
      },
      ...
    ],
    "data": [
      {
        "row_number": 1,
        "cells": [
          {
            "row": 1,
            "column": 0,
            "value": "Laptop"
          },
          ...
        ]
      },
      ...
    ]
  }
}
```

**Example usage:**
```bash
curl -X POST "http://localhost:8000/detect-tables" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/sales_report.xlsx"
```

**Python example:**
```python
import requests

url = "http://localhost:8000/detect-tables"
files = {"file": open("data/sales_report.xlsx", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Detected {result['summary']['header_cells']} headers")
print(f"Detected {result['summary']['data_cells']} data cells")
```

##### POST /predict (Legacy)
Simple prediction endpoint that returns raw prediction array.

**Request:** Excel file upload

**Response:** JSON with predicted table cell locations

**Example usage:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_file.xlsx"
```

### Viewing Experiments

View MLflow experiment tracking:

```bash
mlflow ui
```

Or use the provided script:
```bash
./start_mlflow.sh
```

Then navigate to `http://localhost:5000` in your browser.

### Starting Apache Airflow

Initialize and start Airflow for pipeline orchestration:

```bash
./start_airflow.sh
```

Then navigate to `http://localhost:8080` (login: admin/admin).

## Features Description

### Structural Features
- **Cell Position**: Row and column indices
- **Empty/Numeric Indicators**: Binary flags for cell content type
- **Density Metrics**: Row and column fill ratios
- **Cell Text**: Raw cell content

### Semantic Features
- **Sentence Embeddings**: 384-dimensional vectors from `all-MiniLM-L6-v2` model
- **Context Understanding**: Captures semantic meaning of cell text
- **Zero Embeddings**: For empty or numeric-only cells

## Model Performance

Models are evaluated using:
- Accuracy
- F1 Score
- Cross-validation metrics

Results are logged and can be compared in MLflow UI.

## Technology Stack

- **Python 3.8+**
- **pandas & openpyxl**: Excel file processing
- **scikit-learn**: Machine learning models
- **sentence-transformers**: Semantic feature extraction
- **FastAPI**: REST API serving
- **MLflow**: Experiment tracking
- **Airflow**: Pipeline orchestration

## Troubleshooting

### Header Detection Not Working

**Problem**: "It's not able to detect headers at all"

**Solution**: Make sure you've completed the following steps:

1. **Check if the API server is running:**
   ```bash
   curl http://localhost:8000/health
   ```
   Expected output: `{"status":"healthy","model_loaded":true}`

2. **If model_loaded is false**, train the models:
   ```bash
   python features/feature_extractor.py
   python model/train_model.py
   ```

3. **Restart the API server:**
   ```bash
   # Stop the server (Ctrl+C if running in foreground)
   # Or kill the process if running in background
   
   # Start fresh
   uvicorn serving.app:app --host 0.0.0.0 --port 8000
   ```

4. **Test header detection:**
   ```bash
   python example_detect_tables.py data/sales_report.xlsx
   ```

**Note**: The system now includes a rule-based fallback that automatically detects headers in row 0 if the ML model fails. This ensures robust header detection even with class-imbalanced training data.

### Model Not Found

If you see "No pre-trained model found", run:
```bash
python model/train_model.py
```

### Missing Dependencies

If you encounter import errors, install dependencies:
```bash
pip install -r requirements.txt
```

## Best Practices

- Keep Excel files in the `data/` directory
- Review extracted features before training
- Compare multiple models using MLflow
- Use the API for production deployments
- Monitor model performance on new data

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes.

---

For questions or issues, please open an issue on GitHub.
