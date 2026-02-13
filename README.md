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

- **POST /predict**: Upload an Excel file to detect table cells
  - Request: Excel file upload
  - Response: JSON with predicted table cell locations

Example usage:
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
