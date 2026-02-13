import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import os
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.structural_features import extract_structural_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Excel Table Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'features_df' not in st.session_state:
    st.session_state.features_df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def main():
    """Main application function."""
    
    st.title("📊 Excel Financial Table Detection System")
    st.markdown("**ML-Based Detection of Financial Tables Using Metadata Features**")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Upload & Detect", "Sample Data Explorer", "Feature Analysis", "Model Training", "Services Status"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Upload & Detect":
        show_upload_detect()
    elif page == "Sample Data Explorer":
        show_sample_data()
    elif page == "Feature Analysis":
        show_feature_analysis()
    elif page == "Model Training":
        show_model_training()
    elif page == "Services Status":
        show_services_status()

def show_home():
    """Show home page."""
    st.header("Welcome to Excel Table Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### About This Application
        
        This system uses machine learning to automatically detect and identify 
        financial tables in Excel spreadsheets using metadata and structural features.
        
        **Key Features:**
        - 📤 Upload Excel files for analysis
        - 🔍 Detect table headers and structure
        - 📊 Extract structural and semantic features
        - 🤖 ML-based classification
        - 📈 Visualize detection results
        - 🔬 MLflow experiment tracking
        - 🔄 Airflow pipeline orchestration
        
        **How It Works:**
        1. **Structural Features**: Analyzes cell position, density, and formatting
        2. **Semantic Features**: Understands cell content using NLP
        3. **ML Classification**: Identifies headers and table regions
        """)
    
    with col2:
        st.markdown("""
        ### Getting Started
        
        **Quick Start:**
        1. 📤 Go to "Upload & Detect" to analyze your Excel file
        2. 🔍 Or explore the "Sample Data Explorer" to see examples
        3. 📊 View "Feature Analysis" to understand the detection process
        4. 🤖 Train models in "Model Training" section
        5. 🔍 Check "Services Status" for MLflow and Airflow
        
        **Supported Features:**
        - Excel formats: .xlsx, .xls
        - Multiple sheets support
        - Cell-level analysis
        - Table boundary detection
        
        **Models Available:**
        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        - Support Vector Machine (SVM)
        """)
    
    st.markdown("---")
    
    # Stats section
    st.subheader("📈 System Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    # Check if pre-extracted features exist
    features_path = Path(__file__).parent.parent / "features" / "features.csv"
    if features_path.exists():
        try:
            features_df = pd.read_csv(features_path, nrows=1000)
            col1.metric("Feature Dimensions", f"{features_df.shape[1]}")
            col2.metric("Models Available", "4")
            col3.metric("Test Datasets", "3")
            col4.metric("Status", "✅ Ready")
        except:
            col1.metric("Status", "Ready")
            col2.metric("Models", "4")
            col3.metric("Features", "391")
            col4.metric("Supported", "Excel")
    else:
        col1.metric("Status", "Ready")
        col2.metric("Models", "4")
        col3.metric("Features", "391")
        col4.metric("Supported", "Excel")

def show_upload_detect():
    """Show upload and detection page."""
    st.header("📤 Upload & Detect Tables")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Excel File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an Excel file", 
            type=['xlsx', 'xls'],
            help="Upload an Excel file to detect tables and headers"
        )
        
        if uploaded_file is not None:
            try:
                # Read the Excel file
                df = pd.read_excel(uploaded_file)
                st.session_state.uploaded_data = df
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Display the raw data
                st.subheader("Raw Data Preview")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Data info
                st.subheader("Data Information")
                info_col1, info_col2, info_col3 = st.columns(3)
                info_col1.metric("Rows", df.shape[0])
                info_col2.metric("Columns", df.shape[1])
                info_col3.metric("Missing Values", df.isnull().sum().sum())
                
                # Detect Tables using API
                st.markdown("---")
                st.subheader("🤖 Table Detection with ML Model")
                
                # API Server status check
                api_url = "http://localhost:8000"
                try:
                    import requests
                    health_response = requests.get(f"{api_url}/health", timeout=2)
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        if health_data.get('model_loaded'):
                            st.success("✅ API server is running and model is loaded")
                            api_available = True
                        else:
                            st.warning("⚠️ API server is running but model is not loaded")
                            api_available = False
                    else:
                        st.error("❌ API server error")
                        api_available = False
                except Exception as e:
                    st.error("❌ API server is not running. Please start it with: `uvicorn serving.app:app --host 0.0.0.0 --port 8000`")
                    api_available = False
                
                if api_available and st.button("🔍 Detect Tables with ML Model", type="primary"):
                    with st.spinner("Detecting tables using trained ML model..."):
                        try:
                            import requests
                            
                            # Reset file pointer
                            uploaded_file.seek(0)
                            
                            # Call the API
                            files = {"file": (uploaded_file.name, uploaded_file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                            response = requests.post(f"{api_url}/detect-tables", files=files, timeout=30)
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state.predictions = result
                                
                                st.success("✅ Table detection completed!")
                                
                                # Display summary
                                st.markdown("---")
                                st.subheader("📊 Detection Summary")
                                
                                summary = result.get('summary', {})
                                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                                sum_col1.metric("Total Cells", summary.get('total_cells', 0))
                                sum_col2.metric("Header Cells", summary.get('header_cells', 0))
                                sum_col3.metric("Data Cells", summary.get('data_cells', 0))
                                dims = summary.get('dimensions', {})
                                sum_col4.metric("Dimensions", f"{dims.get('rows', 0)}×{dims.get('columns', 0)}")
                                
                                # Display detected headers
                                st.markdown("---")
                                st.subheader("📌 Detected Headers")
                                headers = result.get('detected_tables', {}).get('headers', [])
                                
                                if headers:
                                    headers_df = pd.DataFrame(headers)
                                    st.dataframe(headers_df, use_container_width=True)
                                    
                                    # Visualize headers on the original data
                                    st.markdown("---")
                                    st.subheader("📋 Headers Highlighted in Data")
                                    
                                    # Create a copy of the dataframe for display
                                    display_df = df.copy()
                                    
                                    # Highlight header row (row 0 typically)
                                    def highlight_headers(row):
                                        return ['background-color: #90EE90' if row.name == 0 else '' for _ in row]
                                    
                                    styled_df = display_df.head(10).style.apply(highlight_headers, axis=1)
                                    st.dataframe(styled_df, use_container_width=True)
                                else:
                                    st.info("No headers detected")
                                
                                # Display sample data rows
                                st.markdown("---")
                                st.subheader("📊 Sample Data Rows")
                                data_rows = result.get('detected_tables', {}).get('data', [])
                                
                                if data_rows:
                                    st.write(f"Total data rows: {len(data_rows)}")
                                    
                                    # Show first few data rows
                                    num_rows_to_show = min(5, len(data_rows))
                                    for i in range(num_rows_to_show):
                                        row = data_rows[i]
                                        with st.expander(f"Row {row['row_number']} ({len(row['cells'])} cells)"):
                                            cells_df = pd.DataFrame(row['cells'])
                                            st.dataframe(cells_df, use_container_width=True)
                                else:
                                    st.info("No data rows detected")
                                
                                # Download JSON result
                                st.markdown("---")
                                st.subheader("💾 Download Results")
                                
                                import json
                                json_str = json.dumps(result, indent=2)
                                st.download_button(
                                    label="📥 Download Detection Results (JSON)",
                                    data=json_str,
                                    file_name=f"table_detection_{uploaded_file.name}.json",
                                    mime="application/json"
                                )
                                
                                # Show raw JSON in expander
                                with st.expander("🔍 View Raw JSON Response"):
                                    st.json(result)
                                
                            else:
                                st.error(f"❌ API Error: {response.status_code} - {response.text}")
                                
                        except Exception as e:
                            st.error(f"❌ Error calling API: {e}")
                            logger.error(f"Error calling detection API: {e}")
                
                # Extract features button (for advanced users)
                st.markdown("---")
                st.subheader("🔬 Advanced: Feature Extraction Only")
                
                with st.expander("Extract Features Without Detection"):
                    if st.button("🔍 Extract Structural Features"):
                        with st.spinner("Extracting features from cells..."):
                            # Extract structural features
                            features_df = extract_structural_features(df)
                            st.session_state.features_df = features_df
                            
                            st.success(f"✅ Extracted {len(features_df)} cell-level features!")
                            
                            # Show feature preview
                            st.dataframe(features_df.head(20), use_container_width=True)
                            
                            # Feature statistics
                            st.subheader("Feature Statistics")
                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            stat_col1.metric("Total Cells", len(features_df))
                            stat_col2.metric("Empty Cells", int(features_df['is_empty'].sum()))
                            stat_col3.metric("Numeric Cells", int(features_df['is_numeric'].sum()))
                            stat_col4.metric("Text Cells", len(features_df) - int(features_df['is_empty'].sum()) - int(features_df['is_numeric'].sum()))
                            
                            # Download features
                            csv = features_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Features CSV",
                                data=csv,
                                file_name="extracted_features.csv",
                                mime="text/csv"
                            )
                
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                logger.error(f"Error processing uploaded file: {e}")
        
        else:
            st.info("👈 Please upload an Excel file to begin analysis")
    
    with col2:
        st.subheader("Detection Process")
        st.markdown("""
        **Steps:**
        
        1️⃣ **Upload File**
        - Choose your Excel file
        - Supports .xlsx and .xls
        
        2️⃣ **Detect Tables**
        - Click "Detect Tables" button
        - ML model analyzes cells
        - Returns JSON with results
        
        3️⃣ **View Results**
        - See detected headers
        - View data organization
        - Download JSON output
        
        **What You Get:**
        - Summary statistics
        - Detected header cells
        - Organized data rows
        - Downloadable JSON
        
        **Complete Cycle:**
        Upload → Detect → JSON Output
        """)

def show_sample_data():
    """Show sample data explorer."""
    st.header("🔍 Sample Data Explorer")
    
    # Get all sample files
    data_dir = Path(__file__).parent.parent / "data"
    sample_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    
    if sample_files:
        st.success(f"✅ Found {len(sample_files)} sample file(s)")
        
        # File selector
        selected_file = st.selectbox(
            "Select a sample file to explore",
            sample_files,
            format_func=lambda x: x.name
        )
        
        try:
            df = pd.read_excel(selected_file)
            
            st.info(f"📊 Loaded: **{selected_file.name}** ({df.shape[0]} rows × {df.shape[1]} columns)")
            
            # Display options
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("Display Options")
                num_rows = st.slider("Number of rows", 5, min(100, len(df)), 20)
                show_info = st.checkbox("Show column info", value=True)
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(df.head(num_rows), use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 Data Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", df.shape[0])
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Missing Values", int(df.isnull().sum().sum()))
            col4.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            
            # Column information
            if show_info:
                st.markdown("---")
                st.subheader("Column Information")
                
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null': df.isnull().sum().values,
                    'Unique': [df[col].nunique() for col in df.columns]
                })
                
                st.dataframe(col_info, use_container_width=True)
            
            # Descriptive statistics
            st.markdown("---")
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading sample data: {e}")
            logger.error(f"Error loading sample data: {e}")
    else:
        st.warning("⚠️ No sample data files found")
        st.info(f"Expected location: {data_dir}")

def show_feature_analysis():
    """Show feature analysis page."""
    st.header("📊 Feature Analysis")
    
    # Load pre-extracted features if available
    features_path = Path(__file__).parent.parent / "features" / "features.csv"
    
    if features_path.exists():
        try:
            # Load with a sample to avoid memory issues
            with st.spinner("Loading pre-extracted features..."):
                sample_df = pd.read_csv(features_path, nrows=2000)
            
            st.success(f"✅ Loaded {len(sample_df)} feature samples with {sample_df.shape[1]} columns")
            
            # Display feature information
            st.subheader("Feature Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Features", sample_df.shape[1])
            col2.metric("Structural Features", 7)
            col3.metric("Semantic Features", 384)
            col4.metric("Sample Size", len(sample_df))
            
            # Show structural features
            st.markdown("---")
            st.subheader("Structural Features (Preview)")
            structural_cols = ['row_idx', 'col_idx', 'is_empty', 'is_numeric', 'row_density', 'col_density']
            if 'label' in sample_df.columns:
                structural_cols.append('label')
            st.dataframe(sample_df[structural_cols].head(20), use_container_width=True)
            
            # Feature distributions
            st.markdown("---")
            st.subheader("Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Label Distribution**")
                if 'label' in sample_df.columns:
                    label_counts = sample_df['label'].value_counts()
                    st.bar_chart(label_counts)
                    st.caption(f"Headers (1): {label_counts.get(1, 0)}, Data (0): {label_counts.get(0, 0)}")
                else:
                    st.info("No labels found in features")
            
            with col2:
                st.markdown("**Cell Type Distribution**")
                cell_types = pd.DataFrame({
                    'Type': ['Empty', 'Numeric', 'Text'],
                    'Count': [
                        int(sample_df['is_empty'].sum()),
                        int(sample_df['is_numeric'].sum()),
                        len(sample_df) - int(sample_df['is_empty'].sum()) - int(sample_df['is_numeric'].sum())
                    ]
                }).set_index('Type')
                st.bar_chart(cell_types)
            
            # Density analysis
            st.markdown("---")
            st.subheader("Density Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Row Density Distribution**")
                density_counts = sample_df['row_density'].value_counts().sort_index()
                st.line_chart(density_counts)
            
            with col2:
                st.markdown("**Column Density Distribution**")
                density_counts = sample_df['col_density'].value_counts().sort_index()
                st.line_chart(density_counts)
            
        except Exception as e:
            st.error(f"❌ Error loading features: {e}")
            logger.error(f"Error in feature analysis: {e}")
    else:
        st.warning("⚠️ No pre-extracted features found")
        st.info("""
        To generate features, run:
        ```bash
        python features/feature_extractor.py
        ```
        
        This will extract features from the sample data and save them to `features/features.csv`.
        """)

def show_model_training():
    """Show model training page."""
    st.header("🤖 Model Training & Evaluation")
    
    st.markdown("""
    Train multiple machine learning models and compare their performance using MLflow tracking.
    """)
    
    # Model information
    st.subheader("Available Models")
    
    models_info = {
        "Logistic Regression": "Fast linear model, good baseline",
        "Random Forest": "Ensemble method, handles non-linear patterns",
        "Gradient Boosting": "Sequential ensemble, high accuracy",
        "SVM": "Support Vector Machine for high-dimensional data"
    }
    
    cols = st.columns(2)
    for idx, (model_name, description) in enumerate(models_info.items()):
        with cols[idx % 2]:
            st.info(f"**{model_name}**\n\n{description}")
    
    st.markdown("---")
    st.subheader("Training Process")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **To train all models:**
        
        1. Run the training script:
        ```bash
        python model/train_model.py
        ```
        
        This will:
        - Load features from `features/features.csv`
        - Train all 4 models
        - Evaluate using accuracy and F1 score
        - Log results to MLflow
        - Save trained models
        
        **View results in MLflow:**
        ```bash
        mlflow ui
        ```
        Then open http://localhost:5000
        """)
        
        # Button to trigger training (informational)
        if st.button("📘 View Training Instructions", type="primary"):
            st.code("""
# Training script
cd /home/runner/work/mid_sem/mid_sem
python model/train_model.py

# View MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
            """, language="bash")
    
    with col2:
        st.markdown("""
        **Metrics:**
        - Accuracy
        - F1 Score
        - Training Time
        - Model Size
        
        **MLflow Tracking:**
        - Parameters
        - Metrics
        - Models
        - Artifacts
        
        **Best Practices:**
        - Compare models
        - Track experiments
        - Version models
        - Reproduce results
        """)
    
    st.markdown("---")
    st.subheader("Performance Expectations")
    
    st.markdown("""
    **Note on Imbalanced Data:**
    
    The dataset is highly imbalanced with very few table headers (label=1) compared to 
    data cells (label=0). This is expected in real-world table detection scenarios.
    
    - **Accuracy**: Will be very high due to class imbalance
    - **F1 Score**: More meaningful metric for this task
    - **Precision/Recall**: Consider for header detection performance
    """)

def show_services_status():
    """Show services status page."""
    st.header("🔧 Services Status")
    
    st.markdown("""
    Monitor the status of MLflow and Apache Airflow services for experiment tracking 
    and pipeline orchestration.
    """)
    
    st.markdown("---")
    
    # MLflow Status
    st.subheader("📊 MLflow - Experiment Tracking")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **MLflow** tracks experiments, parameters, metrics, and models.
        
        **To start MLflow UI:**
        ```bash
        mlflow ui --host 0.0.0.0 --port 5000
        ```
        
        **Access at:** http://localhost:5000
        
        **Features:**
        - Compare model runs
        - View metrics and parameters
        - Download artifacts
        - Model registry
        """)
        
        if st.button("📋 Copy MLflow Start Command"):
            st.code("mlflow ui --host 0.0.0.0 --port 5000", language="bash")
    
    with col2:
        st.info("""
        **Status:** Not Running
        
        Start MLflow to enable:
        - Experiment tracking
        - Model comparison
        - Artifact storage
        - Model versioning
        """)
    
    st.markdown("---")
    
    # Airflow Status
    st.subheader("🔄 Apache Airflow - Pipeline Orchestration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Apache Airflow** orchestrates the Excel table detection pipeline.
        
        **To initialize and start Airflow:**
        ```bash
        # Initialize database
        export AIRFLOW_HOME=/home/runner/work/mid_sem/mid_sem/airflow_home
        airflow db init
        
        # Create admin user
        airflow users create \\
            --username admin \\
            --firstname Admin \\
            --lastname User \\
            --role Admin \\
            --email admin@example.com \\
            --password admin
        
        # Start webserver
        airflow webserver --port 8080 -D
        
        # Start scheduler
        airflow scheduler -D
        ```
        
        **Access at:** http://localhost:8080
        """)
        
        if st.button("📋 Copy Airflow Start Commands"):
            st.code("""
export AIRFLOW_HOME=/home/runner/work/mid_sem/mid_sem/airflow_home
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
airflow webserver --port 8080 -D
airflow scheduler -D
            """, language="bash")
    
    with col2:
        st.info("""
        **Status:** Not Running
        
        Start Airflow to enable:
        - Pipeline scheduling
        - DAG visualization
        - Task monitoring
        - Automated workflows
        """)
    
    st.markdown("---")
    
    # Quick Start Guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    **1. Start MLflow:**
    ```bash
    mlflow ui --host 0.0.0.0 --port 5000 &
    ```
    
    **2. Start Airflow (if needed):**
    ```bash
    export AIRFLOW_HOME=/home/runner/work/mid_sem/mid_sem/airflow_home
    airflow db init
    airflow webserver --port 8080 -D
    airflow scheduler -D
    ```
    
    **3. Run Training Pipeline:**
    ```bash
    python run_pipline.py
    ```
    
    **4. Access Services:**
    - MLflow: http://localhost:5000
    - Airflow: http://localhost:8080 (login: admin/admin)
    - Streamlit UI: http://localhost:8501
    """)
    
    st.markdown("---")
    st.subheader("📖 Documentation Links")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **MLflow:**
        - [Official Documentation](https://mlflow.org/docs/latest/index.html)
        - [Tracking API](https://mlflow.org/docs/latest/tracking.html)
        - [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
        """)
    
    with col2:
        st.markdown("""
        **Apache Airflow:**
        - [Official Documentation](https://airflow.apache.org/docs/)
        - [Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)
        - [DAG Writing](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dags.html)
        """)


if __name__ == "__main__":
    main()
