import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.ml_pipeline import MLPipeline
from configs.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def main():
    """Main application function."""
    
    st.title("🏥 Insurance Premium Prediction System")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    config = get_config()
    
    # Main navigation
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Data Ingestion", "Feature Engineering", "Model Training", "Predictions", "Analytics"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Data Ingestion":
        show_data_ingestion()
    elif page == "Feature Engineering":
        show_feature_engineering()
    elif page == "Model Training":
        show_model_training()
    elif page == "Predictions":
        show_predictions()
    elif page == "Analytics":
        show_analytics()

def show_home():
    """Show home page."""
    st.header("Welcome to Insurance Premium Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### About This Application
        
        This is a comprehensive ML application for predicting insurance premiums.
        
        **Features:**
        - 📥 Data ingestion and normalization
        - 🔧 Advanced feature engineering
        - 🤖 Machine learning model training
        - 🎯 Real-time predictions
        - 📊 Performance analytics
        
        **Supported Models:**
        - Random Forest Regressor
        - Gradient Boosting Regressor
        - Linear Regression
        """)
    
    with col2:
        st.markdown("""
        ### Getting Started
        
        1. **Load Data**: Upload your Excel file in the Data Ingestion section
        2. **Engineer Features**: Configure and create features
        3. **Train Model**: Select a model and train it
        4. **Make Predictions**: Predict on new data
        5. **View Analytics**: Analyze model performance
        
        ### Quick Stats
        """)
        
        if st.session_state.data_loaded and st.session_state.pipeline:
            summary = st.session_state.pipeline.get_pipeline_summary()
            st.metric("Data Shape", f"{summary['data_shape']}")
            st.metric("Model Status", "Trained" if st.session_state.model_trained else "Not Trained")

def show_data_ingestion():
    """Show data ingestion page."""
    st.header("📥 Data Ingestion")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Load Data")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = Path('/tmp') / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Initialize pipeline
                st.session_state.pipeline = MLPipeline({})
                
                # Load data
                data = st.session_state.pipeline.load_data(str(temp_path))
                st.session_state.data_loaded = True
                
                st.success("✅ Data loaded successfully!")
                st.dataframe(data.head(10))
                
                # Data statistics
                st.subheader("Data Statistics")
                st.write(data.describe())
                
                # Data info
                st.subheader("Data Info")
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", data.shape[0])
                col2.metric("Columns", data.shape[1])
                col3.metric("Missing Values", data.isnull().sum().sum())
                
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
        
        else:
            st.info("👈 Please upload an Excel file to begin")
    
    with col2:
        st.subheader("Supported Formats")
        st.markdown("""
        - .xlsx
        - .xls
        
        **Data Requirements:**
        - Columns with numeric and categorical data
        - A target column for regression
        """)

def show_feature_engineering():
    """Show feature engineering page."""
    st.header("🔧 Feature Engineering")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Ingestion section")
        return
    
    pipeline = st.session_state.pipeline
    data = pipeline.data
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Feature Configuration")
        
        # Select target column
        target_col = st.selectbox(
            "Select Target Column",
            data.columns,
            help="Column to predict"
        )
        
        if st.button("Engineer Features", key="engineer_btn"):
            try:
                X = pipeline.engineer_features(target_col)
                st.success("✅ Features engineered successfully!")
                st.dataframe(X.head(10))
                st.metric("Features Count", X.shape[1])
                
            except Exception as e:
                st.error(f"❌ Error engineering features: {e}")
    
    with col2:
        st.subheader("Feature Options")
        st.markdown("""
        - Scaling: Standard or MinMax
        - Encoding: Label encoding for categorical
        - Polynomial features
        - Interaction features
        """)

def show_model_training():
    """Show model training page."""
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["random_forest", "gradient_boosting", "linear_regression"]
        )
        
        # Training button
        if st.button("Train Model", key="train_btn"):
            try:
                with st.spinner("Training model..."):
                    metrics = st.session_state.pipeline.train_model(model_type)
                    st.session_state.model_trained = True
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Train R²", f"{metrics['train_r2']:.4f}")
                col2.metric("Test R²", f"{metrics['test_r2']:.4f}")
                col3.metric("Test RMSE", f"{metrics['test_rmse']:.4f}")
                col4.metric("Test MAE", f"{metrics['test_mae']:.4f}")
                
                # Save model
                config = get_config()
                st.session_state.pipeline.save_model(str(config.MODEL_PATH))
                st.info(f"Model saved to {config.MODEL_PATH}")
                
            except Exception as e:
                st.error(f"❌ Error training model: {e}")
    
    with col2:
        st.subheader("Model Details")
        st.markdown(f"""
        **Selected Model:** {model_type}
        
        Model will be trained on:
        - Features: Engineered
        - Target: Previously selected
        - Test Size: 20%
        """)


def show_predictions():
    """Show predictions page."""
    st.header("🎯 Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first")
        return
    
    st.subheader("Single Prediction")
    
    # Get feature columns
    pipeline = st.session_state.pipeline
    feature_cols = pipeline.features_df.columns
    
    # Create input form
    input_data = {}
    cols = st.columns(3)
    for idx, col in enumerate(feature_cols):
        with cols[idx % 3]:
            input_data[col] = st.number_input(
                f"{col}",
                value=0.0,
                step=0.1
            )
    
    if st.button("Predict", key="predict_btn"):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = pipeline.predict(input_df)
            
            st.success(f"Prediction: **${prediction[0]:.2f}**")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
    
    # Batch prediction
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload data for batch prediction", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_data = pd.read_csv(uploaded_file)
            else:
                batch_data = pd.read_excel(uploaded_file)
            
            if st.button("Predict Batch"):
                predictions = pipeline.predict(batch_data[feature_cols])
                
                result_df = batch_data.copy()
                result_df['Predicted_Premium'] = predictions
                
                st.success("✅ Batch predictions completed!")
                st.dataframe(result_df)
                
                # Download results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error in batch prediction: {e}")


def show_analytics():
    """Show analytics page."""
    st.header("📊 Analytics")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first to view analytics")
        return
    
    pipeline = st.session_state.pipeline
    metrics = pipeline.results.get('training', {})
    
    # Metrics overview
    st.subheader("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train R²", f"{metrics.get('train_r2', 0):.4f}")
    col2.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")
    col3.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.4f}")
    col4.metric("Test MAE", f"{metrics.get('test_mae', 0):.4f}")
    
    # Feature importance
    if hasattr(pipeline.model.model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        feature_importance = pipeline.model.get_feature_importance(
            pipeline.features_df.columns.tolist()
        )
        
        importance_df = pd.DataFrame(feature_importance.items(), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
        
        st.bar_chart(importance_df.set_index('Feature'))


if __name__ == "__main__":
    main()