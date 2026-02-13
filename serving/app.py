from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import openpyxl
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import sys
import io
import os
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.structural_features import extract_structural_features
from features.semantic_features import extract_semantic_features

app = FastAPI(
    title="Excel Table Detection API",
    description="ML-Based Detection of Financial Tables in Excel Using Metadata Features",
    version="1.0.0"
)

# Load the best model from MLflow artifacts
def load_best_model():
    """Load the best performing model from MLflow runs"""
    # Try to load a trained model from mlflow directory
    model_paths = [
        "./mlflow/282570187416666551/models/m-814d16bc57f742d8a7b6b0447842c850/artifacts/model.pkl",
        "./mlflows/126946449638471464/models/m-cded601c5c9d40efb0c71907b80c417e/artifacts/model.pkl",
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            try:
                model = joblib.load(model_path)
                print(f"✅ Loaded model from: {model_path}")
                return model
            except Exception as e:
                print(f"⚠️ Failed to load model from {model_path}: {e}")
                continue
    
    # If no model found, return None
    print("⚠️ No pre-trained model found. Please train a model first using: python model/train_model.py")
    return None

model = load_best_model()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Excel Table Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect-tables": "POST - Upload Excel file to detect tables and get JSON output",
            "/predict": "POST - Simple prediction endpoint (legacy)",
            "/health": "GET - Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def detect_tables(file: UploadFile):
    """Legacy prediction endpoint"""
    wb = openpyxl.load_workbook(file.file)
    sheet = wb.active

    cells = []
    for row in sheet.iter_rows():
        for cell in row:
            cells.append([
                cell.row,
                cell.column,
                int(cell.font.bold if cell.font else 0),
                int(cell.border.top.style is not None)
            ])

    df = pd.DataFrame(cells, columns=["row", "col", "bold", "border"])
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    preds = model.predict(df)

    return {"table_cells": preds.tolist()}

@app.post("/detect-tables")
async def detect_tables_complete(file: UploadFile) -> JSONResponse:
    """
    Complete table detection endpoint
    
    Upload an Excel file and get structured JSON output with:
    - Detected table headers
    - Table data cells
    - Cell locations and content
    - Table structure information
    
    This endpoint performs the complete cycle:
    1. Upload Excel file
    2. Extract structural and semantic features
    3. Use trained ML model to detect table cells
    4. Return structured JSON with detected tables
    """
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train a model first using: python model/train_model.py"
        )
    
    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .xlsx and .xls files are supported.")
    
    try:
        # Read the uploaded Excel file
        contents = await file.read()
        excel_file = io.BytesIO(contents)
        df = pd.read_excel(excel_file)
        
        # Step 1: Extract structural features
        structural_df = extract_structural_features(df)
        
        # Step 2: Extract semantic features
        semantic_vectors = extract_semantic_features(structural_df["text"])
        semantic_df = pd.DataFrame(
            semantic_vectors,
            columns=[f"sem_{i}" for i in range(semantic_vectors.shape[1])]
        )
        
        # Step 3: Combine features (drop 'text' column for prediction)
        features_for_prediction = pd.concat(
            [structural_df.drop(columns=["text"]), semantic_df],
            axis=1
        )
        
        # Step 4: Predict using the trained model
        predictions = model.predict(features_for_prediction)
        
        # Step 5: Structure the output as JSON
        # Combine predictions with cell information
        results_df = structural_df[["row_idx", "col_idx", "text"]].copy()
        results_df["is_header"] = predictions
        
        # Organize into table structure
        headers = []
        data_cells = []
        
        for idx, row in results_df.iterrows():
            cell_info = {
                "row": int(row["row_idx"]),
                "column": int(row["col_idx"]),
                "value": row["text"]
            }
            
            if row["is_header"] == 1:
                headers.append(cell_info)
            else:
                data_cells.append(cell_info)
        
        # Group data by rows for better structure
        data_by_row: Dict[int, List[Dict[str, Any]]] = {}
        for cell in data_cells:
            row_num = cell["row"]
            if row_num not in data_by_row:
                data_by_row[row_num] = []
            data_by_row[row_num].append(cell)
        
        # Convert to list of rows
        table_rows = []
        for row_num in sorted(data_by_row.keys()):
            row_cells = sorted(data_by_row[row_num], key=lambda x: x["column"])
            table_rows.append({
                "row_number": row_num,
                "cells": row_cells
            })
        
        # Create the final response
        response = {
            "status": "success",
            "filename": file.filename,
            "summary": {
                "total_cells": len(results_df),
                "header_cells": int(sum(predictions == 1)),
                "data_cells": int(sum(predictions == 0)),
                "dimensions": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1])
                }
            },
            "detected_tables": {
                "headers": sorted(headers, key=lambda x: (x["row"], x["column"])),
                "data": table_rows
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
