from fastapi import FastAPI, UploadFile
import openpyxl
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
async def detect_tables(file: UploadFile):
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
    preds = model.predict(df)

    return {"table_cells": preds.tolist()}
