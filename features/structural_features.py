import pandas as pd

def extract_structural_features(df):
    """
    Extract structural features from a DataFrame.
    
    Note: This function now properly includes the header row (column names)
    as row_idx=0, and shifts all data rows accordingly.
    """
    features = []
    
    # First, add the header row (column names) as row_idx=0
    for col_idx in range(df.shape[1]):
        header_text = str(df.columns[col_idx])
        
        features.append({
            "row_idx": 0,  # Headers are at row 0
            "col_idx": col_idx,
            "is_empty": 0,  # Headers are never empty (they're column names)
            "is_numeric": 0,  # Column names are typically not numeric
            "row_density": 1.0,  # Header row is always fully populated
            "col_density": df.iloc[:, col_idx].notna().mean(),  # Column density from data
            "text": header_text
        })
    
    # Then add all data rows, starting from row_idx=1
    for data_row_idx in range(df.shape[0]):
        for col_idx in range(df.shape[1]):
            cell = df.iat[data_row_idx, col_idx]
            
            # Adjust row_idx to account for the header row
            actual_row_idx = data_row_idx + 1

            features.append({
                "row_idx": actual_row_idx,
                "col_idx": col_idx,
                "is_empty": int(pd.isna(cell)),
                "is_numeric": int(isinstance(cell, (int, float)) and not pd.isna(cell)),
                "row_density": df.iloc[data_row_idx].notna().mean(),
                "col_density": df.iloc[:, col_idx].notna().mean(),
                "text": "" if pd.isna(cell) else str(cell)
            })

    return pd.DataFrame(features)
