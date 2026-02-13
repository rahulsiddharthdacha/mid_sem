import pandas as pd

def extract_structural_features(df):
    features = []

    for row_idx in range(df.shape[0]):
        for col_idx in range(df.shape[1]):
            cell = df.iat[row_idx, col_idx]

            features.append({
                "row_idx": row_idx,
                "col_idx": col_idx,
                "is_empty": int(pd.isna(cell)),
                "is_numeric": int(isinstance(cell, (int, float))),
                "row_density": df.iloc[row_idx].notna().mean(),
                "col_density": df.iloc[:, col_idx].notna().mean(),
                "text": "" if pd.isna(cell) else str(cell)
            })

    return pd.DataFrame(features)
