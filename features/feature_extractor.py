import pandas as pd
from structural_features import extract_structural_features
from semantic_features import extract_semantic_features

def generate_header_label(row_idx, text, is_numeric):
    """
    Weak supervision rule for header labeling
    
    Headers are typically:
    - In the first row (row_idx == 0)
    - Non-numeric text containing alphabetic characters
    - Descriptive labels for columns
    
    This improved logic identifies headers more accurately by:
    1. Marking all non-numeric cells in row 0 as headers
    2. Also considering text cells with alphabetic content as potential headers
    """
    # First row non-numeric cells are almost always headers
    if row_idx == 0 and not is_numeric:
        return 1
    
    # Text with alphabetic characters (not purely numeric) could be headers
    # This catches column names that might appear in the first row
    text_str = str(text).strip()
    if text_str and not is_numeric and any(c.isalpha() for c in text_str):
        # If it's in the first row, it's definitely a header
        if row_idx == 0:
            return 1
        # Otherwise, it's more likely to be data unless it matches header patterns
        # For now, we'll be conservative and label as data
        return 0
    
    # Numeric or empty cells are data
    return 0


def extract_features(excel_path):
    df = pd.read_excel(excel_path)

    structural_df = extract_structural_features(df)
    structural_df["label"] = structural_df.apply(
        lambda row: generate_header_label(
            row["row_idx"],
            row["text"],
            row["is_numeric"]
        ),
        axis=1
    )

    semantic_vectors = extract_semantic_features(structural_df["text"])

    semantic_df = pd.DataFrame(
        semantic_vectors,
        columns=[f"sem_{i}" for i in range(semantic_vectors.shape[1])]
    )

    final_features = pd.concat(
        [structural_df.drop(columns=["text"]), semantic_df],
        axis=1
    )

    return final_features


if __name__ == "__main__":
    from pathlib import Path
    
    excel_path = "data/sample.xlsx"
    output_path = "features/features.csv"

    features = extract_features(excel_path)
    Path("features").mkdir(exist_ok=True)
    features.to_csv(output_path, index=False)

    print(f"✅ Features saved to {output_path}")

