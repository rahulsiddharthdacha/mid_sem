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
    
    This improved logic identifies headers more accurately:
    - All non-numeric cells in row 0 are marked as headers
    - This ensures column names are always detected as headers
    """
    # First row non-numeric cells are headers (column names)
    if row_idx == 0 and not is_numeric:
        return 1
    
    # All other cells are data
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

