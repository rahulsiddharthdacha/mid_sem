from sentence_transformers import SentenceTransformer
import numpy as np

# Lightweight semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_semantic_features(text_series):
    embeddings = []

    for text in text_series:
        if text.strip() == "" or text.isnumeric():
            embeddings.append(np.zeros(384))
        else:
            embeddings.append(model.encode(text))

    return np.vstack(embeddings)
