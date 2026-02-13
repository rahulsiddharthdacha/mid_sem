import numpy as np

# Lightweight semantic model - lazy loaded
_model = None
_model_load_attempted = False

def get_model():
    """Lazy load the sentence transformer model"""
    global _model, _model_load_attempted
    
    if not _model_load_attempted:
        _model_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Loaded semantic model successfully")
        except Exception as e:
            print(f"⚠️ Could not load semantic model: {e}")
            print("⚠️ Using zero embeddings for semantic features")
            _model = None
    
    return _model

def extract_semantic_features(text_series):
    """Extract semantic features or use zeros if model unavailable"""
    embeddings = []
    model = get_model()

    for text in text_series:
        # If model is not available or text is empty/numeric, use zeros
        if model is None or text.strip() == "" or text.isnumeric():
            embeddings.append(np.zeros(384))
        else:
            embeddings.append(model.encode(text))

    return np.vstack(embeddings)
