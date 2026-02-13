#!/bin/bash

echo "🚀 Starting Streamlit UI..."
echo "================================"

# Install UI requirements if needed
pip install -q -r ui/requirements_ui.txt

# Start Streamlit
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0

echo ""
echo "✅ Streamlit UI started!"
echo "Access at: http://localhost:8501"
