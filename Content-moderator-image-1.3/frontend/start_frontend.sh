#!/bin/bash

echo "🎨 Starting Streamlit Frontend..."
cd /app/frontend  # Ensure correct directory
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
