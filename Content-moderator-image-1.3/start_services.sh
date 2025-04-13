#!/bin/bash

echo "🔄 Starting Ollama server..."
ollama serve &

# Wait for Ollama to start
sleep 10  # Ensure Ollama is running before pulling models

echo "📥 Pulling base model..."
ollama pull wizardlm2:7b
ollama pull gemma3:12b 

echo "🛠 Creating custom moderation model..."
ollama create cyber-moderator-Wlm:7b -f /app/models/Modelfile  # Absolute path
ollama create cyber-moderator-G3:12b  -f /app/models/Modelfile_vision

echo "🚀 Starting Backend..."
bash /app/backend/start.sh &

echo "🎨 Starting Frontend..."
bash /app/frontend/start_frontend.sh
