# Use official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt


# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Expose necessary ports
EXPOSE 8000 8501

#The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501:
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Make scripts executable
RUN chmod +x start_services.sh backend/start.sh frontend/start_frontend.sh

# Use ENTRYPOINT instead of CMD for better control
ENTRYPOINT ["/bin/bash", "start_services.sh"]
