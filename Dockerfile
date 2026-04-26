# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for ML + parsing libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Ensure src is in PYTHONPATH
ENV PYTHONPATH="/app"

# Expose Streamlit port
EXPOSE 8501

# Prevent buffering issues
ENV PYTHONUNBUFFERED=1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]