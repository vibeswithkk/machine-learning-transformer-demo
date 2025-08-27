# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY examples/requirements.txt examples/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r examples/requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn

# Copy application code
COPY . .

# Create demo model if it doesn't exist
RUN python -c "from src.api import initialize_model; initialize_model()" || echo "Model initialization skipped"

# Expose port
EXPOSE 8000

# Run the API server
CMD ["python", "-m", "src.api"]