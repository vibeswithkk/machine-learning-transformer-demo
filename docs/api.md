# REST API Documentation

This document provides detailed information about the REST API for the Machine Learning Transformer Demo.

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET /

Returns basic information about the API.

**Response:**
```json
{
  "message": "Machine Learning Transformer Demo API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### GET /health

Returns the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true,
  "timestamp": "2023-01-01T12:00:00Z"
}
```

### POST /predict

Predicts sentiment for a single text input.

**Request Body:**
```json
{
  "text": "This is a sample text for classification"
}
```

**Response:**
```json
{
  "prediction": "0",
  "confidence": 0.95,
  "probabilities": {
    "0": 0.95,
    "1": 0.03,
    "2": 0.02
  },
  "processing_time_ms": 15.5
}
```

### POST /predict/batch

Predicts sentiment for a batch of text inputs.

**Request Body:**
```json
{
  "texts": [
    "Text 1",
    "Text 2",
    "Text 3"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "0",
      "confidence": 0.95,
      "probabilities": {
        "0": 0.95,
        "1": 0.03,
        "2": 0.02
      },
      "processing_time_ms": 15.5
    }
  ],
  "total_processing_time_ms": 45.2,
  "throughput": 66.4
}
```

### GET /model/info

Returns information about the loaded model.

**Response:**
```json
{
  "model_path": "demo_model.pt",
  "model_version": "unknown",
  "device": "cuda",
  "max_sequence_length": 128,
  "max_batch_size": 32,
  "num_classes": 3,
  "classes": [],
  "cache_size": 0,
  "model_parameters": 123456
}
```

## Error Responses

All endpoints may return the following error responses:

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Prediction failed: <error message>"
}
```

## Running the API

### Using Python

```bash
# Install dependencies
pip install fastapi uvicorn

# Run the API
python -m src.api
```

### Using Docker

```bash
# Build the Docker image
docker build -t ml-transformer-demo .

# Run the container
docker run -p 8000:8000 ml-transformer-demo
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down
```

## API Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great product!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2"]}'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Single prediction
response = requests.post("http://localhost:8000/predict", 
                        json={"text": "This is a great product!"})
print(response.json())

# Batch prediction
response = requests.post("http://localhost:8000/predict/batch",
                        json={"texts": ["Text 1", "Text 2"]})
print(response.json())
```

## Monitoring and Metrics

The API exposes Prometheus metrics at `/metrics` endpoint for monitoring purposes.

### Available Metrics

- `model_predictions_total`: Total number of predictions by class
- `model_prediction_latency_seconds`: Prediction latency histogram
- `model_prediction_confidence`: Prediction confidence histogram
- `model_cache_hit_ratio`: Cache hit ratio gauge
- `model_performance_summary`: Model performance summary

## Security Considerations

- The API does not implement authentication by default
- In production, consider adding:
  - API key authentication
  - Rate limiting
  - Input validation and sanitization
  - HTTPS encryption
  - CORS policy configuration

## Performance Optimization

- The API supports GPU acceleration when available
- Batch processing for multiple texts
- LRU caching for repeated predictions
- Configurable batch size and sequence length limits
- Health checks for monitoring service status

## Troubleshooting

### Model Not Loading

If you receive a "503 Service Unavailable" error:

1. Check if the model file exists
2. Verify the model file is not corrupted
3. Ensure sufficient memory is available
4. Check Docker logs for error messages

### Slow Performance

If predictions are slow:

1. Ensure GPU is available and properly configured
2. Check if batch processing is being used for multiple texts
3. Verify the model is not being reloaded on each request
4. Monitor system resources (CPU, memory, GPU)

### Docker Issues

If Docker containers fail to start:

1. Check Docker logs: `docker logs <container_id>`
2. Verify port conflicts: `docker-compose ps`
3. Ensure sufficient system resources
4. Check file permissions for mounted volumes