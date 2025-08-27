# Deployment Guide

This guide provides instructions for deploying the Machine Learning Transformer Demo in various environments.

## Prerequisites

- Python 3.8 or higher
- Docker (for containerized deployment)
- Kubernetes (for orchestration)
- At least 4GB RAM (8GB recommended for GPU)
- CUDA-compatible GPU (optional, for GPU acceleration)

## Local Deployment

### Using Python

1. Clone the repository:
```bash
git clone https://github.com/yourusername/machine-learning-transformer-demo.git
cd machine-learning-transformer-demo
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the demo:
```bash
python demo_run.py
```

### Using Docker

1. Build the Docker image:
```bash
docker build -t ml-transformer-demo .
```

2. Run the container:
```bash
docker run -p 8000:8000 ml-transformer-demo
```

3. Access the API at `http://localhost:8000`

### Using Docker Compose

1. Start all services:
```bash
docker-compose up -d
```

2. Access services:
   - API: `http://localhost:8000`
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000`

3. Stop services:
```bash
docker-compose down
```

## Cloud Deployment

### AWS Deployment

1. Create an EC2 instance with GPU support (optional):
   - AMI: Deep Learning AMI (Ubuntu)
   - Instance type: p2.xlarge or g4dn.xlarge (for GPU)

2. Install Docker:
```bash
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```

3. Deploy the application:
```bash
docker run -p 8000:8000 ml-transformer-demo
```

4. Configure security groups to allow inbound traffic on port 8000

### Google Cloud Platform (GCP)

1. Create a Compute Engine instance:
   - Machine type: n1-standard-4 or higher
   - GPUs: NVIDIA Tesla T4 (optional)

2. Install Docker:
```bash
sudo apt update
sudo apt install docker.io -y
sudo usermod -a -G docker $USER
```

3. Deploy the application:
```bash
docker run -p 8000:8000 ml-transformer-demo
```

4. Create a firewall rule to allow traffic on port 8000

### Microsoft Azure

1. Create a Virtual Machine:
   - Size: Standard_NC4as_T4_v3 (for GPU) or Standard_D4s_v3
   - Image: Ubuntu Server 20.04 LTS

2. Install Docker:
```bash
sudo apt update
sudo apt install docker.io -y
sudo usermod -a -G docker $USER
```

3. Deploy the application:
```bash
docker run -p 8000:8000 ml-transformer-demo
```

4. Configure Network Security Group to allow inbound traffic on port 8000

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (minikube, GKE, EKS, AKS)
- kubectl configured
- helm (optional)

### Deployment Steps

1. Apply the Kubernetes manifests:
```bash
kubectl apply -f k8s/deployment.yaml
```

2. Check the deployment status:
```bash
kubectl get deployments
kubectl get services
```

3. Access the service:
```bash
kubectl port-forward service/ml-transformer-demo-service 8000:80
```

### Scaling

Scale the deployment to handle more traffic:
```bash
kubectl scale deployment ml-transformer-demo --replicas=3
```

### Updating the Application

1. Build and push a new Docker image:
```bash
docker build -t ml-transformer-demo:v2 .
docker push ml-transformer-demo:v2
```

2. Update the deployment:
```bash
kubectl set image deployment/ml-transformer-demo api=ml-transformer-demo:v2
```

## Monitoring and Observability

### Prometheus Integration

The application exposes Prometheus metrics at `/metrics` endpoint.

1. Configure Prometheus to scrape metrics:
```yaml
scrape_configs:
  - job_name: 'ml-transformer-demo'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboard

1. Access Grafana at `http://localhost:3000` (default credentials: admin/admin)
2. Add Prometheus as a data source
3. Import the provided dashboard JSON or create custom dashboards

### Health Checks

The application provides health check endpoints:
- `/health`: Basic health status
- `/metrics`: Prometheus metrics

Configure your load balancer or orchestration platform to use these endpoints.

## Performance Tuning

### GPU Configuration

To enable GPU acceleration:

1. Ensure CUDA drivers are installed
2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. The application will automatically detect and use GPU if available

### Memory Optimization

Configure memory limits in Docker:
```bash
docker run -p 8000:8000 --memory=4g ml-transformer-demo
```

Or in Kubernetes:
```yaml
resources:
  requests:
    memory: "2Gi"
  limits:
    memory: "4Gi"
```

### Batch Processing

Optimize batch processing by adjusting:
- `max_batch_size` in the inference engine
- Number of concurrent requests
- Sequence length limits

## Security Considerations

### Network Security

- Use HTTPS in production
- Implement proper firewall rules
- Restrict access to monitoring endpoints

### Container Security

- Use minimal base images
- Regularly update dependencies
- Scan images for vulnerabilities

### API Security

- Implement authentication and authorization
- Add rate limiting
- Validate and sanitize all inputs

## Backup and Recovery

### Model Backup

Models are automatically saved during training. To backup:

1. Copy model files from the `models/` directory
2. Export the model registry:
```bash
python -c "from src.registry import get_model_registry; registry = get_model_registry(); registry.export_registry('model_registry_backup.json')"
```

### Configuration Backup

Backup configuration files:
- `config/` directory
- Environment variables
- Docker Compose files

### Disaster Recovery

To restore from backup:

1. Restore model files to the `models/` directory
2. Import the model registry:
```bash
python -c "from src.registry import get_model_registry; registry = get_model_registry(); registry.import_registry('model_registry_backup.json')"
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Check file permissions and model file integrity
2. **GPU not detected**: Verify CUDA installation and PyTorch CUDA support
3. **Memory errors**: Reduce batch size or sequence length
4. **Slow performance**: Enable GPU acceleration or optimize batch processing

### Logs and Debugging

View application logs:
```bash
# Docker
docker logs <container_id>

# Kubernetes
kubectl logs <pod_name>

# Direct Python execution
python demo_run.py 2>&1 | tee app.log
```

### Performance Monitoring

Monitor system resources:
```bash
# CPU and memory usage
htop

# GPU usage (if available)
nvidia-smi

# Disk I/O
iostat
```

## Maintenance

### Regular Updates

1. Update dependencies:
```bash
pip install --upgrade -r requirements.txt
```

2. Rebuild Docker images with updated base images

3. Apply security patches to the operating system

### Model Retraining

1. Prepare new training data
2. Run training script with updated data
3. Register new model version in the registry
4. Deploy updated model

### Performance Review

Regularly review:
- API response times
- Resource utilization
- Error rates
- Cache hit ratios
- Model accuracy metrics

## Support

For issues and questions:
1. Check the documentation and README files
2. Review logs for error messages
3. Search existing issues in the repository
4. Create a new issue with detailed information about the problem