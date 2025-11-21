# MLOps Pipeline Guide

## Docker
### 1. Create Network (Do not exist)
```bash
docker network create mlops
```

### 2. Build Airflow Custom Image
```bash
docker build -t airflow-custom -f airflow/Dockerfile.airflow .
```

### 3. Execute Docker Compose
```bash
docker compose -f docker/docker-compose.core.yml up -d
docker compose -f docker/docker-compose.spark.yml up -d
docker compose -f airflow/docker-compose.airflow.yml up -d
```