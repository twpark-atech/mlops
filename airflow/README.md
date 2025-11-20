# Airflow 설치 방법 (커스텀 이미지 기반)

Airflow DAG에서 필요한 pip 패키지가 방대하므로

먼저 Airflow 커스텀 이미지 빌드 후 Docker Compose 실행한다.

### 1) Airflow 커스텀 이미지 빌드

`Dockerfile.airflow`를 기반으로 커스텀 이미지 생성

```bash
docker build -t airflow-custom -f Dockerfile.airflow .
```

### 2) Docker Compose 실행

커스텀 이미지가 포함된 `docker-compose.airflow.yaml` 실행

```bash
docker compose -f docker-compose.airflow.yaml up -d
```