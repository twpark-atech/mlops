# MLOps Traffic Pipeline

ITS 5분 단위 교통 데이터를 Kafka → MinIO(Data Lake) → Spark(Bronze/Silver/Gold) → PostgreSQL → PyTorch 모델 학습까지 자동화하는 예제입니다. `airflow/docker-compose.airflow.yaml` 로 Airflow 스택을, `docker/docker-compose.core.yml` 로 MinIO/Kafka/Postgres/MLflow 등을 실행할 수 있습니다.

## 디렉터리 개요

| 경로 | 설명 |
| --- | --- |
| `config/*.yml` | MinIO/Kafka/Datalake 기본 설정 및 ingestion job 목록 |
| `src/ingestion/file_ingestor` | ITS ZIP 다운로드 → MinIO 업로드 → Kafka 이벤트 발행 |
| `src/pipelines/*` | Raw→Bronze→Silver→Gold Spark 파이프라인 |
| `src/training/*` | ConvLSTM 모델 및 학습 스크립트 |
| `airflow/dags/its_traffic_full_pipeline.py` | ingestion → Spark 변환 → PostgreSQL → 학습을 한 번에 수행하는 DAG |
| `docker/docker-compose.core.yml` | MinIO, Kafka, Postgres, MLflow 등 핵심 인프라 |
| `airflow/docker-compose.airflow.yaml` | Airflow 클러스터 (CeleryExecutor) |

## 데이터 추출(ingestion) 변경

1. **URL 또는 Kafka 토픽/MinIO 경로를 바꾸려면** `config/ingestion_file.yml` 에서 job 항목을 수정합니다. `url`, `kafka_topic`, `lake_prefix` 등을 원하는 API 혹은 파일로 교체하세요.
2. **코드 레벨 수정**이 필요하면 `src/ingestion/file_ingestor/main.py` 의 `download_zip_in_memory`, `upload_zip_to_datalake`, `ingest_from_url` 등을 편집하면 됩니다. Airflow DAG는 `ingest_from_url()` 을 호출하므로 시그니처를 유지하세요.
3. 새 데이터셋을 추가하고 싶다면 ingestion YAML에 job을 추가하고 DAG `params.ingestion_url_template`를 필요에 맞게 바꿉니다.

## 전처리(Spark 파이프라인) 수정

1. RAW→BRONZE: `src/pipelines/raw_to_bronze.py` 의 `run_its_traffic_raw_to_bronze` 함수에서 입력 경로(`base_raw`), 스키마, 선택 컬럼, 정제 로직을 수정합니다.
2. BRONZE→SILVER: `src/pipelines/bronze_to_silver.py` 에서 링크 필터(`LINKIDS_ALL`), 집계/타임라인 로직을 변경합니다.
3. SILVER→GOLD: `src/pipelines/silver_to_gold.py` 에서 컬럼 매핑, Postgres 연결 정보(환경변수), 삭제/적재 방식을 조정할 수 있습니다.
4. DAG 안에서 같은 함수를 쓰므로 함수 이름/파라미터를 유지한 채 내부 로직을 수정하는 방식이 가장 안전합니다.

## AI 모델 변경

1. **모델 구조**: `src/training/model.py` 의 `TrafficConvLSTM` 을 원하는 PyTorch 모델로 교체하거나 추가합니다.
2. **학습 로직**: `src/training/train_its_traffic_convlstm.py` 의 `TrainConfig`, 전처리 유틸리티, `train_one_epoch`, `evaluate`, `run_training` 을 수정하세요. Airflow DAG는 `run_training(cfg)` 를 호출하므로 이 함수만 유효하게 유지하면 됩니다.
3. **PostgreSQL에서 읽는 컬럼**: `load_gold_from_postgres` 내 SQL을 수정해 원하는 피처를 가져오고, 이어지는 전처리 파이프라인(`ensure_5min_grid`, `impute_missing`)을 데이터 특성에 맞게 조정합니다.

## Scheduler(DAG) 수정

`airflow/dags/its_traffic_full_pipeline.py` 에서 다음을 조정할 수 있습니다.

1. **스케줄/파라미터**: DAG 정의 부분의 `schedule_interval`, `params` (예: `training_window_days`, `ingestion_url_template`)를 변경하세요.
2. **Task 구성**: PythonOperator 단계(ingest, raw_to_bronze, bronze_to_silver, silver_to_gold, train)를 추가/삭제하거나 의존 관계(`>>`)를 바꿀 수 있습니다.
3. **커스텀 파라미터 전달**: 각 task 함수는 `**context` 를 통해 `params`를 받으므로, DAG params에 새 키를 넣고 함수 내부에서 사용하세요.

## 실행 방법

1. **코어 서비스 기동**  
   ```bash
   docker compose -f docker/docker-compose.core.yml up -d
   ```
   MinIO, Kafka, Postgres, MLflow 등이 준비됩니다.

2. **Airflow 스택 기동**  
   - `airflow/docker-compose.airflow.yaml` 의 `_PIP_ADDITIONAL_REQUIREMENTS` 에 `kafka-python`, `six`, `minio`, `psycopg2-binary`, `pandas`, `pyspark`, `torch`, `mlflow`, `boto3`, `pyarrow` 등 DAG 실행에 필요한 라이브러리를 지정했습니다. Airflow 컨테이너 기동 시 자동으로 설치됩니다.
   - `src` 디렉터리를 `/opt/airflow/src` 로 마운트하고 `PYTHONPATH=/opt/airflow:/opt/airflow/src` 를 설정해 DAG가 프로젝트 모듈을 import 할 수 있습니다.
   - 실행:
     ```bash
     docker compose -f airflow/docker-compose.airflow.yaml up -d
     ```
   - 재시작 시에는 `down` 후 `up -d --build` 로 패키지 설치를 다시 적용할 수 있습니다.

3. **Airflow DAG 사용**
   - Airflow UI (`http://localhost:8080`, 기본 계정 `airflow/airflow`)에서 `its_traffic_full_pipeline` DAG를 활성화하거나 CLI로 트리거합니다.
   - CLI 예:
     ```bash
     docker compose -f airflow/docker-compose.airflow.yaml exec airflow-scheduler \
       airflow dags trigger its_traffic_full_pipeline --conf '{"training_window_days":30}'
     ```

4. **환경 변수**
   - MinIO, Kafka, Postgres 접속 정보는 `config/base_config.yml`, `src/common/spark_session.py`, `src/pipelines/silver_to_gold.py`, `TrainConfig` 등에 기본값이 정의되어 있습니다. 실제 배포 환경에 맞춰 컨테이너 환경 변수 또는 `.env` 파일에서 덮어쓰세요.

5. **로그/모니터링**
   - Spark 작업, ingestion, 학습 로그는 각 컨테이너 표준출력과 Airflow task log에서 확인할 수 있습니다.

## 기타 팁

- 새 데이터 소스나 모델을 추가할 때는 동일한 함수 인터페이스를 유지하면 Airflow DAG 수정 없이 교체 가능합니다.
- `_PIP_ADDITIONAL_REQUIREMENTS` 는 Airflow 컨테이너마다 실행 시 pip install 을 수행하므로, 의존성이 늘어날 경우 커스텀 이미지를 빌드하는 방식을 고려하세요.
