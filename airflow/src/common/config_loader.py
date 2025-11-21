# airflow/src/common/config_loader.py
import os
import yaml
from typing import Any, Dict, Iterable, Tuple, Union
from pathlib import Path


def _iter_existing_paths(candidates: Iterable[Path]) -> Iterable[Path]:
    seen = set[Any]()
    for path in candidates:
        if not path:
            continue
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            yield resolved


def _project_root() -> Path:
    resolved = Path(__file__).resolve()
    parents = resolved.parents
    idx = 3 if len(parents) > 3 else len(parents) - 1
    return parents[idx]


def _load_env_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _load_dotenv_if_present() -> None:
    env_path_hint = os.getenv("ENV_FILE_PATH")
    candidates = [
        Path(env_path_hint) if env_path_hint else None,
        Path("/app/.env"),
        Path("/opt/airflow/.env"),
        _project_root() / ".env",
        Path(__file__).resolve().parents[2] / ".env"
    ]
    for env_path in _iter_existing_paths(candidates):
        _load_env_file(env_path)


_load_dotenv_if_present()


def _resolve_config_dir() -> Path:
    config_hint = os.getenv("APP_CONFIG_DIR")
    candidates = [
        Path(config_hint) if config_hint else None,
        Path("/app/config"),
        Path("/opt/airflow/config"),
        _project_root() / "config",
        Path(__file__).resolve().parents[2] / "config"
    ]
    for config_dir in _iter_existing_paths(candidates):
        return config_dir
    raise FileNotFoundError("Unable to locate a config directory (searched /app, /opt/airflow and project root).")


CONFIG_DIR = _resolve_config_dir()


def _coerce_env_value(raw: str, current: Any) -> Any:
    if current is None:
        return raw
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


EnvNames = Union[str, Tuple[str, ...]]
ENV_OVERRIDE_MAP: Tuple[Tuple[Tuple[str, ...], EnvNames], ...] = (
    (("kafka", "bootstrap_servers"), "KAFKA_BOOTSTRAP_SERVERS"),
    (("minio", "endpoint"), "MINIO_ENDPOINT"),
    (("minio", "access_key"), "MINIO_ACCESS_KEY"),
    (("minio", "secret_key"), "MINIO_SECRET_KEY"),
    (("minio", "bucket"), "MINIO_BUCKET"),
    (("minio", "secure"), "MINIO_SECURE"),
    (("datalake", "prefix"), "INGESTION_LAKE_PREFIX"),
    (("datalake", "paths", "raw"), "DATALAKE_RAW_PATH"),
    (("datalake", "paths", "bronze"), "DATALAKE_BRONZE_PATH"),
    (("datalake", "paths", "silver"), "DATALAKE_SILVER_PATH"),
    (("datalake", "paths", "gold"), "DATALAKE_GOLD_PATH"),
    (("ingestion", "url_template"), "INGESTION_URL_TEMPLATE"),
    (("ingestion", "job_name_prefix"), "INGESTION_JOB_NAME_PREFIX"),
    (("ingestion", "lake_prefix"), "INGESTION_LAKE_PREFIX"),
    (("postgres", "host"), "PG_HOST"),
    (("postgres", "port"), "PG_PORT"),
    (("postgres", "db"), "PG_DB"),
    (("postgres", "user"), "PG_USER"),
    (("postgres", "password"), "PG_PASSWORD"),
    (("postgres", "table"), ("PIPELINE_GOLD_TABLE", "ITS_TRAFFIC_GOLD_TABLE")),
    (("training", "job_name"), "TRAINING_JOB_NAME"),
    (("training", "window_days"), "TRAINING_WINDOW_DAYS"),
    (("training", "val_days"), "TRAINING_VAL_DAYS"),
    (("training", "seq_len"), "TRAINING_SEQ_LEN"),
    (("training", "pred_horizon"), "TRAINING_PRED_HORIZON"),
    (("training", "batch_size"), "TRAINING_BATCH_SIZE"),
    (("training", "epochs"), "TRAINING_EPOCHS"),
    (("training", "lr"), "TRAINING_LR"),
    (("training", "grad_clip"), "TRAINING_GRAD_CLIP"),
    (("training", "mlflow_experiment"), "MLFLOW_EXPERIMENT_NAME"),
    (("training", "mlflow_run_name"), "MLFLOW_RUN_NAME"),
    (("training", "mlflow_register_name"), "MLFLOW_REGISTER_NAME"),
    (("training", "minio_bucket"), "TRAINING_MINIO_BUCKET"),
    (("training", "minio_prefix"), "TRAINING_MINIO_PREFIX"),
    (("mlflow", "tracking_uri"), "MLFLOW_TRACKING_URI"),
)


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    for path, env_names in ENV_OVERRIDE_MAP:
        names = (env_names,) if isinstance(env_names, str) else env_names
        override_value = None
        for env_name in names:
            if env_name in os.environ:
                override_value = os.environ[env_name]
                break
        if override_value is None:
            continue
        cursor: Dict[str, Any] = data
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        key = path[-1]
        current = cursor.get(key)
        cursor[key] = _coerce_env_value(override_value, current)
    return data


def load_yaml(filename: str) -> Dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _apply_env_overrides(data)


def load_base_config() -> Dict[str, Any]:
    return load_yaml("base_config.yml")


def load_ingestion_file_config() -> Dict[str, Any]:
    return load_yaml("ingestion_file.yml")


def load_spark_jobs_config() -> Dict[str, Any]:
    return load_yaml("spark_jobs.yml")
