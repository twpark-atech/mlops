# src/common/config_loader.py
import yaml
from pathlib import Path
from typing import Any, Dict

DOCKER_CONFIG_DIR = Path("/app/config")
LOCAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

CONFIG_DIR = DOCKER_CONFIG_DIR if DOCKER_CONFIG_DIR.exists() else LOCAL_CONFIG_DIR

def load_yaml(filename: str) -> Dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def load_base_config() -> Dict[str, Any]:
    return load_yaml("base_config.yml")

def load_ingestion_file_config() -> Dict[str, Any]:
    return load_yaml("ingestion_file.yml")

def load_spark_jobs_config() -> Dict[str, Any]:
    return load_yaml("spark_jobs.yml")