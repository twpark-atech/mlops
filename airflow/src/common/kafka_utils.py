# airflow/src/common/kafka_utils.py
import json
import logging
from kafka import KafkaProducer
from typing import Any, Dict


logger = logging.getLogger(__name__)


def get_kafka_producer(bootstrap_servers: str) -> KafkaProducer:
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    return producer


def send_event(
    producer: KafkaProducer,
    topic: str,
    payload: Dict[str, Any],
    flush: bool = False
) -> None:
    logger.info(f"Sending Kafka message to topic={topic}: {payload}")
    future = producer.send(topic, value=payload)
    future.get(timeout=10)
    if flush:
        producer.flush()