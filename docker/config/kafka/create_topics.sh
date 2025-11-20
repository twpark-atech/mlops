# docker/config/kafka/create_topics.sh
#!/bin/bash
set -e

KAFKA_BROKER="${KAFKA_BROKER:-kafka:9092}"
TOPIC_ITS_RAW="${KAFKA_TOPIC_ITS_RAW:-its_traffic_raw}"

echo "Waiting for Kafka at $KAFKA_BROKER..."

for i in {1..30}; do
    if kafka-topics.sh --bootstrap-server "$KAFKA_BROKER" --list >/dev/null 2>&1; then
        break
    fi
    echo "Kafka not ready yet... {$i}"
    sleep 2
done

echo "Creating topic $TOPIC_ITS_RAW (if not exists)..."
kafka-topics.sh --bootstrap-server "$KAFKA_BROKER" \
    --create --if-not-exists \
    --topic "$TOPIC_ITS_RAW" \
    --replication-factor 1 \
    --partition 3

echo "Kafka topic init done."