-- docker/config/postgres-init.sql
CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.train_dataset(
    id          BIGSERIAL PRIMARY KEY,
    event_time  TIMESTAMP NOT NULL,
    feature_1   DOUBLE PRECISION,
    feature_2   DOUBLE PRECISION,
    feature_3   DOUBLE PRECISION,
    label       INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE ROLE airflow LOGIN PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;