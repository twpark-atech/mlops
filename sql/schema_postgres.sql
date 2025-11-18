-- sql/schema_postgres.sql

CREATE TABLE IF NOT EXISTS its_traffic_5min_gold (
    date        VARCHAR(8)  NOT NULL,
    datetime    TIMESTAMP   NOT NULL,
    linkid      VARCHAR(20) NOT NULL,

    t2_mean     REAL        NOT NULL,
    t1_mean     REAL        NOT NULL,
    self_mean   REAL        NOT NULL,
    f1_mean     REAL        NOT NULL,
    f2_mean     REAL        NOT NULL,

    created_at  TIMESTAMP   NOT NULL DEFAULT NOW(),

    CONSTRAINT pk_its_traffic_5min_gold PRIMARY KEY (date, datetime, linkid)
);

CREATE INDEX IF NOT EXISTS idx_its_traffic_5min_gold_link_date
    ON its_traffic_5min_gold (linkid, date);