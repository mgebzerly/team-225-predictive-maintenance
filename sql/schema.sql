-- ============================================================
-- AI4I 2020 Predictive Maintenance - Relational Schema
-- ============================================================
-- Database: SQLite (file: ai4i2020.db)
-- Tables: machines, sensor_readings, failures, failure_types
-- ============================================================

-- Drop existing tables
DROP TABLE IF EXISTS failure_events;
DROP TABLE IF EXISTS sensor_readings;
DROP TABLE IF EXISTS machines;

-- ============================================================
-- Table: machines
-- One row per unique product_id (machine)
-- ============================================================
CREATE TABLE machines (
    product_id     TEXT PRIMARY KEY,
    type           TEXT NOT NULL CHECK (type IN ('L', 'M', 'H')),
    first_udi      INTEGER NOT NULL,
    last_udi       INTEGER NOT NULL,
    reading_count  INTEGER NOT NULL DEFAULT 1,
    total_failures INTEGER NOT NULL DEFAULT 0,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Table: sensor_readings
-- Time-series sensor data (UDI as sequence proxy)
-- ============================================================
CREATE TABLE sensor_readings (
    udi                    INTEGER PRIMARY KEY,
    product_id             TEXT NOT NULL,
    type                   TEXT NOT NULL CHECK (type IN ('L', 'M', 'H')),
    air_temp_k             REAL NOT NULL,
    process_temp_k         REAL NOT NULL,
    rotational_speed_rpm   INTEGER NOT NULL,
    torque_nm              REAL NOT NULL,
    tool_wear_min          INTEGER NOT NULL,
    machine_failure        INTEGER NOT NULL CHECK (machine_failure IN (0, 1)),
    twf                    INTEGER NOT NULL CHECK (twf IN (0, 1)),
    hdf                    INTEGER NOT NULL CHECK (hdf IN (0, 1)),
    pwf                    INTEGER NOT NULL CHECK (pwf IN (0, 1)),
    osf                    INTEGER NOT NULL CHECK (osf IN (0, 1)),
    rnf                    INTEGER NOT NULL CHECK (rnf IN (0, 1)),
    -- Derived columns
    temp_diff_k            REAL GENERATED ALWAYS AS (process_temp_k - air_temp_k) STORED,
    power_kw               REAL GENERATED ALWAYS AS (torque_nm * rotational_speed_rpm / 9550.0) STORED,
    temp_ratio             REAL GENERATED ALWAYS AS (process_temp_k / air_temp_k) STORED,
    torque_per_rpm         REAL GENERATED ALWAYS AS (torque_nm / (rotational_speed_rpm + 1e-6)) STORED,
    FOREIGN KEY (product_id) REFERENCES machines(product_id)
);

-- Indexes for sensor_readings
CREATE INDEX idx_sensor_product_id ON sensor_readings(product_id);
CREATE INDEX idx_sensor_failure ON sensor_readings(machine_failure);
CREATE INDEX idx_sensor_type ON sensor_readings(type);
CREATE INDEX idx_sensor_wear ON sensor_readings(tool_wear_min);

-- ============================================================
-- Table: failure_events
-- One row per failure event (machine_failure = 1)
-- ============================================================
CREATE TABLE failure_events (
    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    udi              INTEGER NOT NULL UNIQUE,
    product_id       TEXT NOT NULL,
    failure_type     TEXT NOT NULL CHECK (failure_type IN ('TWF', 'HDF', 'PWF', 'OSF', 'RNF')),
    air_temp_k       REAL NOT NULL,
    process_temp_k   REAL NOT NULL,
    rotational_speed_rpm INTEGER NOT NULL,
    torque_nm        REAL NOT NULL,
    tool_wear_min    INTEGER NOT NULL,
    FOREIGN KEY (udi) REFERENCES sensor_readings(udi),
    FOREIGN KEY (product_id) REFERENCES machines(product_id)
);

CREATE INDEX idx_failure_product_id ON failure_events(product_id);
CREATE INDEX idx_failure_type ON failure_events(failure_type);

-- ============================================================
-- View: failure_summary_per_machine
-- Aggregated failure counts per machine
-- ============================================================
CREATE VIEW failure_summary_per_machine AS
SELECT 
    m.product_id,
    m.type,
    m.reading_count,
    COALESCE(SUM(CASE WHEN sr.machine_failure = 1 THEN 1 ELSE 0 END), 0) AS total_failures,
    COALESCE(SUM(sr.twf), 0) AS twf_count,
    COALESCE(SUM(sr.hdf), 0) AS hdf_count,
    COALESCE(SUM(sr.pwf), 0) AS pwf_count,
    COALESCE(SUM(sr.osf), 0) AS osf_count,
    COALESCE(SUM(sr.rnf), 0) AS rnf_count,
    MIN(sr.tool_wear_min) AS min_tool_wear,
    MAX(sr.tool_wear_min) AS max_tool_wear,
    AVG(sr.tool_wear_min) AS avg_tool_wear
FROM machines m
LEFT JOIN sensor_readings sr ON m.product_id = sr.product_id
GROUP BY m.product_id, m.type, m.reading_count;

-- ============================================================
-- View: sensor_stats_per_type
-- Sensor statistics grouped by product type
-- ============================================================
CREATE VIEW sensor_stats_per_type AS
SELECT 
    type,
    COUNT(*) AS reading_count,
    AVG(air_temp_k) AS avg_air_temp_k,
    AVG(process_temp_k) AS avg_process_temp_k,
    AVG(rotational_speed_rpm) AS avg_rotational_speed_rpm,
    AVG(torque_nm) AS avg_torque_nm,
    AVG(tool_wear_min) AS avg_tool_wear_min,
    AVG(temp_diff_k) AS avg_temp_diff_k,
    AVG(power_kw) AS avg_power_kw,
    SUM(machine_failure) AS total_failures,
    100.0 * SUM(machine_failure) / COUNT(*) AS failure_rate_pct
FROM sensor_readings
GROUP BY type;

-- ============================================================
-- View: pre_failure_windows
-- Sensor readings within 10 steps before each failure
-- ============================================================
CREATE VIEW pre_failure_windows AS
SELECT 
    sr.*,
    fe.failure_type,
    fe.udi AS failure_udi,
    (fe.udi - sr.udi) AS steps_before_failure
FROM sensor_readings sr
JOIN failure_events fe ON sr.product_id = fe.product_id
WHERE sr.udi BETWEEN fe.udi - 10 AND fe.udi - 1
  AND sr.machine_failure = 0;

-- ============================================================
-- Trigger: Populate machines table on sensor_readings insert
-- ============================================================
CREATE TRIGGER trg_machines_upsert
AFTER INSERT ON sensor_readings
BEGIN
    INSERT OR IGNORE INTO machines (product_id, type, first_udi, last_udi, reading_count)
    VALUES (NEW.product_id, NEW.type, NEW.udi, NEW.udi, 1);
    
    UPDATE machines
    SET last_udi = NEW.udi,
        reading_count = reading_count + 1,
        total_failures = total_failures + NEW.machine_failure
    WHERE product_id = NEW.product_id;
END;

-- ============================================================
-- Trigger: Populate failure_events on failure insert
-- ============================================================
CREATE TRIGGER trg_failure_events
AFTER INSERT ON sensor_readings
WHEN NEW.machine_failure = 1
BEGIN
    INSERT INTO failure_events (udi, product_id, failure_type, air_temp_k, process_temp_k, rotational_speed_rpm, torque_nm, tool_wear_min)
    SELECT NEW.udi, NEW.product_id, 'TWF', NEW.air_temp_k, NEW.process_temp_k, NEW.rotational_speed_rpm, NEW.torque_nm, NEW.tool_wear_min WHERE NEW.twf = 1
    UNION ALL
    SELECT NEW.udi, NEW.product_id, 'HDF', NEW.air_temp_k, NEW.process_temp_k, NEW.rotational_speed_rpm, NEW.torque_nm, NEW.tool_wear_min WHERE NEW.hdf = 1
    UNION ALL
    SELECT NEW.udi, NEW.product_id, 'PWF', NEW.air_temp_k, NEW.process_temp_k, NEW.rotational_speed_rpm, NEW.torque_nm, NEW.tool_wear_min WHERE NEW.pwf = 1
    UNION ALL
    SELECT NEW.udi, NEW.product_id, 'OSF', NEW.air_temp_k, NEW.process_temp_k, NEW.rotational_speed_rpm, NEW.torque_nm, NEW.tool_wear_min WHERE NEW.osf = 1
    UNION ALL
    SELECT NEW.udi, NEW.product_id, 'RNF', NEW.air_temp_k, NEW.process_temp_k, NEW.rotational_speed_rpm, NEW.torque_nm, NEW.tool_wear_min WHERE NEW.rnf = 1;
END;