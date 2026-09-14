#!/usr/bin/env python3
"""
Load cleaned data into SQLite database - fixed version with proper population.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "ai4i2020.db"
CLEANED_PATH = Path(__file__).parent.parent / "data" / "cleaned" / "ai4i2020_cleaned.csv"
SCHEMA_PATH = Path(__file__).parent.parent / "sql" / "schema.sql"

def main():
    print("=" * 60)
    print("Loading data into SQLite database (fixed)")
    print("=" * 60)
    
    # Load cleaned data
    df = pd.read_csv(CLEANED_PATH)
    print(f"Loaded {len(df)} rows from cleaned data")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop and recreate with simpler schema (no triggers)
    cursor.executescript("""
    DROP TABLE IF EXISTS failure_events;
    DROP TABLE IF EXISTS sensor_readings;
    DROP TABLE IF EXISTS machines;
    
    CREATE TABLE machines (
        product_id     TEXT PRIMARY KEY,
        type           TEXT NOT NULL CHECK (type IN ('L', 'M', 'H')),
        first_udi      INTEGER NOT NULL,
        last_udi       INTEGER NOT NULL,
        reading_count  INTEGER NOT NULL DEFAULT 1,
        total_failures INTEGER NOT NULL DEFAULT 0
    );
    
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
        temp_diff_k            REAL GENERATED ALWAYS AS (process_temp_k - air_temp_k) STORED,
        power_kw               REAL GENERATED ALWAYS AS (torque_nm * rotational_speed_rpm / 9550.0) STORED,
        temp_ratio             REAL GENERATED ALWAYS AS (process_temp_k / air_temp_k) STORED,
        torque_per_rpm         REAL GENERATED ALWAYS AS (torque_nm / (rotational_speed_rpm + 1e-6)) STORED,
        FOREIGN KEY (product_id) REFERENCES machines(product_id)
    );
    
    CREATE INDEX idx_sensor_product_id ON sensor_readings(product_id);
    CREATE INDEX idx_sensor_failure ON sensor_readings(machine_failure);
    CREATE INDEX idx_sensor_type ON sensor_readings(type);
    CREATE INDEX idx_sensor_wear ON sensor_readings(tool_wear_min);
    
    CREATE TABLE failure_events (
        event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        udi              INTEGER NOT NULL,
        product_id       TEXT NOT NULL,
        failure_type     TEXT NOT NULL CHECK (failure_type IN ('TWF', 'HDF', 'PWF', 'OSF', 'RNF')),
        air_temp_k       REAL NOT NULL,
        process_temp_k   REAL NOT NULL,
        rotational_speed_rpm INTEGER NOT NULL,
        torque_nm        REAL NOT NULL,
        tool_wear_min    INTEGER NOT NULL,
        FOREIGN KEY (udi) REFERENCES sensor_readings(udi),
        FOREIGN KEY (product_id) REFERENCES machines(product_id),
        UNIQUE (udi, failure_type)
    );
    
    CREATE INDEX idx_failure_product_id ON failure_events(product_id);
    CREATE INDEX idx_failure_type ON failure_events(failure_type);
    """)
    
    conn.commit()
    print("Schema created successfully")
    
    # Insert sensor_readings
    insert_cols = [
        'udi', 'product_id', 'type', 'air_temp_k', 'process_temp_k',
        'rotational_speed_rpm', 'torque_nm', 'tool_wear_min',
        'machine_failure', 'twf', 'hdf', 'pwf', 'osf', 'rnf'
    ]
    
    df_insert = df[insert_cols].copy()
    df_insert['type'] = df_insert['type'].astype(str)
    df_insert['product_id'] = df_insert['product_id'].astype(str)
    
    batch_size = 1000
    for i in range(0, len(df_insert), batch_size):
        batch = df_insert.iloc[i:i+batch_size]
        placeholders = ','.join(['?' for _ in insert_cols])
        sql = f"INSERT INTO sensor_readings ({','.join(insert_cols)}) VALUES ({placeholders})"
        cursor.executemany(sql, batch.values.tolist())
    
    conn.commit()
    print(f"Inserted {len(df_insert)} sensor readings")
    
    # Populate machines table
    cursor.execute("""
    INSERT INTO machines (product_id, type, first_udi, last_udi, reading_count, total_failures)
    SELECT 
        product_id,
        type,
        MIN(udi) as first_udi,
        MAX(udi) as last_udi,
        COUNT(*) as reading_count,
        SUM(machine_failure) as total_failures
    FROM sensor_readings
    GROUP BY product_id, type
    """)
    conn.commit()
    print("Populated machines table")
    
    # Populate failure_events (one row per failure type per failure)
    failure_data = []
    for _, row in df[df['machine_failure'] == 1].iterrows():
        for fail_type, fail_col in [('TWF', 'twf'), ('HDF', 'hdf'), ('PWF', 'pwf'), ('OSF', 'osf'), ('RNF', 'rnf')]:
            if row[fail_col] == 1:
                failure_data.append((
                    row['udi'], row['product_id'], fail_type,
                    row['air_temp_k'], row['process_temp_k'],
                    row['rotational_speed_rpm'], row['torque_nm'], row['tool_wear_min']
                ))
    
    if failure_data:
        cursor.executemany("""
            INSERT INTO failure_events (udi, product_id, failure_type, air_temp_k, process_temp_k, rotational_speed_rpm, torque_nm, tool_wear_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, failure_data)
        conn.commit()
        print(f"Inserted {len(failure_data)} failure events")
    
    # Create views (drop first if exists)
    cursor.executescript("""
    DROP VIEW IF EXISTS failure_summary_per_machine;
    DROP VIEW IF EXISTS sensor_stats_per_type;
    DROP VIEW IF EXISTS pre_failure_windows;
    
    CREATE VIEW failure_summary_per_machine AS
    SELECT 
        m.product_id,
        m.type,
        m.reading_count,
        m.total_failures,
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
    GROUP BY m.product_id, m.type, m.reading_count, m.total_failures;
    
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
    """)
    
    conn.commit()
    
    # Verify counts
    cursor.execute("SELECT COUNT(*) FROM machines")
    machine_count = cursor.fetchone()[0]
    print(f"Machines table: {machine_count} rows")
    
    cursor.execute("SELECT COUNT(*) FROM sensor_readings")
    reading_count = cursor.fetchone()[0]
    print(f"Sensor readings table: {reading_count} rows")
    
    cursor.execute("SELECT COUNT(*) FROM failure_events")
    failure_count = cursor.fetchone()[0]
    print(f"Failure events table: {failure_count} rows")
    
    cursor.execute("SELECT * FROM failure_summary_per_machine LIMIT 5")
    print("\nSample failure summary:")
    for row in cursor.fetchall():
        print(row)
    
    cursor.execute("SELECT * FROM sensor_stats_per_type")
    print("\nSensor stats per type:")
    for row in cursor.fetchall():
        print(row)
    
    conn.close()
    print(f"\nDatabase saved to: {DB_PATH}")


if __name__ == "__main__":
    main()