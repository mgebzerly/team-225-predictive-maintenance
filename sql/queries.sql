-- 01. Dataset Overview (simple SELECT)
SELECT
    COUNT(*)                                         AS total_readings,
    COUNT(DISTINCT product_id)                      AS total_machines,
    COUNT(DISTINCT type)                            AS type_variants,
    MIN(udi)                                        AS min_udi,
    MAX(udi)                                        AS max_udi,
    ROUND(AVG(air_temp_k), 3)                       AS avg_air_temp_k,
    ROUND(AVG(process_temp_k), 3)                   AS avg_process_temp_k,
    ROUND(AVG(rotational_speed_rpm), 1)             AS avg_rotational_speed_rpm,
    ROUND(AVG(torque_nm), 3)                        AS avg_torque_nm,
    ROUND(AVG(tool_wear_min), 2)                    AS avg_tool_wear_min
FROM sensor_readings;

-- 02. Failure distribution by failure type (JOIN + GROUP BY)
SELECT
    ft.failure_type,
    ft.description,
    COALESCE(fe.event_count, 0)                        AS event_count,
    ROUND(COALESCE(100.0 * fe.event_count / NULLIF(SUM(fe.event_count) OVER (), 0), 0), 2) AS pct_of_all_failures
FROM (
    SELECT 'TWF' AS failure_type, 'Tool Wear Failure' AS description
    UNION ALL SELECT 'HDF', 'Heat Dissipation Failure'
    UNION ALL SELECT 'PWF', 'Power Failure'
    UNION ALL SELECT 'OSF', 'Overstrain Failure'
    UNION ALL SELECT 'RNF', 'Random Failure'
) ft
LEFT JOIN (
    SELECT failure_type, COUNT(*) AS event_count
    FROM failure_events
    GROUP BY failure_type
) fe ON ft.failure_type = fe.failure_type
ORDER BY event_count DESC;

-- 03. Failure rate by product type (JOIN + CTE)
WITH type_stats AS (
    SELECT
        type,
        COUNT(*)                             AS total_readings,
        SUM(machine_failure)                 AS total_failures,
        ROUND(COUNT(DISTINCT product_id), 0) AS unique_machines
    FROM sensor_readings
    GROUP BY type
)
SELECT
    type,
    total_readings,
    total_failures,
    ROUND(100.0 * total_failures / total_readings, 2) AS failure_rate_pct,
    unique_machines,
    CASE
        WHEN type = 'L' THEN 'Low quality variant'
        WHEN type = 'M' THEN 'Medium quality variant'
        WHEN type = 'H' THEN 'High quality variant'
    END AS type_description
FROM type_stats
ORDER BY failure_rate_pct DESC;

-- 04. Machines ranked by total failures (Window function + CTE)
WITH machine_failures AS (
    SELECT
        product_id,
        type,
        SUM(machine_failure) AS failure_count,
        MAX(tool_wear_min)   AS max_tool_wear
    FROM sensor_readings
    GROUP BY product_id, type
)
SELECT
    ROW_NUMBER() OVER (ORDER BY failure_count DESC) AS rank_num,
    DENSE_RANK() OVER (ORDER BY failure_count DESC) AS dense_rank,
    product_id,
    type,
    failure_count,
    max_tool_wear,
    CASE
        WHEN failure_count = 0 THEN 'No failures'
        WHEN failure_count = 1 THEN 'Single failure'
        WHEN failure_count > 1 THEN 'Multiple failures'
    END AS failure_category
FROM machine_failures
WHERE failure_count > 0
ORDER BY failure_count DESC
LIMIT 20;

-- 05. Top 20 highest tool wear readings (Window function, RANK)
WITH wear_ranked AS (
    SELECT
        udi,
        product_id,
        type,
        tool_wear_min,
        rotational_speed_rpm,
        torque_nm,
        machine_failure,
        air_temp_k,
        process_temp_k,
        RANK() OVER (ORDER BY tool_wear_min DESC) AS wear_rank,
        DENSE_RANK() OVER (ORDER BY tool_wear_min DESC) AS wear_dense_rank,
        NTILE(4) OVER (ORDER BY tool_wear_min DESC) AS wear_quartile
    FROM sensor_readings
)
SELECT
    udi,
    product_id,
    type,
    tool_wear_min,
    wear_rank,
    wear_dense_rank,
    wear_quartile,
    rotational_speed_rpm,
    torque_nm,
    machine_failure,
    CASE
        WHEN wear_quartile = 1 THEN 'Highest wear (top 25%)'
        WHEN wear_quartile = 2 THEN 'Above average wear'
        WHEN wear_quartile = 3 THEN 'Below average wear'
        ELSE 'Lowest wear (bottom 25%)'
    END AS wear_category
FROM wear_ranked
WHERE wear_rank <= 20
ORDER BY wear_rank;

-- 06. Sequential readings with running total of failures (Window)
SELECT
    udi,
    product_id,
    type,
    tool_wear_min,
    machine_failure,
    SUM(machine_failure) OVER (
        ORDER BY udi
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_failures,
    LAG(machine_failure, 1, 0) OVER (ORDER BY udi) AS prev_reading_failure,
    LEAD(machine_failure, 1, 0) OVER (ORDER BY udi) AS next_reading_failure,
    CASE
        WHEN machine_failure = 1 AND LAG(machine_failure, 1, 0) OVER (ORDER BY udi) = 0
            THEN 'FAILURE_START'
        WHEN machine_failure = 1 AND LEAD(machine_failure, 1, 0) OVER (ORDER BY udi) = 1
            THEN 'FAILURE_CONTINUES'
        WHEN machine_failure = 0 AND LAG(machine_failure, 1, 0) OVER (ORDER BY udi) = 1
            THEN 'STOP'
        ELSE 'NO_CHANGE'
    END AS transition
FROM sensor_readings
WHERE type = 'L'
  AND udi BETWEEN 4500 AND 4550
ORDER BY udi;

-- 07. Rolling 3-reading averages across all readings (Window)
SELECT
    udi,
    product_id,
    type,
    air_temp_k,
    process_temp_k,
    torque_nm,
    ROUND(AVG(air_temp_k) OVER (
        ORDER BY udi
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 3) AS rolling_avg_air_temp,
    ROUND(AVG(tool_wear_min) OVER (
        ORDER BY udi
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 1) AS rolling_avg_wear,
    machine_failure
FROM sensor_readings
WHERE machine_failure = 1
ORDER BY udi;

-- 08. Failure events with 10-reading pre-failure window (CTE + JOIN)
WITH failure_readings AS (
    SELECT
        fe.udi        AS failure_udi,
        fe.failure_type,
        fe.product_id AS failure_product_id,
        sr.udi        AS pre_udi,
        sr.tool_wear_min,
        sr.torque_nm,
        sr.rotational_speed_rpm,
        sr.air_temp_k,
        sr.process_temp_k,
        (fe.udi - sr.udi) AS steps_before_failure,
        sr.type AS pre_type
    FROM failure_events fe
    JOIN sensor_readings sr
        ON sr.type = fe.product_id
        AND sr.udi < fe.udi
    WHERE sr.machine_failure = 0
    ORDER BY fe.udi, sr.udi DESC
    LIMIT 100
)
SELECT
    failure_type,
    failure_product_id,
    failure_udi,
    pre_udi,
    steps_before_failure,
    tool_wear_min,
    torque_nm,
    rotational_speed_rpm,
    air_temp_k,
    process_temp_k
FROM failure_readings
ORDER BY failure_udi, steps_before_failure DESC
LIMIT 30;

-- 09. Failure rate by tool wear bin (CASE-based binning)
WITH wear_binned AS (
    SELECT
        *,
        CASE
            WHEN tool_wear_min <= 50  THEN '0-50 min'
            WHEN tool_wear_min <= 100 THEN '51-100 min'
            WHEN tool_wear_min <= 150 THEN '101-150 min'
            WHEN tool_wear_min <= 200 THEN '151-200 min'
            ELSE '201+ min'
        END AS wear_bin
    FROM sensor_readings
)
SELECT
    wear_bin,
    COUNT(*)                    AS total_readings,
    SUM(machine_failure)        AS failures,
    ROUND(100.0 * SUM(machine_failure) / COUNT(*), 2) AS failure_rate_pct,
    ROUND(AVG(torque_nm), 3)   AS avg_torque,
    ROUND(AVG(rotational_speed_rpm), 1) AS avg_speed,
    ROUND(AVG(air_temp_k), 2)  AS avg_air_temp
FROM wear_binned
GROUP BY wear_bin
ORDER BY
    CASE wear_bin
        WHEN '0-50 min' THEN 1
        WHEN '51-100 min' THEN 2
        WHEN '101-150 min' THEN 3
        WHEN '151-200 min' THEN 4
        ELSE 5
    END;

-- 10. Failures where temperature differential was elevated (JOIN + CASE)
WITH event_failures AS (
    SELECT DISTINCT
        sr.udi,
        sr.product_id,
        sr.type,
        sr.air_temp_k,
        sr.process_temp_k,
        sr.temp_diff_k,
        sr.rotational_speed_rpm,
        sr.torque_nm,
        sr.tool_wear_min,
        fe.failure_type
    FROM sensor_readings sr
    JOIN failure_events fe ON sr.udi = fe.udi
)
SELECT
    udi,
    product_id,
    type,
    air_temp_k,
    process_temp_k,
    temp_diff_k,
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,
    failure_type,
    CASE
        WHEN temp_diff_k > 10      THEN 'HIGH (>10K)'
        WHEN temp_diff_k > 8.6     THEN 'ELEVATED (>8.6K)'
        WHEN temp_diff_k BETWEEN 7 AND 8.6 THEN 'ABOVE_AVERAGE (7-8.6K)'
        ELSE 'NORMAL'
    END AS temp_diff_category,
    CASE
        WHEN rotational_speed_rpm < 1380 AND temp_diff_k > 8.6
            THEN 'HDF_LIKELY'
        WHEN torque_nm > 72 AND tool_wear_min > 200
            THEN 'OSF_LIKELY'
        WHEN (torque_nm * rotational_speed_rpm / 9550.0) < 3500 OR (torque_nm * rotational_speed_rpm / 9550.0) > 9000
            THEN 'PWF_LIKELY'
        WHEN tool_wear_min > 200
            THEN 'TWF_LIKELY'
        ELSE 'UNCLASSIFIED'
    END AS failure_pattern_hint
FROM event_failures
ORDER BY temp_diff_k DESC
LIMIT 20;

-- 11. Lifecycle comparison: first half vs second half (Window)
WITH lifecycle AS (
    SELECT
        udi,
        product_id,
        type,
        machine_failure,
        tool_wear_min,
        air_temp_k,
        process_temp_k,
        rotational_speed_rpm,
        torque_nm,
        ROW_NUMBER() OVER (ORDER BY udi) AS seq_num,
        NTILE(2) OVER (ORDER BY udi) AS lifecycle_half
    FROM sensor_readings
)
SELECT
    lifecycle_half,
    CASE WHEN lifecycle_half = 1 THEN 'First half' ELSE 'Second half' END AS period,
    COUNT(*) AS readings,
    SUM(machine_failure) AS failures,
    ROUND(100.0 * SUM(machine_failure) / COUNT(*), 2) AS failure_rate_pct,
    ROUND(AVG(tool_wear_min), 1) AS avg_tool_wear,
    ROUND(MIN(tool_wear_min), 1) AS min_wear,
    ROUND(MAX(tool_wear_min), 1) AS max_wear,
    ROUND(AVG(torque_nm), 2) AS avg_torque,
    ROUND(AVG(air_temp_k), 2) AS avg_air_temp,
    ROUND(AVG(process_temp_k), 2) AS avg_process_temp,
    ROUND(AVG(rotational_speed_rpm), 1) AS avg_speed
FROM lifecycle
GROUP BY lifecycle_half
ORDER BY lifecycle_half;

-- 12. Power vs speed correlation by type (CTE + aggregation)
WITH power_by_type AS (
    SELECT
        type,
        ROUND(torque_nm * rotational_speed_rpm / 9550.0, 1) AS power_kw,
        machine_failure,
        tool_wear_min,
        rotational_speed_rpm
    FROM sensor_readings
)
SELECT
    type,
    ROUND(power_kw, 0) AS power_bin,
    COUNT(*) AS readings,
    ROUND(AVG(tool_wear_min), 1) AS avg_wear,
    SUM(machine_failure) AS failures,
    ROUND(100.0 * SUM(machine_failure) / COUNT(*), 2) AS failure_rate_pct
FROM power_by_type
GROUP BY type, power_bin
HAVING power_bin BETWEEN 4000 AND 8000
ORDER BY type, power_bin;

-- 13. Heat Dissipation Failure (HDF) analysis: temp differential
WITH hdf_readings AS (
    SELECT
        sr.udi,
        sr.product_id,
        sr.type,
        sr.air_temp_k,
        sr.process_temp_k,
        sr.temp_diff_k,
        sr.rotational_speed_rpm,
        sr.torque_nm,
        sr.tool_wear_min,
        sr.machine_failure
    FROM failure_events fe
    JOIN sensor_readings sr ON fe.udi = sr.udi
    WHERE fe.failure_type = 'HDF'
)
SELECT
    COUNT(*) AS hdf_event_count,
    ROUND(AVG(air_temp_k), 2)       AS avg_air_temp_k,
    ROUND(AVG(process_temp_k), 2)   AS avg_process_temp_k,
    ROUND(AVG(temp_diff_k), 2)      AS avg_temp_diff_k,
    ROUND(MIN(temp_diff_k), 2)      AS min_temp_diff_k,
    ROUND(MAX(temp_diff_k), 2)      AS max_temp_diff_k,
    ROUND(AVG(rotational_speed_rpm), 1) AS avg_speed_rpm,
    ROUND(MIN(rotational_speed_rpm), 1) AS min_speed_rpm,
    ROUND(AVG(torque_nm), 2)        AS avg_torque_nm,
    ROUND(AVG(tool_wear_min), 1)    AS avg_tool_wear_min,
    ROUND(SUM(machine_failure), 0)  AS total_failures_in_events
FROM hdf_readings;

-- 14. Cross-tab: failure types per product type (CTE + CASE)
WITH product_type_failures AS (
    SELECT
        s.type,
        COUNT(*) AS total_machines,
        SUM(s.twf) AS twf_count,
        SUM(s.hdf) AS hdf_count,
        SUM(s.pwf) AS pwf_count,
        SUM(s.osf) AS osf_count,
        SUM(s.rnf) AS rnf_count,
        SUM(s.machine_failure) AS total_failures,
        COUNT(DISTINCT s.product_id) AS unique_products
    FROM sensor_readings s
    GROUP BY s.type
)
SELECT
    type,
    total_machines,
    unique_products,
    total_failures,
    twf_count,
    hdf_count,
    pwf_count,
    osf_count,
    rnf_count,
    ROUND(100.0 * total_failures / total_machines, 2) AS failure_rate_pct,
    CASE
        WHEN total_failures = 0 THEN 'No failures observed'
        WHEN hdf_count > 0 AND twf_count > 0
            THEN 'HDF + TWF dominant'
        WHEN pwf_count > 0
            THEN 'PWF dominant'
        WHEN osf_count > 0
            THEN 'OSF dominant'
        ELSE 'Mixed failure modes'
    END AS dominant_failure_pattern
FROM product_type_failures
ORDER BY type;

-- 15. Machines with multiple failure types (CTE + HAVING)
WITH multi_failures AS (
    SELECT
        product_id,
        type,
        SUM(CASE WHEN twf = 1 THEN 1 ELSE 0 END) AS twf_events,
        SUM(CASE WHEN hdf = 1 THEN 1 ELSE 0 END) AS hdf_events,
        SUM(CASE WHEN pwf = 1 THEN 1 ELSE 0 END) AS pwf_events,
        SUM(CASE WHEN osf = 1 THEN 1 ELSE 0 END) AS osf_events,
        SUM(CASE WHEN rnf = 1 THEN 1 ELSE 0 END) AS rnf_events,
        SUM(twf + hdf + pwf + osf + rnf) AS total_failure_flags
    FROM sensor_readings
    GROUP BY product_id, type
    HAVING total_failure_flags > 0
)
SELECT
    product_id,
    type,
    twf_events,
    hdf_events,
    pwf_events,
    osf_events,
    rnf_events,
    total_failure_flags,
    CASE
        WHEN twf_events > 0 AND hdf_events > 0 AND pwf_events > 0 AND osf_events > 0 AND rnf_events > 0
            THEN 'ALL_FIVE_TYPES'
        WHEN twf_events + hdf_events + pwf_events + osf_events + rnf_events >= 3
            THEN 'MULTIPLE_FAULTS'
        ELSE 'SINGLE_FAULT'
    END AS fault_complexity,
    CASE
        WHEN twf_events > 0 THEN 'TWF'
        WHEN hdf_events > 0 THEN 'HDF'
        WHEN pwf_events > 0 THEN 'PWF'
        WHEN osf_events > 0 THEN 'OSF'
        WHEN rnf_events > 0 THEN 'RNF'
    END AS primary_failure
FROM multi_failures
ORDER BY total_failure_flags DESC, product_id ASC;

-- 16. Temperature anomaly flag: readings where process_temp > mean + 2SD (CTE)
WITH stats AS (
    SELECT
        AVG(process_temp_k) AS mean_temp,
        SQRT(
            AVG(process_temp_k * process_temp_k) 
            - AVG(process_temp_k) * AVG(process_temp_k)
        ) AS std_temp
    FROM sensor_readings
),
anomalies AS (
    SELECT
        sr.udi,
        sr.product_id,
        sr.type,
        sr.process_temp_k,
        sr.air_temp_k,
        sr.temp_diff_k,
        sr.machine_failure,
        s.mean_temp,
        s.std_temp,
        sr.process_temp_k - s.mean_temp AS deviation,
        ROUND((sr.process_temp_k - s.mean_temp) / NULLIF(s.std_temp, 0), 2) AS z_score
    FROM sensor_readings sr
    CROSS JOIN stats s
    WHERE sr.process_temp_k > s.mean_temp + 2.0 * s.std_temp
)
SELECT
    COUNT(*) AS anomaly_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM sensor_readings), 2) AS anomaly_pct,
    ROUND(AVG(z_score), 2) AS avg_z_score,
    MIN(z_score) AS min_z_score,
    MAX(z_score) AS max_z_score,
    ROUND(100.0 * SUM(machine_failure) / COUNT(*), 2) AS failure_rate_among_anomalies,
    SUM(CASE WHEN type = 'L' THEN 1 ELSE 0 END) AS anomalies_in_L,
    SUM(CASE WHEN type = 'M' THEN 1 ELSE 0 END) AS anomalies_in_M,
    SUM(CASE WHEN type = 'H' THEN 1 ELSE 0 END) AS anomalies_in_H
FROM anomalies;

-- 17. Power analysis: readings outside 3500W–9000W range (CTE + CASE)
WITH power_calc AS (
    SELECT
        udi,
        product_id,
        type,
        torque_nm,
        rotational_speed_rpm,
        ROUND(torque_nm * rotational_speed_rpm / 9550.0, 2) AS power_kw,
        machine_failure,
        tool_wear_min
    FROM sensor_readings
)
SELECT
    CASE
        WHEN power_kw < 3500 THEN 'UNDERLOAD (<3500W)'
        WHEN power_kw > 9000 THEN 'OVERLOAD (>9000W)'
        ELSE 'NORMAL_RANGE'
    END AS power_category,
    type,
    COUNT(*) AS readings,
    SUM(machine_failure) AS failures,
    ROUND(100.0 * SUM(machine_failure) / COUNT(*), 2) AS failure_rate_pct,
    ROUND(AVG(power_kw), 1) AS avg_power_kw,
    MIN(power_kw) AS min_power_kw,
    MAX(power_kw) AS max_power_kw,
    ROUND(AVG(tool_wear_min), 1) AS avg_wear
FROM power_calc
GROUP BY power_category, type
ORDER BY
    CASE power_category
        WHEN 'UNDERLOAD (<3500W)' THEN 1
        WHEN 'OVERLOAD (>9000W)' THEN 2
        ELSE 3
    END,
    type;

-- 18. Failure signal detection: lag and cumulative stats (Window)
WITH machine_signal AS (
    SELECT
        udi,
        product_id,
        type,
        machine_failure,
        tool_wear_min,
        torque_nm,
        air_temp_k,
        process_temp_k,
        temp_diff_k,
        LAG(machine_failure, 1, 0) OVER (ORDER BY udi)   AS prev_fail,
        LAG(machine_failure, 2, 0) OVER (ORDER BY udi)   AS prev2_fail,
        LAG(machine_failure, 5, 0) OVER (ORDER BY udi)   AS prev5_fail,
        ROUND(AVG(air_temp_k) OVER (
            ORDER BY udi ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ), 3) AS rolling_avg_air_5,
        ROUND(AVG(torque_nm) OVER (
            ORDER BY udi ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ), 3) AS rolling_avg_torque_5,
        SUM(machine_failure) OVER (ORDER BY udi) AS cum_failures_total,
        SUM(CASE WHEN tool_wear_min > 200 THEN 1 ELSE 0 END) OVER (ORDER BY udi) AS cum_high_wear
    FROM sensor_readings
)
SELECT
    udi,
    product_id,
    type,
    machine_failure,
    prev_fail,
    prev2_fail,
    prev5_fail,
    rolling_avg_air_5,
    rolling_avg_torque_5,
    cum_failures_total,
    cum_high_wear,
    tool_wear_min,
    temp_diff_k,
    CASE
        WHEN machine_failure = 1
            THEN 'CURRENT_FAILURE'
        WHEN prev_fail = 1
            THEN 'PREVIOUS_HAD_FAILURE'
        WHEN cum_high_wear > 0 AND cum_failures_total = 0
            THEN 'HIGH_WEAR_NO_FAILURE_YET'
        ELSE 'OK'
    END AS signal_status
FROM machine_signal
WHERE machine_failure = 1
   OR prev_fail = 1
   OR (cum_high_wear > 0 AND cum_failures_total = 0)
ORDER BY udi
LIMIT 30;