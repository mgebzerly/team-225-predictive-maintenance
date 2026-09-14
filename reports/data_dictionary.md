## Raw columns

| Column | Type | Description |
|--------|------|-------------|
| UDI | int | Unique row ID, 1–10000 |
| Product ID | text | Machine/product code — L/M/H prefix + 5 digits |
| Type | text | Product quality variant: L (Low), M (Medium), H (High) |
| Air temperature [K] | float | Ambient air temperature, ~296–304 K |
| Process temperature [K] | float | Process temperature, ~306–313 K |
| Rotational speed [rpm] | int | Rotational speed, ~1168–2886 |
| Torque [Nm] | float | Torque, ~1.8–76.2 |
| Tool wear [min] | int | Cumulative tool wear time, 0–250 |
| Machine failure | int (0/1) | Target: any failure occurred |
| TWF | int (0/1) | Tool Wear Failure flag |
| HDF | int (0/1) | Heat Dissipation Failure flag |
| PWF | int (0/1) | Power Failure flag |
| OSF | int (0/1) | Overstrain Failure flag |
| RNF | int (0/1) | Random Failure flag |

Multiple failure types can co-occur. machine_failure = 1 iff any of TWF/HDF/PWF/OSF/RNF = 1.

## Cleaned columns (added)

| Column | Derivation |
|--------|------------|
| temp_diff_k | process_temp_k − air_temp_k |
| power_kw | torque_nm × rotational_speed_rpm / 9550 |
| wear_rate | tool_wear_min / (rotational_speed_rpm + ε) |

Column names converted to snake_case for SQL; Type column becomes categorical.

## Feature-engineered columns (added)

| Column | Description |
|--------|-------------|
| air_temp_k_rolling_mean_10 | 10-row rolling mean of air temperature per product_id |
| air_temp_k_rolling_std_10 | 10-row rolling std of air temperature per product_id |
| process_temp_k_rolling_mean_10 | 10-row rolling mean of process temperature per product_id |
| process_temp_k_rolling_std_10 | 10-row rolling std of process temperature per product_id |
| rotational_speed_rpm_rolling_mean_10 | 10-row rolling mean of rotational speed per product_id |
| rotational_speed_rpm_rolling_std_10 | 10-row rolling std of rotational speed per product_id |
| torque_nm_rolling_mean_10 | 10-row rolling mean of torque per product_id |
| torque_nm_rolling_std_10 | 10-row rolling std of torque per product_id |
| tool_wear_min_rolling_mean_10 | 10-row rolling mean of tool wear per product_id |
| tool_wear_min_rolling_std_10 | 10-row rolling std of tool wear per product_id |
| machine_failure_lag_1 | Previous row machine failure flag (lag 1) |
| machine_failure_lag_2 | Previous row machine failure flag (lag 2) |
| machine_failure_lag_3 | Previous row machine failure flag (lag 3) |
| machine_failure_lag_5 | Previous row machine failure flag (lag 5) |
| machine_failure_lag_10 | Previous row machine failure flag (lag 10) |
| twf_lag_1 | Previous row TWF flag (lag 1) |
| hdf_lag_1 | Previous row HDF flag (lag 1) |
| pwf_lag_1 | Previous row PWF flag (lag 1) |
| osf_lag_1 | Previous row OSF flag (lag 1) |
| rnf_lag_1 | Previous row RNF flag (lag 1) |
| cum_machine_failures | Cumulative failures per product_id |
| temp_ratio | process_temp_k / air_temp_k |
| torque_per_rpm | torque_nm / rotational_speed_rpm |
| tool_wear_bin | Tool wear binned: 0–50, 51–100, 101–150, 151–200, 201–250 min |
| temp_x_speed | temp_diff_k × rotational_speed_rpm |
| torque_x_wear | torque_nm × tool_wear_min |
| type_H | One-hot: Type = H |
| type_L | One-hot: Type = L |
| type_M | One-hot: Type = M |

## Failure type definitions

| Code | Name | Description |
|------|------|-------------|
| TWF | Tool Wear Failure | Tool wear exceeds threshold (200+ min) |
| HDF | Heat Dissipation Failure | Temp diff > 8.6 K and speed < 1380 RPM |
| PWF | Power Failure | Power outside 3500–9000 W range |
| OSF | Overstrain Failure | Torque > 72 Nm and wear > 200 min |
| RNF | Random Failure | Random noise failures |

## Data quality
No missing values, no duplicate UDI. Product IDs are unique per machine (one row each). Class imbalance 96.6%/3.4% — consider stratification for modeling. No temporal column; UDI used as sequence proxy.