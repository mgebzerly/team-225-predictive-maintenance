## Anomaly Detection & Explainability (Part 5)

### Approach

- Autoencoder trained exclusively on **normal** samples (7729 readings).
- Architecture: 11 → 32 → 16 → **8** → 16 → 32 → 11.
- Anomaly score = per-sample reconstruction error (MSE).
- Threshold selected by minimizing business cost (10:1 FN:FP ratio) on training data.

### Test Set Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8356 |
| PR-AUC | 0.2933 |
| Precision | 0.2048 |
| Recall | 0.5000 |
| F1-Score | 0.2906 |
| Business Cost | 472 |
| Threshold | 0.000905 |

### Confusion Matrix

|  | Pred Normal | Pred Failure |
|--|------------|-------------|
| **Actual Normal** | 1800 | 132 |
| **Actual Failure** | 34 | 34 |

### Cost Comparison

| Strategy | Cost |
|----------|------|
| No model (all reactive) | 680 |
| Autoencoder (Part 5) | 472 |
| Random Forest (Part 3) | 108 |

### SHAP Feature Importance (Top 5)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | tool_wear_min | 0.0097 |
| 2 | air_temp_k | 0.0079 |
| 3 | temp_x_speed | 0.0078 |
| 4 | torque_x_wear | 0.0056 |
| 5 | rotational_speed_rpm | 0.0054 |

### Drift Simulation

Simulated gradual sensor degradation: air temperature +5 K and torque +15 Nm
over 10 time windows.

| Window | Drift % | Mean Error | Anomaly Rate |
|--------|---------|------------|-------------|
| 0 | 0% | 0.0005 | 8.3% |
| 1 | 11% | 0.0006 | 12.4% |
| 2 | 22% | 0.0012 | 23.9% |
| 3 | 33% | 0.0022 | 37.9% |
| 4 | 44% | 0.0039 | 57.6% |
| 5 | 56% | 0.0060 | 74.2% |
| 6 | 67% | 0.0083 | 84.4% |
| 7 | 78% | 0.0108 | 91.5% |
| 8 | 89% | 0.0138 | 96.9% |
| 9 | 100% | 0.0172 | 99.4% |

The anomaly rate rose from **8.3%** (no drift) to
**99.4%** (full drift), confirming the autoencoder
catches distribution shifts from sensor degradation.

### Saved Artifacts

- `models/anomaly_autoencoder.keras`
- `models/anomaly_scaler.joblib`
- `models/anomaly_metadata.json`
- `reports/anomaly_metrics.csv`
- `reports/anomaly_drift_results.csv`
