from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


# -----------------------------
# Random Forest
# -----------------------------

RF_MODEL_PATH = MODEL_DIR / "classical_ml_best_pipeline.joblib"
RF_METADATA_PATH = MODEL_DIR / "classical_ml_metadata.json"


def load_rf_model():
    return joblib.load(RF_MODEL_PATH)


def load_rf_metadata():
    with open(RF_METADATA_PATH, "r") as f:
        return json.load(f)


def build_rf_features(
    machine_type,
    air_temp_k,
    process_temp_k,
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,
):
    temp_diff_k = process_temp_k - air_temp_k

    # Formula used by the shared project feature dataset
    power_kw = torque_nm * rotational_speed_rpm / 9550

    temp_ratio = process_temp_k / air_temp_k
    torque_per_rpm = torque_nm / (rotational_speed_rpm + 1e-6)
    temp_x_speed = temp_diff_k * rotational_speed_rpm
    torque_x_wear = torque_nm * tool_wear_min

    return pd.DataFrame([{
        "type": machine_type,
        "air_temp_k": air_temp_k,
        "process_temp_k": process_temp_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
        "temp_diff_k": temp_diff_k,
        "power_kw": power_kw,
        "temp_ratio": temp_ratio,
        "torque_per_rpm": torque_per_rpm,
        "temp_x_speed": temp_x_speed,
        "torque_x_wear": torque_x_wear,
    }])


def predict_rf(features):
    model = load_rf_model()
    metadata = load_rf_metadata()

    probability = float(model.predict_proba(features)[0, 1])
    threshold = float(metadata["decision_threshold"])

    prediction = int(probability >= threshold)

    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
    }


# -----------------------------
# Autoencoder anomaly detector
# -----------------------------

AE_MODEL_PATH = MODEL_DIR / "anomaly_autoencoder.keras"
AE_SCALER_PATH = MODEL_DIR / "anomaly_scaler.joblib"
AE_METADATA_PATH = MODEL_DIR / "anomaly_metadata.json"


def load_autoencoder():
    from tensorflow.keras.models import load_model
    return load_model(AE_MODEL_PATH)


def load_ae_scaler():
    return joblib.load(AE_SCALER_PATH)


def load_ae_metadata():
    with open(AE_METADATA_PATH, "r") as f:
        return json.load(f)


def build_ae_features(
    air_temp_k,
    process_temp_k,
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,
):
    temp_diff_k = process_temp_k - air_temp_k
    power_kw = torque_nm * rotational_speed_rpm / 9550
    temp_ratio = process_temp_k / air_temp_k
    torque_per_rpm = torque_nm / (rotational_speed_rpm + 1e-6)
    temp_x_speed = temp_diff_k * rotational_speed_rpm
    torque_x_wear = torque_nm * tool_wear_min

    return np.array([[
        air_temp_k,
        process_temp_k,
        rotational_speed_rpm,
        torque_nm,
        tool_wear_min,
        temp_diff_k,
        power_kw,
        temp_ratio,
        torque_per_rpm,
        temp_x_speed,
        torque_x_wear,
    ]], dtype=float)


def predict_anomaly(features):
    model = load_autoencoder()
    scaler = load_ae_scaler()
    metadata = load_ae_metadata()

    scaled = scaler.transform(features)

    reconstructed = model.predict(scaled, verbose=0)

    error = float(np.mean((scaled - reconstructed) ** 2))

    threshold = float(metadata["threshold"])

    anomaly = int(error >= threshold)

    return {
        "score": error,
        "threshold": threshold,
        "anomaly": anomaly,
    }

# -----------------------------
# Deep MLP
# -----------------------------

MLP_MODEL_PATH = MODEL_DIR / "deep_mlp_best.keras"
MLP_PREPROCESSOR_PATH = MODEL_DIR / "deep_mlp_preprocessor.joblib"
MLP_CALIBRATOR_PATH = MODEL_DIR / "deep_mlp_calibrator.joblib"
MLP_METADATA_PATH = MODEL_DIR / "deep_mlp_metadata.json"


def load_mlp_model():
    from tensorflow.keras.models import load_model
    return load_model(MLP_MODEL_PATH)


def load_mlp_preprocessor():
    return joblib.load(MLP_PREPROCESSOR_PATH)


def load_mlp_calibrator():
    return joblib.load(MLP_CALIBRATOR_PATH)


def load_mlp_metadata():
    with open(MLP_METADATA_PATH, "r") as f:
        return json.load(f)


def build_mlp_features(
    machine_type,
    air_temp_k,
    process_temp_k,
    rotational_speed_rpm,
    torque_nm,
    tool_wear_min,
):
    temp_diff_k = process_temp_k - air_temp_k

    power_kw = torque_nm * rotational_speed_rpm / 9550

    wear_rate = tool_wear_min / (rotational_speed_rpm + 1e-6)

    temp_ratio = process_temp_k / air_temp_k

    torque_per_rpm = torque_nm / (rotational_speed_rpm + 1e-6)

    torque_toolwear_interaction = torque_nm * tool_wear_min

    return pd.DataFrame([{
        "type": machine_type,
        "air_temp_k": air_temp_k,
        "process_temp_k": process_temp_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
        "temp_diff_k": temp_diff_k,
        "power_kw": power_kw,
        "wear_rate": wear_rate,
        "temp_ratio": temp_ratio,
        "torque_per_rpm": torque_per_rpm,
        "torque_toolwear_interaction": torque_toolwear_interaction,
    }])


def predict_mlp(features):
    model = load_mlp_model()
    preprocessor = load_mlp_preprocessor()
    calibrator = load_mlp_calibrator()
    metadata = load_mlp_metadata()

    processed = preprocessor.transform(features)

    raw_probability = model.predict(processed, verbose=0)

    raw_probability = np.asarray(raw_probability).reshape(-1)

    calibrated_probability = calibrator.predict_proba(
        raw_probability.reshape(-1, 1)
    )[:, 1]

    probability = float(calibrated_probability[0])

    threshold = float(metadata["decision_threshold"])

    prediction = int(probability >= threshold)

    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
    }