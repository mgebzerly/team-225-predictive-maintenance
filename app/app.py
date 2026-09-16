import streamlit as st

from model_utils import (
    build_rf_features,
    predict_rf,
    build_ae_features,
    predict_anomaly,
    build_mlp_features,
    predict_mlp,
)
from logging_config import setup_logging

logger = setup_logging()

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)


st.title("⚙️ Predictive Maintenance")
st.write(
    "Machine failure prediction and anomaly detection using trained ML models."
)


# -----------------------------
# Machine Inputs
# -----------------------------

st.sidebar.header("Machine Information")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp_k = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=298.1,
)

process_temp_k = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=308.6,
)

rotational_speed_rpm = st.sidebar.number_input(
    "Rotational Speed (RPM)",
    min_value=0.0,
    max_value=5000.0,
    value=1500.0,
)

torque_nm = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
)

tool_wear_min = st.sidebar.number_input(
    "Tool Wear (min)",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
)


analyze = st.sidebar.button(
    "🔍 Analyze Machine",
    use_container_width=True,
)


# -----------------------------
# Main
# -----------------------------

if analyze:
    logger.info("Machine analysis started")

    rf_features = build_rf_features(
        machine_type,
        air_temp_k,
        process_temp_k,
        rotational_speed_rpm,
        torque_nm,
        tool_wear_min,
    )

    ae_features = build_ae_features(
        air_temp_k,
        process_temp_k,
        rotational_speed_rpm,
        torque_nm,
        tool_wear_min,
    )
    mlp_features = build_mlp_features(
        machine_type,
        air_temp_k,
        process_temp_k,
        rotational_speed_rpm,
        torque_nm,
        tool_wear_min,
    )

    # Random Forest
    rf_result = predict_rf(rf_features)

    # Autoencoder
    ae_result = predict_anomaly(ae_features)

    # MLP
    mlp_result = predict_mlp(mlp_features)

    # -----------------------------
    # Derived Features
    # -----------------------------

    st.subheader("📊 Machine Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Machine Type", machine_type)
        st.metric("Air Temperature", f"{air_temp_k:.2f} K")

    with col2:
        st.metric("Process Temperature", f"{process_temp_k:.2f} K")
        st.metric("Rotational Speed", f"{rotational_speed_rpm:.0f} RPM")

    with col3:
        st.metric("Torque", f"{torque_nm:.2f} Nm")
        st.metric("Tool Wear", f"{tool_wear_min:.0f} min")

    st.divider()

    # -----------------------------
    # Random Forest Result
    # -----------------------------

    st.subheader("🤖 Random Forest Prediction")

    rf_probability = rf_result["probability"]
    rf_threshold = rf_result["threshold"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Failure Probability",
            f"{rf_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{rf_threshold:.2f}"
        )

    if rf_result["prediction"]:
        st.error("⚠️ Predicted Machine Failure")
    else:
        st.success("✅ No Machine Failure Predicted")

    st.divider()

    # -----------------------------
    # Deep MLP Result
    # -----------------------------

    st.subheader("🧠 Deep MLP Prediction")

    mlp_probability = mlp_result["probability"]
    mlp_threshold = mlp_result["threshold"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Failure Probability",
            f"{mlp_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{mlp_threshold:.2f}"
        )

    if mlp_result["prediction"]:
        st.error("⚠️ Deep MLP predicts machine failure")
    else:
        st.success("✅ Deep MLP predicts no machine failure")

    # -----------------------------
    # Autoencoder Result
    # -----------------------------

    st.subheader("🔎 Anomaly Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Anomaly Score",
            f"{ae_result['score']:.6f}"
        )

    with col2:
        st.metric(
            "Anomaly Threshold",
            f"{ae_result['threshold']:.6f}"
        )

    if ae_result["anomaly"]:
        st.warning("🚨 Machine behavior is classified as anomalous.")
    else:
        st.success("✅ Machine behavior is within the normal range.")

else:
    st.info(
        "Enter the machine measurements in the sidebar and click "
        "**Analyze Machine**."
    )