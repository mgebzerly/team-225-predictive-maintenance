# Feature Engineering and Data Preprocessing Module
import numpy as np
import pandas as pd


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
  """Remove sub-failure indicators to prevent data leakage during training."""
  leakage_cols = ["twf", "hdf", "pwf", "osf", "rnf"]
  existing_leakage = [col for col in leakage_cols if col in df.columns]
  return df.drop(columns=existing_leakage)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
  """Compute mechanical and thermal domain-specific features."""
  df_feat = df.copy()

  if "process_temp_k" in df_feat.columns and "air_temp_k" in df_feat.columns:
    df_feat["temp_diff_k"] = (
        df_feat["process_temp_k"] - df_feat["air_temp_k"]
    )
    df_feat["temp_ratio"] = (
        df_feat["process_temp_k"] / df_feat["air_temp_k"]
    )

  if (
      "rotational_speed_rpm" in df_feat.columns
      and "torque_nm" in df_feat.columns
  ):
    angular_velocity = df_feat["rotational_speed_rpm"] * (2 * np.pi / 60)
    df_feat["power_kw"] = (df_feat["torque_nm"] * angular_velocity) / 1000.0
    df_feat["torque_per_rpm"] = df_feat["torque_nm"] / (
        df_feat["rotational_speed_rpm"] + 1e-5
    )

  if "torque_nm" in df_feat.columns and "tool_wear_min" in df_feat.columns:
    df_feat["torque_toolwear_interaction"] = (
        df_feat["torque_nm"] * df_feat["tool_wear_min"]
    )

  return df_feat


def run_feature_pipeline(
    df: pd.DataFrame, drop_leakage: bool = True
) -> pd.DataFrame:
  """Run the full feature pipeline for downstream model training."""
  df_processed = add_engineered_features(df)
  if drop_leakage:
    df_processed = remove_leakage_features(df_processed)
  return df_processed