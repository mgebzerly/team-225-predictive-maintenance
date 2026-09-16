"""
Part 5 — Anomaly Detection & Explainability

Train an autoencoder on normal-only machine readings to detect anomalies,
explain detections with SHAP, run a cost comparison, and simulate sensor drift.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
FEAT_CSV  = ROOT / "data" / "features" / "ai4i2020_features.csv"
REPORT_DIR = ROOT / "reports"
FIG_DIR    = REPORT_DIR / "figures"
MODEL_DIR  = ROOT / "models"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# same numeric features used in Part 3 (minus the categorical 'type')
FEATURES = [
    "air_temp_k", "process_temp_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min",
    "temp_diff_k", "power_kw", "temp_ratio",
    "torque_per_rpm", "temp_x_speed", "torque_x_wear",
]
TARGET = "machine_failure"

# cost assumptions matching Part 3
FN_COST = 10
FP_COST = 1


# ── helpers ────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(FEAT_CSV)
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    return X, y


def build_autoencoder(n_in):
    """Simple symmetric autoencoder: n→32→16→8→16→32→n."""
    inp = keras.Input(shape=(n_in,))
    x = keras.layers.Dense(32, activation="relu")(inp)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(8, activation="relu")(x)       # bottleneck
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(n_in, activation="linear")(x)

    model = keras.Model(inp, out, name="ae_anomaly")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def recon_error(model, X_sc):
    """Per-sample MSE between input and reconstruction."""
    pred = model.predict(X_sc, verbose=0)
    return np.mean((X_sc - pred) ** 2, axis=1)


def pick_threshold(y_true, errors):
    """Sweep percentiles 80–99 and pick the one with lowest business cost."""
    best_cost, best_t = float("inf"), 0.0
    for pct in range(80, 100):
        t = np.percentile(errors, pct)
        preds = (errors > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = FN_COST * fn + FP_COST * fp
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return best_t


def make_error_fn(model, scaler):
    """Wrap the autoencoder so SHAP can explain reconstruction error."""
    def _fn(X_raw):
        X_sc = scaler.transform(X_raw)
        return recon_error(model, X_sc)
    return _fn


def simulate_drift(X_raw, model, scaler, threshold, n_windows=10):
    """
    Gradually shift air_temp +5K and torque +15Nm over n_windows
    to simulate a cooling-system / mechanical-wear degradation scenario.
    """
    rows = []
    for w in range(n_windows):
        frac = w / (n_windows - 1)
        Xd = X_raw.copy()
        Xd["air_temp_k"]  += 5.0  * frac
        Xd["torque_nm"]   += 15.0 * frac
        # re-derive columns that depend on the shifted ones
        Xd["temp_diff_k"]    = Xd["process_temp_k"] - Xd["air_temp_k"]
        Xd["temp_ratio"]     = Xd["process_temp_k"] / Xd["air_temp_k"]
        Xd["power_kw"]       = Xd["torque_nm"] * Xd["rotational_speed_rpm"] / 9550
        Xd["torque_per_rpm"] = Xd["torque_nm"] / (Xd["rotational_speed_rpm"] + 1e-6)
        Xd["temp_x_speed"]   = Xd["temp_diff_k"] * Xd["rotational_speed_rpm"]
        Xd["torque_x_wear"]  = Xd["torque_nm"] * Xd["tool_wear_min"]

        errs = recon_error(model, scaler.transform(Xd.values))
        rows.append({
            "window": w,
            "drift_fraction": round(frac, 2),
            "mean_error": errs.mean(),
            "anomaly_rate": (errs > threshold).mean(),
        })
    return pd.DataFrame(rows)


# ── main pipeline ──────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # ---- 1. load & split (same seed & ratio as Part 3) ----
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # keep only normal samples for training
    mask_norm = (y_tr == 0).values
    X_norm = X_tr_sc[mask_norm]
    print(f"Training on {X_norm.shape[0]} normal samples "
          f"(held out {(~mask_norm).sum()} failure samples)")

    # ---- 2. train autoencoder ----
    ae = build_autoencoder(X_tr_sc.shape[1])
    hist = ae.fit(
        X_norm, X_norm,
        epochs=50, batch_size=64,
        validation_split=0.1, verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True)])

    # ---- 3. reconstruction errors ----
    tr_err = recon_error(ae, X_tr_sc)
    te_err = recon_error(ae, X_te_sc)

    # ---- 4. cost-optimal threshold (on training fold) ----
    threshold = pick_threshold(y_tr, tr_err)
    print(f"Selected threshold: {threshold:.6f}")

    te_pred = (te_err > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, te_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_te, te_pred, zero_division=0)
    rec  = recall_score(y_te, te_pred, zero_division=0)
    f1   = f1_score(y_te, te_pred, zero_division=0)
    roc  = roc_auc_score(y_te, te_err)
    pr   = average_precision_score(y_te, te_err)
    cost = FN_COST * fn + FP_COST * fp

    print(f"Test  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
          f"ROC={roc:.3f}  PR={pr:.3f}  Cost={cost}")

    # ════════════════════ FIGURES ════════════════════════════════════════

    # fig 1 — training curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist.history["loss"], label="train")
    ax.plot(hist.history["val_loss"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "anomaly_ae_training.png", dpi=150)
    plt.close(fig)

    # fig 2 — error distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(te_err[y_te == 0], bins=60, alpha=0.6, label="Normal", density=True)
    ax.hist(te_err[y_te == 1], bins=60, alpha=0.6, label="Failure", density=True)
    ax.axvline(threshold, color="red", ls="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction Error"); ax.set_ylabel("Density")
    ax.set_title("Error Distribution — Normal vs Failure")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "anomaly_error_dist.png", dpi=150)
    plt.close(fig)

    # fig 3 — ROC curve
    fpr, tpr, _ = roc_curve(y_te, te_err)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Autoencoder (AUC={roc:.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Anomaly Detection — ROC Curve")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "anomaly_roc.png", dpi=150)
    plt.close(fig)

    # fig 4 — confusion matrix
    cm = confusion_matrix(y_te, te_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels=["Normal", "Failure"])
    ax.set_yticks([0, 1], labels=["Normal", "Failure"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Autoencoder — Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "anomaly_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ════════════════════ SHAP ══════════════════════════════════════════

    print("Running SHAP explanations...")
    error_fn = make_error_fn(ae, scaler)

    bg = X_tr.sample(50, random_state=SEED)
    explainer = shap.KernelExplainer(error_fn, bg)
    sample_idx = X_te.sample(100, random_state=SEED).index
    sample = X_te.loc[sample_idx]
    shap_vals = explainer.shap_values(sample, nsamples=100, silent=True)

    # fig 5 — SHAP bar (mean |SHAP|)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([FEATURES[i] for i in order], mean_abs[order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance — Anomaly Detection")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "anomaly_shap_bar.png", dpi=150)
    plt.close(fig)

    # fig 6 — SHAP beeswarm
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_vals, sample, feature_names=FEATURES, show=False)
    plt.title("SHAP Summary — Anomaly Detection", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "anomaly_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ════════════════════ DRIFT SIMULATION ══════════════════════════════

    print("Simulating sensor drift...")
    drift_df = simulate_drift(X_te.copy(), ae, scaler, threshold)
    drift_df.to_csv(REPORT_DIR / "anomaly_drift_results.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(drift_df["window"], drift_df["mean_error"], "o-", color="#2196F3")
    ax1.axhline(threshold, color="red", ls="--", label="Threshold")
    ax1.set_xlabel("Drift Window"); ax1.set_ylabel("Mean Recon. Error")
    ax1.set_title("Reconstruction Error Under Drift"); ax1.legend()

    ax2.bar(drift_df["window"], drift_df["anomaly_rate"] * 100, color="#FF9800")
    ax2.set_xlabel("Drift Window"); ax2.set_ylabel("Anomaly Rate (%)")
    ax2.set_title("Detected Anomalies Under Drift")
    fig.suptitle("Drift: air_temp +5K, torque +15Nm over 10 windows", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "anomaly_drift_sim.png", dpi=150)
    plt.close(fig)

    # ════════════════════ COST COMPARISON ═══════════════════════════════

    no_model_cost = FN_COST * int(y_te.sum())   # miss every failure

    cost_dict = {
        "No Model\n(reactive)": int(no_model_cost),
        "Autoencoder": int(cost),
    }
    ml_path = REPORT_DIR / "classical_ml_metrics.csv"
    if ml_path.exists():
        ml_df = pd.read_csv(ml_path)
        sel = ml_df[ml_df["selected_for_deployment"] == True]
        if not sel.empty:
            cost_dict["Random Forest\n(Part 3)"] = int(sel.iloc[0]["business_cost"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#d9534f", "#5bc0de", "#5cb85c"][:len(cost_dict)]
    bars = ax.bar(cost_dict.keys(), cost_dict.values(), color=colors)
    for b, v in zip(bars, cost_dict.values()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Business Cost ({FN_COST}×FN + {FP_COST}×FP)")
    ax.set_title("Maintenance Strategy Cost Comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "anomaly_cost_comparison.png", dpi=150)
    plt.close(fig)

    # ════════════════════ SAVE ARTIFACTS ════════════════════════════════

    ae.save(MODEL_DIR / "anomaly_autoencoder.keras")
    joblib.dump(scaler, MODEL_DIR / "anomaly_scaler.joblib")

    meta = {
        "model": "Autoencoder (11 -> 32 -> 16 -> 8 -> 16 -> 32 -> 11)",
        "trained_on": "normal samples only",
        "normal_training_samples": int(X_norm.shape[0]),
        "threshold": float(threshold),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "test_roc_auc": round(roc, 4),
        "test_pr_auc": round(pr, 4),
        "test_business_cost": int(cost),
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "features": FEATURES,
        "seed": SEED,
    }
    (MODEL_DIR / "anomaly_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    # metrics csv
    pd.DataFrame([{
        "model": "Autoencoder", "threshold": threshold,
        "roc_auc": roc, "pr_auc": pr,
        "precision": prec, "recall": rec, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "business_cost": cost,
    }]).to_csv(REPORT_DIR / "anomaly_metrics.csv", index=False)

    # ════════════════════ MARKDOWN SUMMARY ══════════════════════════════

    # top 5 SHAP features
    top5 = np.argsort(mean_abs)[::-1][:5]
    shap_rows = ""
    for rank, i in enumerate(top5, 1):
        shap_rows += f"| {rank} | {FEATURES[i]} | {mean_abs[i]:.4f} |\n"

    ml_cost_row = ""
    if "Random Forest\n(Part 3)" in cost_dict:
        ml_cost_row = f"| Random Forest (Part 3) | {cost_dict['Random Forest' + chr(10) + '(Part 3)']} |\n"

    summary = f"""## Anomaly Detection & Explainability (Part 5)

### Approach

- Autoencoder trained exclusively on **normal** samples ({X_norm.shape[0]} readings).
- Architecture: {len(FEATURES)} → 32 → 16 → **8** → 16 → 32 → {len(FEATURES)}.
- Anomaly score = per-sample reconstruction error (MSE).
- Threshold selected by minimizing business cost ({FN_COST}:1 FN:FP ratio) on training data.

### Test Set Results

| Metric | Value |
|--------|-------|
| ROC-AUC | {roc:.4f} |
| PR-AUC | {pr:.4f} |
| Precision | {prec:.4f} |
| Recall | {rec:.4f} |
| F1-Score | {f1:.4f} |
| Business Cost | {cost} |
| Threshold | {threshold:.6f} |

### Confusion Matrix

|  | Pred Normal | Pred Failure |
|--|------------|-------------|
| **Actual Normal** | {tn} | {fp} |
| **Actual Failure** | {fn} | {tp} |

### Cost Comparison

| Strategy | Cost |
|----------|------|
| No model (all reactive) | {no_model_cost} |
| Autoencoder (Part 5) | {cost} |
{ml_cost_row}
### SHAP Feature Importance (Top 5)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
{shap_rows}
### Drift Simulation

Simulated gradual sensor degradation: air temperature +5 K and torque +15 Nm
over 10 time windows.

| Window | Drift % | Mean Error | Anomaly Rate |
|--------|---------|------------|-------------|
"""
    for _, r in drift_df.iterrows():
        summary += (f"| {int(r['window'])} | {r['drift_fraction']:.0%} "
                    f"| {r['mean_error']:.4f} | {r['anomaly_rate']:.1%} |\n")

    summary += f"""
The anomaly rate rose from **{drift_df.iloc[0]['anomaly_rate']:.1%}** (no drift) to
**{drift_df.iloc[-1]['anomaly_rate']:.1%}** (full drift), confirming the autoencoder
catches distribution shifts from sensor degradation.

### Saved Artifacts

- `models/anomaly_autoencoder.keras`
- `models/anomaly_scaler.joblib`
- `models/anomaly_metadata.json`
- `reports/anomaly_metrics.csv`
- `reports/anomaly_drift_results.csv`
"""

    (REPORT_DIR / "anomaly_detection_summary.md").write_text(
        summary, encoding="utf-8")

    print("\n=== Part 5 complete ===")
    print(f"Selected: Autoencoder @ threshold {threshold:.4f}")
    print(f"Test business cost: {cost}")


if __name__ == "__main__":
    main()
