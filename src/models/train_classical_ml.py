"""Part 3: Classical machine learning for predictive maintenance.

Workflow
--------
1. Load the team's shared feature-engineered dataset.
2. Select only an explicit whitelist of safe predictors.
3. Create a stratified train/test split.
4. Compare four classical classifiers with out-of-fold probabilities.
5. Tune the strongest cross-validated model using average precision.
6. Choose an operating threshold on training out-of-fold predictions using a
   transparent false-negative/false-positive cost assumption.
7. Select the deployment model using training/CV evidence only.
8. Evaluate the already-selected model and all comparators once on the held-out
   test set, then save artifacts, metrics, figures, and feature importance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Part 3 deliberately consumes the team's existing shared feature CSV without
# changing Parts 1/2.  Only current-record predictors that are safe for
# supervised failure prediction are allowed into the model.
TARGET_COLUMN = "machine_failure"
APPROVED_FEATURE_COLUMNS = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",
    "power_kw",
    "temp_ratio",
    "torque_per_rpm",
    "temp_x_speed",
    "torque_x_wear",
]

RANDOM_SEED = 42
TEST_FRACTION = 0.20
CROSS_VALIDATION_FOLDS = 3
FALSE_NEGATIVE_COST = 10
FALSE_POSITIVE_COST = 1
TUNING_ITERATIONS = 4

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "features" / "ai4i2020_features.csv"
REPORT_DIRECTORY = PROJECT_ROOT / "reports"
FIGURE_DIRECTORY = REPORT_DIRECTORY / "figures"
MODEL_DIRECTORY = PROJECT_ROOT / "models"


def load_modeling_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the team's feature CSV and select only approved, leakage-safe columns."""
    feature_data = pd.read_csv(FEATURE_DATA_PATH)

    required_columns = APPROVED_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column_name for column_name in required_columns
        if column_name not in feature_data.columns
    ]
    if missing_columns:
        raise ValueError(
            "The shared feature dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    model_features = feature_data.loc[:, APPROVED_FEATURE_COLUMNS].copy()
    target = feature_data[TARGET_COLUMN].astype(int).copy()
    return model_features, target


def build_preprocessor(training_features: pd.DataFrame) -> ColumnTransformer:
    """Build leakage-safe preprocessing learned only inside each training fold."""
    numeric_feature_names = training_features.select_dtypes(
        include=np.number
    ).columns.tolist()
    categorical_feature_names = training_features.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("mode_imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_feature_names),
            ("categorical", categorical_pipeline, categorical_feature_names),
        ],
        verbose_feature_names_out=False,
    )


def build_candidate_models(
    training_features: pd.DataFrame,
) -> dict[str, Pipeline]:
    """Create the four required classical-ML candidate pipelines."""
    base_preprocessor = build_preprocessor(training_features)

    model_estimators = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_SEED,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }

    return {
        model_name: Pipeline(
            steps=[
                ("preprocessor", clone(base_preprocessor)),
                ("classifier", estimator),
            ]
        )
        for model_name, estimator in model_estimators.items()
    }


def get_tuning_space(model_name: str) -> dict[str, list | np.ndarray]:
    """Return a focused hyperparameter search space for the selected model."""
    if model_name == "Logistic Regression":
        return {
            "classifier__C": np.logspace(-2, 2, 12),
            "classifier__solver": ["liblinear", "lbfgs"],
        }
    if model_name == "Decision Tree":
        return {
            "classifier__max_depth": [3, 4, 5, 6, 8, 10, None],
            "classifier__min_samples_leaf": [1, 2, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__criterion": ["gini", "entropy", "log_loss"],
        }
    if model_name == "Random Forest":
        return {
            "classifier__n_estimators": [100, 150, 220, 300],
            "classifier__max_depth": [6, 8, 10, 14, None],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", 0.7],
        }
    return {
        "classifier__learning_rate": [0.03, 0.05, 0.08, 0.10, 0.15],
        "classifier__max_iter": [120, 180, 240],
        "classifier__max_leaf_nodes": [7, 15, 31],
        "classifier__min_samples_leaf": [10, 20, 40],
        "classifier__l2_regularization": [0.0, 0.1, 1.0, 5.0],
    }


def find_cost_sensitive_threshold(
    actual_target: pd.Series | np.ndarray,
    failure_probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Find the threshold that minimizes the stated maintenance cost proxy."""
    actual_target_array = np.asarray(actual_target)
    threshold_rows: list[dict[str, float | int]] = []

    for decision_threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        predicted_target = (
            failure_probabilities >= decision_threshold
        ).astype(int)
        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(
                actual_target_array,
                predicted_target,
                labels=[0, 1],
            ).ravel()
        )
        business_cost = (
            FALSE_NEGATIVE_COST * false_negative
            + FALSE_POSITIVE_COST * false_positive
        )

        threshold_rows.append(
            {
                "threshold": decision_threshold,
                "true_negatives": int(true_negative),
                "false_positives": int(false_positive),
                "false_negatives": int(false_negative),
                "true_positives": int(true_positive),
                "business_cost": int(business_cost),
                "precision": precision_score(
                    actual_target_array,
                    predicted_target,
                    zero_division=0,
                ),
                "recall": recall_score(
                    actual_target_array,
                    predicted_target,
                    zero_division=0,
                ),
                "f1": f1_score(
                    actual_target_array,
                    predicted_target,
                    zero_division=0,
                ),
            }
        )

    threshold_results = pd.DataFrame(threshold_rows)
    best_threshold_row = threshold_results.sort_values(
        [
            "business_cost",
            "false_negatives",
            "false_positives",
            "threshold",
        ],
        ascending=[True, True, True, True],
    ).iloc[0]

    return float(best_threshold_row["threshold"]), threshold_results


def calculate_classification_metrics(
    model_name: str,
    actual_target: pd.Series | np.ndarray,
    failure_probabilities: np.ndarray,
    decision_threshold: float,
) -> dict[str, float | int | str]:
    """Calculate probability and threshold-dependent evaluation metrics."""
    predicted_target = (failure_probabilities >= decision_threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual_target,
        predicted_target,
        labels=[0, 1],
    ).ravel()

    return {
        "model": model_name,
        "threshold": decision_threshold,
        "roc_auc": roc_auc_score(actual_target, failure_probabilities),
        "pr_auc": average_precision_score(actual_target, failure_probabilities),
        "precision": precision_score(
            actual_target,
            predicted_target,
            zero_division=0,
        ),
        "recall": recall_score(
            actual_target,
            predicted_target,
            zero_division=0,
        ),
        "f1": f1_score(actual_target, predicted_target, zero_division=0),
        "false_negative_rate": (
            false_negative / (false_negative + true_positive)
            if false_negative + true_positive
            else np.nan
        ),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
        "business_cost": int(
            FALSE_NEGATIVE_COST * false_negative
            + FALSE_POSITIVE_COST * false_positive
        ),
    }


def create_output_directories() -> None:
    REPORT_DIRECTORY.mkdir(exist_ok=True)
    FIGURE_DIRECTORY.mkdir(exist_ok=True)
    MODEL_DIRECTORY.mkdir(exist_ok=True)


def main() -> None:
    create_output_directories()

    all_features, all_targets = load_modeling_dataset()
    (
        training_features,
        test_features,
        training_targets,
        test_targets,
    ) = train_test_split(
        all_features,
        all_targets,
        test_size=TEST_FRACTION,
        stratify=all_targets,
        random_state=RANDOM_SEED,
    )

    split_summary = pd.DataFrame(
        {
            "split": ["train", "test"],
            "rows": [len(training_features), len(test_features)],
            "failures": [int(training_targets.sum()), int(test_targets.sum())],
            "failure_rate": [training_targets.mean(), test_targets.mean()],
        }
    )
    split_summary.to_csv(
        REPORT_DIRECTORY / "classical_ml_split.csv",
        index=False,
    )

    stratified_cv = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    candidate_models = build_candidate_models(training_features)

    cross_validation_rows: list[dict[str, float | int | str]] = []
    out_of_fold_probabilities: dict[str, np.ndarray] = {}
    selected_thresholds: dict[str, float] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}

    # Stage 1: compare all baseline candidates using out-of-fold probabilities.
    for model_name, model_pipeline in candidate_models.items():
        print(f"Cross-validating: {model_name}", flush=True)
        failure_probabilities = cross_val_predict(
            model_pipeline,
            training_features,
            training_targets,
            cv=stratified_cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        out_of_fold_probabilities[model_name] = failure_probabilities

        decision_threshold, threshold_results = find_cost_sensitive_threshold(
            training_targets,
            failure_probabilities,
        )
        selected_thresholds[model_name] = decision_threshold
        threshold_tables[model_name] = threshold_results

        threshold_row = threshold_results.loc[
            threshold_results["threshold"].eq(decision_threshold)
        ].iloc[0]
        cross_validation_rows.append(
            {
                "model": model_name,
                "cv_pr_auc": average_precision_score(
                    training_targets,
                    failure_probabilities,
                ),
                "cv_roc_auc": roc_auc_score(
                    training_targets,
                    failure_probabilities,
                ),
                "cost_threshold": decision_threshold,
                "oof_business_cost": int(threshold_row["business_cost"]),
                "oof_precision": float(threshold_row["precision"]),
                "oof_recall": float(threshold_row["recall"]),
                "oof_f1": float(threshold_row["f1"]),
            }
        )

    baseline_cv_results = pd.DataFrame(cross_validation_rows).sort_values(
        "cv_pr_auc",
        ascending=False,
    )

    # Stage 2: tune the strongest baseline model by CV average precision.
    tuning_target_model_name = str(baseline_cv_results.iloc[0]["model"])
    print(f"Tuning: {tuning_target_model_name}", flush=True)
    hyperparameter_search = RandomizedSearchCV(
        estimator=clone(candidate_models[tuning_target_model_name]),
        param_distributions=get_tuning_space(tuning_target_model_name),
        n_iter=TUNING_ITERATIONS,
        scoring="average_precision",
        cv=stratified_cv,
        random_state=RANDOM_SEED,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    hyperparameter_search.fit(training_features, training_targets)

    tuning_results = pd.DataFrame(hyperparameter_search.cv_results_)[
        ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ].copy()
    tuning_results["params"] = tuning_results["params"].map(
        lambda parameters: json.dumps(
            parameters,
            sort_keys=True,
            default=str,
        )
    )
    tuning_results.sort_values("rank_test_score").to_csv(
        REPORT_DIRECTORY / "classical_ml_tuning_results.csv",
        index=False,
    )

    baseline_pr_auc = float(
        baseline_cv_results.loc[
            baseline_cv_results["model"].eq(tuning_target_model_name),
            "cv_pr_auc",
        ].iloc[0]
    )
    tuning_improved_pr_auc = (
        float(hyperparameter_search.best_score_) > baseline_pr_auc
    )
    retained_tuned_model = (
        hyperparameter_search.best_estimator_
        if tuning_improved_pr_auc
        else clone(candidate_models[tuning_target_model_name])
    )

    retained_model_probabilities = cross_val_predict(
        retained_tuned_model,
        training_features,
        training_targets,
        cv=stratified_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    (
        retained_model_threshold,
        retained_model_threshold_results,
    ) = find_cost_sensitive_threshold(
        training_targets,
        retained_model_probabilities,
    )

    out_of_fold_probabilities[tuning_target_model_name] = (
        retained_model_probabilities
    )
    selected_thresholds[tuning_target_model_name] = retained_model_threshold
    threshold_tables[tuning_target_model_name] = retained_model_threshold_results
    candidate_models[tuning_target_model_name] = retained_tuned_model

    # Rebuild the CV summary so the tuned/retained version replaces its baseline.
    final_cv_rows: list[dict[str, float | int | str]] = []
    for model_name in candidate_models:
        model_probabilities = out_of_fold_probabilities[model_name]
        decision_threshold = selected_thresholds[model_name]
        threshold_results = threshold_tables[model_name]
        threshold_row = threshold_results.loc[
            threshold_results["threshold"].eq(decision_threshold)
        ].iloc[0]
        final_cv_rows.append(
            {
                "model": model_name,
                "cv_pr_auc": average_precision_score(
                    training_targets,
                    model_probabilities,
                ),
                "cv_roc_auc": roc_auc_score(
                    training_targets,
                    model_probabilities,
                ),
                "cost_threshold": decision_threshold,
                "oof_business_cost": int(threshold_row["business_cost"]),
                "oof_precision": float(threshold_row["precision"]),
                "oof_recall": float(threshold_row["recall"]),
                "oof_f1": float(threshold_row["f1"]),
                "tuned_candidate": model_name == tuning_target_model_name,
                "tuning_retained": (
                    model_name == tuning_target_model_name
                    and tuning_improved_pr_auc
                ),
            }
        )

    final_cv_results = pd.DataFrame(final_cv_rows).sort_values(
        ["oof_business_cost", "cv_pr_auc", "oof_recall"],
        ascending=[True, False, False],
    )
    final_cv_results.to_csv(
        REPORT_DIRECTORY / "classical_ml_cv_results.csv",
        index=False,
    )

    # Deployment model is selected before looking at held-out test labels.
    selected_model_name = str(final_cv_results.iloc[0]["model"])
    selected_model_threshold = float(
        final_cv_results.iloc[0]["cost_threshold"]
    )

    combined_threshold_results = []
    for model_name, threshold_results in threshold_tables.items():
        model_threshold_results = threshold_results.copy()
        model_threshold_results.insert(0, "model", model_name)
        combined_threshold_results.append(model_threshold_results)
    pd.concat(combined_threshold_results, ignore_index=True).to_csv(
        REPORT_DIRECTORY / "classical_ml_threshold_search.csv",
        index=False,
    )

    # Stage 3: fit all candidate models once on the full training set and report
    # held-out performance. Test results do not change the selected model.
    fitted_models: dict[str, Pipeline] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    test_metric_rows: list[dict[str, float | int | str | bool]] = []

    for model_name, model_pipeline in candidate_models.items():
        print(f"Final fit/test: {model_name}", flush=True)
        fitted_model = clone(model_pipeline)
        fitted_model.fit(training_features, training_targets)
        fitted_models[model_name] = fitted_model

        failure_probabilities = fitted_model.predict_proba(test_features)[:, 1]
        test_probabilities[model_name] = failure_probabilities

        model_metrics = calculate_classification_metrics(
            model_name,
            test_targets,
            failure_probabilities,
            selected_thresholds[model_name],
        )
        model_metrics["selected_for_deployment"] = (
            model_name == selected_model_name
        )
        test_metric_rows.append(model_metrics)

    test_results = pd.DataFrame(test_metric_rows).sort_values(
        ["selected_for_deployment", "business_cost", "pr_auc"],
        ascending=[False, True, False],
    )
    test_results.to_csv(
        REPORT_DIRECTORY / "classical_ml_metrics.csv",
        index=False,
    )

    prediction_output = pd.DataFrame(
        {
            "row_index": test_features.index.to_numpy(),
            "actual_machine_failure": test_targets.to_numpy(),
        }
    )
    for model_name, failure_probabilities in test_probabilities.items():
        safe_model_name = (
            model_name.lower().replace(" ", "_").replace("-", "_")
        )
        prediction_output[f"{safe_model_name}_probability"] = (
            failure_probabilities
        )
        prediction_output[f"{safe_model_name}_prediction"] = (
            failure_probabilities >= selected_thresholds[model_name]
        ).astype(int)
    prediction_output.to_csv(
        REPORT_DIRECTORY / "classical_ml_test_predictions.csv",
        index=False,
    )

    selected_fitted_model = fitted_models[selected_model_name]
    joblib.dump(
        selected_fitted_model,
        MODEL_DIRECTORY / "classical_ml_best_pipeline.joblib",
    )

    model_metadata = {
        "selected_model": selected_model_name,
        "selection_basis": (
            "Lowest out-of-fold training business cost, with CV PR-AUC and "
            "OOF recall as tie-breakers; held-out test metrics were not used "
            "for model selection."
        ),
        "decision_threshold": selected_model_threshold,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "cost_ratio_note": (
            "10:1 is a project assumption for demonstrating cost-sensitive "
            "decision making, not a measured industrial maintenance cost."
        ),
        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
        "test_fraction": TEST_FRACTION,
        "random_seed": RANDOM_SEED,
        "tuning_target_model": tuning_target_model_name,
        "tuning_improved_cv_pr_auc": tuning_improved_pr_auc,
        "best_tuning_parameters": hyperparameter_search.best_params_,
        "target": TARGET_COLUMN,
        "approved_input_features": APPROVED_FEATURE_COLUMNS,
        "source_dataset": str(FEATURE_DATA_PATH.relative_to(PROJECT_ROOT)),
    }
    (MODEL_DIRECTORY / "classical_ml_metadata.json").write_text(
        json.dumps(model_metadata, indent=2, default=str),
        encoding="utf-8",
    )

    # Permutation importance gives an original-feature-level explanation for the
    # selected full pipeline and works for both numeric and categorical inputs.
    importance_result = permutation_importance(
        selected_fitted_model,
        test_features,
        test_targets,
        scoring="average_precision",
        n_repeats=3,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": test_features.columns,
            "importance_mean": importance_result.importances_mean,
            "importance_std": importance_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    feature_importance.to_csv(
        REPORT_DIRECTORY / "classical_ml_feature_importance.csv",
        index=False,
    )

    # ROC curves.
    plt.figure(figsize=(8, 6))
    for model_name, failure_probabilities in test_probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            test_targets,
            failure_probabilities,
        )
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{model_name} "
                f"({roc_auc_score(test_targets, failure_probabilities):.3f})"
            ),
        )
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Classical ML — ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIRECTORY / "classical_ml_roc_curves.png", dpi=180)
    plt.close()

    # Precision-recall curves.
    plt.figure(figsize=(8, 6))
    for model_name, failure_probabilities in test_probabilities.items():
        precision_values, recall_values, _ = precision_recall_curve(
            test_targets,
            failure_probabilities,
        )
        plt.plot(
            recall_values,
            precision_values,
            label=(
                f"{model_name} "
                f"({average_precision_score(test_targets, failure_probabilities):.3f})"
            ),
        )
    plt.axhline(
        test_targets.mean(),
        linestyle="--",
        linewidth=1,
        label="Failure prevalence",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Classical ML — Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIRECTORY / "classical_ml_pr_curves.png", dpi=180)
    plt.close()

    # Cost curve for the deployment-selected model, based only on training OOF.
    selected_threshold_table = threshold_tables[selected_model_name]
    plt.figure(figsize=(8, 5))
    plt.plot(
        selected_threshold_table["threshold"],
        selected_threshold_table["business_cost"],
    )
    plt.axvline(selected_model_threshold, linestyle="--")
    plt.xlabel("Decision Threshold")
    plt.ylabel(
        f"OOF cost = {FALSE_NEGATIVE_COST}×FN + "
        f"{FALSE_POSITIVE_COST}×FP"
    )
    plt.title(f"Cost-Sensitive Threshold — {selected_model_name}")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "classical_ml_threshold_cost.png",
        dpi=180,
    )
    plt.close()

    # Confusion matrix for the already-selected deployment model on the test set.
    selected_test_predictions = (
        test_probabilities[selected_model_name] >= selected_model_threshold
    ).astype(int)
    selected_confusion_matrix = confusion_matrix(
        test_targets,
        selected_test_predictions,
        labels=[0, 1],
    )
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(selected_confusion_matrix)
    for (row_index, column_index), value in np.ndenumerate(
        selected_confusion_matrix
    ):
        axis.text(
            column_index,
            row_index,
            str(value),
            ha="center",
            va="center",
        )
    axis.set_xticks([0, 1], labels=["No failure", "Failure"])
    axis.set_yticks([0, 1], labels=["No failure", "Failure"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"{selected_model_name} — Confusion Matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIRECTORY / "classical_ml_best_confusion_matrix.png",
        dpi=180,
    )
    plt.close(figure)

    # Feature importance figure.
    top_feature_importance = feature_importance.head(12).sort_values(
        "importance_mean"
    )
    plt.figure(figsize=(8, 6))
    plt.barh(
        top_feature_importance["feature"],
        top_feature_importance["importance_mean"],
        xerr=top_feature_importance["importance_std"],
    )
    plt.xlabel("Decrease in Average Precision after permutation")
    plt.title(f"Permutation Importance — {selected_model_name}")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "classical_ml_feature_importance.png",
        dpi=180,
    )
    plt.close()

    summary_lines = [
        "# Classical ML (Part 3) — Final Summary",
        "",
        f"- Dataset: `{FEATURE_DATA_PATH.relative_to(PROJECT_ROOT)}`.",
        f"- Rows: {len(all_features):,}; failure prevalence: {all_targets.mean():.2%}.",
        f"- Approved predictors: {len(APPROVED_FEATURE_COLUMNS)} safe current-record features.",
        "- Leakage protection: explicit feature whitelist excludes IDs, direct failure-mode labels, and target-derived history.",
        f"- Validation: {CROSS_VALIDATION_FOLDS}-fold stratified CV on the training split; {TEST_FRACTION:.0%} held-out test set.",
        "- Models: Logistic Regression, Decision Tree, Random Forest, Hist Gradient Boosting.",
        f"- Hyperparameter search target: **{tuning_target_model_name}**; improved CV PR-AUC: **{tuning_improved_pr_auc}**.",
        f"- Deployment model selected from training/CV only: **{selected_model_name}**.",
        f"- Deployment threshold: **{selected_model_threshold:.2f}**.",
        f"- Cost assumption: {FALSE_NEGATIVE_COST}:1 for false negative vs false positive.",
        "",
        "## Cross-validated model selection",
        "",
        final_cv_results.to_markdown(index=False),
        "",
        "## Held-out test comparison",
        "",
        test_results.to_markdown(index=False),
        "",
        "## Important limitations",
        "",
        "- The 10:1 cost ratio is illustrative, not measured from a real factory.",
        "- AI4I is a synthetic benchmark, so test performance must not be presented as real industrial reliability.",
        "- The shared feature file also contains rolling/lag/target-derived columns from earlier project work. Part 3 leaves that file unchanged and excludes those columns through an explicit approved-feature whitelist.",
        "",
        "Saved deployment artifact: `models/classical_ml_best_pipeline.joblib`.",
    ]
    (REPORT_DIRECTORY / "classical_ml_summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nCross-validated selection:")
    print(final_cv_results.to_string(index=False), flush=True)
    print("\nHeld-out test results:")
    print(test_results.to_string(index=False), flush=True)
    print(
        f"\nSelected for deployment: {selected_model_name} "
        f"@ threshold {selected_model_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
