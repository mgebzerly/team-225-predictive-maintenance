# Classical ML (Part 3) — Final Summary

- Dataset: `data/features/ai4i2020_features.csv`.
- Rows: 10,000; failure prevalence: 3.39%.
- Approved predictors: 12 safe current-record features.
- Leakage protection: explicit feature whitelist excludes IDs, direct failure-mode labels, and target-derived history.
- Validation: 3-fold stratified CV on the training split; 20% held-out test set.
- Models: Logistic Regression, Decision Tree, Random Forest, Hist Gradient Boosting.
- Hyperparameter search target: **Random Forest**; improved CV PR-AUC: **False**.
- Deployment model selected from training/CV only: **Random Forest**.
- Deployment threshold: **0.33**.
- Cost assumption: 10:1 for false negative vs false positive.

## Cross-validated model selection

| model                  |   cv_pr_auc |   cv_roc_auc |   cost_threshold |   oof_business_cost |   oof_precision |   oof_recall |   oof_f1 | tuned_candidate   | tuning_retained   |
|:-----------------------|------------:|-------------:|-----------------:|--------------------:|----------------:|-------------:|---------:|:------------------|:------------------|
| Random Forest          |    0.882866 |     0.977702 |             0.33 |                 495 |        0.753289 |     0.845018 | 0.796522 | True              | False             |
| Hist Gradient Boosting |    0.873021 |     0.973629 |             0.1  |                 495 |        0.635135 |     0.867159 | 0.733229 | False             | False             |
| Decision Tree          |    0.831587 |     0.941169 |             0.76 |                 528 |        0.699387 |     0.841328 | 0.763819 | False             | False             |
| Logistic Regression    |    0.451815 |     0.930333 |             0.84 |                1315 |        0.377828 |     0.616236 | 0.468443 | False             | False             |

## Held-out test comparison

| model                  |   threshold |   roc_auc |   pr_auc |   precision |   recall |       f1 |   false_negative_rate |   true_negatives |   false_positives |   false_negatives |   true_positives |   business_cost | selected_for_deployment   |
|:-----------------------|------------:|----------:|---------:|------------:|---------:|---------:|----------------------:|-----------------:|------------------:|------------------:|-----------------:|----------------:|:--------------------------|
| Random Forest          |        0.33 |  0.984807 | 0.876096 |    0.681818 | 0.882353 | 0.769231 |              0.117647 |             1904 |                28 |                 8 |               60 |             108 | True                      |
| Hist Gradient Boosting |        0.1  |  0.971174 | 0.893739 |    0.491935 | 0.897059 | 0.635417 |              0.102941 |             1869 |                63 |                 7 |               61 |             133 | False                     |
| Decision Tree          |        0.76 |  0.898619 | 0.82169  |    0.647059 | 0.808824 | 0.718954 |              0.191176 |             1902 |                30 |                13 |               55 |             160 | False                     |
| Logistic Regression    |        0.84 |  0.944236 | 0.467718 |    0.412371 | 0.588235 | 0.484848 |              0.411765 |             1875 |                57 |                28 |               40 |             337 | False                     |

## Important limitations

- The 10:1 cost ratio is illustrative, not measured from a real factory.
- AI4I is a synthetic benchmark, so test performance must not be presented as real industrial reliability.
- The shared feature file also contains rolling/lag/target-derived columns from earlier project work. Part 3 leaves that file unchanged and excludes those columns through an explicit approved-feature whitelist.

Saved deployment artifact: `models/classical_ml_best_pipeline.joblib`.