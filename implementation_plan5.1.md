# Phase 5 — Process-Aware Advisory Optimizer

## Background

The Phase 4 soft sensor (Model A + Model B) predicts Total C4 slippage from the current plant state. Phase 5 closes the loop: given that plant state, it recommends the **steam and reflux set-points** that minimise Total C4 while respecting process constraints and utility economics.

The key architectural upgrade over a naive grid search is that the optimizer does **not** hold temperatures and pressure frozen. Instead it:

1. Predicts how Bottom Temp, Tray Temp, and Column Pressure would change at `t+1` for every candidate `(steam, reflux)` pair — using three lightweight **surrogate process models** (T1, T2, T3).
2. Uses the predicted process conditions for safety constraint checking.
3. Feeds those predicted process conditions into the frozen soft sensor to obtain a physically coherent Total C4 estimate.
4. Evaluates candidates using a spec-first two-stage logic, computes a recommendation confidence score, and outputs the suggestion to the operator.

---

## Phase 5.1A — Surrogate Process Models (T1, T2, T3)

### Overview

Three XGBoost/CatBoost/LightGBM regressors trained on the historical dataset to predict the value of Bottom Temp, Tray Temp, and Column Top Pressure at horizon `t+1`. We start with `t+1` models to validate prediction accuracy before extending to multi-step models (`t+2`, `t+3`) in Phase 5.1B if metrics permit.

---

### 5.1.1 — `notebooks/build_surrogate_dataset.py` [NEW]

**Purpose**: Load `data/features.parquet`, create target columns for horizon `t+1` only, and save `data/surrogate_data.parquet`.

**Logic**:
Targets are created per-block using negative shifts to avoid leakage across gaps:
```python
for block_val, group in df.groupby("Data_Block"):
    group["bottom_temp_future_t1"] = group["Column_Bottom_Temp"].shift(-1)
    group["tray_temp_future_t1"]   = group["Control_Tray_Temp"].shift(-1)
    group["pressure_future_t1"]    = group["Column_Top_Pressure"].shift(-1)
```
The last row of each block gets `NaN` and is dropped before training.

**Feature set used for all surrogate targets**:
Same ~19 mass/energy balance ratios, deviations, and 1-hour lag variables from `data/features.parquet`.

**Output**: `data/surrogate_data.parquet`

---

### 5.1.2 — `notebooks/train_surrogate_models.py` [NEW]

**Purpose**: Train XGBoost, CatBoost, and LightGBM models for each of the 3 targets. For each target, select the algorithm with the highest Block 4 test $R^2$, and save the winner to `models/surrogates/`.

**Naive Baseline Comparison**:
For each target, compute and print metrics for a naive baseline where $\hat{y}_{t+1} = y_t$ (current value):
- $R^2_{\text{naive}}$
- $\text{MAE}_{\text{naive}}$

We will compare surrogate model performance against this baseline to ensure real predictive value.

**Training & Generalization Strategy**:
- **Train Set**: Blocks 1, 2, 3 (stuck/shutdown rows removed)
- **Test Set**: Block 4 (held out)
- **Model Constraints**: Shallow trees (`max_depth=4`) to prevent overfitting.
- **Optuna Tuning**: Triggered automatically (30 trials) for any target failing the baseline $R^2$ threshold.

**Acceptance Thresholds (Block 4 Test $R^2$)**:

| Target | Horizon `t+1` |
|---|---|
| Bottom Temp (T1) | > 0.80 |
| Tray Temp (T2) | > 0.75 |
| Pressure (T3) | > 0.70 |

**Feature Importance Export**:
Save the feature importances for the winning models as:
- `models/surrogates/surrogate_feature_importance_bottom.csv`
- `models/surrogates/surrogate_feature_importance_tray.csv`
- `models/surrogates/surrogate_feature_importance_pressure.csv`

**Output**:
- `models/surrogates/{target}_t1_model.pkl` (3 serialized model files)
- `models/surrogates/surrogate_results.json` (comprehensive performance metrics and naive baseline comparisons)

---

### 5.1.3 — `notebooks/surrogate_diagnostics.py` [NEW]

**Purpose**: Generate diagnostics and plots for all 3 models to verify generalization.

**Visual Outputs** (saved to `experiments/diagnostics/surrogates/`):
- Actual vs. Predicted scatter plots
- Residual histograms & residual vs. time plots

---

## Build Order

1. **Step 1**: Create and run `build_surrogate_dataset.py` (t+1 targets only).
2. **Step 2**: Create and run `train_surrogate_models.py` (XGB/LGB/CB, naive baselines, feature importances CSV).
3. **Step 3**: Create and run `surrogate_diagnostics.py` (Actual vs Pred, residuals plots).
4. **Step 4**: Report metrics and save final `models/surrogates/surrogate_results.json`.
