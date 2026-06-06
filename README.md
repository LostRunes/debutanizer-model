# Debutanizer C4 Slippage Drift Mitigation & Soft Sensor Model

This repository contains the machine learning soft-sensor pipeline and adaptive modeling framework designed to predict and minimize C4 slippage in the bottom stream of an industrial Debutanizer column.

---

## 📌 Project Overview
* **Objective**: Minimize C4 slippage (C4H8 & C4H6) in the C5+ bottom product stream (reducing variation from 0.8–1.5% down to a spec target of <0.5 mol%).
* **Problem**: 
  * The physical analyzer has a 12-minute cycle time, introducing lag and wide variation.
  * Analyzer readings are historically prone to reliability issues.
  * Significant feed and operating variability (pressure setpoint changes and seasonal column temperature shifts) cause standard ML models to suffer from **severe concept drift** and prediction sign inversion.

---

## 🏗️ Solution Architecture
1. **Data Collection**: Retrieves column process parameters from Exaquantum and periodic laboratory/analyzer feedback data.
2. **AI Soft Sensor**: 
   * **Model A (`C4H8_Bottom`)**: Gradient Boosting Regression (XGBoost/LightGBM/CatBoost) using mass/energy balance ratios, short-term rolling deviations, and a leakage-free calibration anchor.
   * **Model B (`C4H6_Bottom`)**: Evaluated baselines and delta correction models. Concluded that a pure Campaign-Anchor-based tracking mechanism ($\text{R}^2 = 0.96$) outperformed hybrid ML correction.
3. **Real-time Operator Recommendation Dashboard**: Live predictions, deviation trends, and recommendation instructions calculated with loss estimates (in INR/hr).

---

## 📊 Process Details & Input Variables
The debutanizer column separates mixed C4s (top product) from C5s and heavier components (bottom product). 
* **Feed Location**: Feed enters the column on level control at the 17th tray.
* **Reboiling Duty**: Provided via Low Pressure (desuperheater) steam.
* **Condensation**: Column vapors are condensed via cooling water and collected in the reflux drum before being split into reflux flow and distillate.
* **Process Inputs**:
  * Feed Flow, Reflux Flow, and Reboiling Steam Flow.
  * Column Top, Bottom, and Control Tray Temperatures.
  * Column Top Pressure.
  * Bottom Analyzer baseline value (Calibration Anchor).

---

## 🛠️ Feature Engineering & Drift Mitigation
Standard models using absolute temperatures overfit to specific pressure regimes. When column pressure changed from **4.19 bar** to **3.98 bar** between campaigns, the boiling point of the mixture changed, reversing the temperature-to-concentration relationships.

To mitigate this, the pipeline applies:
* **Dimensionless Ratios**: `Reflux_Ratio` and `Steam_Feed_Ratio` to represent fundamental mass and energy balances.
* **Rolling Deviations (`_dev24h`)**: Subtracts the 24-hour rolling average from flow, temperature, and pressure measurements. This extracts high-frequency transient signals and strips out long-term campaign/fouling drift.
* **Leak-Free Campaign Anchor**: Integrates the last known analyzer measurement using a 12-hour forward-fill limit (`shift(1).ffill(limit=12)`), preventing lookahead target leakage while providing a dynamic calibration baseline.
* **Gap-Aware Time-Series Resampling**: Features are computed on a continuous hourly grid and reindexed back, preventing values from leaking across campaign gaps.

---

## 📈 Model Performance Summary (Block 4 Test Set)

### Model A (`C4H8_Bottom`) — Robust 8-Feature Configuration
By training on Blocks 1-3 and testing on the held-out Block 4 dataset, the robust, anti-drift configuration achieved:

| Model | CV $R^2$ (Train Blocks 1-3) | Test $R^2$ (Block 4) | Test MAE (wt%) | Top Feature |
| :--- | :---: | :---: | :---: | :--- |
| **Tuned LightGBM** | **0.7087** | **0.9147** | **0.0494 wt%** | `C4H8_campaign_anchor` |
| **Tuned XGBoost** | **0.7037** | **0.9074** | **0.0516 wt%** | `C4H8_campaign_anchor` |
| **Tuned CatBoost** | **0.7181** | **0.9030** | **0.0524 wt%** | `C4H8_campaign_anchor` |

### Model B (`C4H6_Bottom`) — Target Audit & Baselines
* **Findings**: Block 1 represented a cold reboiler regime (mean C4H6: 0.208 wt%), whereas Block 4 represented a low-concentration campaign (mean C4H6: 0.0057 wt%). 
* **Evaluation**: Applying machine learning on top of the anchor degraded prediction accuracy due to the extremely small delta variance.
* **Optimal Strategy**: The 12-hour raw persistence anchor alone provides a highly accurate state estimator.
  * **Anchor-only (12h)**: $R^2 = \mathbf{0.9606}$, $\text{MAE} = \mathbf{0.0005}$ wt%.
  * **Anchor + XGBoost**: $R^2 = 0.9010$, $\text{MAE} = 0.0011$ wt%.

---

## 📂 Repository Structure
```
├── DEBUTANIZER-model/
│   ├── data/                           # Data storage directory
│   ├── configs/                        # JSON configurations for Model A and Model B features
│   ├── models/                         # Serialized models and optimization parameters
│   ├── inference/                      # Scripts for real-time inference and fallback logic
│   │   ├── predict_c4h8.py             # Model A inference runner
│   │   ├── predict_c4h6.py             # Model B anchor tracker
│   │   └── predict_total_c4.py         # Unified online prediction script (Model A + B)
│   ├── notebooks/                      # Verification and audit notebooks
│   │   ├── tune_robust_xgb.py          # Optuna hyperparameter optimization script
│   │   ├── verify_anchor_leakage.py    # Baseline target leakage verification
│   │   ├── run_robust_checks.py        # Model A diagnostic checks and SHAP generation
│   │   ├── model_b_target_audit.py     # Model B campaign and target audit
│   │   └── model_b_inversion_check.py  # Model B robust validation checks
│   ├── data_preprocessing.py           # Preprocessing script (winsorizing, stuck periods)
│   ├── feature_engineering.py          # Time-series resampling and deviation feature generation
│   ├── requirements.txt                # Project dependencies
│   └── README.md                       # Project documentation
```

---

## 🚀 Execution & Replication Pipeline

Follow the execution sequence below to run the pipeline:

```bash
# Step 1: Preprocess raw Excel dataset
python data_preprocessing.py

# Step 2: Generate dynamic features & deviations
python feature_engineering.py

# Step 3: Run Optuna hyperparameter tuning for Model A
python notebooks/tune_robust_xgb.py

# Step 4: Verify leakage checks and Block 3 metrics
python notebooks/verify_anchor_leakage.py

# Step 5: Freeze Model A production release assets
python notebooks/freeze_model_A.py

# Step 6: Audit Model B target distributions
python notebooks/model_b_target_audit.py

# Step 7: Analyze Model B anchor coverage and age metrics
python notebooks/model_b_anchor_audit.py

# Step 8: Run Model B baseline and anchor evaluations
python notebooks/anchor_only_baselines.py

# Step 9: Verify delta corrections for Model B
python notebooks/model_b_delta_model.py

# Step 10: Run campaign inversion checks for Model B anchor
python notebooks/model_b_inversion_check.py

# Step 11: Execute combined DCS online inference
python inference/predict_total_c4.py
```

---

## 🔮 Future Scope
* **Real-time Integration**: Deployment to production historians/DCS platforms such as Seeq.
* **Closed-loop Control**: Integration with Advanced Process Control (APC) for automated reflux/steam manipulation.
* **Scale-out**: Extending similar soft-sensor architectures to neighboring distillation systems.
