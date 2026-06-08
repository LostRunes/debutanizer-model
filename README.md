# IOCL Debutanizer Column C4 Slippage AI Soft-Sensor & Advisory Optimizer

This repository contains the production-ready machine learning soft-sensor pipeline and process-aware adaptive advisory optimizer designed to predict, monitor, and minimize C4 slippage in the bottom stream of an industrial Debutanizer column at the IOCL Refinery.

---

## 📌 Project Overview & Objectives

*   **Objective**: Minimize C4 slippage (C4H8 & C4H6) in the C5+ bottom product stream, reducing variation from 0.8–1.5 wt% down to the product spec target of **<0.50 wt%**.
*   **The Challenges**:
    *   **Analyzer Lag**: The online gas chromatograph (GC) has a 12-minute cycle time, introducing a blind window for rapid process upsets.
    *   **Analyzer Reliability**: Analyzer readings are prone to extended "stuck" flat-line periods.
    *   **Severe Concept Drift**: Operational changes (e.g., column pressure setpoint drops, feed quality shifts, reboiler heat deviations) between campaigns lead to complete correlation sign reversals, causing standard ML models to suffer from severe prediction inversion.
*   **The Solution**: A hierarchical framework integrating:
    1.  **Model A (`C4H8_Bottom`)**: A robust, drift-resistant 8-feature XGBoost regressor utilizing mass/energy ratios, 24-hour dynamic deviations, and a leak-free calibration anchor.
    2.  **Model B (`C4H6_Bottom`)**: A campaign-anchor persistence tracker ($R^2 = 0.96$). (Systematic evaluations proved adding ML delta corrections degrades performance due to high autocorrelation and low concentration levels).
    3.  **Process-Aware Advisory Optimizer**: A physics-constrained local grid search optimizer that utilizes lightweight CatBoost surrogate models to predict column thermal/pressure response before estimating C4 slippage reduction, adhering strictly to safety envelopes.
    4.  **Operator Dashboard**: A NiceGUI-based web application providing real-time predictions, column health indicators, safety confidence rankings, and actionable advisory controls.

---

## 🏗️ System Architecture

The advisory optimizer operates in a multi-stage process-aware loop, chaining surrogate process models with the frozen soft-sensors to generate physically realistic recommendations:

```
       Candidate Setpoint Move (Steam Flow Δ, Reflux Flow Δ)
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│ 1. Surrogate Delta Process Models (T1, T2, T3)             │
│    - Predicts t+1 deviations/deltas for Bottom Temp,       │
│      Tray Temp, and Column Top Pressure                    │
│    - Bottom Temp uses dev24h targets to block drift        │
└────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│ 2. Uncertainty Safety Buffer Constraint Checks             │
│    - Evaluates predicted conditions + model MAE            │
│      against hard limits (Bottom Temp < 115°C, Press < 5bar)│
└────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│ 3. Soft-Sensor C4 Slippage Estimation                      │
│    - Constructs Model A features using predicted T, P      │
│    - Model A predicts C4H8 wt%; Model B anchors C4H6 wt%   │
│    - Calculates predicted Total C4 slippage                │
└────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────┐
│ 4. Two-Stage Spec-First Optimization Selection             │
│    - Stage 1: Filter candidates meeting product spec       │
│    - Stage 2: Minimize cost (Economic) or C4 (Spec)        │
└────────────────────────────────────────────────────────────┘
                                │
                                ▼
                   Operator Recommendation Output
```

---

## 📊 Debutanizer Column Process & Variables

The column separates mixed C4s (distillate) from C5+ components (bottom product). 

*   **Feed Inlet**: Enters at the 17th tray under level control (~60–70 TPH).
*   **Reboiler Heat**: Provided by Low-Pressure (LP) desuperheated steam.
*   **Reflux System**: Vapors are condensed via cooling water and split into reflux flow (~80–104 TPH) and distillate.
*   **Input Variables**:
    *   **Manipulated Variables (MVs)**: Reboiling Steam Flow (TPH), Reflux Flow (TPH).
    *   **Disturbance Variables (DVs)**: Feed Flow (TPH).
    *   **State Indicators (SVs)**: Column Bottom Temperature (°C), Control Tray Temperature (°C), Column Top Pressure (bar).
    *   **Target Variables**: C4H8 Bottom (wt%), C4H6 Bottom (wt%), and Total C4 (wt%).

---

## 🕒 Operational Campaigns (Block Structure)

The dataset contains 11,343 hours of plant operations divided into 4 distinct operating campaigns:
*   **Block 1**: Cold reboiler startup regime (mean C4H6 ~0.208 wt%). Fundamentally different thermodynamics.
*   **Block 2 & 3**: Hot fractionation regime. Normal operation used for training.
*   **Block 4**: Recent campaign with lower operating pressure (3.98 bar vs 4.19 bar in training). Held-out test set for drift verification.

---

## 🛠️ Feature Engineering & Drift Mitigation

Standard models suffer from **concept drift** because absolute temperatures correlate differently under different pressure regimes. For instance, when column pressure dropped from **4.19 bar to 3.98 bar** in Block 4, the bubble point of the mixture dropped by 3–5°C, reversing the temperature-to-concentration relationships and causing standard ML models to fail.

To achieve robust generalization, the pipeline applies:
1.  **Dimensionless Ratios**: Normalizes reflux and steam flows by feed (`Reflux_Ratio`, `Steam_Feed_Ratio`) to represent fundamental mass and energy balances.
2.  **Rolling Deviations (`_dev24h`)**: Subtracts the 24-hour rolling average from temperatures, pressures, and flows (e.g., `Column_Bottom_Temp_dev24h = Bottom_Temp - RollMean24h(Bottom_Temp)`). This isolates high-frequency transient signals and strips out long-term campaign shifts.
3.  **Leak-Free Campaign Anchor**: Integrates the last known valid analyzer measurement using a 1-hour delay (`shift(1)`) and a forward-fill limit.
4.  **Gap-Aware Resampling**: Resamples each block to a continuous hourly grid before computing lags/rolling averages, preventing values from leaking across campaign shutdown gaps.

---

## 🔬 Model Development & Performance

### 1. Model A (`C4H8_Bottom`) Soft-Sensor
Model A is trained on Blocks 1-3 using the **Robust 8-Feature Configuration** (Anchor + Ratios + Deviations) and tested on the held-out Block 4.

*   **Algorithm**: Tuned XGBoost (`max_depth = 3` to prevent memorizing campaign/temporal proxies).
*   **Optuna Tuning**: 5-fold TimeSeriesSplit CV.
*   **Validation Results**:
    *   **Block 3 Test Set**: $R^2 = \mathbf{0.7694}$, $\text{MAE} = \mathbf{0.0817}$ wt%.
    *   **Block 4 Test Set (Held-out)**: $R^2 = \mathbf{0.9074}$, $\text{MAE} = \mathbf{0.0516}$ wt%.
*   **DCS Fallback Logic**:
    *   *Level 1 (Green)*: Anchor available within a 72-hour limit $\rightarrow$ XGBoost prediction.
    *   *Level 2 (Yellow)*: Anchor stale (72h to 168h) $\rightarrow$ Uses a 24-hour rolling mean of previous model predictions.
    *   *Level 3 (Red)*: Hard timeout (> 168h or startup) $\rightarrow$ Default to the historical Block 4 target mean (**0.480 wt%**).

### 2. Model B (`C4H6_Bottom`) Soft-Sensor
*   **Evaluation**: Because C4H6 (butadiene) standard deviation is extremely small in Block 4 (mean ~0.0057 wt%), training a delta ML correction model on top of the anchor degraded performance due to fitting noise ($R^2$ dropped from $0.96$ to $0.90$).
*   **Decision**: Model B is implemented as a pure **12-hour persistence anchor** ($R^2 = \mathbf{0.9606}$, $\text{MAE} = \mathbf{0.0005}$ wt%).

---

## 🔮 Process-Aware Advisory Optimizer (Phase 5)

Chains three CatBoost surrogate models to predict column response 1-hour ahead:
*   **T1 (Bottom Temp)**: $R^2 = 0.764$, $\text{MAE} = \pm0.68^\circ\text{C}$ (Uses dev24h target to block drift).
*   **T2 (Control Tray Temp)**: $R^2 = 0.908$, $\text{MAE} = \pm1.99^\circ\text{C}$.
*   **T3 (Top Pressure)**: $R^2 = 0.949$, $\text{MAE} = \pm0.014\text{ bar}$.

### Safety and Optimization Constraints
*   **Safety Limits**: Reject candidates exceeding `115.0°C` bottom temperature (with a $\pm0.69^\circ\text{C}$ buffer) or `5.0 bar` top pressure (with a $\pm0.014\text{ bar}$ buffer).
*   **Search Space**: Local grid search around current values: Steam Flow ($\pm2.0\text{ TPH}$ at $0.2\text{ TPH}$ steps), Reflux Flow ($\pm10.0\text{ TPH}$ at $1.0\text{ TPH}$ steps).
*   **Two Optimization Modes** (`configs/economics.json`):
    *   **SPEC Mode**: Minimizes Total C4 slippage.
    *   **ECONOMIC Mode**: Minimizes utility cost (steam consumption + reflux pumping power) while keeping C4 slippage below the $0.50$ wt% limit.

### Validation Performance (100 Block 4 Out-of-Spec Snapshots)
*   **Recommendation Feasibility Rate**: **86.0%** (14.0% rejected because the column was already at safety limits).
*   **Safety Limit Violations**: **0.0%** (proven safety buffer constraint enforcement).
*   **Avg. C4 Reduction**: **17.4%** in SPEC mode, **16.7%** in ECONOMIC mode.
*   **Avg. Utility Cost Change**: Save **$-₹2.67/\text{hr}$** under Economic mode compared to a slight increase in Spec mode.

---

## 💻 NiceGUI Operator Dashboard (Phase 6)

The web interface (`http://localhost:8080`) provides operators with a comprehensive control-room style display:
*   **KPI Panel**: Color-coded tiles indicating Total C4, C4H8, C4H6, Steam, Reflux, and Column Health.
*   **Analyzer Staleness Tracker**: Displays time elapsed since the last valid GC reading, highlighting active fallback states.
*   **Timeline Scrubber**: Allows operators to load historical states from Block 4 to inspect model and optimizer outputs.
*   **Live Optimizer Page**: Visualizes current vs recommended setpoints, predicted temperature responses, safety confidence levels (High/Medium/Low based on thermal headroom), and utility benefits.
*   **Settings Editor**: Interactive configuration form to update cost coefficients and safety limits in real-time.

---

## 📂 Repository Structure

```
├── DEBUTANIZER-model/
│   ├── configs/                        # System configuration files
│   │   ├── economics.json              # Optimization modes & price coefficients
│   │   ├── model_A_features.json       # Feature configuration for Model A
│   │   └── model_B_features.json       # Feature configuration for Model B
│   ├── data/                           # Process datasets
│   │   ├── clean_data.parquet          # Cleaned historian data
│   │   ├── features.parquet            # Engineered model features
│   │   └── surrogate_data.parquet      # Surrogate training data
│   ├── debutanizer_dashboard/          # NiceGUI Dashboard application code
│   │   ├── app.py                      # Main entry point for NiceGUI app
│   │   ├── pages/                      # Dashboard page views (overview, optimizer, trends, etc.)
│   │   ├── components/                 # Reusable UI cards, tables, and Plotly charts
│   │   └── services/                   # Backend services (state, predictions, optimization)
│   ├── docs/                           # Project technical documentation
│   │   ├── 01_problem_statement.md     # Problem statement & context
│   │   ├── 07_final_architecture.md    # Final pipeline architecture details
│   │   └── optimizer_summary.md        # Comprehensive summary of Phase 5
│   ├── inference/                      # Real-time inference scripts
│   │   ├── predict_c4h8.py             # Model A inference runner
│   │   ├── predict_c4h6.py             # Model B anchor tracker
│   │   └── predict_total_c4.py         # Unified online prediction script
│   ├── models/                         # Serialized model assets
│   │   ├── final/                      # Frozen Model A assets (.pkl, .json)
│   │   └── surrogates/                 # Trained surrogate models & importance files
│   ├── notebooks/                      # Development & optimization scripts
│   │   ├── train_surrogate_models.py   # Surrogate model training script
│   │   ├── optimizer_v2_physics_aware.py# Main advisory optimizer logic
│   │   └── optimizer_validation.py     # Optimizer batch validator
│   ├── reports/                        # Final performance reports
│   │   ├── model_A_final_summary.md    # Model A validation report
│   │   └── optimizer_final_summary.md  # Phase 5 executive summary
│   ├── data_preprocessing.py           # Raw Excel dataset preprocessing pipeline
│   ├── feature_engineering.py          # Feature creation & gap-aware resampling
│   ├── generate_full_documentation.py  # Script to compile the master DOCX report
│   └── README.md                       # Project master documentation
```

---

## 🚀 Execution & Replication Pipeline

Execute the pipeline using the following steps:

```bash
# 1. Preprocess raw data from Excel file
python data_preprocessing.py

# 2. Perform feature engineering & generate dynamic deviations
python feature_engineering.py

# 3. Train process surrogate models
python notebooks/train_surrogate_models.py

# 4. Run surrogate model diagnostics
python notebooks/surrogate_diagnostics.py

# 5. Run batch validation on the advisory optimizer
python notebooks/optimizer_validation.py

# 6. Execute combined online DCS inference
python inference/predict_total_c4.py

# 7. Start the NiceGUI Operator Dashboard
python debutanizer_dashboard/app.py
```

---

## 🔮 Future Scope

1.  **Multi-Step Safety Horizon**: Extend surrogates to predict 2–4 hours ahead to identify thermal peaks.
2.  **DCS closed-loop control**: Integrate recommendations directly into Advanced Process Control (APC) systems.
3.  **Seeq Integration**: Deploy model endpoints directly into the plant's Seeq server to overlay predictions onto DCS consoles.
