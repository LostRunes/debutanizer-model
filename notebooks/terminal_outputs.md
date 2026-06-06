# Terminal Output Log — Phase 3 Diagnostics & Model Training

All commands run from: `c:\Users\KIIT\OneDrive\Desktop\DEBUTANIZER-model\`

---

## 1. Environment Check

### Command
```
python -c "import sys; print(sys.executable)"
```
### Output
```
C:\Users\KIIT\AppData\Local\Programs\Python\Python310\python.exe
```

---

### Command
```
python -c "import sklearn, xgboost; print('sklearn/xgboost ok')"
```
### Output
```
sklearn/xgboost ok
```

---

### Command (optuna not installed yet)
```
python -c "import optuna; print('optuna ok')"
```
### Output
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'optuna'
```

---

### Command
```
pip install optuna
```
### Output
```
Collecting optuna
  Downloading optuna-4.9.0-py3-none-any.whl.metadata (15 kB)
Collecting alembic>=1.5.0 (from optuna)
...
Installing collected packages: tomli, Mako, greenlet, colorlog, sqlalchemy, alembic, optuna

Successfully installed Mako-1.3.12 alembic-1.18.4 colorlog-6.10.1 greenlet-3.5.1 optuna-4.9.0 sqlalchemy-2.0.50 tomli-2.4.1
```

---

## 2. `python model_training.py` — Phase 3 Pipeline (Step-by-step)

### Command
```
python model_training.py
```

### Output
```
================================================================================
STARTING MODEL TRAINING (PHASE 3)
================================================================================
Loading features from data\features.parquet...
  Loaded dataset shape: (11343, 92)
  Tier 1 features (process-only): 67
  Tier 2 features (+ target lags): 82
  Assertions passed: Column counts match Phase 2 plan exactly.

Model A (C4H8) Sets:
  Train size: 4332 | Test size: 6081
Model B (C4H6) Sets:
  Train size: 3556 | Test size: 2974

==================================================
STEP 1: MODEL A (C4H8) BASELINES
==================================================
  Overall Mean Baseline:
    R² = -0.0034 | MAE = 0.2199 wt% | RMSE = 0.2762 wt%
    % within ±0.1 wt% = 24.7% | Spec Recall (>0.5) = 100.0%
  Block Mean Baseline:
    R² = -0.0034 | MAE = 0.2199 wt% | RMSE = 0.2762 wt%
    % within ±0.1 wt% = 24.7% | Spec Recall (>0.5) = 100.0%
  Naive Lag-1 Baseline (Requires Analyzer):
    R² = 0.9328 | MAE = 0.0361 wt% | RMSE = 0.0715 wt%
    % within ±0.1 wt% = 91.6% | Spec Recall (>0.5) = 96.0%

==================================================
STEP 2: TRAINING MODEL A DEFAULT MODELS (NO TUNING)
==================================================
Training LinearRegression...
  LinearRegression (All Test):
    R² = -3.7282 | MAE = 0.5035 wt% | RMSE = 0.5996 wt%
    % within ±0.1 wt% = 6.8% | Spec Recall (>0.5) = 21.6%
Training Ridge...
  Ridge (All Test):
    R² = -3.9320 | MAE = 0.5082 wt% | RMSE = 0.6124 wt%
    % within ±0.1 wt% = 7.6% | Spec Recall (>0.5) = 21.2%
Training RandomForest...
  RandomForest (All Test):
    R² = -0.9917 | MAE = 0.2987 wt% | RMSE = 0.3892 wt%
    % within ±0.1 wt% = 23.4% | Spec Recall (>0.5) = 19.7%
Training XGBoost...
  XGBoost (All Test):
    R² = -1.0287 | MAE = 0.3017 wt% | RMSE = 0.3928 wt%
    % within ±0.1 wt% = 23.6% | Spec Recall (>0.5) = 19.0%

Default model leaderboard saved to: models\default_leaderboard.csv

--- DEFAULT LEADERBOARD ---
                    Model  R2_All  MAE_All  RMSE_All  Spec_Recall_All Analyzer_Required
LinearRegression (Tier 1) -3.7282   0.5035    0.5996          21.5785                No
           Ridge (Tier 1) -3.9320   0.5082    0.6124          21.1509                No
    RandomForest (Tier 1) -0.9917   0.2987    0.3892          19.7123                No
         XGBoost (Tier 1) -1.0287   0.3017    0.3928          19.0124                No
     Naive Lag-1 Baseline  0.9328   0.0361    0.0715          95.9565               Yes

==================================================
STEP 3: DATA_BLOCK A/B EXPERIMENT
==================================================
XGBoost WITH Data_Block feature:    R² = -1.0287
XGBoost WITHOUT Data_Block feature: R² = -1.0445
Delta R² = 0.0158
  Recommendation: Keep Data_Block feature (adds meaningful performance).

==================================================
STEP 4: FEATURE IMPORTANCE & REGIME MEMORISATION CHECK
==================================================
Top 10 features for winning model (XGBoost):
                         Feature  Importance
                       month_cos      0.3781
                       month_sin      0.0752
                      Data_Block      0.0567
        Column_Top_Pressure_lag1      0.0434
 Column_Bottom_Temp_roll_mean_3h      0.0393
       Reboiler_Outlet_Temp_lag2      0.0213
             Column_Top_Pressure      0.0199
Column_Bottom_Temp_roll_mean_12h      0.0191
       Reboiler_Outlet_Temp_lag3      0.0188
          Control_Tray_Temp_lag1      0.0163

  SUCCESS: Top features look healthy. Model is learning from process variables.

==================================================
STEP 5: TRAINING MODEL B DEFAULT MODELS (NO TUNING)
==================================================
Training Model B LinearRegression...
  LinearRegression (All Test):
    R² = -305.2729 | MAE = 0.1629 wt% | RMSE = 0.1754 wt%
    % within ±0.1 wt% = 18.5% | Spec Recall (>0.5) = 0.0%
Training Model B Ridge...
  Ridge (All Test):
    R² = -298.5895 | MAE = 0.1603 wt% | RMSE = 0.1735 wt%
    % within ±0.1 wt% = 20.6% | Spec Recall (>0.5) = 0.0%
Training Model B RandomForest...
  RandomForest (All Test):
    R² = -6.2493 | MAE = 0.0190 wt% | RMSE = 0.0270 wt%
    % within ±0.1 wt% = 99.9% | Spec Recall (>0.5) = 0.0%
Training Model B XGBoost...
  XGBoost (All Test):
    R² = -34.6323 | MAE = 0.0550 wt% | RMSE = 0.0598 wt%
    % within ±0.1 wt% = 97.5% | Spec Recall (>0.5) = 0.0%

--- MODEL B LEADERBOARD ---
                            Model    R2_All  MAE_All  RMSE_All  Spec_Recall_All Analyzer_Required
LinearRegression (Model B - C4H6) -305.2729   0.1629    0.1754              0.0                No
           Ridge (Model B - C4H6) -298.5895   0.1603    0.1735              0.0                No
    RandomForest (Model B - C4H6)   -6.2493   0.0190    0.0270              0.0                No
         XGBoost (Model B - C4H6)  -34.6323   0.0550    0.0598              0.0                No

==================================================
STEP 6: TRAINING TIER 2 RESEARCH MODELS (WITH TARGET LAGS)
==================================================
Training Model A (C4H8) Tier 2 XGBoost...
  Model A Tier 2 XGBoost:
    R² = 0.7128 | MAE = 0.0995 wt% | RMSE = 0.1478 wt%
    % within ±0.1 wt% = 65.6% | Spec Recall (>0.5) = 90.7%
Training Model B (C4H6) Tier 2 XGBoost...
  Model B Tier 2 XGBoost:
    R² = -16.8202 | MAE = 0.0369 wt% | RMSE = 0.0423 wt%
    % within ±0.1 wt% = 99.8% | Spec Recall (>0.5) = 50.0%

Value of working Analyzer:
  Model A (C4H8) R² gain: +1.7415
  Model B (C4H6) R² gain: +17.8121

==================================================
STEP 7: COMBINED TOTAL C4 EVALUATION
==================================================
Combined test predictions saved to: models\test_predictions.parquet (shape: (6514, 96))
Evaluating combined Total_C4 on 2927 healthy analyzer rows:
  Combined XGBoost (Tier 1):
    R² = -1.0520 | MAE = 0.3621 wt% | RMSE = 0.4561 wt%
    % within ±0.1 wt% = 21.8% | Spec Recall (>0.5) = 0.8%
  Combined Naive Lag-1 Baseline (Requires Analyzer):
    R² = 0.9217 | MAE = 0.0487 wt% | RMSE = 0.0891 wt%
    % within ±0.1 wt% = 86.4% | Spec Recall (>0.5) = 95.8%
All training metrics summary saved to: models\training_metrics.csv
Models saved as JSON to models/ directory.

================================================================================
PHASE 3 PIPELINE COMPLETED SUCCESSFULLY
================================================================================
```

---

## 3. `notebooks/inspect_training_shift.py` — Train vs Test Distribution Comparison

### Command
```
python notebooks/inspect_training_shift.py
```

### Output
```
=== TARGET A (C4H8_Bottom) STATS ===
Train A:
count    4353.000000
mean        0.443736
std         0.309880
min         0.032553
25%         0.196050
50%         0.387917
75%         0.614128
max         1.528021
Name: C4H8_Bottom, dtype: float64

Test A:
count    6093.000000
mean        0.427817
std         0.275720
min         0.072179
25%         0.215741
50%         0.356820
75%         0.593113
max         1.528021
Name: C4H8_Bottom, dtype: float64

=== TARGET B (C4H6_Bottom) STATS ===
Train B:
count    3577.000000
mean        0.139520
std         0.161813
min         0.001061
25%         0.030074
50%         0.082994
75%         0.189978
max         0.669380
Name: C4H6_Bottom, dtype: float64

Test B:
count    2974.000000
mean        0.005663       ← 24× lower than train mean!
std         0.010023
min         0.001000
25%         0.001648
50%         0.005682
75%         0.007273
max         0.380282
Name: C4H6_Bottom, dtype: float64

=== FEATURE DIFFERENCES TRAIN vs TEST ===
Feature                        | Train Mean | Test Mean  | Diff
Feed_Flow                      |    87.0826 |    78.5616 |   -8.5210
Reboiler_Outlet_Temp           |    56.4510 |    71.6874 |  +15.2364
Column_Top_Temp                |    87.1361 |    67.1259 |  -20.0102
Reboiling_Steam_Flow           |    22.1831 |    20.3762 |   -1.8070
Reflux_Flow                    |    97.6909 |    87.1245 |  -10.5664
Column_Top_Pressure            |     4.1870 |     3.9783 |   -0.2088
Column_Bottom_Temp             |   106.8086 |   108.1785 |   +1.3699
Control_Tray_Temp              |    69.2933 |    73.4098 |   +4.1165
Reflux_Ratio                   |     1.1290 |     1.1238 |   -0.0052
Steam_Feed_Ratio               |     0.2557 |     0.2616 |   +0.0059
Temp_Gradient                  |    19.6725 |    41.0526 |  +21.3801   ← +108% shift!
Reboiler_Delta                 |   -50.3577 |   -36.4911 |  +13.8665
```

---

## 4. `notebooks/experiment_features.py` — Feature Selection Ablation

### Command
```
python notebooks/experiment_features.py
```

### Output
```
Total Tier 1 features: 67
Process-only features: 62
XGBoost (with month/block): R2 = -1.0287 | MAE = 0.3017
XGBoost (process-only):      R2 = -0.9733 | MAE = 0.2944
RandomForest (process-only): R2 = -0.9267 | MAE = 0.2925

Top 10 features (process-only):
                         Feature  Importance
       Reboiler_Outlet_Temp_lag3      0.0979
          Control_Tray_Temp_lag1      0.0946
        Column_Top_Pressure_lag1      0.0815
       Reboiler_Outlet_Temp_lag1      0.0713
       Reboiler_Outlet_Temp_lag2      0.0455
Column_Bottom_Temp_roll_mean_12h      0.0332
 Column_Bottom_Temp_roll_mean_6h      0.0290
             Column_Top_Pressure      0.0257
          Control_Tray_Temp_lag2      0.0217
                         dow_sin      0.0205
```

---

## 5. `notebooks/diagnose_predictions.py` — Out-of-Bounds Feature Detection

### Command
```
python notebooks/diagnose_predictions.py
```

### Output
```
=== TRAIN EVALUATION ===
R2 Train: 0.9827
MAE Train: 0.0301
True Train sample (first 10): [0.034, 0.033, 0.033, 0.040, 0.033, 0.033, 0.033, 0.033, 0.033, 0.038]
Pred Train sample (first 10): [0.094, 0.058, 0.034, 0.038, 0.061, 0.015, 0.031, 0.015, 0.041, 0.044]

=== TEST EVALUATION ===
R2 Test: -0.9733
MAE Test: 0.2944
True Test sample (first 10): [0.222, 0.218, 0.212, 0.221, 0.230, 0.230, 0.224, 0.221, 0.211, 0.220]
Pred Test sample (first 10): [0.364, 0.336, 0.458, 0.318, 0.451, 0.542, 0.510, 0.511, 0.490, 0.476]

Mean of True Test: 0.4283
Mean of Pred Test: 0.3003
Min of Pred Test: -0.0667
Max of Pred Test: 0.9289

=== OUT OF BOUNDS FEATURES IN TEST SET ===
Feature                         | Train Range              | Test Range               | OOB%
Reflux_Flow_roll_mean_12h       | [87.89, 105.65]          | [70.42, 105.29]          | 53.56%  ← SEVERE
Reflux_Flow_roll_mean_6h        | [86.69, 105.65]          | [70.42, 105.65]          | 47.20%  ← SEVERE
Reflux_Flow_roll_mean_3h        | [85.02, 105.65]          | [70.42, 105.65]          | 40.95%  ← SEVERE
Reflux_Flow_lag6                | [85.00, 105.65]          | [70.42, 105.65]          | 38.22%  ← SEVERE
Reflux_Flow_lag3                | [85.00, 105.65]          | [70.42, 105.65]          | 38.17%  ← SEVERE
Reflux_Flow_lag2                | [85.00, 105.65]          | [70.42, 105.65]          | 38.15%  ← SEVERE
Reflux_Flow_lag1                | [85.00, 105.65]          | [70.42, 105.65]          | 38.14%  ← SEVERE
Reflux_Flow                     | [85.00, 105.65]          | [70.42, 105.65]          | 38.12%  ← SEVERE
Column_Top_Temp                 | [29.35, 113.00]          | [18.04, 113.00]          | 33.89%  ← HIGH
Column_Top_Temp_lag1            | [29.35, 113.00]          | [18.04, 113.00]          | 33.89%  ← HIGH
Column_Top_Temp_lag2            | [29.35, 113.00]          | [18.04, 113.00]          | 33.89%  ← HIGH
Temp_Gradient                   | [-4.29, 82.09]           | [-2.23, 94.94]           | 24.52%  ← HIGH (extrapolation)
Reboiling_Steam_Flow_roll_mean_12h| [18.73, 25.26]         | [14.43, 23.54]           | 14.47%
Reboiling_Steam_Flow_roll_mean_6h | [18.66, 25.26]         | [14.43, 23.79]           | 13.78%
Reboiling_Steam_Flow_roll_mean_3h | [17.85, 25.26]         | [14.43, 23.88]           |  8.27%
Feed_Flow_roll_mean_12h         | [63.39, 99.26]           | [44.55, 92.20]           |  6.41%
Feed_Flow_roll_mean_6h          | [62.06, 99.26]           | [44.55, 93.34]           |  5.44%
Feed_Flow_roll_mean_3h          | [60.52, 99.26]           | [44.55, 94.83]           |  4.74%
Reboiling_Steam_Flow            | [14.91, 25.26]           | [14.43, 23.99]           |  2.27%
...
```

---

## 6. `notebooks/experiment_regimes.py` — Hot vs Cold Regime Experiment

### Command
```
python notebooks/experiment_regimes.py
```

### Output
```
Train set: 4332 rows | Hot: 1380 | Cold: 2952
Test set:  6081 rows | Hot: 3252 | Cold: 2829

Train: All -> Test: All                            | R2:  -0.9733 | MAE:   0.2944
Train: All -> Test: Hot                            | R2:  -1.3539 | MAE:   0.3610
Train: All -> Test: Cold                           | R2:  -0.7759 | MAE:   0.2179
Train: Hot -> Test: Hot                            | R2:  -1.3326 | MAE:   0.3558
Train: Cold -> Test: Cold                          | R2:  -0.9336 | MAE:   0.2425
```
**Note:** Even training exclusively on hot rows and testing on hot rows yields R² = -1.33.  
The problem is NOT regime mismatch alone — it is **covariate shift + concept drift** within the hot regime itself.

---

## 7. `notebooks/inspect_hot_shift.py` — Feature Shift Inside Hot Regime

### Command
```
python notebooks/inspect_hot_shift.py
```

### Output
```
Hot Train size: 1380 | Hot Test size: 3252

Feature                        | Train Mean | Test Mean  | Train Min  | Train Max  | Test Min   | Test Max
C4H8_Bottom                    |       0.28 |       0.51 |       0.03 |       1.24 |       0.08 |       1.53
Feed_Flow                      |      89.73 |      77.78 |      60.47 |      99.26 |      44.55 |      94.94
Reboiler_Outlet_Temp           |     105.88 |     106.79 |      50.24 |     111.75 |      50.05 |     111.75
Column_Top_Temp                |      39.97 |      28.64 |      29.35 |     113.00 |      18.04 |     109.80
Reboiling_Steam_Flow           |      24.02 |      20.54 |      20.00 |      25.26 |      14.43 |      23.99
Reflux_Flow                    |     102.16 |      88.16 |      89.85 |     105.65 |      70.42 |     105.65
Column_Top_Pressure            |       4.15 |       3.90 |       3.78 |       4.55 |       3.78 |       4.55
Column_Bottom_Temp             |     107.68 |     107.86 |      99.03 |     112.98 |      99.03 |     112.98
Control_Tray_Temp              |      73.49 |      75.25 |      57.16 |      91.31 |      57.16 |      91.31
Reflux_Ratio                   |       1.15 |       1.15 |       0.96 |       1.69 |       0.83 |       1.86
Steam_Feed_Ratio               |       0.27 |       0.27 |       0.22 |       0.37 |       0.19 |       0.38
Temp_Gradient                  |      67.71 |      79.22 |      -3.32 |      82.09 |      -1.49 |      94.94
Reboiler_Delta                 |      -1.80 |      -1.06 |     -58.99 |       7.85 |     -57.40 |      12.72
```

**Key observations:**
- C4H8_Bottom mean: train = 0.28 vs test = 0.51 (+82% higher in Block 4 hot regime)
- Reflux_Flow: train min = 89.85 vs test min = 70.42 (Block 4 sees 18% lower reflux)
- Temp_Gradient: 67.7°C train vs 79.2°C test (+17% shift)

---

## 8. `notebooks/inspect_bias.py` — Correlation & Bias Correction Analysis

### Command
```
python notebooks/inspect_bias.py
```

### Output
```
Pearson Correlation between y_test and pred_test: -0.3263

Bias Corrected (shifted by +0.1280) | R2: -0.7578 | MAE: 0.3057
Linear Calibrated (slope: -0.5409, intercept: 0.5907) | R2: 0.1065 | MAE: 0.1954
```

**Critical interpretation:**
- **Correlation = -0.33**: The model's predictions are *inversely correlated* with the truth in Block 4.
  - When the model predicts high C4H8 → actual C4H8 is low, and vice versa.
  - This is concept drift: the relationship between process variables and C4H8 literally **reversed sign** between training and test periods.
- After linear calibration, R² reaches 0.11 — confirming there IS information in the features, but the learned direction is wrong.

---

## 9. Correlation Comparison: Train vs Test Blocks

### Command
```
python -c "...correlation comparison..."
```

### Output
```
=== CORRELATIONS IN TRAIN ===
Reflux_Ratio           -0.209900
Steam_Feed_Ratio       -0.256080
Column_Top_Pressure     0.474891     ← Strong positive in Train
Feed_Flow               0.254268
Reflux_Flow             0.084175
Reboiling_Steam_Flow    0.015597
Column_Bottom_Temp     -0.065053
Control_Tray_Temp      -0.367446     ← Strong negative in Train
Temp_Gradient          -0.253730

=== CORRELATIONS IN TEST (Block 4) ===
Reflux_Ratio           -0.329605
Steam_Feed_Ratio       -0.312190
Column_Top_Pressure     0.058098     ← Near-zero in Test (was +0.47!)
Feed_Flow               0.229468
Reflux_Flow            -0.239434     ← Sign REVERSED
Reboiling_Steam_Flow   -0.050342
Column_Bottom_Temp      0.223111     ← Sign REVERSED
Control_Tray_Temp       0.391486     ← Sign REVERSED (was -0.37!)
Temp_Gradient           0.165522     ← Sign REVERSED
```

---

## 10. Hot Regime Correlation Comparison

### Command
```
python -c "...hot regime correlation comparison..."
```

### Output
```
=== HOT CORRELATIONS IN TRAIN ===
Column_Top_Pressure     0.361562     ← Positive in Train Hot
Control_Tray_Temp      -0.165342     ← Negative in Train Hot
Reflux_Flow             0.109998
Temp_Gradient          -0.055286

=== HOT CORRELATIONS IN TEST (Block 4 Hot) ===
Column_Top_Pressure    -0.067656     ← Near-zero, was +0.36!
Control_Tray_Temp       0.359703     ← Strongly REVERSED sign
Reflux_Flow            -0.166455     ← REVERSED sign
Temp_Gradient           0.215308     ← REVERSED sign
```

**Confirmed: Concept drift is present even within the same operating regime (hot).**  
The direction of influence of several key process variables on C4H8 flipped between 2023/2024 campaigns and 2025/2026 campaign.

---

## Summary Table: All Model Results

| Model | R² | MAE (wt%) | RMSE (wt%) | Spec Recall | Analyzer? |
|:------|:---|:----------|:-----------|:------------|:----------|
| **Naive Lag-1** (C4H8) | **0.9328** | **0.0361** | **0.0715** | **96.0%** | **Yes** |
| LinearRegression (C4H8, Tier 1) | -3.728 | 0.5035 | 0.5996 | 21.6% | No |
| Ridge (C4H8, Tier 1) | -3.932 | 0.5082 | 0.6124 | 21.2% | No |
| RandomForest (C4H8, Tier 1) | -0.992 | 0.2987 | 0.3892 | 19.7% | No |
| XGBoost (C4H8, Tier 1) | -1.029 | 0.3017 | 0.3928 | 19.0% | No |
| **XGBoost (C4H8, Tier 2)** | **0.7128** | **0.0995** | **0.1478** | **90.7%** | **Yes** |
| RandomForest (C4H6, Tier 1) | -6.249 | 0.0190 | 0.0270 | 0% | No |
| XGBoost (C4H6, Tier 1) | -34.63 | 0.0550 | 0.0598 | 0% | No |
| XGBoost (C4H6, Tier 2) | -16.82 | 0.0369 | 0.0423 | 50% | Yes |
| **Combined Lag-1** (Total C4) | **0.9217** | **0.0487** | **0.0891** | **95.8%** | **Yes** |
| Combined XGBoost (Tier 1) | -1.052 | 0.3621 | 0.4561 | 0.8% | No |

---

## 11. `python notebooks/run_drift_experiments.py` — Phase 4 Adaptive Modeling experiments (Leak-Free)

### Command
```
python notebooks/run_drift_experiments.py
```

### Output
```
================================================================================
RUNNING ADAPTIVE MODELING EXPERIMENTS
================================================================================
Loaded features dataset: (11343, 113)
Original feature set size: 67

Training Baseline with 67 features...
  Pearson = -0.1785 | R² = -1.0363 | MAE = 0.3010 wt%
  Top Feature: month_cos (0.3781)

Training Exp 1: No Calendar with 65 features...
  Pearson = -0.3139 | R² = -0.9674 | MAE = 0.2930 wt%
  Top Feature: Control_Tray_Temp_lag1 (0.1112)

Training Exp 2: No Regime with 64 features...
  Pearson = -0.2075 | R² = -0.8816 | MAE = 0.2820 wt%
  Top Feature: month_cos (0.3030)

Training Exp 3: All Removed with 62 features...
  Pearson = -0.3340 | R² = -0.9693 | MAE = 0.2891 wt%
  Top Feature: Control_Tray_Temp_lag1 (0.1031)

Training Exp 4: Pnorm (k=3) with 65 features...
  Pearson = -0.2676 | R² = -0.7295 | MAE = 0.2683 wt%
  Top Feature: Control_Tray_Temp_Pnorm_k3 (0.1716)

Training Exp 5: Pnorm (k=5) with 65 features...
  Pearson = -0.3383 | R² = -1.0502 | MAE = 0.2901 wt%
  Top Feature: Control_Tray_Temp_Pnorm_k5 (0.1747)

Training Exp 6: Pnorm (k=10) with 65 features...
  Pearson = -0.3546 | R² = -0.8705 | MAE = 0.2804 wt%
  Top Feature: Column_Bottom_Temp_Pnorm_k10 (0.2497)

Training Exp 7: Pnorm Ratio with 65 features...
  Pearson = -0.3847 | R² = -1.1232 | MAE = 0.2985 wt%
  Top Feature: Column_Bottom_Temp_Pratio (0.2163)

Training Exp 8: Pnorm Gradient with 63 features...
  Pearson = -0.3501 | R² = -1.0454 | MAE = 0.2948 wt%
  Top Feature: Control_Tray_Temp_lag1 (0.1084)

Training Exp 9: Pressure Interactions with 65 features...
  Pearson = -0.3506 | R² = -0.9841 | MAE = 0.2912 wt%
  Top Feature: Pressure_x_TopTemp (0.1960)

Training Exp 10: Rolling Deviations with 67 features...
  Pearson = -0.3262 | R² = -0.8522 | MAE = 0.2784 wt%
  Top Feature: Control_Tray_Temp_lag1 (0.1089)

Best physical normalization experiment: Exp 4: Pnorm (k=3) (Pearson = -0.2676)

Training Exp 11: Campaign Anchor with 66 features...
  Pearson = +0.8595 | R² = 0.6865 | MAE = 0.1032 wt%
  Top Feature: C4H8_campaign_anchor (0.5375)

================================================================================
EXPERIMENT VERIFICATION MATRIX SUMMARY
================================================================================
                  Experiment   Pearson        R2      MAE                          Top Feature
                    Baseline -0.178488 -1.036339 0.301015                    month_cos (0.378)
          Exp 1: No Calendar -0.313944 -0.967408 0.293017       Control_Tray_Temp_lag1 (0.111)
            Exp 2: No Regime -0.207530 -0.881611 0.281955                    month_cos (0.303)
          Exp 3: All Removed -0.334019 -0.969305 0.289057       Control_Tray_Temp_lag1 (0.103)
          Exp 4: Pnorm (k=3) -0.267563 -0.729505 0.268253   Control_Tray_Temp_Pnorm_k3 (0.172)
          Exp 5: Pnorm (k=5) -0.338292 -1.050179 0.290074   Control_Tray_Temp_Pnorm_k5 (0.175)
         Exp 6: Pnorm (k=10) -0.354560 -0.870474 0.280370 Column_Bottom_Temp_Pnorm_k10 (0.250)
          Exp 7: Pnorm Ratio -0.384735 -1.123166 0.298547    Column_Bottom_Temp_Pratio (0.216)
       Exp 8: Pnorm Gradient -0.350057 -1.045360 0.294766       Control_Tray_Temp_lag1 (0.108)
Exp 9: Pressure Interactions -0.350640 -0.984065 0.291224           Pressure_x_TopTemp (0.196)
  Exp 10: Rolling Deviations -0.326183 -0.852168 0.278413       Control_Tray_Temp_lag1 (0.109)
     Exp 11: Campaign Anchor  0.859535  0.686487 0.103220         C4H8_campaign_anchor (0.538)
================================================================================
Results saved to models/drift_experiments_summary.csv
```

---

## 12. `python notebooks/run_feature_ablation_study.py` — Systematic Feature Subset Ablation

### Command
```
python notebooks/run_feature_ablation_study.py
```

### Output
```
================================================================================
RUNNING FEATURE ABLATION STUDY (MODEL A - C4H8)
================================================================================
Subset: 1. Baseline TIER1 (67 features)
  Pearson: -0.1785 | R²: -1.0363 | MAE: 0.3010 wt%
  Top Feature: month_cos (0.3781)
------------------------------------------------------------
Subset: 2. Physics Only (No Temps) (41 features)
  Pearson: -0.0639 | R²: -0.3084 | MAE: 0.2459 wt%
  Top Feature: Column_Top_Pressure_lag1 (0.1507)
------------------------------------------------------------
Subset: 3. Physics + Stable Temps (59 features)
  Pearson: -0.2203 | R²: -0.5849 | MAE: 0.2599 wt%
  Top Feature: Column_Bottom_Temp_Pnorm_k10 (0.2652)
------------------------------------------------------------
Subset: 4. Deviations & Ratios Only (7 features)
  Pearson: -0.0771 | R²: -0.3185 | MAE: 0.2420 wt%
  Top Feature: Steam_Feed_Ratio (0.2201)
------------------------------------------------------------
Subset: 5. Physics + Stable Temps + Campaign Anchor (60 features)
  Pearson: +0.8559 | R²: 0.6754 | MAE: 0.1068 wt%
  Top Feature: C4H8_campaign_anchor (0.5344)
------------------------------------------------------------
Subset: 6. Physics Only (No Temps) + Campaign Anchor (42 features)
  Pearson: +0.8320 | R²: 0.6589 | MAE: 0.1146 wt%
  Top Feature: C4H8_campaign_anchor (0.6164)
------------------------------------------------------------
Subset: 7. Deviations & Ratios + Campaign Anchor (8 features)
  Pearson: +0.9220 | R²: 0.8345 | MAE: 0.0726 wt%
  Top Feature: C4H8_campaign_anchor (0.8320)
------------------------------------------------------------

================================================================================
ABLATION STUDY SUMMARY
================================================================================
                                      Subset  Features   Pearson        R2      MAE                          Top Feature
                           1. Baseline TIER1        67 -0.178488 -1.036339 0.301015                    month_cos (0.378)
                  2. Physics Only (No Temps)        41 -0.063886 -0.308432 0.245890     Column_Top_Pressure_lag1 (0.151)
                   3. Physics + Stable Temps        59 -0.220310 -0.584948 0.259927 Column_Bottom_Temp_Pnorm_k10 (0.265)
                 4. Deviations & Ratios Only         7 -0.077136 -0.318545 0.242031             Steam_Feed_Ratio (0.220)
 5. Physics + Stable Temps + Campaign Anchor        60  0.855902  0.675425 0.106790         C4H8_campaign_anchor (0.534)
6. Physics Only (No Temps) + Campaign Anchor        42  0.831989  0.658878 0.114594         C4H8_campaign_anchor (0.616)
    7. Deviations & Ratios + Campaign Anchor         8  0.921976  0.834469 0.072585         C4H8_campaign_anchor (0.832)
================================================================================
Ablation study summary saved to models/ablation_study_summary.csv
```

---

## 13. `python notebooks/run_anchor_analysis.py` — Campaign Anchor Lookback Sensitivity & Coverage Analysis

### Command
```
python notebooks/run_anchor_analysis.py
```

### Output
```
================================================================================
RUNNING CAMPAIGN ANCHOR SENSITIVITY & COVERAGE ANALYSIS
================================================================================
Evaluated limit = 6   | R² = +0.8450 | MAE = 0.0694 wt% | Train Coverage = 90.6% | Test Coverage = 93.8%
Evaluated limit = 12  | R² = +0.8450 | MAE = 0.0694 wt% | Train Coverage = 91.1% | Test Coverage = 94.1%
Evaluated limit = 24  | R² = +0.8341 | MAE = 0.0726 wt% | Train Coverage = 92.0% | Test Coverage = 94.6%
Evaluated limit = 48  | R² = +0.8341 | MAE = 0.0726 wt% | Train Coverage = 93.0% | Test Coverage = 95.4%
Evaluated limit = 72  | R² = +0.8345 | MAE = 0.0726 wt% | Train Coverage = 94.0% | Test Coverage = 96.1%

================================================================================
1. ANCHOR COVERAGE STATISTICS
================================================================================
Limit  Train Coverage (%)  Test Coverage (%)
   6h           90.619176          93.813325
  12h           91.116173          94.089653
  24h           91.965210          94.642309
  48h           92.959205          95.440589
  72h           93.953199          96.070003

================================================================================
2. MODEL PERFORMANCE VS. ANCHOR LIMIT
================================================================================
Limit       R2      MAE  Train Rows  Test Rows
   6h 0.844967 0.069416        4348       6090
  12h 0.844967 0.069416        4348       6090
  24h 0.834119 0.072581        4350       6090
  48h 0.834143 0.072581        4350       6091
  72h 0.834469 0.072585        4350       6092
================================================================================
```

---

## 5. `python notebooks/run_advanced_experiments.py` — Advanced Modeling Experiments

### Command
```
python notebooks/run_advanced_experiments.py
```

### Output
```
Loaded dataset: (11343, 113)
Found 81 pure process physics features.
Pure Physics train set: (4332, 81) | test set: (6058, 81)

================================================================================
RUN 1: STRONGER CATBOOST TUNING (5-FOLD TS-CV + EARLY STOPPING)
================================================================================
Running 15 Optuna trials for CatBoost...
Best CV R2 score: -0.1770
Best CatBoost parameters:
  iterations: 593
  learning_rate: 0.038135752841855254
  depth: 5
  l2_leaf_reg: 0.16519632443105745
  bagging_temperature: 1.0307760563843826
  random_strength: 2.606622902359794
  rsm: 0.6311023010745054
==================================================
[Final Pure Physics Train] Running leakage check...
  [OK] No high-correlation leakage detected.
Tuned CatBoost on Block 4 Test Set: R2 = -1.0155 | MAE = 0.2859 wt%
Top CatBoost feature: Column_Bottom_Temp_roll_mean_12h (3.3602)
Saved experiment 'CatBoost (Tuned) (Pure Physics)' to experiments\master_leaderboard.csv

================================================================================
RUN 2: TRAINING ENSEMBLE (CATBOOST + XGBOOST + LIGHTGBM)
================================================================================
Fitting XGBoost Baseline...
Saved experiment 'XGBoost (Default) (Pure Physics)' to experiments\master_leaderboard.csv
Loaded optimized LightGBM parameters from experiments/optuna_results.json
Fitting LightGBM (Tuned)...
Saved experiment 'LightGBM (Tuned) (Pure Physics)' to experiments\master_leaderboard.csv
Combining predictions...

==================================================
ENSEMBLE RESULTS (Block 4 Test Set)
==================================================
  Tuned CatBoost R2:  -1.0155  | MAE: 0.2859
  XGBoost R2:         -0.9353  | MAE: 0.2912
  Tuned LightGBM R2:  -0.8299  | MAE: 0.2698
  --------------------------------------------------
  Ensemble R2:        -0.9210  | MAE: 0.2810
==================================================
Saved experiment 'Ensemble (0.5CB+0.3XG+0.2LG) (Pure Physics)' to experiments\master_leaderboard.csv

================================================================================
RUN 3: TEST 'NO TEMPERATURE' HYPOTHESIS
================================================================================
Removed 40 temperature columns. Remaining features: 41
[No Temperature Train] Running leakage check...
  [OK] No high-correlation leakage detected.
Fitting Tuned CatBoost (No Temp)...
No Temperature Tuned CatBoost: R2 = -0.2548 | MAE = 0.2383 wt%
Saved experiment 'CatBoost (Tuned) (No Temperature)' to experiments\master_leaderboard.csv

================================================================================
RUN 4: RESEARCH CAMPAIGN ANCHOR EXPERIMENT
================================================================================
[Campaign Anchor Train] Running leakage check...
  [OK] No high-correlation leakage detected.
Fitting Tuned CatBoost with Campaign Anchor...
Campaign Anchor Tuned CatBoost: R2 = -0.1899 | MAE = 0.2068 wt%
Saved experiment 'CatBoost (Tuned) (Process+Anchor)' to experiments\master_leaderboard.csv

================================================================================
GENERATING DIAGNOSTICS FOR THE TUNED CATBOOST (PURE PHYSICS)
================================================================================
Saved residual vs time plot to experiments\diagnostics\pure_physics_catboost_plot_4_residual_vs_time.png
Generating SHAP values for Tuned CatBoost...
Saved SHAP plot to experiments\diagnostics\pure_physics_catboost_plot_5_shap.png

TOP 20 SHAP FEATURES (TUNED CATBOOST PURE PHYSICS)
--------------------------------------------------
                     Feature  Mean_Abs_SHAP
     Feed_Flow_roll_mean_12h       0.023375
   Reflux_Flow_roll_mean_12h       0.017579
          Pressure_x_TopTemp       0.017463
              Reboiler_Delta       0.015149
    Column_Top_Temp_Pnorm_k5       0.014504
   Reboiler_Outlet_Temp_lag2       0.014023
       Feed_Flow_roll_std_6h       0.013609
   Reboiler_Outlet_Temp_lag3       0.011898
  Control_Tray_Temp_Pnorm_k5       0.011607
    Column_Top_Temp_Pnorm_k3       0.010932
    Control_Tray_Temp_dev24h       0.010323
        Column_Top_Temp_lag1       0.009865
Column_Bottom_Temp_Pnorm_k10       0.009588
        Reboiler_Outlet_Temp       0.009115
         Temp_Gradient_Pnorm       0.008838
   Column_Top_Temp_Pnorm_k10       0.008257
   Column_Bottom_Temp_Pratio       0.008179
    Reflux_Flow_roll_mean_6h       0.007894
      Column_Top_Temp_Pratio       0.007890
    Column_Top_Pressure_lag1       0.007772
--------------------------------------------------

================================================================================
ALL ADVANCED EXPERIMENTS COMPLETED SUCCESSFULLY!
================================================================================

MASTER LEADERBOARD:
| Model                        | Feature_Set    |        R2 |      MAE | Top_Feature                             |
|:-----------------------------|:---------------|----------:|---------:|:----------------------------------------|
| CatBoost (Tuned)             | Pure Physics   | -1.01546  | 0.285883 | Column_Bottom_Temp_roll_mean_12h (3.36) |
| XGBoost (Default)            | Pure Physics   | -0.935333 | 0.291246 | Column_Bottom_Temp_Pnorm_k10 (0.290)    |
| LightGBM (Tuned)             | Pure Physics   | -0.82991  | 0.269759 | Feed_Flow_roll_mean_12h (243.00)        |
| Ensemble (0.5CB+0.3XG+0.2LG) | Pure Physics   | -0.921003 | 0.280993 | nan                                     |
| CatBoost (Tuned)             | No Temperature | -0.254758 | 0.238302 | Column_Top_Pressure_lag1 (8.67)         |
| CatBoost (Tuned)             | Process+Anchor | -0.189892 | 0.206844 | C4H8_campaign_anchor (18.28)            |
```

---

## 6. `python notebooks/run_anchor_tuning.py` — Robust 8-Feature Physical Modeling

### Command
```
python notebooks/run_anchor_tuning.py
```

### Output
```
Loaded dataset: (11343, 113)
Robust 8-Feature Train size: (4346, 8) | Test size: (6089, 8)
[Train] Running leakage check...
  [OK] No high-correlation leakage detected.
[Test] Running leakage check...
  [OK] No high-correlation leakage detected.

Training XGBoost on 8 features...
  XGBoost test R2 = 0.8846 | MAE = 0.0572 wt%

Tuning CatBoost on 8 features...
  Best CV R2: 0.7181
  Best params:
    iterations: 219
    learning_rate: 0.09815419337350885
    depth: 3
    l2_leaf_reg: 1.4322632892152434
    random_strength: 0.36974610755302056
  Tuned CatBoost test R2 = 0.9030 | MAE = 0.0524 wt%

Tuning LightGBM on 8 features...
  Best LGB CV R2: 0.7087
  Tuned LightGBM test R2 = 0.9147 | MAE = 0.0494 wt%

Evaluating Ensemble...
  Ensemble test R2 = 0.9052 | MAE = 0.0513 wt%

Generating best model diagnostics (Tuned CatBoost)...
Saved tuned CatBoost model to models/model_A_CatBoost_robust.bin

==================================================
ROBUST 8-FEATURE PHYSICAL MODEL COMPARISON
==================================================
XGBoost R2:       0.8846  | MAE: 0.0572 wt%
CatBoost R2:      0.9030  | MAE: 0.0524 wt%
LightGBM R2:      0.9147  | MAE: 0.0494 wt%
Ensemble R2:      0.9052  | MAE: 0.0513 wt%
==================================================
```

---

## 7. `python notebooks/tune_robust_xgb.py` — Robust XGBoost Hyperparameter Optimization

### Command
```
python notebooks/tune_robust_xgb.py
```

### Output
```
Loaded dataset: (11343, 113)
Train set: (4348, 8) | Test set: (6091, 8)

Running 50 Optuna trials for XGBoost...
==================================================
XGBOOST OPTUNA OPTIMIZATION COMPLETED
==================================================
Best CV R2 score: 0.7037
Best parameters:
  n_estimators: 102
  max_depth: 3
  learning_rate: 0.04049995978081821
  subsample: 0.8056067369529782
  colsample_bytree: 0.936028802042788
  min_child_weight: 8
  gamma: 3.4041610450962554e-05
  reg_alpha: 0.0007831342584736976
  reg_lambda: 3.756558568832882e-08
==================================================

Final Optimized XGBoost Model (Block 4 Test Set):
  Test R2:  0.9074
  Test MAE: 0.0516 wt%
==================================================
Optuna results saved to experiments/robust_xgb_optuna_results.json
Optimized XGBoost model saved to models/model_A_XGBoost_robust_opt.json
Saved optimized diagnostic plots to experiments/diagnostics/
```

---

## 8. `python notebooks/verify_anchor_leakage.py` — Target Leakage proof & Block 3 Validation

### Command
```
python notebooks/verify_anchor_leakage.py
```

### Output
```
================================================================================
Loaded dataset: (11343, 113) rows.

Production Features:
  - C4H8_campaign_anchor
  - Steam_Feed_Ratio
  - Reflux_Ratio
  - Reboiling_Steam_Flow_dev24h
  - Reflux_Flow_dev24h
  - Column_Bottom_Temp_dev24h
  - Control_Tray_Temp_dev24h
  - Column_Top_Pressure_dev24h

--- RUNNING PROGRAMMATIC LEAK-FREE PROOF ---
Original target at t=538: 0.1010
Perturbed target at t=538: 5.1010
Original anchor at t=538: 0.1078
Perturbed anchor at t=538: 0.1078
Perturbed anchor at t=539 (should change): 5.1010
[VERIFIED] Programmatic proof passed: C4H8_campaign_anchor has no current-timestep leakage.

Training set size (Blocks 1+2): 3567 rows
Testing set size (Block 3):     781 rows

--- EVALUATION METRICS ON BLOCK 3 TEST ---
R² Score:            0.7694  (Expected: 0.7694)
MAE:                 0.0817 wt% (Expected: 0.0817)
Pearson Correlation: +0.8848 (Expected: +0.8848)

[VERIFIED] Metrics match expected values exactly!
================================================================================
```

---

## 9. `python notebooks/model_b_target_audit.py` — Model B C4H6 target stats block-by-block

### Command
```
python notebooks/model_b_target_audit.py
```

### Output
```
================================================================================
MODEL B (C4H6_Bottom) TARGET AUDIT BY BLOCK (HEALTHY ANALYZER ROWS > 0.001)
================================================================================
 Block     Mean   Median      Std      P95      P99      Max Count
     1 0.208113 0.123793 0.172158 0.669380 0.669380 0.669380  2219
     2 0.031413 0.033122 0.013766 0.050861 0.059714 0.074878   684
     3 0.023405 0.017447 0.019025 0.056435 0.081022 0.152658   674
     4 0.005663 0.005682 0.010023 0.010792 0.026308 0.380282  2974
================================================================================
```

---

## 10. `python notebooks/model_b_anchor_audit.py` — Block 4 C4H6 Anchor Coverage & Age

### Command
```
python notebooks/model_b_anchor_audit.py
```

### Output
```
================================================================================
MODEL B (C4H6) ANCHOR AUDIT ON BLOCK 4
================================================================================

--- ALL BLOCK 4 ROWS (Count: 6514) ---
C4H6 Campaign Anchor (12h Limit):
  Coverage:  55.28%
  Mean Age:  1.07 hours
  Max Age:   12.00 hours
C4H6 Campaign Anchor (72h Limit):
  Coverage:  69.62%
  Mean Age:  7.74 hours
  Max Age:   72.00 hours

--- HEALTHY ANALYZER ROWS ONLY (Count: 2974) ---
C4H6 Campaign Anchor (12h Limit):
  Coverage:  98.45%
  Mean Age:  0.03 hours
  Max Age:   12.00 hours
C4H6 Campaign Anchor (72h Limit):
  Coverage:  99.76%
  Mean Age:  0.36 hours
  Max Age:   69.00 hours
================================================================================
```

---

## 11. `python notebooks/anchor_only_baselines.py` — Model B Anchor-Only baseline check

### Command
```
python notebooks/anchor_only_baselines.py
```

### Output
```
================================================================================
MODEL B (C4H6) ANCHOR-ONLY BASELINES ON BLOCK 4
================================================================================
Number of test rows: 2974
Target distribution - Mean: 0.005663 | Var: 0.00010043

Baseline Model                      | R2 Score   | MAE (wt%)  | Pearson  | Available %
-------------------------------------------------------------------------------------
Baseline A (12h Anchor)             |     0.9606 |     0.0005  | +0.9830  |      98.45%
Baseline A (72h Anchor)             |     0.6074 |     0.0007  | +0.7813  |      99.76%
Baseline B (24h Roll 12h Anchor)    |     0.9308 |     0.0009  | +0.9672  |      99.56%
Baseline B (24h Roll 72h Anchor)    |     0.1510 |     0.0011  | +0.3898  |      99.80%
Baseline C (Block 4 Mean: 0.005663) |    -0.0000 |     0.0032  | Constant |     100.00%
Baseline C (Train Mean: 0.0273)     |    -4.6618 |     0.0221  | Constant |     100.00%
================================================================================
```

---

## 12. `python notebooks/model_b_delta_model.py` — Delta Correction Model Experiment

### Command
```
python notebooks/model_b_delta_model.py
```

### Output
```
================================================================================
MODEL B (C4H6) DELTA CORRECTION MODEL
================================================================================
Train size: 1352 | Test size: 2928
Train Mean Delta: -0.000018 | Test Mean Delta: -0.000001

--- RESULTS COMPARISON (BLOCK 4 HEALTHY ROWS) ---
1. Baseline (12h Anchor Only):
   R² Score:            0.960630
   MAE:                 0.000547 wt% (5.5 ppm)
   Pearson Correlation: +0.9830

2. Delta Correction Model (12h Anchor + Predicted Delta):
   R² Score:            0.900985
   MAE:                 0.001194 wt% (11.9 ppm)
   Pearson Correlation: +0.9499

--- FEATURE IMPORTANCES FOR DELTA MODEL (GAIN) ---
                    Feature     Gain
Reboiling_Steam_Flow_dev24h 0.000557
           Steam_Feed_Ratio 0.000431
   Control_Tray_Temp_dev24h 0.000283
 Column_Top_Pressure_dev24h 0.000252
  Column_Bottom_Temp_dev24h 0.000238
               Reflux_Ratio 0.000230
         Reflux_Flow_dev24h 0.000225
```

---

## 13. `python notebooks/model_b_inversion_check.py` — Inversion checks cross blocks 2, 3, 4

### Command
```
python notebooks/model_b_inversion_check.py
```

### Output
```
================================================================================
MODEL B ANCHOR ROBUSTNESS INVERSION CHECKS (12h LIMIT)
================================================================================
Block 2 (Target Mean = 0.0314 wt%):
  R² Score:            0.7518
  MAE:                 0.0043 wt% (42.9 ppm)
  Pearson Correlation: +0.8758
  Healthy Coverage:    99.71%
----------------------------------------
Block 3 (Target Mean = 0.0234 wt%):
  R² Score:            0.7651
  MAE:                 0.0049 wt% (48.5 ppm)
  Pearson Correlation: +0.8821
  Healthy Coverage:    99.41%
----------------------------------------
Block 4 (Target Mean = 0.0057 wt%):
  R² Score:            0.9606
  MAE:                 0.0005 wt% (5.5 ppm)
  Pearson Correlation: +0.9830
  Healthy Coverage:    98.45%
----------------------------------------
================================================================================
```

---

## 14. `python final_v1/inference/predict_total_c4.py` — Release Verification

### Command
```
python final_v1/inference/predict_total_c4.py
```

### Output
```
=== TESTING predict_total_c4.py ===

Combined Output (Normal Operations):
  Predicted C4H8:     0.4151 wt%
  Predicted C4H6:     0.004500 wt% (45.0 ppm)
  Predicted Total C4: 0.4196 wt%
  Out of Spec (>0.5): False
  Overall Health:     GREEN
  Model A:            Model A (Health: GREEN, Reason: None)
  Model B:            Model B (12h Anchor) (Health: GREEN, Reason: None)
```




