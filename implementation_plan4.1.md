# Debutanizer C4 Slippage Model - Implementation Plan

---

## STATUS: Phase 1 Locked

**Output**: `data/clean_data.parquet` — 11,343 rows x **24 columns** (final, no further changes)
**Audit scripts**: `notebooks/analyze_data.py`, `notebooks/analyze_constraints.py`, `notebooks/inspect_clean_data.py`

> [!NOTE]
> Phase 1 is closed. Three bugs were fixed after the initial run (see F8). The parquet is the ground truth for all downstream work.

---

## Phase 1 Findings: What the Clean Data Actually Shows

These findings come from running `notebooks/inspect_clean_data.py` against the parquet output.
They update and replace some assumptions made during planning.

### F1. There Are 4 Data Blocks, Not 3

The gap structure is more complex than anticipated:

| Block | Start | End | Rows | Duration |
|-------|-------|-----|------|----------|
| 1 | 2023-04-16 | 2023-08-31 | 3,288 | 137 days |
| — | **Gap: 376 days** | — | — | — |
| 2 | 2024-09-11 | 2024-10-11 | 738 | 30.7 days |
| — | **Gap: ~43 hours** | — | — | — |
| 3 | 2024-10-13 | 2024-11-15 | 803 | 33.4 days |
| — | **Gap: 258 days** | — | — | — |
| 4 | 2025-08-01 | 2026-04-30 | 6,514 | 272 days |

Blocks 2 and 3 are separated by only 43 hours — they were originally treated as one "Block 2" but the gap exceeds the 24h threshold. They likely represent the same operating campaign. **Decision needed**: treat Blocks 2+3 as one campaign or keep separate. For lag computation, they must stay separate (43h gap = 43 missing lag rows).

> [!NOTE]
> Block 4 has 2 non-1h gaps (max 10 hours) within it. These are minor data dropouts, not campaign boundaries. Lag features around those 2 points will be NaN-padded and excluded from training.

### F2. The Bimodal Temperature Split is Entirely Block-Driven

This is the single most important finding from the inspection.

| Block | Reboiler Temp (mean) | Regime |
|-------|---------------------|--------|
| 1 | 35.9°C | **100% cold** (below 50°C) |
| 2 | 108.0°C | **100% hot** (above 50°C) |
| 3 | 93.2°C | Mixed (15% cold, 85% hot) |
| 4 | 71.7°C | Mixed (~49% cold, ~51% hot) |

**Block 1 is entirely in the cold reboiler regime. Block 2 is entirely in the hot regime.**

This means the "bimodal distribution" is not two concurrent operating modes — it is the difference between *operating periods*. The column was running differently in 2023 than in 2024-2026 (possibly different feed composition, different season, or equipment changes).

**Implications**:
1. **No K-Means needed** — regime is essentially captured by the block label or the actual temperature values. The tree model will learn `if Reboiler_Outlet_Temp < 50` as a natural split, which is the right behaviour.
2. The `Data_Block` label itself carries regime information and should be included as a categorical feature in the model.
3. C4 slippage between cold and hot regimes is nearly identical (cold: 0.52 wt%, hot: 0.46 wt%), so the regime is not a strong predictor of C4 level on its own — but it may interact with other variables in ways the tree will capture.

### F3. C4H6 Analyzer Often Freezes at Zero — This Is Not a Valid Value

During stuck periods, the C4H6 analyzer statistics are:

```
C4H6 STUCK periods:  mean=0.010, median=0.000, P75=0.000
C4H6 NORMAL periods: mean=0.072, median=0.009, P75=0.081
```

The median stuck value is **exactly 0.000**. When the C4H6 analyzer freezes, it usually freezes at or near zero — meaning it is **not** reporting a real low-C4H6 condition, it is reporting nothing. This is critical for Model B:

- **Original plan**: Train Model B (C4H6 predictor) on non-stuck rows only
- **Refined**: Additionally check that C4H6 training rows have `C4H6_Bottom > 0.001` (exclude frozen-at-zero readings). There are rows where the analyzer reads zero and is not flagged as stuck (because the run length is < 12). These zeros may still be invalid.
- **Practical impact**: Model B's training set will be significantly smaller (~4,000-5,000 rows). This is acceptable; XGBoost handles this well.

### F4. C4H8 Stuck Readings Cluster at Two Extreme Values

During C4H8 stuck periods:
```
C4H8 STUCK periods: mean=0.603, P25=0.034, P50=0.034, P75=1.262
```

The 25th and 50th percentiles are 0.034, but the 75th is 1.262 — indicating the analyzer freezes at two very different levels: near its minimum and near its maximum. This is unusual. It suggests the C4H8 analyzer sometimes gets stuck reporting a false-high just as often as a false-low.

**Implication for Model A (C4H8 predictor)**: The 7.9% stuck C4H8 rows are not uniformly distributed in value space. If we train on all rows including stuck, we are teaching the model that some process conditions correspond to extreme-high or extreme-low C4H8 when they may not. Consider also filtering out C4H8 stuck rows from Model A training (not just the target but the label noise). Use the `C4H8_Bottom_stuck` flag.

### F5. Block 1 Has Significantly Higher C4 Slippage Than Later Blocks

| Block | Mean Total_C4 | % Above 0.5 spec |
|-------|--------------|-----------------|
| **1** | **0.602** | **59.6%** |
| 2 | 0.337 | 17.3% |
| 3 | 0.330 | 19.9% |
| 4 | 0.482 | 34.9% |

Block 1 has 60% of readings above spec — nearly double Block 4's rate. This is the earliest data (2023). Either the column was being operated more poorly in 2023, or the operating conditions were genuinely harder (higher feed, lower steam/reflux ratios).

Checking the per-block process stats:
- Block 1 Feed_Flow mean: **85.8 TPH** vs Block 4: **78.6 TPH** — Block 1 was running higher throughput
- Block 1 Reflux_Flow mean: **95.7 TPH** vs Blocks 2/3: **102.2 / 101.9 TPH** — lower reflux in Block 1
- Block 1 Reboiling_Steam mean: **21.3 TPH** vs Blocks 2/3: **23.8 / 24.3 TPH** — significantly less steam

**This physically explains the higher slippage in Block 1**: more feed, less steam, less reflux = worse separation. The model should learn this pattern from the data.

**Implication for train/test split**: We had planned Block 1+2 train, Block 4 test. Given that Block 1 has systematically different operating conditions *and* different C4 behavior, keeping it in training is correct — it gives the model exposure to high-slippage operating conditions that Block 4 also shows 35% of the time.

### F6. Winsorisation Clipped Exactly 114 Rows Per Column

Every single column had exactly 114 rows clipped (57 at the low end for some, 57 at the high end for others, or 114 at one end and 0 at the other). This is unlikely to be a coincidence. These 114 rows are likely the **same rows** — a cluster of process upsets or abnormal operating periods that manifest as extreme values across all variables simultaneously.

**Action for feature engineering**: Flag these 114 rows as `is_extreme_event = True`. Check whether C4 slippage during these periods is systematically different. If these are real upset events, the model needs to learn from them; if they are sensor errors, they should be excluded.

### F7. Sampling Is Perfect Within Blocks (With 2 Minor Exceptions)

- Blocks 1, 2, 3: **zero** non-1h intervals — perfectly regular hourly data
- Block 4: **2 gaps** of up to 10 hours

The 2 gaps in Block 4 are manageable. Lag features touching those gaps will be NaN and excluded during training (standard time-series treatment).

### F8. Three Bugs Fixed After Initial Run — Final Parquet Has 24 Columns

After the first clean run was inspected, three issues were identified and fixed before Phase 1 was locked.

**Bug 1 — Floating-point equality in stuck detection (real bug, not style):**
The original `series.diff().eq(0)` misses cases where a historian (Exaquantum, OSI PI) writes the same reading with tiny floating-point noise — e.g., `0.07980082929134369` vs `0.0798008292913436`. Replaced with `np.isclose()` using a relative tolerance of `1e-6`.

Impact measured: C4H6 stuck count increased from **4,125 → 4,140 rows** (+15 rows that were being silently missed). Small but real.

**Bug 2 — Exact-zero shutdown detection:**
The original `== 0` comparison misses soft shutdowns where historian writes `0.001` during ramp-down. Replaced with `< SHUTDOWN_THRESHOLD` (currently 0.5 TPH/degC).

Impact on this dataset: still **56 rows removed** (confirming those rows were hard zeros). The fix future-proofs against different historian export settings.

**Addition — `Analyzer_Health` categorical column:**
Derived from `max(hours_since_C4H6_change, hours_since_C4H8_change)` and bucketed:

| Status | Threshold | Rows | % of dataset |
|--------|-----------|------|--------------|
| GOOD | Both changed within 12h | 7,928 | **69.9%** |
| WARNING | At least one unchanged 12-24h | 579 | 5.1% |
| BAD | At least one unchanged >24h | 2,836 | **25.0%** |

**25% of the dataset has at least one analyzer in BAD state.** This is immediately useful to operators as a standalone indicator on the dashboard, independent of C4 prediction. Costs nothing to compute since `hours_since_*` is already produced by stuck detection.

---

## Model Constraints & Operating Limits

> [!CAUTION]
> The optimizer must **never** recommend values outside safe operating limits. Safety constraints are **absolute ceilings** — they override the optimization objective.

### Safety Ceilings (Hard-Coded, Non-Negotiable)

| Variable | Safety Ceiling | Consequence If Exceeded |
|----------|---------------|-------------------------|
| **Column Bottom Temp** | **<= 115 degC** | Alarm / potential shutdown |
| **Column Top Pressure** | **<= 5.0 kg/cm2g** | Pressure trip / emergency depressuring |

```python
# Hard-coded in optimizer - NOT configurable, NOT overridable
SAFETY_CEILINGS = {
    'Column_Bottom_Temp': 115.0,   # degC - absolute max, alarm threshold
    'Column_Top_Pressure': 5.0,    # kg/cm2g - trip threshold
}
```

### Manipulable Variables (Optimizer Can Adjust)

| Variable | Unit | Hard Min | Rec. Min | Mean | Rec. Max | Hard Max | Max d/hr |
|----------|------|----------|----------|------|----------|----------|----------|
| Reboiling Steam Flow | TPH | 14.4 | 18.0 | 21.0 | 24.4 | 25.3 | +/-2.0 |
| Reflux Flow | TPH | 70.4 | 80.0 | 91.1 | 103.9 | 105.7 | +/-5.0 |

### Monitored Variables (Optimization Constraints)

| Variable | Unit | Rec. Min | Mean | Rec. Max | Safety Ceiling |
|----------|------|----------|------|----------|----------------|
| Column Bottom Temp | degC | 102.6 | 107.0 | 111.5 | **115.0 (absolute)** |
| Control Tray Temp | degC | 59.1 | 71.6 | 89.7 | -- |
| Column Top Pressure | kg/cm2g | 3.85 | 4.05 | 4.45 | **5.0 (absolute)** |

### Clarification Needed: "+/-50%" Constraint

> [!WARNING]
> Stakeholders mentioned "plus or minus 50%" for model recommendations. Three possible interpretations:
>
> | Interpretation | Steam Example (current = 21 TPH) | Effective Range |
> |---------------|----------------------------------|----------------|
> | +/-50% of **current value** | 21 +/- 10.5 | 10.5 - 31.5 TPH |
> | +/-50% of **operating range** (18-24.4) | midpoint +/- 3.2 | 17.9 - 24.3 TPH |
> | +/-50% of **mean** | 21 +/- 10.5 | 10.5 - 31.5 TPH |
>
> **Current implementation**: Data-derived P5-P95 ranges with +/-2 TPH/hr rate limit. Confirm with IOCL which interpretation is intended.

---

## STATUS: Phase 2 Locked

**Output**: `data/features.parquet` — 11,343 rows × **92 columns** (final, no further changes)
**Audit scripts**: `notebooks/analyze_phase2.py`, `notebooks/verify_features.py`, `notebooks/check_extreme_rows.py`

> [!NOTE]
> Phase 2 is closed. Feature set verified: no block leakage, 664 NaN cells all at block boundaries (correct), extreme events flagged correctly.

---

## Phase 2 Findings: What Feature Engineering Revealed

### F9. Extreme Events Are 1,392 Rows — Not 114

The plan said "114 winsorised rows" because the per-tail, per-column clip count is 114. But 114 is not the number of unique rows affected — it is the count per tail per column. Because each column clips different rows at each tail, the union of all clipped rows is **1,392 rows (12.3% of the dataset)**.

| Block | Extreme Events | % of Block |
|-------|---------------|------------|
| 1     | 480           | 14.6%      |
| 2     | 237           | **32.1%**  |
| 3     | 209           | **26.0%**  |
| 4     | 466           | 7.2%       |

**Key observations:**
- Blocks 2 and 3 have a disproportionately high extreme-event rate (32% and 26%) despite being the shortest blocks. This may indicate the plant was in a transitional or unstable operating period after the 376-day gap in 2024.
- C4 during extreme events: mean **0.622 wt%** (vs. 0.479 for normal rows) — extreme events have ~30% higher C4 on average.
- 47.7% of extreme-event rows exceed the 0.5 spec limit vs. 38.8% for normal rows.
- Extreme events are spread across all 4 blocks (not concentrated in one block), so they likely represent real process upsets, not sensor errors.

**Decision: Keep all 1,392 extreme-event rows in training.** The model needs to learn behaviour at operating boundaries. The `is_extreme_event` flag is preserved for post-hoc analysis and potential sample-weighting experiments.

### F10. Temp_Gradient is Bimodal — It is a Regime Indicator, Not a Continuous Feature

The `Temp_Gradient = Column_Bottom_Temp − Column_Top_Temp` has a striking bimodal distribution:

```
P50: -0.112   (near-zero — cold regime, Bottom ≈ Top)
P60: 64.334   (large positive — hot regime, Bottom >> Top)
```

P40 is -0.26, P60 is 64.3 — the entire middle 20% of the distribution jumps 64°C. This is because:
- **Block 1 (cold)**: Temp_Gradient mean = **-0.39°C** (bottom and top nearly equal — column not fully operational)
- **Block 2 (hot)**: Temp_Gradient mean = **+70.21°C** (normal fractionation — large thermal gradient)

This means Temp_Gradient provides very strong regime information for the tree model, but is essentially a restatement of `Data_Block` for Blocks 1 and 2. XGBoost will handle this via natural splits, but it's worth knowing the feature is not continuously informative.

Similarly, **Reboiler_Delta** (Reboiler_Outlet_Temp − Column_Bottom_Temp):
- Block 1: mean **-70.46°C** (reboiler outlet much colder than column bottom — abnormal)
- Block 2: mean **-0.30°C** (reboiler outlet ≈ column bottom — normal tight control)

This extreme regime difference across blocks further confirms that `Data_Block` is the single most important splitting feature for the tree.

### F11. Strongest Correlations With Total_C4 Come From Ratios, Not Raw Variables

Linear (Pearson) correlations of Tier 1 features with `Total_C4`:

| Feature | Abs. Correlation |
|---------|------------------|
| `Steam_Feed_Ratio` | **0.3239** |
| `Reflux_Ratio` | **0.2847** |
| `Column_Top_Pressure` | 0.2422 |
| `Feed_Flow` | 0.2146 |
| `Reflux_Flow` | 0.0989 |
| `Reboiler_Outlet_Temp` | 0.0634 |
| `Column_Bottom_Temp` | 0.0181 |

**Key insight:** The two engineered ratios (`Steam_Feed_Ratio`, `Reflux_Ratio`) are the strongest linear predictors — stronger than any raw variable. This validates the feature engineering decision. Raw `Reboiling_Steam_Flow` has correlation 0.055 but `Steam_Feed_Ratio` has 0.324 — a **6× improvement** from normalising by feed.

However, linear correlations understate tree model performance. XGBoost will use the lags and rolling stats in non-linear combinations. Expect the model's effective signal to be significantly higher than 0.32.

### F12. NaN Profile Is Exactly as Expected — No Surprises

Total NaN cells: **664** across 51 columns.

- All NaNs are at block boundaries (first 12 rows of each block hold lags up to lag-12)
- Each block contributes exactly 12 NaN rows (one per block × 4 blocks = 48 NaN-affected rows total)
- No NaNs in any other position — confirms within-block lag/rolling logic is correct
- Rows with any NaN per block: exactly 12 per block (same 12 rows — the first N rows where N = max lag = 12)
- These 48 rows will be dropped during model training (`dropna()`) — immaterial given training set sizes

**Verification confirmed:** No data leakage across block boundaries. Lag-1 and rolling-std values are NaN for every first row of each block.

### F13. Confirmed Training Set Sizes (After All Filters + NaN Drop)

| Model | Train (Blocks 1-3) | Test (Block 4) | Notes |
|-------|-------------------|----------------|-------|
| **Model A** (C4H8) | **4,332 rows** | **6,081 rows** | Excludes C4H8_stuck rows; drops 12 NaN rows per block |
| **Model B** (C4H6) | **3,556 rows** | **2,974 rows** | Excludes C4H6_stuck AND C4H6 < 0.001; smaller valid set |

> [!WARNING]
> Model B's training set (3,556 rows) is ~18% the size of Model A's. XGBoost's default `n_estimators=100` may be sufficient, but use `early_stopping_rounds=30` and monitor validation loss carefully. Stronger regularisation (`reg_alpha=1`, `reg_lambda=5`, lower `learning_rate=0.05`) recommended as starting point.

Extreme events in Model A training set: **903 rows (20.8%)** — well-distributed, will be kept.

### F14. Tier 1 vs Tier 2 Feature Sets Confirmed

| Tier | Features | Description |
|------|----------|-------------|
| **Tier 1** (Production) | **67** | Process-only: current values + lags + rolling + ratios + diffs + time + block |
| **Tier 2** (Research) | **82** | Tier 1 + 15 target lags (C4H8 lag 1-12, C4H6 lag 1-3) |

The 15 additional target-lag columns are already in `features.parquet` but will only be selected during Tier 2 training.

### F15. Operator Instability Captured by Rolling Std Features

Rolling std of key manipulable variables (6h window):

| Feature | Mean | Std | P95 (high instability) |
|---------|------|-----|------------------------|
| `Reboiling_Steam_Flow_roll_std_6h` | 0.257 TPH | 0.241 | **0.703 TPH** |
| `Reflux_Flow_roll_std_6h` | 0.273 TPH | 0.698 | **1.458 TPH** |
| `Feed_Flow_roll_std_6h` | 2.131 TPH | 1.890 | **6.216 TPH** |

These features capture **operator behaviour patterns** — erratic steam or reflux adjustments over 6 hours signal poor control. P95 of Feed instability is 6.2 TPH/6h — at those moments, feed variability is 3× the mean steam flow. The dashboard can surface this as a direct operator coaching metric.

---

## Phase 2 Outputs (Locked)

**Feature set produced by `feature_engineering.py`:**

| Category | Planned | Actual | Notes |
|----------|---------|--------|-------|
| Core process vars | 8 | 8 | ✓ |
| Process lags | ~32 | 32 | ✓ Steam×5, Reflux×4, Feed×3, BotTemp×3, CTTray×4, ReboilerTemp×3, TopTemp×2, TopPress×2 |
| Rolling means | 12 | 12 | ✓ |
| Rolling stds | 6 | 6 | ✓ |
| Engineered ratios | 4 | 4 | ✓ |
| Rate-of-change | 4 | 4 | ✓ |
| Time cyclical | 6 | 6 | ✓ |
| Block label | 1 | 1 | ✓ |
| **Tier 1 Total** | ~73 | **67** | Slight difference from plan: `Data_Block` counted as 1, no dummy encoding |
| Target lags (Tier 2) | 15 | 15 | C4H8×12 + C4H6×3 |
| **Tier 2 Total** | ~88 | **82** | ✓ |
| Metadata flags | — | 5 | `is_extreme_event` + stuck flags + hours_since + Analyzer_Health |

---

## STATUS: Phase 3 Complete (Partially — Critical Failure Diagnosed)

**Output**: `models/default_leaderboard.csv`, `models/model_A_C4H8.json`, `models/model_B_C4H6.json`, `models/training_metrics.csv`, `models/test_predictions.parquet`
**Audit scripts added**: `notebooks/inspect_training_shift.py`, `notebooks/experiment_features.py`, `notebooks/experiment_regimes.py`, `notebooks/diagnose_predictions.py`, `notebooks/inspect_hot_shift.py`, `notebooks/inspect_bias.py`

> [!CAUTION]
> **All Tier 1 (process-only) models fail on Block 4 test set.** This is not a tuning problem — it is a fundamental data problem. Optuna tuning of defaults is **BLOCKED** until the root cause is understood and a strategy is chosen. See F16–F20 below.

---

## Phase 3 Findings: What Model Training Revealed

### F16. Default Model Results (Block 4 Test Set)

#### Model A (C4H8) Baselines
| Baseline | R² | MAE | RMSE | Spec Recall | Analyzer? |
|---------|-----|-----|------|-------------|-----------|
| Overall Mean | -0.0034 | 0.2199 | 0.2762 | 100% | No |
| Block Mean | -0.0034 | 0.2199 | 0.2762 | 100% | No |
| **Naive Lag-1** | **0.9328** | **0.0361** | **0.0715** | **96.0%** | **Yes** |

#### Model A (C4H8) ML Models — Tier 1 (Process-Only)
| Model | R² | MAE | RMSE | Spec Recall | Analyzer? |
|-------|-----|-----|------|-------------|-----------|
| LinearRegression | -3.728 | 0.5035 | 0.5996 | 21.6% | No |
| Ridge (α=1.0) | -3.932 | 0.5082 | 0.6124 | 21.2% | No |
| RandomForest | -0.992 | 0.2987 | 0.3892 | 19.7% | No |
| XGBoost | -1.029 | 0.3017 | 0.3928 | 19.0% | No |

#### Model A (C4H8) — Tier 2 (With Target Lags, Requires Analyzer)
| Model | R² | MAE | RMSE | Spec Recall | Analyzer? |
|-------|-----|-----|------|-------------|-----------|
| **XGBoost Tier 2** | **0.7128** | **0.0995** | **0.1478** | **90.7%** | **Yes** |

#### Model B (C4H6) — Tier 1 & Tier 2
| Model | R² | MAE | RMSE | Pct ±0.1 | Analyzer? |
|-------|-----|-----|------|-----------|-----------|
| LinearRegression | -305.3 | 0.1629 | 0.1754 | 18.5% | No |
| Ridge | -298.6 | 0.1603 | 0.1735 | 20.6% | No |
| RandomForest | -6.249 | 0.0190 | 0.0270 | **99.9%** | No |
| XGBoost Tier 1 | -34.63 | 0.0550 | 0.0598 | 97.5% | No |
| XGBoost Tier 2 | -16.82 | 0.0369 | 0.0423 | 99.8% | Yes |

> [!NOTE]
> Model B's MAE of 0.019 wt% (RandomForest) seems excellent, but R² = -6.25 because Block 4 C4H6 baseline variance is only 0.0001 wt² — the target is nearly constant at ~0.006 wt%. The model cannot beat a mean prediction in such a narrow range.

#### Combined Total C4 Evaluation
| Model | R² | MAE | RMSE | Spec Recall | Analyzer? |
|-------|-----|-----|------|-------------|-----------|
| **Combined Naive Lag-1** | **0.9217** | **0.0487** | **0.0891** | **95.8%** | **Yes** |
| Combined XGBoost (Tier 1) | -1.052 | 0.3621 | 0.4561 | 0.8% | No |

---

### F17. Root Cause 1 — Covariate Shift (Extrapolation)

Between Block 3 (Nov 2024) and Block 4 (Aug 2025), the plant's operating envelope shifted significantly. The model is being asked to predict at conditions it has never seen:

| Feature | Train Min | Test Min | % Test Rows Out-of-Bounds |
|---------|-----------|----------|--------------------------|
| `Reflux_Flow` (all lags/rolling) | 85.0 TPH | 70.4 TPH | **38–54%** of test rows |
| `Column_Top_Temp` (all lags) | 29.35°C | 18.04°C | **34%** of test rows |
| `Temp_Gradient` | -4.29°C | upper extrapolates to 94.94°C | **24.5%** of test rows |
| `Reboiling_Steam_Flow` (rolling) | 18.73 TPH | 14.43 TPH | **14%** of test rows |
| `Feed_Flow` (rolling) | 63.39 TPH | 44.55 TPH | **6%** of test rows |

The reflux rate in Block 4 runs significantly lower than in any previous block. Tree-based models (XGBoost, RandomForest) will extrapolate the last seen leaf value when encountering out-of-bounds inputs — producing systematically biased predictions.

Key mean shifts between training and testing periods:
- Feed_Flow: -8.5 TPH (−10%)
- Reflux_Flow: -10.6 TPH (−11%)
- Column_Top_Temp: -20°C (−23%)
- Temp_Gradient: +21.4°C (+109%)
- Reboiler_Outlet_Temp: +15.2°C (+27%)

---

### F18. Root Cause 2 — Concept Drift (Correlation Sign Reversal)

The **relationships between process variables and C4H8** are not consistent between training blocks and Block 4. Several correlations literally reversed sign:

| Feature | Corr with C4H8 in Train | Corr with C4H8 in Test | Status |
|---------|------------------------|------------------------|--------|
| `Column_Top_Pressure` | +0.475 | +0.058 | Near-zero — lost signal |
| `Control_Tray_Temp` | **-0.367** | **+0.391** | **REVERSED** |
| `Reflux_Flow` | +0.084 | **-0.239** | **REVERSED** |
| `Column_Bottom_Temp` | -0.065 | **+0.223** | **REVERSED** |
| `Temp_Gradient` | -0.254 | **+0.166** | **REVERSED** |

This holds even **within the hot regime only** (Reboiler_Outlet_Temp ≥ 50°C):

| Feature | Train Hot | Test Hot | Status |
|---------|-----------|----------|--------|
| `Control_Tray_Temp` | -0.165 | **+0.360** | **REVERSED** |
| `Column_Top_Pressure` | +0.362 | -0.068 | Near-zero |
| `Reflux_Flow` | +0.110 | **-0.166** | **REVERSED** |

**Chemical Engineering Explanation**: The operating pressure dropped from 4.19 kg/cm²g (train) to 3.98 kg/cm²g (test). At lower column pressure, the bubble/dew point temperatures shift downward. A control tray temperature of 75°C at 3.90 bar corresponds to a heavier composition than 75°C at 4.15 bar. Without pressure-compensated temperature features, raw temperatures carry completely different composition signals across campaigns.

Additionally, the plant changed its operating strategy: Block 4 uses lower reflux (~87 TPH vs ~102 TPH) at lower steam (~20.4 TPH vs ~22.2 TPH) and lower feed (~78.6 vs ~87.1 TPH) simultaneously — a different throughput-efficiency trade-off than 2023/2024 campaigns.

---

### F19. Root Cause 3 — C4H6 Target Collapse in Block 4

The C4H6 target distribution in Block 4 is qualitatively different from training blocks:

| Period | C4H6 Mean | C4H6 Std | % > 0.1 wt% |
|--------|-----------|----------|-------------|
| Train (Blocks 1-3) | 0.1395 | 0.1618 | Large fraction |
| Test (Block 4) | **0.0057** | **0.0100** | Near zero |

Block 4's C4H6 is ~24× lower than the training mean. The range is so narrow (0.001–0.38 wt%, with P75 at 0.007 wt%) that no model can produce meaningful R² — any prediction within ±0.01 wt% of zero will be "accurate" by MAE but R² will be negative because the variance is essentially zero.

This may indicate one of:
1. A column operational improvement that permanently reduced C4H6 slippage (e.g. catalyst replacement, tray repair)
2. A feed composition change (less C4H6 in the feed in 2025–2026)
3. The C4H6 analyzer in Block 4 may still be partially stuck or recalibrated

> [!WARNING]
> **Model B (C4H6 predictor) cannot be meaningfully trained or evaluated until the Block 4 C4H6 situation is clarified with IOCL.** If Block 4's C4H6 is genuinely near-zero due to operational improvements, then the production target itself may have changed and Total C4 ≈ C4H8 alone is sufficient.

---

### F20. Bias Diagnostic — Predictions Are Inversely Correlated with Truth

Running Pearson correlation between `y_test` and `pred_test`:

```
Pearson Correlation: -0.3263
```

The predictions are **inversely correlated** with the truth. When the model predicts high C4H8, the actual C4H8 is low. This is strong evidence that concept drift (correlation reversal) is the dominant effect, not just extrapolation.

Linear calibration experiment:
- After fitting a linear `y_actual ~ y_pred` calibrator, R² improves from -0.97 to **+0.11**
- This confirms there IS information in the predictions, but the model learned the wrong direction

---

### F21. Data_Block Feature — Marginal Benefit

Data_Block A/B experiment result:
- XGBoost WITH Data_Block: R² = -1.0287
- XGBoost WITHOUT Data_Block: R² = -1.0445
- Delta R² = 0.0158

Block label adds only 0.016 R² improvement and does not solve the generalization problem. The top feature in the trained model was `month_cos` (importance = 0.378), which the model latched onto as a proxy for campaign identity. This is indirect evidence of campaign memorization — the model learned "time of year" instead of "process physics."

---

### F22. Tier 2 Model Works — But Requires the Analyzer

XGBoost Tier 2 (with C4H8 target lags) achieves R² = 0.7128 on Block 4 because:
- `C4H8_Bottom_lag1` effectively acts as a **self-correcting bias term**
- It tells the model the current baseline level of C4H8 before prediction
- This absorbs the campaign-level mean shift that the process variables cannot compensate for
- Without the analyzer lag, the model is "flying blind" across the 258-day gap

This quantifies the value of the working analyzer:
- **Tier 1 (process-only)**: R² ≈ -1.03 (useless in production)
- **Tier 2 (with analyzer lag-1)**: R² = 0.7128 (useful)
- **Gap**: 1.74 R² points — the analyzer provides the campaign-baseline correction

---

## Phase 3 — Current Recommended Next Steps

### Option A: Domain Adaptation (Recommended)

Build a **mean-corrected soft sensor** that adjusts for campaign baseline shifts without requiring the full analyzer history:

1. **Pressure-compensated temperature features**: Add `Column_Top_Temp_adj = Column_Top_Temp - f(Column_Top_Pressure)` — this normalizes temperatures to a common pressure baseline, potentially restoring the correlations across campaigns.

2. **Campaign mean correction layer**: During deployment, use the last N available analyzer readings (even if sparse) to estimate a campaign bias offset. Apply this as a post-processing correction to Tier 1 predictions. This is a practical soft-sensor design for IOCL: even if the analyzer is broken for 30 days, its last valid reading provides an anchor for the bias layer.

3. **Rolling mean residual feature**: Add `C4H8_rolling_mean_24h` as a feature — not the lag-1 (which requires working analyzer every hour) but a rolling mean over the last 24 valid readings. This captures campaign baseline without strict lag-1 dependency.

### Option B: Accept Tier 2 as Production Model

If IOCL can guarantee the analyzer provides at least one valid reading per day (even if unreliable hourly), Tier 2 achieves R² = 0.7128. The framing changes:
- **"Soft sensor (Tier 2)"**: Works with daily analyzer updates, not hourly
- **"Soft sensor (Tier 1)"**: Fallback for fully broken analyzer periods

### Option C: Retrain with Block 4 Data (When Available)

If Block 4 represents the new normal operating regime, the model should be retrained with Block 4 data included in training. This requires IOCL to provide fresh labeled data from the current campaign before deployment.

---

## Phase 3 — File Inventory (Completed)

| File | Purpose | Status |
|------|---------|--------|
| `model_training.py` | Phase 3 pipeline | **Done** |
| `models/default_leaderboard.csv` | Default model comparison (pre-tuning) | **Done** |
| `models/model_A_C4H8.json` | Default XGBoost for C4H8 (not tuned) | Done (not production-ready) |
| `models/model_B_C4H6.json` | Default XGBoost for C4H6 (not tuned) | Done (not production-ready) |
| `models/training_metrics.csv` | All model metrics | **Done** |
| `models/test_predictions.parquet` | Test set predictions | **Done** |
| `notebooks/inspect_training_shift.py` | Covariate shift analysis | **Done** |
| `notebooks/experiment_features.py` | Feature ablation study | **Done** |
| `notebooks/experiment_regimes.py` | Hot/cold regime isolation | **Done** |
| `notebooks/diagnose_predictions.py` | Out-of-bounds feature detection | **Done** |
| `notebooks/inspect_hot_shift.py` | Hot regime distribution shift | **Done** |
| `notebooks/inspect_bias.py` | Correlation & bias analysis | **Done** |
| `notebooks/terminal_outputs.md` | All terminal outputs recorded | **Done** |

---

## Phase 4 Plan: Adaptive Modeling Strategy & Concept Drift Mitigation

**Status: Planning stage — awaiting user approval.**

We will address the concept drift (correlation sign reversal) and covariate shift (extrapolation) by implementing a systematic set of preprocessing and feature-engineering enhancements, followed by a sequence of target ablation and domain adaptation experiments.

---

### Step 4.1 — Feature Engineering & Preprocessing Enhancements

To strengthen the features and prevent campaign memorization, we will implement the following changes in `feature_engineering.py`:

#### 1. Preprocessing: Gap-Aware Lag Protection
Within Block 4, there are two time gaps of up to 10 hours. To prevent `lag1` or other lags from referencing data across these gaps (e.g. associating a row at 9:00 PM with a row at 11:00 AM after a 10-hour gap), we will implement a robust resampling-based shift method:
- For each `Data_Block`, we temporarily set `DateTime` as the index and resample to a strict 1-hour grid (`asfreq('1h')`).
- Lags and rolling features are computed on this continuous hourly grid, automatically producing `NaN`s for missing hours and correctly propagating them through subsequent lags.
- The grid is then reindexed back to the original timestamps, dropping any artificially inserted rows. This ensures the correct number of rows is maintained while making all lag features gap-aware.

#### 2. Feature Engineering: Pressure-Normalized Temperatures
To address the bubble/dew point shifts caused by operating pressure differences across blocks (from mean 4.19 to 3.98 kg/cm²g), we will create pressure-normalized versions of the key temperatures using three systematic forms:
- **Form 1: Linear Subtraction**
  $$T_{norm} = T - (P_{Top} - P_{ref}) \times k$$
  Using $P_{ref} = 4.05$ (training mean pressure) and testing $k \in \{3.0, 5.0, 10.0\}$.
- **Form 2: Ratio**
  $$T_{ratio} = \frac{T}{P_{Top}}$$
- **Form 3: Gradient Normalized**
  $$TGrad_{norm} = \frac{Column\_Bottom\_Temp - Column\_Top\_Temp}{P_{Top}}$$

We will evaluate Form 1 (at all three $k$ values), Form 2, and Form 3. The correct normalization is the one that makes the train/test correlation of the normalized temperature vs $C4H8$ most consistent across blocks.

#### 3. Feature Engineering: Remove Campaign Proxies
We will remove features that allow the model to memorize campaigns rather than learning physics:
- **Calendar features**: `month_sin`, `month_cos` (and optionally check day-of-week if it acts as a proxy).
- **Regime features**: `Data_Block`, `Temp_Gradient`, `Reboiler_Delta` (these will be removed or replaced by their pressure-normalized variants).

#### 4. Feature Engineering: Pressure Interaction Features
Tree models can utilize interaction features to represent temperature behavior shifting with pressure:
- `Pressure_x_TopTemp`
- `Pressure_x_BottomTemp`
- `Pressure_x_ControlTrayTemp`

#### 5. Feature Engineering: Relative-to-Rolling-Baseline Features
To help the model adapt to campaign drift, we will compute deviations from 24-hour rolling averages rather than absolute values:
- $Var_{dev} = Var - Var_{rolling\_mean\_24h}$ for Steam, Reflux, Bottom Temp, Control Tray Temp, and Top Pressure.

---

### Step 4.2 — Systematic Experiment Sequence

We will train Model A ($C4H8$) XGBoost on the training set (Blocks 1-3) and evaluate on the Block 4 test set. We will track the Pearson correlation of prediction vs actuals as our primary metric (to restore directionality from $-0.326$ to positive) and $R^2$ as our secondary metric.

We will run these experiments in order:

#### Phase A: Campaign Proxy Ablation
1. **Experiment 1 (No Calendar)**: Remove only calendar proxies (`month_sin`, `month_cos`).
2. **Experiment 2 (No Regime)**: Remove only regime proxies (`Data_Block`, `Temp_Gradient`, `Reboiler_Delta`).
3. **Experiment 3 (All Removed)**: Remove both calendar and regime proxies together.
   - *Goal*: Verify which proxies were responsible for the negative correlation, check what becomes the new top feature to ensure no surrogate campaign proxy was learned (e.g. lag of calendar features).

#### Phase B: Physical Normalization & Interactions
4. **Experiment 4 (Pnorm Linear k=3)**: Add linear pressure-compensated temperatures with $k=3.0$.
5. **Experiment 5 (Pnorm Linear k=5)**: Add linear pressure-compensated temperatures with $k=5.0$.
6. **Experiment 6 (Pnorm Linear k=10)**: Add linear pressure-compensated temperatures with $k=10.0$.
7. **Experiment 7 (Pnorm Ratio)**: Add temperature-to-pressure ratio features.
8. **Experiment 8 (Pnorm Gradient)**: Add pressure-normalized temperature gradients.
9. **Experiment 9 (Pressure Interactions)**: Add pressure-temperature interaction features.
10. **Experiment 10 (Rolling Deviations)**: Add relative-to-rolling-baseline features.

#### Phase C: Campaign Anchor
11. **Experiment 11 (Campaign Anchor)**: Once Pearson correlation is positive and stable, introduce the sparse campaign anchor (last valid analyzer reading, carried forward up to 72 hours) and measure the additional lift.

---

### Step 4.3 — Verification Metric Tracking Table

We will construct and populate this tracking table across all runs:

| Experiment | Pearson (Test) | $R^2$ (Test) | MAE (Test) | Top Feature | Notes |
|------------|----------------|--------------|------------|-------------|-------|
| **Baseline** | -0.1785 | -1.0363 | 0.3010 wt% | `month_cos` (0.378) | Default XGBoost |
| **Exp 1: No Calendar** | -0.3139 | -0.9674 | 0.2930 wt% | `Control_Tray_Temp_lag1` (0.111) | Removing time proxies |
| **Exp 2: No Regime** | -0.2075 | -0.8816 | 0.2820 wt% | `month_cos` (0.303) | Removing block/gradient proxies |
| **Exp 3: All Removed** | -0.3340 | -0.9693 | 0.2891 wt% | `Control_Tray_Temp_lag1` (0.103) | Pure physics-only baseline |
| **Exp 4: Pnorm (k=3)** | -0.2676 | -0.7295 | 0.2683 wt% | `Control_Tray_Temp_Pnorm_k3` (0.172) | Linear top pressure correction |
| **Exp 5: Pnorm (k=5)** | -0.3383 | -1.0502 | 0.2901 wt% | `Control_Tray_Temp_Pnorm_k5` (0.175) | Linear top pressure correction |
| **Exp 6: Pnorm (k=10)** | -0.3546 | -0.8705 | 0.2804 wt% | `Column_Bottom_Temp_Pnorm_k10` (0.250) | Linear top pressure correction |
| **Exp 7: Pnorm Ratio** | -0.3847 | -1.1232 | 0.2985 wt% | `Column_Bottom_Temp_Pratio` (0.216) | Temperature / Pressure ratios |
| **Exp 8: Pnorm Gradient** | -0.3501 | -1.0454 | 0.2948 wt% | `Control_Tray_Temp_lag1` (0.108) | Temp gradient / Pressure ratio |
| **Exp 9: Pressure Interactions** | -0.3506 | -0.9841 | 0.2912 wt% | `Pressure_x_TopTemp` (0.196) | Interaction terms |
| **Exp 10: Rolling Deviations** | -0.3262 | -0.8522 | 0.2784 wt% | `Control_Tray_Temp_lag1` (0.109) | 24h rolling deviations |
| **Exp 11: Campaign Anchor** | **+0.8595** | **+0.6865** | **0.1032 wt%** | `C4H8_campaign_anchor` (0.538) | Leak-free (shifted by 1) |

---

## Open Questions for IOCL (Updated)

> [!IMPORTANT]
> These questions block Phase 4 design decisions:

1. **Block 4 C4H6**: C4H6 in Block 4 is ~24× lower than training data (mean 0.0057 vs 0.14 wt%). Is this a real operational improvement, or a measurement/calibration issue with the C4H6 analyzer in Block 4?

2. **Pressure change**: Column top pressure dropped from mean 4.19 (train) to 3.98 kg/cm²g (test). Was this intentional? Was there a setpoint change or control system reconfiguration between 2024 and 2025?

3. **Reflux reduction**: Average reflux in Block 4 is ~87 TPH vs ~102 TPH in earlier blocks — 15% lower. Was this an intentional energy-saving operating strategy change?

4. **Training data availability**: Can IOCL provide labeled data from Block 4 (with valid analyzer readings) for retraining? Even 2–3 weeks of representative Block 4 data in training would likely resolve the generalization issue.

5. **Sparse anchor acceptable**: Is a deployment model acceptable that requires at least one valid analyzer reading per 72 hours (not continuous hourly) to work effectively? This enables the campaign anchor feature without depending on hourly availability.

---

## File Inventory (Full — All Phases)

| File | Purpose | Phase | Status |
|------|---------|-------|--------|
| `data_preprocessing.py` | Phase 1 pipeline | 1 | ✅ Done |
| `data/clean_data.parquet` | 11,343 rows × 24 cols | 1 | ✅ Done |
| `feature_engineering.py` | Phase 2 pipeline | 2 | ✅ Done |
| `data/features.parquet` | 11,343 rows × 92 cols | 2 | ✅ Done |
| `model_training.py` | Phase 3 pipeline | 3 | ✅ Done |
| `models/default_leaderboard.csv` | Default model comparison | 3 | ✅ Done |
| `models/model_A_C4H8.json` | XGBoost C4H8 (default, not prod-ready) | 3 | ✅ Done |
| `models/model_B_C4H6.json` | XGBoost C4H6 (default, not prod-ready) | 3 | ✅ Done |
| `models/training_metrics.csv` | All metrics summary | 3 | ✅ Done |
| `models/test_predictions.parquet` | Block 4 predictions | 3 | ✅ Done |
| `notebooks/terminal_outputs.md` | All Phase 3 terminal outputs | 3 | ✅ Done |
| `notebooks/inspect_training_shift.py` | Train vs test distribution | 3 | ✅ Done |
| `notebooks/experiment_features.py` | Feature ablation | 3 | ✅ Done |
| `notebooks/experiment_regimes.py` | Regime isolation | 3 | ✅ Done |
| `notebooks/diagnose_predictions.py` | OOB feature detection | 3 | ✅ Done |
| `notebooks/inspect_hot_shift.py` | Hot regime shift analysis | 3 | ✅ Done |
| `notebooks/inspect_bias.py` | Correlation/bias analysis | 3 | ✅ Done |
| `optimizer.py` | Phase 5 | 5 | ⏳ Pending |
| `drift_detection.py` | Phase 6 | 6 | ⏳ Pending |
| `app.py` | Dashboard | 7 | ⏳ Pending |
| `config.py` | Central config + constraints | 7 | ⏳ Pending |
