# 06. Drift Analysis

## 1. Covariate Shift (Input Extrapolation)
Between the training blocks (2023–2024) and Block 4 (2025–2026), the operating envelope of the column shifted significantly:
*   **Reflux Flow**: The average reflux rate dropped from $97.7$ TPH to $87.1$ TPH. In Block 4, **38% to 54%** of reflux measurements were completely out-of-bounds relative to the training history.
*   **Column Top Temp**: Running at lower pressures dropped the average top temperature from $87.1$ °C to $67.1$ °C. **34%** of test rows were out-of-bounds.
*   **Tree-Based Extrapolation Limit**: Trees predict flat leaves when inputs exceed training bounds, leading to systematically biased composition predictions.

## 2. Concept Drift (Correlation Sign Reversal)
Because boiling temperatures depend on pressure, raw temperature correlations with composition completely reversed sign across campaigns:

| Feature | Corr in Train (Blocks 1-3) | Corr in Test (Block 4) | Status |
| :--- | :---: | :---: | :--- |
| `Control_Tray_Temp` | **-0.367** | **+0.391** | **REVERSED** |
| `Reflux_Flow` | **+0.084** | **-0.239** | **REVERSED** |
| `Column_Bottom_Temp` | **-0.065** | **+0.223** | **REVERSED** |
| `Temp_Gradient` | **-0.254** | **+0.166** | **REVERSED** |

Without thermodynamic adjustments, any model trained on absolute temperatures was forced to make predictions in the wrong physical direction, leading to a negative Pearson correlation coefficient of **$-0.326$**.

## 3. Drift Resolution Rationale
To ensure the soft-sensor generalizes to unseen campaigns:
*   **Dimensionless Ratios**: Normalizing Reflux and Steam flows by Feed flow makes the model invariant to column throughput and scale changes.
*   **24-Hour Deviations (`dev24h`)**: Using deviations from recent rolling averages instead of absolute values bypasses setpoint drift and sensor fouling, retaining stable physical directions (e.g. steam deviation always correlates positively with stripping efficiency).
*   **Leak-Free Campaign Anchor**: Accounts for slow-moving feed composition shifts.
