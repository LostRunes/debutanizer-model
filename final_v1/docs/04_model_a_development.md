# 04. Model A (C4H8_Bottom) Development

## 1. Initial Failure & Campaign Memorization
*   **The Baseline Failure**: The baseline XGBoost trained on absolute values scored $R^2 \approx -1.03$ on the Block 4 test set. 
*   **Calendar Proxy Memorization**: SHAP analysis revealed the model was heavily relying on calendar features (`month_cos` importance = $0.378$) as surrogates for campaign labels, memorizing historical campaigns rather than learning physics.

## 2. The Control-Loop "Rot" Discovery
We discovered that column temperature correlations with pressure reverse signs between training and testing. 
*   **Cascade Controller Interference**: In normal operation, pressure build-ups trigger automated cascade controllers to cool the control tray, resulting in a large negative slope of **$-37.8$ °C/bar** between `Control_Tray_Temp` and pressure.
*   **The Overfitting Loop**: The model learned this control-loop correlation as physical separation truth, causing it to fail completely when operators shifted setpoints in Block 4.

## 3. Robust 8-Feature Physical Design
To eliminate control-loop dependencies, we restricted Model A to **8 process-only and dynamic calibration features**:
1.  `C4H8_campaign_anchor` (72h limit, Shift-1 leak-free)
2.  `Steam_Feed_Ratio`
3.  `Reflux_Ratio`
4.  `Reboiling_Steam_Flow_dev24h`
5.  `Reflux_Flow_dev24h`
6.  `Column_Bottom_Temp_dev24h`
7.  `Control_Tray_Temp_dev24h`
8.  `Column_Top_Pressure_dev24h`

## 4. Hyperparameter Optimization & Shallow Trees
We ran 50 Optuna trials using a 5-fold `TimeSeriesSplit` cross-validation on Train Blocks 1-3.
*   **Tuning Space**: Optimized XGBoost parameters (learning rate, estimators, tree depth, regularizations).
*   **Shallow Tree Constraint**: The optimal tree depth was found to be extremely shallow: **`max_depth = 3`**. This forces the trees to learn simple, monotonic physical relationships rather than complex, overfitted splits.

## 5. Final Results
*   **Validation Split (Blocks 1+2 $\rightarrow$ Block 3)**: $R^2 = \mathbf{0.7694}$ | MAE = **0.0817 wt%**
*   **Production Split (Blocks 1-3 $\rightarrow$ Block 4)**: $R^2 = \mathbf{0.9074}$ | MAE = **0.0516 wt%**
