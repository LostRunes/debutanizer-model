# 05. Model B (C4H6_Bottom) Development

## 1. Mismatch & Target Collapse
*   **The Baseline Failure**: ML models trained on all process features performed very poorly, scoring $R^2 \approx -34.6$.
*   **The Mismatch**: C4H6 composition drops from $0.14$ wt% in Block 1 to just **$0.0057$ wt%** in Block 4. 

## 2. The Power of the Campaign Anchor
Due to the microscopic concentration, butadiene slippage behaves as a slow-moving, flat state variable rather than a rapid process response.
*   **Baseline Audit**: Evaluating the 1-hour-shifted campaign anchor (`C4H6_campaign_anchor_12h`) alone achieves:
    *   **$R^2$ Score**: **0.9606**
    *   **MAE**: **0.0005 wt% (5.5 ppm)**
    *   **Pearson Correlation**: **+0.9830**
*   This proved that almost all predictive information is contained in the analyzer's memory.

## 3. The Delta Correction Experiment
We evaluated whether a machine learning model could learn to predict high-frequency corrections (Deltas) around the anchor to improve metrics:
$$\text{Delta} = C_4H_6\text{ Bottom} - \text{C4H6 Campaign Anchor}$$
*   **Result**: 
    *   The delta model **degraded performance** compared to the raw anchor: $R^2$ dropped to **0.9010** and MAE doubled to **11.9 ppm**.
    *   Process flows and temperatures introduced fitting noise rather than genuine separation signal.
*   **Decision**: Bypassed machine learning entirely for Model B. Deployed a deterministic, analyzer-tracking state estimator.

## 4. Robustness checks across blocks (12h Limit Anchor)
To confirm robustness, we evaluated the anchor performance across the older campaigns:
*   **Block 2 (Target Mean = 0.0314 wt%)**: $R^2 = \mathbf{0.7518}$ | $MAE = \mathbf{42.9}$ ppm | Coverage = $99.71\%$
*   **Block 3 (Target Mean = 0.0234 wt%)**: $R^2 = \mathbf{0.7651}$ | $MAE = \mathbf{48.5}$ ppm | Coverage = $99.41\%$
*   **Block 4 (Target Mean = 0.0057 wt%)**: $R^2 = \mathbf{0.9606}$ | $MAE = \mathbf{5.5}$ ppm | Coverage = $98.45\%$
