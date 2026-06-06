# 09. Future Optimizer Integration

This document outlines how the soft-sensor predictions will interface with the downstream **Reflux & Steam Optimizer** on the column.

## 1. Optimization Objective
The primary economic goal is to **minimize reboiling steam consumption** (utility energy costs) while keeping the bottoms composition strictly within specification limits.

$$\text{Minimize } J = \text{Reboiling Steam Flow (TPH)}$$

$$\text{Subject to: } \text{Predicted Total C4}_t \le 0.40\text{ wt\%}$$

> [!NOTE]
> We recommend a safety margin threshold of **$0.40$ wt%** rather than the hard limit of $0.50$ wt% to protect against high-frequency process upsets and analyzer delays.

## 2. Decision Variables & Control Bounds
The optimizer can manipulate Reflux and Steam flows within safe operational bounds:

| Variable | Operating Min | Midpoint (Mean) | Operating Max | Rate-of-Change Limit |
| :--- | :---: | :---: | :---: | :---: |
| **Reboiling Steam Flow** | 18.0 TPH | 21.0 TPH | 24.4 TPH | $\pm 2.0$ TPH/hour |
| **Reflux Flow** | 80.0 TPH | 91.1 TPH | 103.9 TPH | $\pm 5.0$ TPH/hour |

## 3. Hard Safety Ceilings
The optimization algorithm must check and never exceed the following safety limits:
*   `Column_Bottom_Temp` $\le 115.0$ °C (Exceeding this triggers alarm/shutdown).
*   `Column_Top_Pressure` $\le 5.0\text{ kg/cm}^2\text{g}$ (Exceeding this triggers relief trip).

## 4. How the Optimizer Queries the Models
To optimize settings at hour $t+1$:
1.  **Simulate Process Iterations**: The optimizer generates candidate adjustments for Reflux Flow and Steam Flow within rate limits (e.g. $+1.0$ TPH reflux, $-0.5$ TPH steam).
2.  **Form Feature Vectors**:
    *   Compute the new dimensionless `Reflux_Ratio` and `Steam_Feed_Ratio` based on the candidate settings and current feed rate.
    *   Re-calculate `dev24h` deviations based on the candidate setting relative to the past 23-hour history.
    *   Maintain the current campaign anchor values `C4H8_campaign_anchor` and `C4H6_campaign_anchor` (which act as baseline constants during the optimization timestep).
3.  **Evaluate Predictions**: Query `predict_total_c4` to retrieve the virtual Total C4 composition.
4.  **Find the Optimal Settings**: The optimizer selects the candidate setting that minimizes steam flow while satisfying all constraints.

## 5. Feed Quality Adaptation
The biggest operational risk is unmeasured changes in the upstream feed composition. If the feed gets heavier, Butene slip will naturally increase. 
*   **The Anchor's Role**: The campaign anchor represents the current feed quality baseline. When a new analyzer reading arrives, the anchor shifts.
*   **Self-Correction**: This shifts the model's base prediction level, forcing the optimizer to increase reflux or steam immediately to restore specification limits, creating a closed-loop control system.
