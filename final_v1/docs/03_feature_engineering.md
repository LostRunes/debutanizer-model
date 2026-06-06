# 03. Feature Engineering

## 1. Gap-Aware Lag & Rolling Windows
Because of large campaign gaps (ranging from 43 hours to 376 days), computing standard lag and rolling statistics directly on the raw row indices would incorrectly carry data across column shutdowns.
*   **Hourly Resampling**: Each block is resampled to a continuous hourly datetime grid. 
*   **Protected Lags**: Lags and rolling windows are calculated on the grid, ensuring gaps represent missing values (`NaN`).
*   **Reindexing**: The grid is reindexed back to the original timestamps, dropping the shutdown periods but preserving valid time-series gaps.

## 2. Mass & Energy Balance Ratios
Standard flow rates are strongly dependent on column throughput, which shifts significantly across campaigns. To normalize this, we compute dimensionless ratios:
*   **Reflux Ratio**: Captures column cooling input per feed mass unit.
$$\text{Reflux Ratio} = \frac{\text{Reflux Flow (TPH)}}{\text{Feed Flow (TPH)}}$$
*   **Steam-Feed Ratio**: Captures heat input per feed mass unit, representing energy balance.
$$\text{Steam Feed Ratio} = \frac{\text{Reboiling Steam Flow (TPH)}}{\text{Feed Flow (TPH)}}$$

## 3. Relative-to-Rolling-Baseline Features (`dev24h`)
Absolute temperatures, flows, and pressures drift over time due to seasonal variations, operator setpoint changes, or exchanger fouling. To bypass this, we compute deviations from recent rolling means:
*   **Formula**:
$$\text{Feature}_{\text{dev24h}} = \text{Feature}_t - \text{Rolling Mean (last 24h)}$$
*   **Variables**: Computed for Reboiling Steam Flow, Reflux Flow, Bottom Temperature, Control Tray Temperature, and Top Pressure. Deviations maintain consistent correlation signs with the target across all campaigns.

## 4. Pressure-Normalized Temperatures
According to thermodynamic laws, boiling points shift based on pressure (bubble/dew point behavior). A raw temperature value represents completely different compositions at different pressures.
*   **Thermodynamic Correction**: 
$$\text{Temp}_{\text{Pnorm}} = \text{Temp}_t - (\text{Column Top Pressure}_t - P_{\text{ref}}) \times k$$
*   We use a reference pressure $P_{\text{ref}} = 4.05\text{ kg/cm}^2\text{g}$ (the training mean) and fit $k$-factors ($3, 5, 10$ °C/bar) matching bubble point slopes.
*   **Gradient Normalization**: Normalizes temperature differences across the column by pressure:
$$\text{Temp Gradient Pnorm} = \frac{\text{Column Bottom Temp} - \text{Column Top Temp}}{\text{Column Top Pressure}}$$
