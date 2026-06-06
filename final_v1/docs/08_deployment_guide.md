# 08. Deployment Guide

This guide details the integration, execution, and verification steps for deploying the Debutanizer virtual soft-sensor.

## 1. Prerequisites & Dependencies
The deployment environment must have Python 3.8+ installed with the following packages:
```bash
pip install pandas numpy xgboost scikit-learn
```

## 2. Directory Layout (`final_v1/`)
Ensure the following folder structure is deployed to the SCADA/DCS interface server:
```text
final_v1/
├── configs/
│   ├── model_A_features.json
│   └── model_B_features.json
├── models/
│   ├── model_A_final_v1.pkl
│   └── model_A_final_v1.json
├── inference/
│   ├── predict_c4h8.py
│   ├── predict_c4h6.py
│   └── predict_total_c4.py
└── notebooks/
    └── verify_anchor_leakage.py
```

## 3. SCADA / DCS API Integration
The virtual analyzer reads live process values hourly. In your SCADA/DCS data collection script, call the unified `predict_total_c4` function:

```python
import pandas as pd
from inference.predict_total_c4 import predict_total_c4

# 1. Retrieve the last 24 hours of hourly process data from historian
# Columns required: ['Feed_Flow', 'Reboiling_Steam_Flow', 'Reflux_Flow', 'Column_Bottom_Temp', 'Control_Tray_Temp', 'Column_Top_Pressure']
process_data_24h = pd.DataFrame({
    'Feed_Flow': [...],             # 24 floats
    'Reboiling_Steam_Flow': [...],  # 24 floats
    'Reflux_Flow': [...],           # 24 floats
    'Column_Bottom_Temp': [...],    # 24 floats
    'Control_Tray_Temp': [...],     # 24 floats
    'Column_Top_Pressure': [...]    # 24 floats
})

# 2. Retrieve last known analyzer readings and hours since they were updated
latest_c4h8 = {
    "value": 0.435,      # last valid laboratory or GC value
    "hours_ago": 3       # elapsed hours since that sample was taken
}

latest_c4h6 = {
    "value": 0.0045,
    "hours_ago": 3
}

# 3. Retrieve historical model predictions from the local database for fallback
prev_c4h8_predictions = [0.421, 0.433, 0.435]  # list of recent predictions
prev_c4h6_predictions = [0.0044, 0.0046, 0.0045]

# 4. Execute Prediction
prediction_result = predict_total_c4(
    process_history=process_data_24h,
    latest_valid_c4h8=latest_c4h8,
    latest_valid_c4h6=latest_c4h6,
    previous_c4h8_preds=prev_c4h8_predictions,
    previous_c4h6_preds=prev_c4h6_predictions,
    model_a_pkl_path="models/model_A_final_v1.pkl"
)

# Output structure
print("Prediction Health:", prediction_result["prediction_health"])
print("Predicted Total C4:", prediction_result["predicted_total_c4"])
```

## 4. Verification Check
Run the verification test suite directly on the target machine before going live:
```bash
python final_v1/inference/predict_total_c4.py
```
This should print a `GREEN` health status and show valid non-negative compositions.
