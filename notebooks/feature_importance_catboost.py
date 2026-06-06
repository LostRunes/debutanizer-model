"""
notebooks/feature_importance_catboost.py
========================================
Extracts and audits the top 20 features from the winning CatBoost model.
"""

import os
import pandas as pd
from catboost import CatBoostRegressor

# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
MODEL_FILE = os.path.join("models", "model_A_CatBoost.bin")

def main():
    df = pd.read_parquet(FEATURES_FILE)
    
    TARGET_LAG_COLS = (
        [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
        [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
    )
    META_COLS = [
        "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
        "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
        "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
        "Analyzer_Health", "is_extreme_event", "Data_Block",
    ] + TARGET_LAG_COLS
    
    feature_cols = [c for c in df.columns if c not in META_COLS]
    
    # Load CatBoost model
    model = CatBoostRegressor()
    model.load_model(MODEL_FILE)
    
    importances = model.get_feature_importance()
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    print("=" * 80)
    print("TOP 20 CATBOOST FEATURES IMPORTANCE")
    print("=" * 80)
    print(imp_df.head(20).to_string(index=False))
    print("=" * 80)
    
    # Let's write the classification audit
    audit = []
    for idx, row in imp_df.head(20).iterrows():
        feat = row["Feature"]
        imp = row["Importance"]
        
        # Classification
        classification = "Other Physics"
        if "Steam" in feat or "Steam_Feed" in feat:
            classification = "Physics (Steam)"
        elif "Reflux" in feat:
            classification = "Physics (Reflux)"
        elif "Pressure" in feat:
            classification = "Physics (Pressure)"
        elif "Bottom_Temp" in feat:
            classification = "Physics (Bottom Temp)"
        elif "month" in feat or "hour" in feat or "dow" in feat:
            classification = "Suspicious (Time/Calendar proxy)"
        elif "Temp_Gradient" in feat or "Reboiler_Delta" in feat:
            classification = "Regime/Gradient Indicator"
            
        audit.append({
            "Feature": feat,
            "Importance": round(float(imp), 4),
            "Classification": classification
        })
        
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv("experiments/catboost_feature_importance_audit.csv", index=False)
    print("\nFeature importance audit saved to experiments/catboost_feature_importance_audit.csv")

if __name__ == "__main__":
    main()
