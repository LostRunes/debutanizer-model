import pandas as pd

df = pd.read_parquet("data/features.parquet")
mA_filter = ~df["C4H8_Bottom_stuck"]

train_df = df[df["Data_Block"].isin([1, 2, 3]) & mA_filter]
test_df = df[(df["Data_Block"] == 4) & mA_filter]

TARGET_LAG_COLS = (
    [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
    [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
)
TIME_PROXIES = ["month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"]
META_COLS = [
    "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
    "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
    "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
    "Analyzer_Health", "is_extreme_event", "Data_Block"
] + TARGET_LAG_COLS + TIME_PROXIES

physics_feats = [c for c in df.columns if c not in META_COLS]
temp_related_cols = [c for c in physics_feats if any(tk in c for tk in ["Temp", "Delta", "Gradient"])]
physics_no_temps = [c for c in physics_feats if c not in temp_related_cols]

corrs = []
for col in physics_no_temps:
    tr_corr = train_df[col].corr(train_df["C4H8_Bottom"])
    te_corr = test_df[col].corr(test_df["C4H8_Bottom"])
    corrs.append({
        "Feature": col,
        "Train_Corr": tr_corr,
        "Test_Corr": te_corr,
        "Diff": abs(tr_corr - te_corr),
        "Same_Sign": (tr_corr * te_corr) > 0
    })

corrs_df = pd.DataFrame(corrs).sort_values(by="Diff", ascending=False)
print("=== TOP 20 CORRELATION CHANGES (NO TEMP PROCESS FEATURES) ===")
print(corrs_df.head(20).to_string(index=False))

print("\n=== FEATURES WITH SIGN REVERSAL ===")
print(corrs_df[~corrs_df["Same_Sign"]].to_string(index=False))
