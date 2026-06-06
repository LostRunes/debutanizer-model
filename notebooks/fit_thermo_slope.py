import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_parquet("data/features.parquet")

# Filter for hot regime (Reboiler_Outlet_Temp >= 50) in Blocks 1-3 to get normal operating data
normal_mask = (df["Data_Block"].isin([1, 2, 3])) & (df["Reboiler_Outlet_Temp"] >= 50) & (~df["C4H8_Bottom_stuck"])
normal_df = df[normal_mask].dropna(subset=["Column_Top_Pressure", "Column_Bottom_Temp", "Control_Tray_Temp", "Column_Top_Temp"])

print(f"Number of normal operating rows: {len(normal_df)}")

for t_col in ["Column_Top_Temp", "Control_Tray_Temp", "Column_Bottom_Temp"]:
    X = normal_df[["Column_Top_Pressure"]].values
    y = normal_df[t_col].values
    
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    r2 = reg.score(X, y)
    
    print(f"Regression of {t_col} vs Column_Top_Pressure:")
    print(f"  Slope (dT/dP): {slope:.4f} °C / (kg/cm²g)")
    print(f"  Intercept:     {reg.intercept_:.4f}")
    print(f"  R² score:      {r2:.4f}")
    print("-" * 40)
