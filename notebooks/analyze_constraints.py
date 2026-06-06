import pandas as pd
import numpy as np

df = pd.read_excel(r'9.DB DATA -B.xlsx', sheet_name='Sheet2')
df2 = df.iloc[2:].copy()
df2.columns = ['DateTime', 'Feed_Flow', 'Reboiler_Outlet_Temp', 'Column_Top_Temp',
                'Reboiling_Steam_Flow', 'Reflux_Flow', 'Column_Top_Pressure',
                'Column_Bottom_Temp', 'Control_Tray_Temp', 'C4H6_Bottom', 'C4H8_Bottom']
df2 = df2.reset_index(drop=True)
for col in df2.columns[1:]:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')
df2['DateTime'] = pd.to_datetime(df2['DateTime'], errors='coerce')

# Remove shutdown rows
mask = ~(df2[['Feed_Flow', 'Reboiler_Outlet_Temp', 'Column_Top_Temp',
              'Reboiling_Steam_Flow', 'Reflux_Flow']].sum(axis=1) == 0)
df2 = df2[mask].reset_index(drop=True)

# === DERIVE OPERATING LIMITS FROM DATA ===
# Use 1st-99th percentile as "normal operating range"
# Use min-max as "hard limits"
# Use 5th-95th as "recommended operating range"

manipulable = ['Reboiling_Steam_Flow', 'Reflux_Flow']
monitored = ['Feed_Flow', 'Reboiler_Outlet_Temp', 'Column_Top_Temp', 
             'Column_Top_Pressure', 'Column_Bottom_Temp', 'Control_Tray_Temp']

print("=" * 90)
print(f"{'Variable':<25} {'Min':>8} {'P1':>8} {'P5':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
print("=" * 90)

all_vars = manipulable + monitored
for var in all_vars:
    vals = df2[var]
    pcts = vals.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    marker = " <-- MANIPULABLE" if var in manipulable else ""
    print(f"{var:<25} {pcts.iloc[0]:8.2f} {pcts.iloc[1]:8.2f} {pcts.iloc[2]:8.2f} {pcts.iloc[3]:8.2f} {pcts.iloc[4]:8.2f} {pcts.iloc[5]:8.2f} {pcts.iloc[6]:8.2f} {pcts.iloc[7]:8.2f} {pcts.iloc[8]:8.2f}{marker}")

# === WHAT HAPPENS AT LOW/HIGH C4 SLIPPAGE? ===
df2['Total_C4'] = df2['C4H6_Bottom'] + df2['C4H8_Bottom']

print("\n\n=== OPERATING CONDITIONS WHEN C4 IS LOW (< 0.3 wt%) vs HIGH (> 0.8 wt%) ===")
low_c4 = df2[df2['Total_C4'] < 0.3]
high_c4 = df2[df2['Total_C4'] > 0.8]
print(f"\nLow C4 rows: {len(low_c4)}, High C4 rows: {len(high_c4)}")

print(f"\n{'Variable':<25} {'Low C4 Mean':>12} {'High C4 Mean':>13} {'Diff':>8} {'Direction':<15}")
print("-" * 75)
for var in all_vars:
    low_mean = low_c4[var].mean()
    high_mean = high_c4[var].mean()
    diff = high_mean - low_mean
    direction = "Higher -> More C4" if diff > 0 else "Lower -> More C4"
    print(f"{var:<25} {low_mean:12.2f} {high_mean:13.2f} {diff:+8.2f}  {direction}")

# === CROSS-CORRELATION BETWEEN MANIPULABLE VARS ===
print("\n\n=== CORRELATION BETWEEN STEAM AND REFLUX ===")
print(f"Correlation: {df2['Reboiling_Steam_Flow'].corr(df2['Reflux_Flow']):.4f}")

# === SAFE OPERATING ENVELOPE - SUGGESTED LIMITS ===
print("\n\n=== SUGGESTED OPERATING CONSTRAINTS FOR OPTIMIZER ===")
print("(Based on P5-P95 of historical data as 'safe range', P1-P99 as 'hard limits')")
for var in manipulable:
    p5 = df2[var].quantile(0.05)
    p95 = df2[var].quantile(0.95)
    p1 = df2[var].quantile(0.01)
    p99 = df2[var].quantile(0.99)
    mean = df2[var].mean()
    print(f"\n  {var}:")
    print(f"    Recommended range: {p5:.2f} - {p95:.2f}")
    print(f"    Hard limits:       {p1:.2f} - {p99:.2f}")
    print(f"    Mean:              {mean:.2f}")
    print(f"    +/- 50% of mean:   {mean*0.5:.2f} - {mean*1.5:.2f}")

for var in ['Column_Bottom_Temp', 'Control_Tray_Temp', 'Column_Top_Pressure']:
    p5 = df2[var].quantile(0.05)
    p95 = df2[var].quantile(0.95)
    p1 = df2[var].quantile(0.01)
    p99 = df2[var].quantile(0.99)
    mean = df2[var].mean()
    print(f"\n  {var}:")
    print(f"    Recommended range: {p5:.2f} - {p95:.2f}")
    print(f"    Hard limits:       {p1:.2f} - {p99:.2f}")
    print(f"    Mean:              {mean:.2f}")
