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
df2['Total_C4'] = df2['C4H6_Bottom'] + df2['C4H8_Bottom']

# Remove shutdown rows (all process vars = 0)
mask = ~(df2[['Feed_Flow', 'Reboiler_Outlet_Temp', 'Column_Top_Temp',
              'Reboiling_Steam_Flow', 'Reflux_Flow']].sum(axis=1) == 0)
df2 = df2[mask].reset_index(drop=True)
print(f"After removing shutdowns: {len(df2)} rows")

# ============ LAGGED CORRELATIONS ============
process_vars = ['Feed_Flow', 'Reboiler_Outlet_Temp', 'Column_Top_Temp',
                'Reboiling_Steam_Flow', 'Reflux_Flow', 'Column_Top_Pressure',
                'Column_Bottom_Temp', 'Control_Tray_Temp']

print("\n=== LAGGED CORRELATIONS WITH Total_C4 ===")
print("(Data is hourly, so lag 1 = 1 hour)")
header = f"{'Variable':<25}" + "".join([f"  lag{l}" for l in [0, 1, 2, 3, 4, 6, 12]])
print(header)
for var in process_vars:
    row = f"{var:<25}"
    for lag in [0, 1, 2, 3, 4, 6, 12]:
        c = df2[var].shift(lag).corr(df2['Total_C4'])
        row += f"  {c:+.4f}"
    print(row)

# ============ ENGINEERED FEATURES ============
df2['Reflux_Ratio'] = df2['Reflux_Flow'] / df2['Feed_Flow'].replace(0, np.nan)
df2['Temp_Diff'] = df2['Column_Bottom_Temp'] - df2['Column_Top_Temp']
df2['Steam_Feed_Ratio'] = df2['Reboiling_Steam_Flow'] / df2['Feed_Flow'].replace(0, np.nan)

print("\n=== ENGINEERED FEATURE CORRELATIONS ===")
for var in ['Reflux_Ratio', 'Temp_Diff', 'Steam_Feed_Ratio']:
    row = f"{var:<25}"
    for lag in [0, 1, 2, 3, 4, 6, 12]:
        c = df2[var].shift(lag).corr(df2['Total_C4'])
        row += f"  {c:+.4f}"
    print(row)

# ============ AUTOCORRELATION ============
print("\n=== AUTOCORRELATION OF Total_C4 ===")
for lag in [1, 2, 3, 6, 12, 24]:
    ac = df2['Total_C4'].autocorr(lag)
    print(f"  lag {lag}h: {ac:.4f}")

# ============ BIMODAL DISTRIBUTIONS ============
print("\n=== REBOILER OUTLET TEMP DISTRIBUTION ===")
print(df2['Reboiler_Outlet_Temp'].describe())
low = (df2['Reboiler_Outlet_Temp'] < 50).sum()
high = (df2['Reboiler_Outlet_Temp'] >= 50).sum()
print(f"Below 50C: {low} ({low/len(df2)*100:.1f}%)")
print(f"Above 50C: {high} ({high/len(df2)*100:.1f}%)")

print("\n=== COLUMN TOP TEMP DISTRIBUTION ===")
low2 = (df2['Column_Top_Temp'] < 50).sum()
high2 = (df2['Column_Top_Temp'] >= 50).sum()
print(f"Below 50C: {low2} ({low2/len(df2)*100:.1f}%)")
print(f"Above 50C: {high2} ({high2/len(df2)*100:.1f}%)")

# ============ CHECK FOR REGIME CHANGES ============
print("\n=== MONTHLY AVG Total_C4 (trend check) ===")
df2.set_index('DateTime', inplace=True)
monthly = df2['Total_C4'].resample('M').agg(['mean', 'std', 'count'])
for idx, row in monthly.iterrows():
    print(f"  {idx.strftime('%Y-%m')}: mean={row['mean']:.4f}, std={row['std']:.4f}, n={int(row['count'])}")

# ============ STUCK ANALYZER ANALYSIS ============
df2 = df2.reset_index()
print("\n=== STUCK ANALYZER READINGS ===")
# Count consecutive same readings
c4h8_same = df2['C4H8_Bottom'].diff().eq(0)
c4h6_same = df2['C4H6_Bottom'].diff().eq(0)

# Find max consecutive same
def max_consec_same(series):
    groups = (series.diff().ne(0)).cumsum()
    return groups.value_counts().max()

print(f"C4H6 max consecutive identical: {max_consec_same(df2['C4H6_Bottom'])}")
print(f"C4H8 max consecutive identical: {max_consec_same(df2['C4H8_Bottom'])}")

# What % of C4H6 readings are repeats?
c4h6_repeats = c4h6_same.sum()
c4h8_repeats = c4h8_same.sum()
print(f"C4H6 repeat readings: {c4h6_repeats}/{len(df2)} ({c4h6_repeats/len(df2)*100:.1f}%)")
print(f"C4H8 repeat readings: {c4h8_repeats}/{len(df2)} ({c4h8_repeats/len(df2)*100:.1f}%)")
