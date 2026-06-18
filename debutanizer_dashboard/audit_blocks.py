"""Cross-block boundary audit."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'debutanizer_dashboard')
from services.state_service import state

df = state.df
b4_start = state.block4_indices[0]
print("Block 4 starts at idx:", b4_start)

history = df.loc[b4_start - 24:b4_start]
print("History slice Data_Block values:", history["Data_Block"].unique().tolist())
print("History DateTime range:", str(history["DateTime"].min()), "to", str(history["DateTime"].max()))
print()

b3 = df[df["Data_Block"] == 3]
b4 = df[df["Data_Block"] == 4]

print("Block 3 last idx:", b3.index[-1], "| DateTime:", str(b3["DateTime"].iloc[-1]))
print("Block 4 first idx:", b4.index[0], "| DateTime:", str(b4["DateTime"].iloc[0]))
gap = b4["DateTime"].iloc[0] - b3["DateTime"].iloc[-1]
print("Time gap between Block 3 and Block 4:", str(gap))

print()
print("=== HISTORY SLICING BUG ===")
print("get_current_history() does: df.loc[current_idx-24 : current_idx]")
print("When current_idx = block4_indices[0], the -24 goes into BLOCK 3!")
print("This means the initial history contains", len(history[history["Data_Block"] == 3]), "rows from Block 3 and", len(history[history["Data_Block"] == 4]), "rows from Block 4")
print("These are from DIFFERENT campaigns separated by 9+ months - invalid mixing!")

print()
print("=== TREND PAGE SCOPE ===")
# The trends page shows df.loc[current_idx-24:current_idx]
# Only 25 rows (24 hours) -- should show full block 4 for historical trends view
print("Trends page only shows 24-row window (just 24 hours of data)")
print("Full Block 4 has", len(b4), "rows spanning", str(b4["DateTime"].min()), "to", str(b4["DateTime"].max()))
print("A full historical view needs to show ALL of block 4, not just 24 hours")
