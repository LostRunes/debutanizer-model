import pandas as pd
import matplotlib.pyplot as plt
import os

print("Generating diagnostic scatter plot...")
df = pd.read_parquet("data/features.parquet")

plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df["Reboiler_Outlet_Temp"],
    df["Column_Top_Temp"],
    c=df["Total_C4"],
    cmap="viridis",
    alpha=0.6,
    edgecolors="w",
    linewidths=0.2
)

plt.colorbar(scatter, label="Total_C4 (wt.%)")
plt.xlabel("Reboiler Outlet Temp (°C)")
plt.ylabel("Column Top Temp (°C)")
plt.title("Diagnostic Scatter Plot: Reboiler Temp vs Top Temp (colored by Total_C4)")
plt.grid(True, linestyle="--", alpha=0.5)

plot_path = os.path.join("notebooks", "diagnostic_scatter.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved to {plot_path}")
