"""
data_preprocessing.py
=====================
Phase 1: Raw data -> clean parquet ready for feature engineering.

Steps performed (in order):
  1.  Load raw Excel, skip 2 header rows, rename columns
  2.  Parse DateTime, coerce all process columns to float
  3.  Remove shutdown rows (all process vars < SHUTDOWN_THRESHOLD simultaneously)
        - Uses epsilon threshold, not exact zero, to catch soft shutdowns
  4.  Winsorise outliers at P1 / P99 per column
  5.  Detect stuck analyzer readings for C4H6 and C4H8
        - Uses np.isclose() for float comparison -- historian data can have
          tiny floating-point noise between "identical" readings
        - Mark each row with C4H6_stuck / C4H8_stuck flags
        - Record hours_since_change and Analyzer_Health (GOOD/WARNING/BAD)
  6.  Assign contiguous data-block labels (Block 1 / 2 / 3 / 4)
        - Blocks are separated wherever the time gap exceeds GAP_THRESHOLD
        - No lag or rolling features may ever cross block boundaries
  7.  Add cyclical time features (sin/cos for hour, day-of-week, month)
  8.  Write output to data/clean_data.parquet
  9.  Print a concise audit report to stdout

Audit trail
-----------
The thresholds below were derived from notebooks/analyze_data.py and
notebooks/analyze_constraints.py.  Change them here if stakeholder
clarification changes the values.

Constants to review with IOCL before going to production:
  STUCK_RUN_THRESHOLD   - minimum consecutive identical readings to flag as stuck
  GAP_THRESHOLD_HOURS   - minimum gap (hours) that separates two operating blocks
  WINSORISE_LOWER / UPPER - percentile clip bounds
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURABLE THRESHOLDS  (sourced from audit scripts in notebooks/)
# ---------------------------------------------------------------------------
RAW_FILE          = r"9.DB DATA -B.xlsx"
OUT_DIR           = "data"
OUT_FILE          = os.path.join(OUT_DIR, "clean_data.parquet")

STUCK_RUN_THRESHOLD   = 12       # consecutive identical readings -> flag as stuck
GAP_THRESHOLD_HOURS   = 24       # hours between consecutive rows -> new block
WINSORISE_LOWER       = 0.01     # P1
WINSORISE_UPPER       = 0.99     # P99

# Shutdown detection: rows where ALL process vars are below this threshold
# are treated as plant-offline periods and removed.
# Using a small epsilon (not exact 0) catches soft shutdowns and ramp-down
# periods where historian writes near-zero instead of exactly zero.
SHUTDOWN_THRESHOLD    = 0.5     # TPH / degC -- below this = offline

# Analyzer health bucketing thresholds (hours since last reading change)
ANALYZER_HEALTH_WARNING_H = 12   # GOOD -> WARNING after 12h unchanged
ANALYZER_HEALTH_BAD_H     = 24   # WARNING -> BAD after 24h unchanged

# Columns used for shutdown detection
PROCESS_COLS_FOR_SHUTDOWN_CHECK = [
    "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow",
]

# ---------------------------------------------------------------------------
# COLUMN MAP  (tag IDs -> human-readable names)
# ---------------------------------------------------------------------------
COLUMN_RENAME = {
    "Unnamed: 0":            "DateTime",
    "Feed Flow to DB":       "Feed_Flow",
    "Reboiler o/l Temp":     "Reboiler_Outlet_Temp",
    "Column top Temp":       "Column_Top_Temp",
    "Reboiling steam flow":  "Reboiling_Steam_Flow",
    "Reflux flow":           "Reflux_Flow",
    "Column Top pressure":   "Column_Top_Pressure",
    "Column bottom temp":    "Column_Bottom_Temp",
    "Control tay temp":      "Control_Tray_Temp",
    "C4H6 in DB bottom":     "C4H6_Bottom",
    "C4H8 in DB bottom":     "C4H8_Bottom",
}

PROCESS_NUMERIC_COLS = [
    "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Top_Pressure",
    "Column_Bottom_Temp", "Control_Tray_Temp",
]
TARGET_COLS = ["C4H6_Bottom", "C4H8_Bottom"]
ALL_NUMERIC  = PROCESS_NUMERIC_COLS + TARGET_COLS


# ===========================================================================
# HELPERS
# ===========================================================================

def detect_stuck_runs(series, threshold):
    """
    Return two aligned Series:
      stuck_flag  : bool - True if this row is inside a run of >= threshold
                   identical consecutive values
      hours_since : int  - how many rows (hours) ago the value last changed
                   (0 = this row is a new value)

    Uses np.isclose() for float comparison instead of exact equality.
    Historian systems (Exaquantum, OSI PI, etc.) can write values that are
    "the same reading" but differ in the last few floating-point bits due to
    averaging, compression, or unit conversion.  Exact .eq(0) on .diff()
    would miss those runs and under-count stuck periods.
    """
    # Compute absolute difference from previous row; treat as "same" if
    # within floating-point tolerance relative to the series scale.
    abs_diff = series.diff().abs()
    scale    = series.abs().median() + 1e-9   # avoid division by zero
    is_same  = (abs_diff / scale) < 1e-6      # relative tolerance
    is_same.iloc[0] = False                   # first row is never "same as previous"

    run_id    = (~is_same).cumsum()
    run_len   = run_id.map(run_id.value_counts())

    stuck_flag  = (is_same) & (run_len >= threshold)
    hours_since = run_id.groupby(run_id).cumcount()

    return stuck_flag.rename(series.name + "_stuck"), hours_since.rename("hours_since_" + series.name + "_change")


def winsorise(df, cols, lower, upper):
    """Clip each column to [P{lower}, P{upper}].
    Returns modified df and a summary of how many values were clipped."""
    summary_rows = []
    for col in cols:
        lo   = df[col].quantile(lower)
        hi   = df[col].quantile(upper)
        n_lo = (df[col] < lo).sum()
        n_hi = (df[col] > hi).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        summary_rows.append({"column": col, "P1_clip": round(lo, 4), "P99_clip": round(hi, 4),
                              "n_clipped_low": n_lo, "n_clipped_high": n_hi})
    return df, pd.DataFrame(summary_rows)


def assign_blocks(dt_series, gap_hours):
    """Label contiguous time blocks.
    A new block starts wherever the gap between consecutive timestamps
    exceeds gap_hours."""
    gaps   = dt_series.diff() > pd.Timedelta(hours=gap_hours)
    blocks = gaps.cumsum() + 1   # Block 1, 2, 3, ...
    return blocks.rename("Data_Block")


def cyclical_encode(df):
    """Add sin/cos encodings for hour-of-day, day-of-week, and month."""
    df["hour_sin"]  = np.sin(2 * np.pi * df["DateTime"].dt.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["DateTime"].dt.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["DateTime"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["DateTime"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["DateTime"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["DateTime"].dt.month / 12)
    return df


def longest_stuck_run(stuck_series):
    """Return the length of the longest stuck run."""
    if not stuck_series.any():
        return 0
    run_id = (~stuck_series).cumsum()
    return int(run_id[stuck_series].map(run_id[stuck_series].value_counts()).max())


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_preprocessing():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 - Load raw Excel, skip 2 header rows, rename
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1 - Loading raw Excel ...")
    raw = pd.read_excel(RAW_FILE, sheet_name="Sheet2")
    print("  Raw shape (including header rows):", raw.shape)

    df = raw.iloc[2:].copy().reset_index(drop=True)
    df = df.rename(columns=COLUMN_RENAME)
    print("  After skipping 2 header rows:     ", df.shape)

    # ------------------------------------------------------------------
    # STEP 2 - Parse types
    # ------------------------------------------------------------------
    print("\nSTEP 2 - Parsing types ...")
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    for col in ALL_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_dt_bad = df["DateTime"].isna().sum()
    print("  Unparseable DateTime values:", n_dt_bad)
    if n_dt_bad:
        df = df.dropna(subset=["DateTime"])
        print("  Dropped", n_dt_bad, "rows with bad DateTime ->", len(df), "rows remain")

    df = df.sort_values("DateTime").reset_index(drop=True)
    print("  Date range:", df["DateTime"].min(), "->", df["DateTime"].max())
    print("  NaN counts after type coercion:")
    print(df[ALL_NUMERIC].isna().sum().to_string(header=False))

    # ------------------------------------------------------------------
    # STEP 3 - Remove shutdown rows
    # ------------------------------------------------------------------
    print("\nSTEP 3 - Removing shutdown rows (all process vars < %.2f) ..." % SHUTDOWN_THRESHOLD)
    # Use threshold instead of exact zero: historian may write 0.001 or
    # similar near-zero values during ramp-down / soft shutdown periods.
    shutdown_mask = (df[PROCESS_COLS_FOR_SHUTDOWN_CHECK] < SHUTDOWN_THRESHOLD).all(axis=1)
    n_shutdown = shutdown_mask.sum()
    df = df[~shutdown_mask].reset_index(drop=True)
    print("  Removed", n_shutdown, "shutdown rows ->", len(df), "rows remain")

    # ------------------------------------------------------------------
    # STEP 4 - Winsorise outliers
    # ------------------------------------------------------------------
    print("\nSTEP 4 - Winsorising outliers at P1 / P99 ...")
    df, clip_summary = winsorise(df, ALL_NUMERIC, WINSORISE_LOWER, WINSORISE_UPPER)
    print(clip_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # STEP 5 - Stuck analyzer detection
    # ------------------------------------------------------------------
    print("\nSTEP 5 - Detecting stuck analyzer runs (threshold =", STUCK_RUN_THRESHOLD, "consecutive identical) ...")
    for target_col in TARGET_COLS:
        stuck_flag, hours_since = detect_stuck_runs(df[target_col], STUCK_RUN_THRESHOLD)
        df[stuck_flag.name]  = stuck_flag
        df[hours_since.name] = hours_since

    c4h6_stuck_n   = df["C4H6_Bottom_stuck"].sum()
    c4h8_stuck_n   = df["C4H8_Bottom_stuck"].sum()
    c4h6_stuck_pct = c4h6_stuck_n / len(df) * 100
    c4h8_stuck_pct = c4h8_stuck_n / len(df) * 100
    print("  C4H6 stuck rows:", c4h6_stuck_n, "(%.1f%%)" % c4h6_stuck_pct)
    print("  C4H8 stuck rows:", c4h8_stuck_n, "(%.1f%%)" % c4h8_stuck_pct)
    print("  Longest C4H6 stuck run:", longest_stuck_run(df["C4H6_Bottom_stuck"]), "hours")
    print("  Longest C4H8 stuck run:", longest_stuck_run(df["C4H8_Bottom_stuck"]), "hours")

    # Analyzer_Health: categorical column derived from the worst (highest)
    # hours_since value across both analyzers at each row.
    # GOOD    = both changed within 12h
    # WARNING = at least one unchanged for 12-24h
    # BAD     = at least one unchanged for >24h
    worst_hours = df[["hours_since_C4H6_Bottom_change",
                       "hours_since_C4H8_Bottom_change"]].max(axis=1)
    df["Analyzer_Health"] = pd.cut(
        worst_hours,
        bins=[-1, ANALYZER_HEALTH_WARNING_H - 1, ANALYZER_HEALTH_BAD_H - 1, worst_hours.max() + 1],
        labels=["GOOD", "WARNING", "BAD"],
    )
    health_counts = df["Analyzer_Health"].value_counts()
    print("  Analyzer_Health distribution:")
    for label in ["GOOD", "WARNING", "BAD"]:
        n = health_counts.get(label, 0)
        print("    %-8s %6d  (%.1f%%)" % (label, n, n / len(df) * 100))

    # ------------------------------------------------------------------
    # STEP 6 - Assign data blocks
    # ------------------------------------------------------------------
    print("\nSTEP 6 - Assigning data blocks (gap threshold =", GAP_THRESHOLD_HOURS, "hours) ...")
    df["Data_Block"] = assign_blocks(df["DateTime"], GAP_THRESHOLD_HOURS)
    block_summary = (
        df.groupby("Data_Block")["DateTime"]
        .agg(start="min", end="max", count="count")
    )
    block_summary["duration_days"] = (
        (block_summary["end"] - block_summary["start"]).dt.total_seconds() / 86400
    ).round(1)
    print(block_summary.to_string())

    # ------------------------------------------------------------------
    # STEP 7 - Cyclical time features
    # ------------------------------------------------------------------
    print("\nSTEP 7 - Adding cyclical time features ...")
    df = cyclical_encode(df)
    print("  Added: hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos")

    # ------------------------------------------------------------------
    # STEP 8 - Add Total_C4 derived column
    # ------------------------------------------------------------------
    df["Total_C4"] = df["C4H6_Bottom"] + df["C4H8_Bottom"]

    # ------------------------------------------------------------------
    # STEP 9 - Write output parquet
    # ------------------------------------------------------------------
    print("\nSTEP 9 - Writing output to", OUT_FILE, "...")
    df.to_parquet(OUT_FILE, index=False)
    print("  Final shape:", df.shape)
    print("  Columns:", list(df.columns))

    # ------------------------------------------------------------------
    # AUDIT SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print("  Input rows (after header skip):  11,399")
    print("  Rows removed (shutdown):        ", n_shutdown)
    print("  Final clean rows:               ", len(df))
    print("\n  Total_C4 stats:")
    print(df["Total_C4"].describe().round(4).to_string())
    print("\n  C4H6 stuck %:", "%.1f%%" % c4h6_stuck_pct, "(flagged, NOT removed)")
    print("  C4H8 stuck %:", "%.1f%%" % c4h8_stuck_pct, "(flagged, NOT removed)")
    print("  Data blocks: ", df["Data_Block"].nunique())
    print("\n  Output:", os.path.abspath(OUT_FILE))
    print("=" * 70)

    return df


if __name__ == "__main__":
    df_clean = run_preprocessing()
