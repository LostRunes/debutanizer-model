# 02. Dataset Understanding

## Dataset Overview
The dataset contains **11,343 hours** of hourly process data and laboratory analyzer measurements from the Debutanizer Column, representing plant operations spanning 2023 to 2026. 

## Campaign Gaps & Data Blocks
The dataset is split by significant shutdown periods (gaps) into **4 distinct operating campaigns (Blocks)**:

| Block | Start Date | End Date | Rows | Duration (Days) | Reboiler Mean Temp |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Block 1** | 2023-04-16 | 2023-08-31 | 3,288 | 137.0 | 35.9 °C (Cold) |
| — | **Gap: 376 Days** | — | — | — | — |
| **Block 2** | 2024-09-11 | 2024-10-11 | 738 | 30.7 | 108.0 °C (Hot) |
| — | **Gap: 43 Hours** | — | — | — | — |
| **Block 3** | 2024-10-13 | 2024-11-15 | 803 | 33.4 | 93.2 °C (Mixed) |
| — | **Gap: 258 Days** | — | — | — | — |
| **Block 4** | 2025-08-01 | 2026-04-30 | 6,514 | 272.0 | 71.7 °C (Mixed) |

## Analyzer Quality & Stuck Detections
The physical analyzer readings suffer from periods of freezing (sensor stuck) where the same value is written for consecutive hours. Stuck sequences of length $\ge 12$ hours were identified using a relative tolerance of `1e-6` and filtered:
*   **C4H8_Bottom**: Stuck readings constitute **7.9%** of the dataset, clustering at two extreme readings ($0.034$ wt% and $1.262$ wt%).
*   **C4H6_Bottom**: Stuck readings constitute a massive **33.0%** of the dataset, freezing at exactly zero ($0.000$ wt%). Zeros are treated as invalid measurements because butadiene is physically always present in small amounts during separation.
*   **Analyzer Health Categories**: Derived from the elapsed hours since the last analyzer change:
    *   `GOOD` (Both changed within 12h): 69.9% of dataset.
    *   `WARNING` (Unchanged for 12-24h): 5.1% of dataset.
    *   `BAD` (At least one analyzer flatlined >24h): 25.0% of dataset.

## Target Distribution Shifts (Train vs. Test)
*   **Model A (C4H8_Bottom)**: Generalizes stably across campaigns, with a mean of **0.444 wt%** in Train (Blocks 1-3) and **0.428 wt%** in Test (Block 4).
*   **Model B (C4H6_Bottom)**: Suffers a severe **target collapse**. Its mean is **0.1395 wt%** in Train, but drops 24-fold to **0.0057 wt%** in Block 4. This target collapse is driven by Block 1's cold reboiler campaign (non-fractionating bottoms) where butadiene slip was extremely high (mean 0.208 wt%). In contrast, Block 4 operates in a highly efficient fractionation regime, keeping butadiene slip near zero.
