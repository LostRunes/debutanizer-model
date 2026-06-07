Objectives 
  “AI Based Model to Minimize C4 Slippage in DEBUTANIZER” 

Problem Statement
  C4 slippage in C5+ product stream varies 0.8 % to 1.5% (Spec 0.5 M%)
  Manual operation based on experience 
  Analyzer cycle time 12 minutes (Wide variation between sampling)
 Analyzer reading not reliable
  Feed and operating variability not handled optimally 








Input variables 
  Top & Bottom temperature 
  Reboiler outlet temperature 
  Reboiler steam flow
  Reflux flow
  Feed flow
 Control tray temperature 
 Tray Temperature
 Bottom Analyzer
 Re-boiling steam flow

Soft Sensor Model
  Model : you decide the model
  Feature Engineering : Steam flow, Reflux ratio, Temp. diff, feed flow
  Output : Predicted C4 wt.% in DB bottom 

Dashboard Design
  Live C4 prediction 
  Actual vs predicted trends for C4 
  Operator recommendations with loss INR/hr calculations

Optimization Strategy 
  Dynamic adjustment of reflux and steam 
  Maintain optimal bottom temperature 
  Balance energy vs recovery

Future Scope
 Real time deployment in Seeq/other options to be explored.
 Closed loop optimization with APC. 
 Extend to other columns. 


Solution architecture 
  Tailing tower data from Exaquantum and lab data.
  AI based soft sensor for C4 prediction
  Real time optimization along with operator instructions.



Process Details:
To separate mixed C4s from C5s and heavier. 
DP bottom is fed on level control to the 17th tray of debutanizer
Reboiling duty is provided by LP (desuperheater) steam.
Column vapors are condensed with cooling water and collected in reflux drum.
Mixed C4s after meeting reflux requirement are sent for further processing to: 
            Butadiene Extraction Unit.
            C4 hydrogenation Unit. 
            OSBL Storage.





[2:15 AM, 6/5/2026] Random: Sir, a quick update on the debutanizer model — Phase 1 (data preprocessing) is complete. Some findings from inspecting the clean data that I wanted to flag before proceeding:
1. The data has 4 blocks, not 3. Blocks 2 and 3 are only 43 hours apart so they're likely the same operating campaign, but lag features still can't cross that boundary.
2. The bimodal temperature distribution we saw earlier turns out to be entirely block-driven — Block 1 (2023) ran in a cold reboiler regime throughout, Block 2 (2024) ran hot throughout. Not two concurrent modes.
3. The C4H6 analyzer freezes at exactly zero during stuck periods — median stuck value is 0.000. So those readings aren't just stale, they're physically invalid. We'll filter these out from the C4H6 model training in addition to the flagged stuck runs.
4. The C4H8 analyzer gets stuck at both extremes — near its minimum (0.034) and near its maximum (1.26), not just low. Both will be filtered from training.
5. Block 1 (2023) had significantly worse C4 slippage (60% above spec vs 35% in recent data), which is physically explained by higher feed throughput and lower steam/reflux in that period.
One clarification I still need before finalizing the optimizer: regarding the ±50% constraint mentioned earlier — did you mean ±50% of the current live value, or ±50% of the defined operating range? For example, if steam is currently at 21 TPH, those two interpretations give very different ranges (10.5–31.5 TPH vs roughly 18–24 TPH). Currently using the conservative data-derived ranges (steam: 18–24.4 TPH, reflux: 80–103.9 TPH) — please confirm if a different interpretation is intended.
Also flagging three preprocessing thresholds that were set based on data analysis — these should be confirmed before production deployment:

STUCK_RUN_THRESHOLD = 12 (consecutive identical readings before flagging as stuck)
GAP_THRESHOLD_HOURS = 24 (hours between rows before treating as a new block)
Winsorization bounds: P1/P99

Proceeding to Phase 2 (feature engineering) now. Will update again when model training results are available.
[8:35 AM, 6/5/2026] Deepesh Sharma Sir: Sir, a quick update on the debutanizer model — Phase 1 (data preprocessing) is complete. Some findings from inspecting the clean data that I wanted to flag before proceeding:
The data has 4 blocks, not 3. Blocks 2 and 3 are only 43 hours apart so they're likely the same operating campaign, but lag features still can't cross that boundary.
The bimodal temperature distribution we saw earlier turns out to be entirely block-driven — Block 1 (2023) ran in a cold reboiler regime throughout, Block 2 (2024) ran hot throughout. Not two concurrent modes.
The C4H6 analyzer freezes at exactly zero during stuck periods — median stuck value is 0.000. So those readings aren't just stale, they're physically invalid. We'll filter these out from the C4H6 model training in 
Ill reply in lunch okay
[5:06 PM, 6/5/2026] Deepesh Sharma Sir: Point 1 : Blocks 2 aur 3 bhale hi 43 hours apart hain, par unhe alag hi treat karna hoga.
Point 2: Lag features banate waqt boundary strict rkho (groupby('Block_ID')). Block 1 (2023) ka cold reboiler aur Block 2 (2024) ka hot reboiler regime XGBoost ko dono scenarios seekhne me help karega. Model me Block_ID ko ek categorical feature ki tarah mat daalna, balki model ko variables ke physical relationship (Steam vs Bottom Temp) se hi seekhne dena, taki wo generalized rahe
Point 3 & 4: drop that data
Point 5 Z: dont drop anything