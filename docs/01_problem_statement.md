# 01. Problem Statement

## Objective
The primary goal of this project is to build a high-fidelity, real-time soft-sensor system to predict the concentration of **Total C4** composition (comprising Butene $C_4H_8$ and Butadiene $C_4H_6$) slippage in the bottom product of the **Debutanizer Column** in a petrochemical refinery.

$$\text{Total C4 (wt\%)} = C_4H_8\text{ (wt\%)} + C_4H_6\text{ (wt\%)}$$

## Industrial Context & Challenges
1.  **Product Specification Constraints**: The bottom product of the Debutanizer Column has a strict quality specification limit of **$\le 0.50$ wt%** for Total C4 slippage. Exceeding this limit leads to off-spec products.
2.  **Downstream Impact (Butadiene Poisoning)**: High butadiene ($C_4H_6$) levels in the column bottoms are highly detrimental as butadiene acts as a catalyst poison in downstream units (such as polymerization plants). 
3.  **Economic Penalties (Butene Loss)**: Over-fractionating the column bottoms to reduce Butene ($C_4H_8$) slippage consumes excessive reboiling steam, resulting in significant utility costs and carbon emissions. Accurate real-time control allows optimal separation efficiency.
4.  **Feedback Delay (GC Analyzers vs. Soft-Sensors)**: The physical gas chromatograph (GC) analyzer has an inherent feedback delay of **2 to 4 hours** (and is frequently offline or stuck due to sensor fouling). A soft-sensor provides virtual, real-time hourly estimates ($t$) enabling immediate closed-loop cascade control or manual adjustments by operators.
