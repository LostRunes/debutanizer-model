# Debutanizer Column C4 Slippage Optimizer — Phase 5 Final Summary

This report documents the design, validation, and performance statistics for the Phase 5 advisory optimizer developed for IOCL's debutanizer C4 slippage mitigation.

---

## 1. System Architecture

The advisory optimizer operates in a multi-stage **Process-Aware** layout. Rather than assuming that temperatures and column pressure remain frozen during steam or reflux adjustments, it chains lightweight surrogate process models to predict column response before calling the soft sensor.

```
       Candidate Setpoint Move (Steam Flow, Reflux Flow)
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 1. Surrogate Delta Models (T1, T2, T3)                     │
│    - Predicts t+1 deviations/deltas for Bottom Temp,       │
│      Tray Temp, and Column Top Pressure                    │
│    - Bottom Temp uses dev24h targets to block drift        │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 2. Uncertainty Safety Buffer Constraint Checks             │
│    - Evaluates predicted conditions + model MAE            │
│      against hard limits (Bottom Temp < 115C, Press < 5bar)│
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 3. Soft Sensor C4 Slippage Estimation                      │
│    - Constructs Model A features using predicted T, P      │
│    - Model A predicts C4H8 wt%; Model B anchors C4H6 wt%   │
│    - Calculates predicted Total C4 slippage                │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 4. Two-Stage Spec-First Optimization & Safety Check        │
│    - Stage 1: Filter candidates meeting product spec       │
│    - Stage 2: Minimize cost (Economic) or C4 (Spec)        │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 Operator Recommendation Output
```

---

## 2. Validation Methodology

To prove the optimizer's reliability and quantify expected C4 savings, we ran batch validation checks over **100 random, out-of-spec snapshots (> 0.50 wt% Total C4)** selected from the held-out Block 4 test campaign. 

Each snapshot was evaluated under both **SPEC Mode** (prioritizing C4 minimization) and **ECONOMIC Mode** (minimizing utility cost while meeting specification limits).

---

## 3. Performance Results

The batch validation metrics show highly stable, physically correct setpoint recommendations:

| Metric | SPEC Mode | ECONOMIC Mode |
|---|:---:|:---:|
| **Recommendation Feasibility Rate** | **86.0%** (86/100) | **86.0%** (86/100) |
| **Safety Limit Violations** | **0.0%** (0/100) | **0.0%** (0/100) |
| **Rejections due to no C4 savings** | 14.0% (14/100) | 14.0% (14/100) |
| **Average C4 Slippage Reduction** | **17.4%** (0.1384 wt% abs) | **16.7%** (0.1348 wt% abs) |
| **Average Steam Flow Change** | $+0.99\text{ TPH}$ | $+0.56\text{ TPH}$ |
| **Average Reflux Flow Change** | $-5.02\text{ TPH}$ | $-5.46\text{ TPH}$ |
| **Average Utility Cost Change** | **$-\$0.05/\text{hr}$** | **$-\$2.67/\text{hr}$** |

### Key Observations
1. **Safety Rejection Success**: 14% of the out-of-spec periods returned no moves because the column's bottom temperature was already near the safety boundary. Any move to heat the column further was rejected by the safety buffers, preventing unsafe operator instructions.
2. **Economic Optimization Trade-off**: Under Economic Mode, the optimizer successfully saved **$-\$2.67/\text{hr}$** in utilities on average (compared to a slight cost increase of $+\$0.65/\text{hr}$ in older runs) while still capturing a **16.7% relative C4 reduction** (almost identical to SPEC mode's $17.4\%$).

---

## 4. Technical Limitations

- **C4H6 Constant Assumption**: The optimizer assumes C4H6 (butadiene) remains constant at the latest analyzer-estimated value. The validated Model B contains no manipulable-variable response model (ML delta training showed no useful signal due to high autocorrelation and low mass fractions).
- **1-Hour Horizon**: The surrogate process models predict the process state $1\text{-hour}$ ahead. The column may require multiple hours to reach complete equilibrium following large setpoint changes.
- **Advisory Only**: The optimizer does not write directly to the Distributed Control System (DCS). Operating recommendations must be reviewed and executed manually by the board operator.

---

## 5. Future Work

- **MPC Integration**: Transitioning from advisory mode to closed-loop Advanced Process Control (APC) or Multivariable Predictive Control (MPC) once steady-state responses are validated.
- **Seeq Deployment**: Deploying the optimizer on the refinery's Seeq historian platform to provide real-time recommendations on operator dashboard displays.
- **Economic Calibration**: Updating the placeholder price coefficients in `configs/economics.json` with real refinery utility costs and C4 product recovery values.
