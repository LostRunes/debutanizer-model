---
title: "DEBUTANIZER model - AI Model for C4 Optimization"
source: "https://chatgpt.com/g/g-p-6a208af67de881919073b905fc780bc2-debutanizer-model/c/6a208c53-a9dc-8322-83e9-f061ed7c8458"
---

**You:**

i have been assigned an ML project, here is the info on it:

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
  Model : think yourself
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


## Introduction to the Problem

The current system involves monitoring bottom temperatures and analyzing gas composition. Light gases, such as butane, tend to rise. The output of the process, particularly regarding a component valued at 5 rupees, is often analyzed. There's an analyzer that provides readings every 12 minutes, but its reliability is questionable and it can sometimes be faulty. Senior personnel provide interpretations, but even their analyses can be incorrect, though they offer some insights. Lab reports are considered the most accurate source for verifying the input and output. The primary issue is 'seepage,' where lighter gases, due to incorrect combinations, slip downwards instead of rising. This contaminates the heavier gases at the bottom and leads to waste that requires cleaning. The wasted gas, like butane, is costly. Significant financial losses, potentially 1-2 crores per hour (or even 9-10 crores), can occur due to seepage if not detected promptly, especially if lab reports take 2 hours to process.

## Current Solutions and Their Limitations

Operators try to mitigate issues by adjusting parameters like speed and water temperature within limits based on their knowledge and specifications. An analyzer provides readings on various parameters every 12 minutes to indicate if things are okay. However, this analyzer is not entirely trustworthy and lacks perfect accuracy. The most accurate information comes from lab results, which test the internal components based on temperature readings to predict and confirm the correctness of the process. The goal is to configure parameters to minimize leakage.

## Data Preparation and Processing

The provided data will require pre-processing as some readings are faulty, and certain expected values are missing. Approximately 1200 rows of data might be cut during pre-processing. Manual selection of entries and clarification on which ones are needed is also required.

## Project Requirements

The project aims to address two main needs:

1. **Predictive Model:** Develop a predictive model that forecasts the extent of gas leakage based on the current process values.
2. **Real-time Simulator:** Create a simulator that, in real-time, can take suggested values and simulate their impact on future leakage reduction within minutes. This should indicate how much leakage would decrease if those values were implemented. The system should eventually be automated, but these two core functionalities are the immediate focus. The readings from the current system are reliable over 12-minute intervals, which should be considered when building the model.

## Future Considerations

Further steps involve checking the seepage of gas from the TV bottom to the main bottom and providing recommendations to operators. The full details of these requirements have been communicated separately and will be provided for further clarification and action points. This includes a detailed written explanation that will be shared to guide the next steps. 

so now i have to make a whole ML model for this, the dataset provided is not preprocesssed and there are some faulty values too i think, so gotta do all that too

* * *

**ChatGPT:**

This is actually a very good industrial ML project, because it is **not just a prediction problem**. It is a combination of:

1. **Soft Sensor Development** (predict C4 wt.% without waiting for analyzer/lab)
2. **Data Cleaning & Validation**
3. **Process Optimization**
4. **Decision Support Dashboard**
5. **Future APC Integration**

If I were building this from scratch for an internship/project review, I would structure it exactly like this.

* * *

# What You Are Really Building

Current situation:

```
Column Running
       ↓
Analyzer gives reading every 12 min
       ↓
Analyzer sometimes wrong
       ↓
Lab result comes after ~2 hours
       ↓
Operator adjusts based on experience
```

Problem:

```
C4 slips into bottom product

Target:
C4 < 0.5 wt%

Actual:
0.8 - 1.5 wt%
```

Financial impact:

```
Higher C4 slip
      ↓
Loss of valuable butane
      ↓
Off-spec product
      ↓
Reprocessing cost
      ↓
Huge INR loss
```

* * *

# Deliverable 1: Soft Sensor

This is the most important part.

Input:

```
Top Temperature
Bottom Temperature
Control Tray Temperature
Tray Temperature
Reboiler Outlet Temperature
Steam Flow
Reflux Flow
Feed Flow
Bottom Analyzer Reading
```

Output:

```
Predicted C4 wt%
```

Instead of waiting:

```
12 min analyzer
or
2 hr lab result
```

Model predicts:

```
Current C4 wt%
every minute
```

* * *

# Step 1: Understand Your Dataset

Before touching ML:

Create a report:

### Missing values

```Python
df.isnull().sum()
```

* * *

### Duplicate rows

```Python
df.duplicated().sum()
```

* * *

### Data types

```Python
df.info()
```

* * *

### Statistical summary

```Python
df.describe()
```

* * *

# Step 2: Remove Faulty Data

You already mentioned:

> Analyzer faulty  
> Some values missing  
> Bad readings exist

This is normal in process industries.

Examples:

### Steam flow negative

Impossible.

```Python
SteamFlow < 0
```

Remove.

* * *

### Temperature = 9999

Sensor fault.

Remove.

* * *

### Sudden spikes

Example:

```
Top Temp

84
85
85
300
86
```

300 is impossible.

Use:

```Python
IQR
```

or

```Python
Isolation Forest
```

for anomaly detection.

* * *

# Step 3: Time Alignment

This is VERY IMPORTANT.

Most students miss this.

Column behavior is delayed.

Example:

```
Steam change at 10:00

C4 composition changes at 10:20
```

Not instantly.

So create lag features.

Example:

```Python
Steam_t-1
Steam_t-2
Steam_t-3

Reflux_t-1
Reflux_t-2
```

If data interval is 12 minutes:

```
t-1 = 12 min ago
t-2 = 24 min ago
```

This often improves prediction dramatically.

* * *

# Step 4: Feature Engineering

Don't just use raw values.

Create process-engineering features.

* * *

### Reflux Ratio

Very important.

```Python
Reflux Ratio =
Reflux Flow / Feed Flow
```

* * *

### Steam to Feed Ratio

```Python
Steam Flow / Feed Flow
```

* * *

### Temperature Difference

```Python
Bottom Temp - Top Temp
```

* * *

### Reboiler Driving Force

```Python
Reboiler Temp - Bottom Temp
```

* * *

### Moving Average

```Python
Steam MA(3)
Steam MA(5)
```

Smooths noise.

* * *

### Rate of Change

```Python
Δ Steam

Current Steam
-
Previous Steam
```

Helps capture operator actions.

* * *

# Step 5: Choose Models

Don't start with deep learning.

Industrial datasets are usually small.

Start:

## Model 1

Linear Regression

Baseline.

* * *

## Model 2

Random Forest

Usually works surprisingly well.

* * *

## Model 3

XGBoost

This is what I expect will win.

Reasons:

* handles nonlinearities
* robust to noise
* works with tabular data
* interpretable

* * *

## Model 4

LightGBM

Alternative to XGBoost.

* * *

# Step 6: Evaluation

Metrics:

### MAE

```Python
mean_absolute_error
```

Most understandable.

Example:

```
Actual = 0.90

Prediction = 0.95

Error = 0.05
```

* * *

### RMSE

Penalizes big mistakes.

* * *

### R²

Good for presentations.

* * *

Target:

```
R² > 0.85

MAE < 0.05 wt%
```

would look excellent.

* * *

# Step 7: Explainability

Management will ask:

> Why is C4 increasing?

Use:

## SHAP

Shows feature importance.

Example:

```
Top contributors:

1 Steam Flow
2 Reflux Ratio
3 Bottom Temp
4 Feed Flow
```

Very impressive in reviews.

* * *

# Deliverable 2: Real-Time Simulator

This is the second major component.

Current:

```
Current conditions
↓
Prediction = 1.2%
```

Now operator asks:

```
What if I increase reflux?
```

Simulator answers:

```
Reflux +5%

Predicted C4:

1.2%
↓
0.8%

Savings:
₹45,000/hr
```

* * *

How?

Take trained model.

Change selected inputs.

Run prediction again.

Compare outputs.

Example:

```Python
current = model.predict(X)

X["Reflux"] *= 1.05

new = model.predict(X)
```

* * *

# Optimization Engine

Instead of operator guessing.

Let algorithm search.

Variables:

```
Steam
Reflux
Bottom Temp
```

Constraints:

```
Steam <= Max Steam

Bottom Temp <= Safety Limit

Reflux <= Pump Limit
```

Objective:

```
Minimize Predicted C4
```

Use:

```Python
scipy.optimize
```

or

```Python
Optuna
```

* * *

Output:

```
Recommended Actions

Increase Reflux by 4%

Decrease Steam by 2%

Expected C4:
0.95 → 0.48

Expected Savings:
₹32,000/hr
```

* * *

# Dashboard Architecture

Use:

### Backend

Python

```Python
Pandas
Scikit-Learn
XGBoost
SHAP
```

* * *

### Dashboard

I recommend:

```Python
Streamlit
```

because:

* quick
* easy
* professional

* * *

Dashboard Sections

### Live Status

```
Current C4 Prediction

0.62 wt%
```

Green/Yellow/Red.

* * *

### Trend

```
Actual vs Predicted
```

Line graph.

* * *

### Recommendations

```
Increase Reflux +3%

Expected C4:
0.62 → 0.48
```

* * *

### Loss Calculator

```
Current Loss

₹21,400/hr
```

* * *

### SHAP Importance

```
Steam Flow      █████████
Reflux Ratio    ████████
Bottom Temp     ██████
```

* * *

# What Reviewers Will Love

Most students stop here:

```
Input
↓
Random Forest
↓
Output
```

You should show:

```
Data Cleaning
↓
Feature Engineering
↓
Soft Sensor
↓
Simulator
↓
Optimizer
↓
Dashboard
```

That looks like a real refinery digitalization project.

* * *

My expectation from your project:

**Best model:** XGBoost Regressor

**Most important engineered features:**

* Reflux Ratio
* Steam/Feed Ratio
* Bottom-Top Temperature Difference
* Lagged Steam Flow
* Lagged Reflux Flow

**Final architecture:**

```
Exaquantum Historian
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
XGBoost Soft Sensor
        ↓
C4 Prediction
        ↓
What-If Simulator
        ↓
Optimization Engine
        ↓
Streamlit Dashboard
        ↓
Operator Recommendation
```

Once you receive the actual dataset (CSV/Excel), I can help you build the entire pipeline step-by-step: preprocessing, anomaly removal, feature engineering, model training, optimization logic, and the Streamlit dashboard.

* * *

**You:**

![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000a1f07209baa32dc194147ce3&ts=494638&p=fs&cid=1&sig=6b480e993bc00b6f0dac60b87a3aa9b441b228b2e7e130213ace5b61e0e2a0d4&v=0)btw the dataset provided is actual data from iocl, i have been given the iocl personal whole year's worth of data to train this, this is more than just a silly little project, this is a whole job, so i gotta be good with this

here is a small sample of the data:

2025-08-07 13:00:00.000	92.45732241	38.85262996	108.9877828	20.11382799	82.11866471	4.287959667	108.5623377	66.09068275	0	0.390708178
2025-08-07 14:00:00.000	92.72592699	40.58936631	109.5748062	20.43307832	82.97264735	4.335522901	109.0748162	67.76915316	4.71289E-05	0.549724765
2025-08-07 15:00:00.000	87.93038181	42.0655196	110.2496254	19.6528696	83.02192472	4.300133468	109.8425107	68.73799285	0.00405252	0.485159576
2025-08-07 16:00:00.000	90.52078509	42.61096269	111.1152749	19.98396607	82.98032992	4.333603229	110.6775702	69.21659663	0.002766937	0.43690402
2025-08-07 17:00:00.000	89.47017145	42.3305776	111.2756321	19.90480249	82.99131409	4.365852978	110.8086546	68.98484482	0.000665503	0.386107332
2025-08-07 18:00:00.000	88.98890151	42.14962992	110.764417	19.80930231	83.01169569	4.367541376	110.3502884	68.7151523	0.00211657	0.367531146
2025-08-07 19:00:00.000	88.73336143	41.8197755	110.8595475	19.73863986	82.98548912	4.361158138	110.5647766	69.09024965	0.000464833	0.374611423
2025-08-07 20:00:00.000	90.15434013	41.20249712	111.0503143	19.90903845	82.95834268	4.355294662	110.7435573	68.92367352	4.21875E-05	0.400257356
2025-08-07 21:00:00.000	90.27088863	40.7034448	111.0303942	19.91808211	83.01550745	4.328663055	110.7584421	69.05808744	4.21875E-05	0.376315385
2025-08-07 22:00:00.000	89.36149676	40.30320855	110.8585325	19.86378333	82.99483918	4.300042466	110.5560355	69.00616498	4.21875E-05	0.367412267
2025-08-07 23:00:00.000	88.40975362	39.97689849	110.6223513	19.80303288	83.02019307	4.258117885	110.2672775	69.03111201	4.21875E-05	0.35824759
2025-08-08 00:00:00.000	88.15973288	39.54372814	110.5503382	19.83069757	82.99195071	4.220256708	110.1542576	69.09282203	4.21875E-05	0.344097417
2025-08-08 01:00:00.000	87.37150171	39.04847371	110.3638251	19.74382817	82.97973308	4.200533444	109.9910094	68.87824948	4.21875E-05	0.339819364
2025-08-08 02:00:00.000	85.9420402	38.5908407	110.5417806	19.66978473	82.97593001	4.192237376	110.1603395	69.10943139	4.21875E-05	0.329434604
2025-08-08 03:00:00.000	87.93527853	38.3248434	110.3827002	19.84327571	82.96512024	4.197954926	109.9937287	68.874916	4.21875E-05	0.326425989
2025-08-08 04:00:00.000	87.6397836	38.19068461	110.5496955	19.81647013	83.01524559	4.188252422	110.1779487	69.02522253	4.21875E-05	0.340924865
2025-08-08 05:00:00.000	87.2124453	37.95286304	110.5080281	19.92105278	83.00237692	4.192230353	110.1290684	69.02435709	4.21875E-05	0.325657129
2025-08-08 06:00:00.000	89.08614103	37.51687076	110.2696907	20.02634162	82.9704415	4.189146271	109.79663	68.83870078	4.21875E-05	0.315220803
2025-08-08 07:00:00.000	86.92515103	37.06246904	110.5389515	19.87723818	83.00351496	4.180698875	110.0480505	69.07644638	4.21875E-05	0.313331043
2025-08-08 08:00:00.000	88.07686882	36.79993594	109.8085094	19.96146857	82.99664022	4.138507176	109.3118028	69.04181381	4.21875E-05	0.323280088
2025-08-08 09:00:00.000	83.1759831	37.01224693	109.3863051	19.53128039	83.01871831	4.0805607	108.8552486	69.42245594	4.21875E-05	0.334021444
2025-08-08 10:00:00.000	80.10071377	37.63663002	109.3126371	19.32890111	83.02138755	4.084442131	108.7594037	69.22908715	4.21875E-05	0.366225662
2025-08-08 11:00:00.000	63.26660854	38.26160763	107.6065933	16.96465954	83.06218519	3.977729649	107.558154	70.62037927	3.17227E-05	0.354703633
2025-08-08 12:00:00.000	76.81881998	38.82776267	104.5955141	18.39790452	83.00590929	4.096628959	103.9428394	69.259875	1.53477E-16	0.392872441
2025-08-08 13:00:00.000	76.43309037	39.3205624	102.5881888	17.49820466	82.9458578	4.076146917	102.0284724	71.55686515	1.4559E-16	0.479000989
2025-08-08 14:00:00.000	74.86020125	40.03218867	104.7295489	18.05289799	83.00500153	4.118651475	104.2554591	74.60460909	0	0.401097857
2025-08-08 15:00:00.000	81.22985816	40.5332929	110.1899968	19.74377047	83.01590979	4.19373225	109.4561862	75.81521945	0	0.662078548
2025-08-08 16:00:00.000	83.0221578	41.02343714	110.894216	19.31553966	83.01616513	4.169121739	110.5982462	81.9955804	0	0.630336174
2025-08-08 17:00:00.000	82.15644949	41.50304298	109.7501138	19.42618768	82.99557668	4.161390516	109.3654881	82.70603546	0	0.680003114
2025-08-08 18:00:00.000	82.37499334	41.61781781	109.5924104	19.40608609	83.04406102	4.152902699	109.243524	86.3969393	0	0.659546088
2025-08-08 19:00:00.000	82.14838889	41.19135984	108.7485236	19.05378323	83.06047293	4.144463467	108.5173558	89.67782988	0	0.584021219
2025-08-08 20:00:00.000	82.76917772	40.39884217	106.6332891	18.56628142	82.94350287	4.161491202	106.4180436	87.99280469	0	0.534080041
2025-08-08 21:00:00.000	84.74745176	39.58085955	107.2644353	19.12663806	82.99899132	4.19412537	107.0132704	89.09266539	0	0.593811329
2025-08-08 22:00:00.000	86.06647939	38.81943625	107.2286239	19.3833599	83.00067067	4.153148861	107.0279029	89.78598796	0	0.552422331
2025-08-08 23:00:00.000	84.17047371	38.2205891	107.0027215	19.52797434	82.43283466	4.112114951	106.7095853	90.42436338	0	0.593358709
2025-08-09 00:00:00.000	86.22040316	37.75519519	107.2264272	18.66426308	78.26387924	4.012287199	107.0243589	90.29973248	0	0.503865017
2025-08-09 01:00:00.000	77.52741174	37.36869331	105.3240728	17.41925288	78.02354375	3.937152523	105.1903702	87.80757486	0	0.567688981
2025-08-09 02:00:00.000	78.23208728	37.01858994	103.1132491	17.36557037	83.02953659	4.007114832	103.1081822	84.37389602	0	0.555280945
2025-08-09 03:00:00.000	74.80110596	36.65720148	104.5436925	18.18651256	83.93562421	4.078321143	104.2256802	86.64704147	0	0.48815388
2025-08-09 04:00:00.000	75.09332811	36.42552336	105.6001747	18.39566735	84.06492507	4.089177129	105.2504972	84.379832	0	0.706597117
2025-08-09 05:00:00.000	74.72708576	36.34148788	105.9750229	18.40926864	83.95155553	4.088424506	105.6731443	84.62373165	0	0.776819621
2025-08-09 06:00:00.000	74.77207445	36.25409119	105.5926642	18.45795507	84.05395011	4.080750385	105.350075	84.33855769	0	0.729083638
2025-08-09 07:00:00.000	79.10543405	36.09815653	105.4009231	18.82581629	83.92557807	4.083823915	105.1962972	83.95622334	0	0.756422097
2025-08-09 08:00:00.000	82.96530523	36.06355453	108.0701504	20.03884154	84.00307691	4.124878031	107.7698489	89.37494225	0	0.65085483
2025-08-09 09:00:00.000	86.12224392	35.93821796	106.7167579	20.06126485	84.00661092	4.111979575	106.5202922	89.11590558	0	0.634383837
2025-08-09 10:00:00.000	79.30602688	34.03003028	106.6937314	19.72091641	84.00070119	4.025474534	106.4716265	89.46907129	0	0.458426049
2025-08-09 11:00:00.000	81.5147155	31.78712511	105.339103	20.43327421	83.87241882	3.957101335	105.0306231	89.41857244	0	0.468088978
2025-08-09 12:00:00.000	78.97375408	30.66010636	106.0873404	20.14619829	85.00348136	3.92541763	105.7728442	89.78532853	0	0.376247758
2025-08-09 13:00:00.000	79.54361725	30.51917267	106.0673294	20.13661575	86.66641235	3.899803162	105.8208923	89.65803528	0	0.392596543
2025-08-09 14:00:00.000	80.5458209	31.35370601	105.4045911	20.18751766	87.75478366	3.989597496	105.1661073	89.3493867	0	0.466206182
2025-08-09 15:00:00.000	76.83297613	33.24674528	105.2654476	20.49681445	87.8817678	4.136677579	104.923531	89.59712065	0	0.50410541
2025-08-09 16:00:00.000	78.21718171	33.79472953	106.5579827	20.50260224	90.4279187	4.161730229	106.1928973	89.23893303	0	0.61157854
2025-08-09 17:00:00.000	70.4159005	33.99406622	110.103828	20.26525241	88.12693441	4.153666079	109.5455998	90.56813896	0	0.512070052
2025-08-09 18:00:00.000	68.79159209	34.02064482	109.6937689	19.26234677	88.01516936	4.102050916	109.2710699	89.58746345	0	0.407973368
2025-08-09 19:00:00.000	75.81161941	34.12638091	110.6139155	20.22534117	87.9964696	4.156922718	110.0370948	88.39736972	0	0.340973226
2025-08-09 20:00:00.000	73.47783474	34.02006465	114.2933398	20.0490936	88.0899318	4.139568545	113.7732685	89.96356595	0	0.283832517
2025-08-09 21:00:00.000	73.85570759	33.70382827	111.463643	20.15260215	87.98156281	4.140212957	111.1154938	85.82255305	0	0.298702367
2025-08-09 22:00:00.000	71.59178559	33.58828225	107.4215	18.60267082	88.02571361	4.086122198	107.4081253	70.13252444	0	0.275544368
2025-08-09 23:00:00.000	72.92698362	33.55923682	111.7658287	20.76183836	87.89732828	4.180762042	110.8848424	71.85125121	0	0.283410861
2025-08-10 00:00:00.000	75.66507683	33.33877077	114.2400592	20.12995844	88.09116647	4.133162118	113.709516	80.0676305	0	0.2823365
2025-08-10 01:00:00.000	73.45867139	33.23711506	112.309202	19.64680613	87.99199354	4.103376153	111.8247394	76.36248421	0	0.28047355
2025-08-10 02:00:00.000	75.18681679	33.39620871	108.3879871	19.56108001	88.01038079	4.119710026	107.6404691	64.10739251	0	0.215215359
2025-08-10 03:00:00.000	73.46731943	33.4809539	107.6000647	19.79264684	87.96562066	4.11646074	107.1093132	62.51334346	0	0.26187707
2025-08-10 04:00:00.000	72.58781381	33.49277636	108.2601139	19.8300016	88.06442942	4.100109394	107.6755577	63.4327784	0	0.338694375
2025-08-10 05:00:00.000	72.21446597	33.42731639	108.8790666	19.73147938	88.04265132	4.088685158	108.2917909	64.09135127	0	0.315451475
2025-08-10 06:00:00.000	73.31923968	33.41145875	108.5814177	19.74460434	87.92843528	4.100238016	108.0346262	64.62855868	0	0.283301547
2025-08-10 07:00:00.000	73.17265187	33.4953158	108.4702588	19.73304888	87.94571234	4.095565207	107.8339147	66.66800243	0	0.283278651
2025-08-10 08:00:00.000	72.80361703	33.5644309	108.6341247	19.71479024	87.9779011	4.077703727	107.9279856	66.96488809	0	0.286408727
2025-08-10 09:00:00.000	73.39027575	33.98224338	108.5993686	19.74735428	87.9607555	4.076767412	107.9237726	66.67024287	0	0.329672358
2025-08-10 10:00:00.000	72.70883173	34.90222896	108.7540266	19.70364765	87.98699347	4.106442536	108.1144453	66.40055752	0	0.345765555
2025-08-10 11:00:00.000	72.86110051	35.8147817	108.8190986	19.66672344	87.98358307	4.135034988	108.2099879	65.62295548	0	0.366377355
2025-08-10 12:00:00.000	75.27853349	36.35253224	108.6625909	19.79386299	88.03093027	4.178815013	108.0082403	66.21677246	0	0.340593642
2025-08-10 13:00:00.000	74.94615326	36.75829065	108.6471844	19.82769808	87.94780466	4.196804633	107.8142477	65.99230284	0	0.3600183
2025-08-10 14:00:00.000	74.57125928	37.18744015	108.8540811	19.74733599	87.99515899	4.214336802	108.2387707	65.64344042	0	0.38192747
2025-08-10 15:00:00.000	74.59610082	38.5886471	108.6303114	19.79623371	88.01183279	4.246620214	108.0772951	65.58009447	0	0.384161135
2025-08-10 16:00:00.000	77.74758553	39.52774817	108.7595848	20.01851492	88.00968488	4.283190338	108.1937806	64.93343117	0.009296709	0.432876092
2025-08-10 17:00:00.000	74.26346101	39.43379135	108.1034784	19.69784277	88.02696784	4.264867886	107.9322442	63.09217958	0.013663739	0.396503181
2025-08-10 18:00:00.000	78.64751853	39.3787929	108.0744703	20.14036242	88.01189287	4.298790182	107.7310988	63.21171056	0.007256512	0.3894164
2025-08-10 19:00:00.000	79.44824321	39.16027726	108.5847599	20.06326835	87.97367091	4.310008981	108.1685506	63.49834792	0.003699873	0.372816975
2025-08-10 20:00:00.000	79.12793713	38.64033782	108.7820869	20.00192034	88.00580636	4.326032327	108.269285	65.92488942	7.8125E-07	0.442496692
2025-08-10 21:00:00.000	78.19181076	38.19542468	108.9205952	19.86463279	88.05144819	4.337625966	108.461003	65.84125716	7.8125E-07	0.419613303
2025-08-10 22:00:00.000	77.99074133	37.80430999	108.4670911	19.86434115	87.98761908	4.341674195	107.9768773	65.60481064	7.8125E-07	0.376241476
2025-08-10 23:00:00.000	79.03406359	37.1071657	108.1777335	19.91373168	87.92259947	4.329891747	107.727291	65.39437881	7.8125E-07	0.393426358
2025-08-11 00:00:00.000	81.63315724	36.40340213	108.952396	20.07895967	87.99583957	4.349031385	108.4810484	65.60248402	7.8125E-07	0.349117954
2025-08-11 01:00:00.000	81.25184959	35.78390317	107.8538055	20.18058745	87.99768245	4.345735565	107.3548546	64.52271202	7.8125E-07	0.328276263
2025-08-11 02:00:00.000	79.32300116	35.14816656	108.1586774	20.0912745	87.98613274	4.312214418	107.7050555	63.4654521	7.8125E-07	0.305651551
2025-08-11 03:00:00.000	79.69459845	34.8686753	107.9942041	20.20433472	87.9690663	4.286655171	107.4836232	63.23466981	7.8125E-07	0.312287943
2025-08-11 04:00:00.000	82.53889616	34.86508844	108.479445	20.27730221	87.97849487	4.277610316	107.9605041	64.00896621	7.8125E-07	0.26031322
2025-08-11 05:00:00.000	80.56580214	34.90594728	108.51677	20.18213074	88.00023937	4.25270671	107.988175	63.79740892	7.8125E-07	0.276127719
2025-08-11 06:00:00.000	82.28970835	34.93358657	107.939977	20.3328256	88.00343703	4.232114323	107.418178	63.07726577	7.8125E-07	0.304838247
2025-08-11 07:00:00.000	81.43628855	34.90072829	108.4178021	20.38951101	88.03644566	4.202379689	107.7204476	62.92845914	7.8125E-07	0.286077555
2025-08-11 08:00:00.000	80.08062113	34.90312719	108.5190842	20.27029494	88.00833893	4.174644874	107.9000322	62.86186282	7.8125E-07	0.288119655
2025-08-11 09:00:00.000	82.69312207	34.99304841	108.1981296	20.47286636	87.93147385	4.177828832	107.5906823	62.95155473	7.8125E-07	0.30923002
2025-08-11 10:00:00.000	79.84659763	35.32394307	109.0071035	20.29722775	87.9716282	4.175769856	108.2619086	63.33089658	7.8125E-07	0.305012097
2025-08-11 11:00:00.000	79.94560468	35.81665694	108.801885	20.28025511	88.01768547	4.18355936	108.0475888	62.89740341	7.8125E-07	0.312267829
2025-08-11 12:00:00.000	64.42583319	36.32051972	109.509181	19.48244788	88.10122038	4.105094976	108.8263439	62.49765167	7.81249923420546E-07	0.333224318
2025-08-11 13:00:00.000	65.20096896	36.73877853	109.9565621	19.15181191	88.05944287	4.08501438	109.3177489	64.93445158	7.8125E-07	0.332191508

* * *

**ChatGPT:**

Love, this is not a college ML project anymore.

This is essentially a **mini industrial digital twin / soft sensor project** using actual refinery historian data. If this works well, it could genuinely save money and eventually get deployed into operations.

Looking at your sample, a few things immediately jump out.

* * *

# First Observation: Your Target Variable

You have two analyzer outputs:

```
C4H6 in DB bottom
C4H8 in DB bottom
```

Need clarification from IOCL:

### Which one represents "C4 slippage"?

Because your project statement says:

```
Output:
Predicted C4 wt.% in DB bottom
```

but the data contains:

```
C4H6 = Butadiene
C4H8 = Butene
```

and "total C4" may be:

```
C4H6 + C4H8 + other C4s
```

Before doing ANYTHING:

Ask process engineer:

> What exactly is the target variable?
> 
> * C4H6?
> * C4H8?
> * Total C4?
> * Lab measured C4 wt%?
> 
> Which tag is considered the official KPI?

This is the most important question of the entire project.

* * *

# Second Observation: The Data Looks Like Historian Data

This is not laboratory data.

This is:

```
Exaquantum Historian
```

hourly sampled.

I can see:

```
2025-08-07 13:00
2025-08-07 14:00
2025-08-07 15:00
```

1-hour intervals.

Meaning:

```
8760 rows/year
```

approximately.

That's actually decent for XGBoost.

* * *

# Third Observation: There Are Obvious Faulty Values

Look:

```
C4H6

0
0
0
0
0
```

then

```
0.009
0.013
0.007
```

then

```
7.8125E-07
```

This screams:

```
Analyzer lower detection limit
```

or

```
Analyzer stuck value
```

* * *

Also:

```
1.53477E-16
```

is basically:

```
0
```

physically impossible measurement precision.

Those need handling.

* * *

# Fourth Observation: There Are Process Upsets

Notice:

```
Column top temp

110
109
108
107
106
105
104
102
```

then suddenly recovers.

This is not random noise.

This is:

```
Operating disturbance
```

or

```
Feed composition change
```

or

```
Operator intervention
```

These periods are GOLD.

Your model learns from these.

Don't automatically remove them.

* * *

# What I Would Do First

Before touching ML:

Create an EDA notebook.

* * *

## Step 1

Read data.

```Python
df = pd.read_csv(...)
```

* * *

## Step 2

Convert timestamp.

```Python
df["Date"] = pd.to_datetime(df["Date"])
```

* * *

## Step 3

Sort.

```Python
df = df.sort_values("Date")
```

* * *

## Step 4

Missing values report.

```Python
df.isnull().sum()
```

* * *

## Step 5

Zero count report.

```Python
(df == 0).sum()
```

* * *

## Step 6

Plot everything.

For every column:

```Python
plt.plot(...)
```

You will immediately spot:

* sensor failures
* stuck values
* maintenance periods
* shutdowns

* * *

# Fifth Observation: Time Delay Will Matter

This is huge.

A debutanizer doesn't respond instantly.

Example:

```
Increase steam now
```

C4 reduction may appear:

```
30 min later
1 hour later
2 hours later
```

So your best features may actually be:

```
Steam(t-1)
Steam(t-2)
Steam(t-3)

Reflux(t-1)
Reflux(t-2)

Bottom Temp(t-1)
```

not the current values.

* * *

# Sixth Observation: Engineer Features

Raw variables:

```
Feed
Steam
Reflux
Top Temp
Bottom Temp
Pressure
```

Good.

But process engineers think in ratios.

Create:

### Reflux Ratio

```
Reflux / Feed
```

* * *

### Steam Ratio

```
Steam / Feed
```

* * *

### Temperature Spread

```
Bottom Temp - Top Temp
```

* * *

### Relative Volatility Indicator

```
Control Tray Temp - Top Temp
```

* * *

### Pressure-Temperature Interaction

```
Pressure × Top Temp
```

* * *

These will likely outperform raw values.

* * *

# Seventh Observation: Don't Start With Deep Learning

If I were your mentor I'd forbid it initially.

Try:

### Linear Regression

baseline

* * *

### Random Forest

good baseline

* * *

### XGBoost

likely winner

* * *

### LightGBM

alternative winner

* * *

For refinery historian data:

```
XGBoost
```

wins ridiculously often.

* * *

# Eighth Observation: This Can Become a Recommendation Engine

Not just prediction.

Suppose:

Current:

```
Predicted C4 = 0.82%
```

Spec:

```
0.50%
```

Optimizer searches:

```
Steam
Reflux
Bottom Temp
```

and outputs:

```
Increase reflux by 3%

Increase steam by 2%

Expected C4:
0.82 → 0.46
```

This is exactly what IOCL management will care about.

* * *

# What Would Impress IOCL The Most

Not:

```
Accuracy = 95%
```

Nobody in operations cares.

They care about:

```
Current loss:
₹1,20,000/hr

Recommended action:
Increase reflux 4%

Expected loss:
₹25,000/hr
```

That gets attention.

* * *

# What I Think Your Final Architecture Should Be

```
Raw Exaquantum Data
          ↓
Data Cleaning
          ↓
Sensor Validation
          ↓
Feature Engineering
          ↓
Lag Features
          ↓
XGBoost Soft Sensor
          ↓
Predicted C4 Slip
          ↓
What-If Simulator
          ↓
Optimization Engine
          ↓
Operator Recommendation
          ↓
Dashboard
```

And because this is actual IOCL data and not a toy dataset, I would spend **60% of the project effort on data cleaning and process understanding**, **20% on feature engineering**, and only **20% on model training**.

In industrial ML, the model is often the easiest part. The real challenge is figuring out which values are genuine process behavior and which are analyzer faults, maintenance events, sensor drift, or historian artifacts.

The next thing I'd do is get answers from IOCL on:

1. Exact target variable (what defines C4 slippage).
2. Whether analyzer values or lab values are available.
3. How often lab results come.
4. Whether there is a column status/shutdown flag.
5. How many total rows you actually have.

Once you have the full dataset, we can design a proper preprocessing pipeline and model architecture like an actual process data scientist would.

* * *

**You:**

Pasted markdown.mdFilei got the data analyzed, this is what i got:
i will work on getting the answers of the question, before proceeding, i did not understand hte 4th question

* * *

**ChatGPT:**

The analysis is actually quite good. Whoever did it understands time-series process data better than most ML beginners.

Pasted markdown

About **Question 4**, here's what it means in refinery/process language:

### Q4: Two Operating Regimes

The analyzer found:

```
Reboiler Outlet Temp:
58% of values below 50°C
42% above 50°C

Column Top Temp:
also split into two groups
```

Normally, process variables tend to form one broad distribution.

For example:

```
Top Temperature

103
104
105
106
107
108
109
```

This is one operating regime.

But your data apparently looks more like:

```
Group 1:
30°C
35°C
40°C
45°C

Group 2:
100°C
105°C
110°C
115°C
```

Notice there is almost nothing in between.

That's called a **bimodal distribution**.

* * *

## Why this matters

It means the column is not always being operated in the same way.

Possible reasons:

### 1. Different Feed Composition

Sometimes feed contains:

```
More light components
```

Other times:

```
More heavy components
```

The column behaves differently.

* * *

### 2. Different Product Targets

Sometimes operations may prioritize:

```
Maximum recovery
```

Other times:

```
Minimum energy usage
```

Leading to different temperature ranges.

* * *

### 3. Plant Startup / Shutdown

You might have:

```
Normal operation
```

and

```
Startup stabilization
```

mixed together.

* * *

### 4. Equipment Modifications

Remember you have:

```
2023
2024
2025
2026
```

data.

Maybe:

* tray replaced
* steam control changed
* APC enabled/disabled
* feed source changed

The process after modification may look completely different.

* * *

## Why your analyst asked this

Suppose:

### Regime A

```
Top Temp = 40°C
Bottom Temp = 105°C

C4 = 0.4%
```

### Regime B

```
Top Temp = 110°C
Bottom Temp = 105°C

C4 = 0.4%
```

Same C4.

Completely different temperatures.

The model gets confused.

* * *

## What you should ask the process engineer

Something like:

> We observed two distinct operating regions in the temperature data.
> 
> Is the debutanizer operated under different modes, feed compositions, seasonal conditions, startup/shutdown conditions, or product specifications that could explain these separate temperature ranges?

* * *

## Something more important than Q4

Looking at the report, I would ask this first:

### "Why are temperatures sometimes around 30–40°C and other times 100–110°C?"

Because that is huge.

For a debutanizer, a jump like:

```
Top Temp
35°C → 110°C
```

is not a normal fluctuation.

It suggests one of:

1. Wrong sensor scaling
2. Different tag calibration
3. Different operating mode
4. Different units during some periods
5. Data export issue

Before building any model, I would verify that those temperature values are physically correct.

* * *

### One thing I disagree with in the report

This statement:

> Total C4 = C4H6 + C4H8

I would **NOT assume this yet**.

You must confirm from IOCL.

Because C4 slippage may include:

```
Butadiene (C4H6)
Butenes (C4H8)
Butane (C4H10)
Isobutylene
Other C4s
```

The actual KPI might be:

```
Total C4 in DB Bottom
```

which may not equal:

```
C4H6 + C4H8
```

So before training anything, get clarification on:

1. What exactly is the target?
2. Is there lab data available?
3. Why do temperatures show two distinct regimes?
4. What caused the 13-month data gap?
5. Which variables can operators actually manipulate? (steam, reflux, pressure, etc.)

Of these, **Question 1 (target variable)** is the most critical. If that is wrong, the entire ML model will be optimizing the wrong thing.

* * *

**You:**

here is the new plan:
# AI-Based Model to Minimize C4 Slippage in Debutanizer

## Dataset Analysis Findings

### Overview
- **Dataset**: [9.DB DATA -B.xlsx](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/9.DB%20DATA%20-B.xlsx) — 11,399 hourly rows (after removing 2 header rows)
- **Date Range**: April 2023 → April 2026 (~3 years)
- **Columns**: 8 process variables + 2 target outputs (C4H6 wt%, C4H8 wt%)
- **Sampling**: Hourly (1-hour median interval), only 2 gaps > 2 hours

### Data Columns

| Column | Tag ID | Unit | Description |
|--------|--------|------|-------------|
| Feed Flow | 1145FIC1101 | TPH | Feed flow to debutanizer |
| Reboiler Outlet Temp | 1150TI1107 | °C | Reboiler outlet temperature |
| Column Top Temp | 1150TI1202 | °C | Column top temperature |
| Reboiling Steam Flow | 1150FIC1101 | TPH | LP steam flow |
| Reflux Flow | 1150FIC1202 | TPH | Reflux flow |
| Column Top Pressure | 1150PIC1101 | kg/cm²g | Column top pressure |
| Column Bottom Temp | 1150TI1105 | °C | Column bottom temperature |
| Control Tray Temp | 1150TIC1101 | °C | Control tray temperature |
| **C4H6 in DB Bottom** | 1150AI14011F | Wt% | Butadiene slippage (target 1) |
| **C4H8 in DB Bottom** | 1150AI14011G | Wt% | Butylene slippage (target 2) |

### Target Variable: Total C4 Slippage (C4H6 + C4H8)

| Stat | Value |
|------|-------|
| Mean | 0.50 wt% |
| Median | 0.40 wt% |
| Std | 0.37 wt% |
| Min | 0.00007 wt% |
| Max | 2.75 wt% |
| **% exceeding 0.5 spec** | **39.9%** |
| % exceeding 1.0 | 10.9% |
| % exceeding 1.5 | 1.5% |

> [!IMPORTANT]
> Nearly **40% of the time**, the plant exceeds the 0.5 wt% spec — confirming the problem statement is real and significant.

---

### Critical Data Quality Issues Found

#### 1. 🚨 13-Month Data Gap (Missing Data, Not Shutdown)
The monthly breakdown reveals the data is NOT continuous:
- **Block 1**: Apr 2023 – Aug 2023 (5 months, ~3,288 rows)
- **GAP**: Sep 2023 – Aug 2024 (13 months — data not recorded/exported)
- **Block 2**: Sep 2024 – Nov 2024 (3 months, ~1,541 rows)
- **GAP**: Dec 2024 (missing)
- **Block 3**: Aug 2025 – Apr 2026 (9 months, ~6,514 rows)

> [!WARNING]
> Since this is a data gap (not a plant shutdown), we must treat blocks independently for lag feature creation. No lag features should span across block boundaries.

#### 2. 🚨 Bimodal Distribution in Key Variables
- **Reboiler Outlet Temp**: 58% below 50°C, 42% above 107°C — **two operating regimes** (cause unknown; model will auto-detect via clustering or the data itself)
- **Column Top Temp**: 41% below 50°C, 59% above 50°C — same bimodal pattern
- **Strategy**: Add binary operating_regime feature via K-Means clustering; train model across both regimes — tree-based models handle this naturally

#### 3. 🚨 Stuck Analyzer Readings (Unreliable C4H6 Analyzer)
- **C4H6**: 42.7% of readings are repeats; max 889 consecutive identical readings (~37 days!)
- **C4H8**: 7.6% repeats; max 333 consecutive identical readings (~14 days)

> [!IMPORTANT]
> **Strategy for handling unreliable C4H6**: Since the requirement is to predict **Total C4 (C4H6 + C4H8)**, we'll use a **two-model ensemble approach**:
> 1. **Model A** — Predict C4H8 directly (reliable analyzer, good training signal)
> 2. **Model B** — Predict C4H6 directly, but trained **only on non-stuck readings** (filter out runs of >12 consecutive identical values; ~6,500 valid rows remain)
> 3. **Total C4 = Model A output + Model B output**
> 
> This way, each model learns from its best-quality data rather than training on a noisy combined target.

#### 4. Plant Shutdown Periods
- **56 rows** where ALL process variables = 0 simultaneously — removed before modeling

#### 5. Outliers
- Feed Flow: 3.0% outliers, Reboiling Steam: 2.3%, Total C4: 2.0%
- Not extreme — capped at 1st/99th percentile (not removed)

---

### Correlation & Time-Delay Analysis

#### What Drives High C4 Slippage?
Comparison of operating conditions when C4 is **low (<0.3%)** vs **high (>0.8%)**:

| Variable | Low C4 Mean | High C4 Mean | Difference | Interpretation |
|----------|-------------|--------------|------------|----------------|
| Feed Flow | 80.3 TPH | 85.8 TPH | **+5.4** | Higher feed → more slippage |
| Reflux Flow | 92.8 TPH | 90.6 TPH | **-2.2** | Less reflux → more slippage |
| Reboiling Steam | 21.2 TPH | 20.8 TPH | **-0.3** | Less steam → more slippage |
| Reboiler Outlet Temp | 66.2°C | 56.9°C | **-9.3** | Lower temp → more slippage |
| Column Top Pressure | 4.01 | 4.15 | **+0.14** | Higher pressure → more slippage |

> [!NOTE]
> **Steam and Reflux are highly correlated** (r = 0.88). The optimizer must account for this — changes to one will likely require coordinated changes to the other.

#### Autocorrelation of Total_C4 (Very High!)
| Lag | Autocorrelation |
|-----|----------------|
| 1h | **0.938** |
| 6h | 0.794 |
| 12h | 0.724 |
| 24h | 0.724 |

Past C4 values are the strongest predictor. Naive baseline (C4(t) = C4(t-1)) achieves R² ≈ 0.88.

---

## Model Constraints & Operating Limits

> [!CAUTION]
> The optimizer must **never** recommend values outside safe operating limits. Extreme recommendations could cause column flooding, foaming, pressure surges, or emergency shutdown.

### Constraint Strategy

1. **Hard Limits** (from data P1-P99 + refinery literature): Model output is clipped — **never** exceeded
2. **Recommended Operating Range** (from data P5-P95): Optimizer searches **only within this range**
3. **Rate-of-Change Limits**: Maximum allowed change per hour — prevents sudden dangerous swings (the ±50% idea expressed as a rate constraint)

### Manipulable Variables (Optimizer Can Adjust)

| Variable | Unit | Hard Min | Rec. Min | Mean | Rec. Max | Hard Max | Max Δ/hr |
|----------|------|----------|----------|------|----------|----------|----------|
| Reboiling Steam Flow | TPH | 14.4 | 18.0 | 21.0 | 24.4 | 25.3 | ±2.0 |
| Reflux Flow | TPH | 70.4 | 80.0 | 91.1 | 103.9 | 105.7 | ±5.0 |

### Monitored Variables (Constraints — Optimizer Must Respect)

| Variable | Unit | Hard Min | Rec. Min | Mean | Rec. Max | Hard Max | Trip/Alarm |
|----------|------|----------|----------|------|----------|----------|------------|
| Column Bottom Temp | °C | 99.0 | 102.6 | 107.0 | 111.5 | 113.0 | >115°C alarm |
| Control Tray Temp | °C | 57.2 | 59.1 | 71.6 | 89.7 | 91.3 | — |
| Column Top Pressure | kg/cm²g | 3.78 | 3.85 | 4.05 | 4.45 | 4.55 | >5.0 trip |
| Feed Flow | TPH | 44.6 | 65.4 | 83.3 | 95.3 | 99.3 | Disturbance (not controlled) |

### Constraint Sources

| Source | What It Provides |
|--------|-----------------|
| **Historical data (P5-P95)** | The "normal" operating envelope — values within which the plant has successfully operated 90% of the time |
| **Historical data (P1-P99)** | Hard limits — only 1% of data falls outside these |
| **Refinery literature** | Typical debutanizer overhead temp: 40-70°C, bottoms: 100-150°C, pressure: 5-16 bar. Our data (3.8-4.6 kg/cm²g ≈ 3.7-4.5 bar) operates on the lower end — consistent with LP steam reboiling |
| **Safety considerations** | Flooding risk at high reflux, fouling risk at high reboiler temp (>125°C), pressure trip at >5 kg/cm²g |

### Rate-of-Change Constraints (Preventing Sudden Swings)
Instead of a blanket ±50% of current value, we use **maximum allowed step change per hour**:

Steam: max ±2.0 TPH/hr   (≈10% of mean, prevents thermal shock to reboiler)
Reflux: max ±5.0 TPH/hr  (≈5% of mean, prevents flooding/weeping)

These were derived from the 95th percentile of actual hourly changes observed in the data, so they represent the range operators historically adjust within.

---

## Resolved Questions

| Question | Resolution |
|----------|-----------|
| Target variable | **Total C4 (C4H6 + C4H8)** — Two-model ensemble: predict C4H8 and C4H6 separately, sum for Total C4 |
| Cost formula | Placeholder: Loss_INR_per_hr = max(0, C4_predicted - 0.5) × Feed_Flow × 1000 × C4_price_per_kg (calibrate later with actual economics) |
| Data gap | Missing data export, not shutdown. Handle by splitting into contiguous blocks |
| Operating regimes | Unknown cause. Model will auto-detect via the data; tree-based model handles bimodal distributions naturally |
| Constraints | Data-driven limits (P5-P95 for optimizer, P1-P99 as hard limits) validated against refinery literature |

---

## Proposed Changes

### Phase 1: Data Preprocessing Pipeline

#### [NEW] [data_preprocessing.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/data_preprocessing.py)

Complete data cleaning pipeline:

1. **Parse & rename columns** (tag IDs → human-readable names)
2. **Remove header rows** (rows 0-1 contain tag IDs and units)
3. **Remove shutdown rows** (56 rows where all process vars = 0)
4. **Handle stuck analyzer readings**:
   - Detect runs of >12 consecutive identical C4H6 readings → mark as C4H6_stuck = True
   - For **Model B training**: exclude stuck rows (keep ~6,500 valid rows)
   - For **Model A**: C4H8 is reliable, use all rows
5. **Outlier capping**: Winsorize at 1st/99th percentile
6. **Identify data blocks**: Label contiguous time blocks (Block 1/2/3) to prevent lag features from crossing gaps
7. **Create time features**: hour_of_day, day_of_week, month (cyclical sine/cosine encoding)
8. **Output**: Clean parquet file ready for feature engineering

---

### Phase 2: Feature Engineering

#### [NEW] [feature_engineering.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/feature_engineering.py)

**Lagged Process Variables** (for each of 8 process vars):
- {var}_lag1, {var}_lag2, {var}_lag3 (1-3 hour lags)
- {var}_lag6, {var}_lag12 (slower response variables)
- **Only within contiguous blocks** — no lag across data gaps

**Lagged Target Variables** (critical — autocorrelation 0.94):
- C4H8_lag1, C4H8_lag2, C4H8_lag3, C4H8_lag6, C4H8_lag12
- C4H6_lag1, C4H6_lag2, C4H6_lag3 (for non-stuck periods only)

**Rolling Statistics**:
- {var}_rolling_mean_3h, _6h, _12h
- {var}_rolling_std_3h (variability = instability)

**Engineered Ratios**:
- Reflux_Ratio = Reflux / Feed
- Steam_Feed_Ratio = Steam / Feed
- Temp_Gradient = Bottom - Top
- Reboiler_Delta = Reboiler_Outlet - Bottom

**Rate-of-Change** (captures dynamic behavior):
- {var}_diff1 = {var}(t) - {var}(t-1) for Steam, Reflux, Feed, Bottom Temp

**Operating Regime** (auto-detected):
- Binary feature from K-Means on [Reboiler_Outlet_Temp, Column_Top_Temp]

Total estimated features: **~80-100** → pruned via feature importance

---

### Phase 3: Model Training

#### [NEW] [model_training.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/model_training.py)

**Two-Model Ensemble Architecture**:

┌──────────────────┐
Process Features ──┤  Model A (XGBoost)├──► Predicted C4H8 ──┐
 + Lag features    │  (all rows)      │                      │
                   └──────────────────┘                      ├──► Total C4
                   ┌──────────────────┐                      │
Process Features ──┤  Model B (XGBoost)├──► Predicted C4H6 ──┘
 + Lag features    │  (non-stuck only)│
                   └──────────────────┘

**Training Strategy**:
1. **Block-aware split**: Block 1+2 for training, Block 3 for test (most recent = most realistic)
2. **Time-series CV**: Expanding window within training set
3. **Baseline**: Naive lag-1 predictor (R² ≈ 0.88) — must beat this
4. **Hyperparameter tuning**: Optuna Bayesian optimization
5. **Constraint-aware validation**: Verify model never predicts C4 < 0 or > 3.0 wt%

**Metrics**: MAE, RMSE, R², MAPE, % within ±0.1 wt% of actual

---

### Phase 4: Constrained Optimization Engine

#### [NEW] [optimizer.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/optimizer.py)

**Constrained optimizer** that respects all operating limits:

python
# Pseudo-code for optimizer
minimize:  predicted_C4(steam, reflux, ...)
subject to:
    18.0 <= steam <= 24.4                    # Recommended range
    80.0 <= reflux <= 103.9                  # Recommended range
    |steam_new - steam_current| <= 2.0       # Rate constraint
    |reflux_new - reflux_current| <= 5.0     # Rate constraint
    predicted_bottom_temp <= 111.5           # Safety constraint
    predicted_pressure <= 4.45               # Safety constraint

- Uses scipy.optimize.minimize with bounds and constraints
- Falls back to grid search if optimization fails
- **Energy penalty term**: total_cost = C4_loss_cost - energy_savings (balance recovery vs energy)
- Outputs: recommended setpoints, expected C4 reduction, energy trade-off, INR savings

---

### Phase 5: Streamlit Dashboard

#### [NEW] [app.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/app.py)

**Tab 1: Live Prediction**
- Process value inputs (manual or CSV upload)
- C4 prediction gauge (green < 0.3, yellow 0.3-0.5, red > 0.5)
- SHAP waterfall for feature contributions

**Tab 2: Actual vs Predicted Trends**
- Overlay chart of actual vs predicted C4
- Model accuracy metrics (MAE, RMSE, R²)
- Anomaly flag when predicted ≠ actual (analyzer may be stuck)

**Tab 3: Operator Recommendations**
- Current vs optimal operating point
- Recommended steam/reflux adjustments **with constraint limits shown**
- **Constraint violation warnings** (if current values are near limits)
- Loss calculator: INR/hr at current C4 level
- "What-if" simulator: sliders (bounded by operating limits) to test changes

**Tab 4: Analytics & Reports**
- Monthly C4 trends, feature importance
- Operating regime analysis
- PDF report generation (reportlab)

---

### Phase 6: Supporting Files

#### [MODIFY] [requirements.txt](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/requirements.txt)
Add: scikit-learn, optuna, shap, plotly, joblib, scipy

#### [NEW] [config.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/config.py)
Central configuration including:
- Column mappings and tag IDs
- **Operating constraints table** (hard limits, recommended ranges, rate limits)
- Cost parameters (C4 price, energy costs)
- Model hyperparameters

#### [DELETE] [analyze_data.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/analyze_data.py)
#### [DELETE] [analyze_constraints.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/analyze_constraints.py)
Temporary analysis scripts — findings captured in this document.

---

## Verification Plan

### Automated Tests
1. **Preprocessing**: No NaN/inf in output, correct row count, correct dtypes
2. **Feature engineering**: Lag features correctly aligned, no look-ahead bias, no cross-block leakage
3. **Model performance**:
   - Must beat naive lag-1 baseline (R² > 0.88)
   - Target: MAE < 0.05 wt%, R² > 0.92
   - Cross-validation stability: std(R²) < 0.05
4. **Constraint enforcement**: Verify optimizer never recommends values outside hard limits
5. **Dashboard**: All tabs load, charts render, predictions match model output

### Manual Verification
- Upload original Excel data → verify predictions visually match trends
- Test what-if simulator with known scenarios
- Verify constraint warnings trigger correctly near limits
- Generate PDF report and check completeness

---

## Execution Order

mermaid
graph LR
    A[Phase 1: Preprocessing] --> B[Phase 2: Feature Eng.]
    B --> C[Phase 3: Model Training]
    C --> D[Phase 4: Optimizer]
    D --> E[Phase 5: Dashboard]
    E --> F[Phase 6: Polish & Docs]

Estimated effort: ~4-6 hours total.

* * *

**ChatGPT:**

This plan is **much stronger** than the previous one. It's starting to look like something a refinery data science team would actually review.

That said, there are a few things I would change before you commit to implementation.

* * *

# 1. Biggest Risk: Using Lagged C4 as Input

You wrote:

```
C4H8_lag1, lag2, lag3...
C4H6_lag1, lag2, lag3...
```

Technically this will give amazing accuracy.

But ask yourself:

> Why is IOCL building a soft sensor?

Because:

```
Analyzer is unreliable
Analyzer updates slowly
Lab is delayed
```

If tomorrow the analyzer freezes for 37 days:

```
C4H6 = constant
```

your model loses one of its strongest features.

You could accidentally build:

```
Predict C4 using previous C4
```

instead of:

```
Predict C4 using process variables
```

which defeats the purpose.

* * *

My recommendation:

### Model 1 (Production Soft Sensor)

Use ONLY:

```
Feed
Steam
Reflux
Pressure
Temperatures
Ratios
Lagged process variables
```

NO lagged target.

* * *

### Model 2 (Research Model)

Allow:

```
C4 lag features
```

Then compare.

This will show management:

```
Soft Sensor R² = 0.87

Soft Sensor + Analyzer History R² = 0.95
```

Very valuable information.

* * *

# 2. I Would Not Use K-Means Yet

You wrote:

```
Operating Regime = KMeans
```

Sounds good on paper.

But with only:

```
Top Temp
Reboiler Temp
```

you may create artificial regimes.

First check:

```Python
sns.scatterplot(
    x=reboiler_temp,
    y=top_temp,
    hue=target
)
```

If clusters clearly exist:

then KMeans.

Otherwise:

skip it.

* * *

Tree models already naturally split:

```
if temp > 50:
    ...
else:
    ...
```

without needing explicit clustering.

* * *

# 3. Ensemble for C4H6 + C4H8

This is actually smart.

I like:

```
Model A → C4H8
Model B → C4H6
Total = A + B
```

Because the quality of the analyzers differs.

This is much better than:

```
Total C4 = raw sum
```

and training a single model.

* * *

# 4. Optimizer Needs a Reality Check

Right now:

```
Optimizer:
Steam
Reflux
↓
Minimize C4
```

Problem:

the optimizer may discover:

```
Steam = max
Reflux = max
```

gives lowest C4.

Which operators already know.

* * *

Need objective:

```
Loss =
C4 Loss Cost
+
Steam Cost
+
Reflux Cost
```

Not:

```
Minimize C4 only
```

Otherwise optimizer becomes:

```
Throw energy at the problem.
```

* * *

# 5. Your Performance Target Is Dangerous

You wrote:

```
R² > 0.92
MAE < 0.05
```

I would remove those.

Industrial datasets are messy.

Instead:

```
Goal:
Beat lag baseline.

Stretch goal:
R² > 0.85
```

If you promise:

```
0.92
```

and get:

```
0.88
```

you look like you failed.

Even though 0.88 may be excellent.

* * *

# 6. Add Feature Importance Validation

This is something managers love.

After training:

Top features should make physical sense.

Example:

Good:

```
1 Reflux Ratio
2 Steam/Feed Ratio
3 Top Pressure
4 Bottom Temperature
```

Bad:

```
1 Month
2 DayOfWeek
3 Hour
```

If nonsense features dominate:

model is probably learning artifacts.

* * *

# 7. Add Drift Detection

Since you have:

```
2023
2024
2025
2026
```

I'd add:

```
Population Stability Index (PSI)
```

or at least:

```
Feature distribution monitoring
```

because refinery conditions change.

A model trained on 2023 may degrade in 2026.

This becomes a great "Future Scope" slide.

* * *

# 8. The 4–6 Hours Estimate Is Fantasy 😭

Realistically:

### Data Cleaning

1–2 days

### Feature Engineering

1 day

### Training + Tuning

1 day

### Optimizer

1 day

### Dashboard

1–2 days

Total:

```
5–7 working days
```

for a solid result.

* * *

# What I Would Present to IOCL

Final architecture:

```
Historian Data
        ↓
Data Validation
        ↓
Analyzer Fault Detection
        ↓
Feature Engineering
        ↓
Soft Sensor Model
        ↓
C4H6 Prediction

Historian Data
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Soft Sensor Model
        ↓
C4H8 Prediction

C4H6 + C4H8
        ↓
Total C4 Slip

Total C4 Slip
        ↓
Constrained Optimizer
        ↓
Recommended Steam/Reflux

Recommended Steam/Reflux
        ↓
Expected Savings (₹/hr)
```

The strongest parts of your current plan are:

✅ Treating C4H6 and C4H8 separately  
✅ Block-aware time series handling  
✅ Safety constraints  
✅ Rate-of-change constraints  
✅ Actual refinery operating limits

The thing I'd change immediately is:

❌ Don't make lagged C4 values a dependency of the production soft sensor.

A true soft sensor should still work when the analyzer is dead, frozen, or unavailable—which is exactly the situation this project is supposed to solve.

* * *

**You:**

# AI-Based Model to Minimize C4 Slippage in Debutanizer

## Dataset Analysis Findings

### Overview
- **Dataset**: [9.DB DATA -B.xlsx](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/9.DB%20DATA%20-B.xlsx) — 11,399 hourly rows (after removing 2 header rows)
- **Date Range**: April 2023 → April 2026 (~3 years)
- **Columns**: 8 process variables + 2 target outputs (C4H6 wt%, C4H8 wt%)
- **Sampling**: Hourly (1-hour median interval), only 2 gaps > 2 hours

### Data Columns

| Column | Tag ID | Unit | Description |
|--------|--------|------|-------------|
| Feed Flow | 1145FIC1101 | TPH | Feed flow to debutanizer |
| Reboiler Outlet Temp | 1150TI1107 | °C | Reboiler outlet temperature |
| Column Top Temp | 1150TI1202 | °C | Column top temperature |
| Reboiling Steam Flow | 1150FIC1101 | TPH | LP steam flow |
| Reflux Flow | 1150FIC1202 | TPH | Reflux flow |
| Column Top Pressure | 1150PIC1101 | kg/cm²g | Column top pressure |
| Column Bottom Temp | 1150TI1105 | °C | Column bottom temperature |
| Control Tray Temp | 1150TIC1101 | °C | Control tray temperature |
| **C4H6 in DB Bottom** | 1150AI14011F | Wt% | Butadiene slippage (target 1) |
| **C4H8 in DB Bottom** | 1150AI14011G | Wt% | Butylene slippage (target 2) |

### Target Variable: Total C4 Slippage (C4H6 + C4H8)

| Stat | Value |
|------|-------|
| Mean | 0.50 wt% |
| Median | 0.40 wt% |
| Std | 0.37 wt% |
| Min | 0.00007 wt% |
| Max | 2.75 wt% |
| **% exceeding 0.5 spec** | **39.9%** |
| % exceeding 1.0 | 10.9% |
| % exceeding 1.5 | 1.5% |

> [!IMPORTANT]
> Nearly **40% of the time**, the plant exceeds the 0.5 wt% spec — confirming the problem statement is real and significant.

---

### Critical Data Quality Issues Found

#### 1. 🚨 13-Month Data Gap (Missing Data, Not Shutdown)
- **Block 1**: Apr 2023 – Aug 2023 (5 months, ~3,288 rows)
- **GAP**: Sep 2023 – Aug 2024 (13 months — data not recorded/exported)
- **Block 2**: Sep 2024 – Nov 2024 (3 months, ~1,541 rows)
- **GAP**: Dec 2024 (missing)
- **Block 3**: Aug 2025 – Apr 2026 (9 months, ~6,514 rows)

> [!WARNING]
> Lag features must never cross block boundaries.

#### 2. Bimodal Distribution in Key Variables
- **Reboiler Outlet Temp**: 58% below 50°C, 42% above 107°C
- **Column Top Temp**: 41% below 50°C, 59% above 50°C
- Cause unknown. **No K-Means clustering** — tree models handle bimodal splits naturally via if temp > threshold. A scatter plot will be generated as a validation artifact to visually confirm whether true clusters exist.

#### 3. 🚨 Stuck Analyzer Readings
- **C4H6**: 42.7% repeats; max 889 consecutive identical readings (~37 days!)
- **C4H8**: 7.6% repeats; max 333 consecutive identical (~14 days)
- This is **the reason** we're building a soft sensor.

#### 4. Shutdowns & Outliers
- 56 all-zero rows removed (plant offline)
- Outliers capped at P1/P99 (not removed — process upsets carry information)

---

### What Drives High C4 Slippage?

| Variable | Low C4 (<0.3%) | High C4 (>0.8%) | Δ | Physical Interpretation |
|----------|---------------|-----------------|---|------------------------|
| Feed Flow | 80.3 TPH | 85.8 TPH | **+5.4** | Higher feed overwhelms separation |
| Reflux Flow | 92.8 TPH | 90.6 TPH | **-2.2** | Less reflux = poorer fractionation |
| Reboiling Steam | 21.2 TPH | 20.8 TPH | **-0.3** | Less energy = less stripping |
| Reboiler Outlet Temp | 66.2°C | 56.9°C | **-9.3** | Colder reboiler = insufficient vaporization |
| Column Top Pressure | 4.01 | 4.15 | **+0.14** | Higher pressure = harder separation |

> [!NOTE]
> **Steam and Reflux are highly correlated** (r = 0.88). Optimizer must handle this — they move together.

---

## Model Architecture: Two-Tier Strategy

> [!CAUTION]
> **Why NOT use lagged C4 as input in the production model?**
>
> IOCL is building this soft sensor because the analyzer is unreliable (stuck for 37 days), slow (12-min cycle), and lab results are delayed (2 hrs). If we use C4_lag1 as a feature, we get amazing R² — but we've built a model that says *"tomorrow's C4 ≈ today's C4"*. The moment the analyzer freezes, our strongest feature becomes stale, and the model collapses.
>
> The production soft sensor must predict C4 **purely from process variables** — that's the whole point.

### Model Tier 1: Production Soft Sensor (Deployed)

**Purpose**: Replace/supplement the unreliable analyzer. Must work even when analyzer is frozen.

**Features (process-only)**:
- Current process variables: Feed, Steam, Reflux, Pressure, Temperatures
- Lagged **process** variables: Steam_lag1..lag3, Reflux_lag1..lag3, Feed_lag1..lag3, etc.
- Rolling stats: Steam_rolling_mean_3h, Feed_rolling_std_6h, etc.
- Engineered ratios: Reflux/Feed, Steam/Feed, Temp gradients
- Rate-of-change: Steam_diff1, Reflux_diff1, etc.
- Time features: hour, day_of_week, month (cyclical)
- **NO lagged C4 target values**

**Sub-models** (ensemble for Total C4):
- **Model A**: Predict C4H8 (reliable analyzer → all rows for training)
- **Model B**: Predict C4H6 (unreliable → trained only on non-stuck readings)
- **Total C4 = Model A + Model B**

### Model Tier 2: Research Model (Comparison Only)

**Purpose**: Show management the value of analyzer history when it's working.

**Features**: Everything from Tier 1 **plus** lagged C4 values (C4H8_lag1..lag12, C4H6_lag1..lag3)

**Expected outcome**:
Production Soft Sensor R² = ~0.85
Research Model R²          = ~0.93+

This comparison is extremely valuable for presentations:
- Proves the soft sensor works without the analyzer
- Quantifies exactly how much analyzer history helps
- Justifies investment in better analyzer hardware (if the gap is large)

---

## Model Constraints & Operating Limits

> [!CAUTION]
> The optimizer must **never** recommend values outside safe operating limits.

### Manipulable Variables (Optimizer Can Adjust)

| Variable | Unit | Hard Min | Rec. Min | Mean | Rec. Max | Hard Max | Max Δ/hr |
|----------|------|----------|----------|------|----------|----------|----------|
| Reboiling Steam Flow | TPH | 14.4 | 18.0 | 21.0 | 24.4 | 25.3 | ±2.0 |
| Reflux Flow | TPH | 70.4 | 80.0 | 91.1 | 103.9 | 105.7 | ±5.0 |

### Monitored Variables (Constraints)

| Variable | Unit | Hard Min | Rec. Min | Mean | Rec. Max | Hard Max | Trip/Alarm |
|----------|------|----------|----------|------|----------|----------|------------|
| Column Bottom Temp | °C | 99.0 | 102.6 | 107.0 | 111.5 | 113.0 | >115°C alarm |
| Control Tray Temp | °C | 57.2 | 59.1 | 71.6 | 89.7 | 91.3 | — |
| Column Top Pressure | kg/cm²g | 3.78 | 3.85 | 4.05 | 4.45 | 4.55 | >5.0 trip |
| Feed Flow | TPH | 44.6 | 65.4 | 83.3 | 95.3 | 99.3 | Disturbance |

### Rate-of-Change Constraints
Steam: max ±2.0 TPH/hr  (≈10% of mean, prevents thermal shock)
Reflux: max ±5.0 TPH/hr (≈5% of mean, prevents flooding/weeping)

Derived from the 95th percentile of actual hourly changes in the data.

---

## Resolved Questions

| Question | Resolution |
|----------|-----------|
| Target | Total C4 (C4H6 + C4H8) — Two-model ensemble |
| Cost formula | Placeholder: Loss = max(0, C4 - 0.5) × Feed × 1000 × price_per_kg |
| Data gap | Missing data export. Split into contiguous blocks |
| Operating regimes | Unknown cause. No K-Means — trees handle naturally |
| Constraints | Data-driven (P5-P95 / P1-P99) + refinery literature |

---

## Proposed Changes

### Phase 1: Data Preprocessing

#### [NEW] [data_preprocessing.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/data_preprocessing.py)

1. Parse & rename columns
2. Remove 2 header rows + 56 shutdown rows
3. Detect stuck analyzer readings (runs of >12 identical C4H6 values) → mark with C4H6_stuck flag
4. Outlier capping at P1/P99
5. Label contiguous time blocks (Block 1/2/3)
6. Cyclical time features (hour, day_of_week, month via sin/cos)
7. Output: clean parquet file

---

### Phase 2: Feature Engineering

#### [NEW] [feature_engineering.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/feature_engineering.py)

**Process-only features** (for Production Soft Sensor):

| Category | Features | Count |
|----------|----------|-------|
| Lagged process vars | {var}_lag1..lag3 for 8 vars + _lag6, _lag12 for slow vars | ~50 |
| Rolling stats | {var}_rolling_mean_3h/6h/12h, _rolling_std_3h | ~32 |
| Ratios | Reflux/Feed, Steam/Feed, Temp gradients, Reboiler delta | 4 |
| Rate-of-change | {var}_diff1 for Steam, Reflux, Feed, Bottom Temp | 4 |
| Time | hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos | 6 |
| **Total** | | **~96** |

**Additional features for Research Model only**:
- C4H8_lag1..lag12, C4H6_lag1..lag3 (~8 more features)

All lag/rolling features computed **within blocks only** — no cross-gap leakage.

---

### Phase 3: Model Training

#### [NEW] [model_training.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/model_training.py)

**Architecture**:

PRODUCTION SOFT SENSOR (Tier 1)
─────────────────────────────────
Process features ──┬── Model A (XGBoost) ──► C4H8_pred ──┐
(~96 features)     │   (all rows)                        ├──► Total C4
                   └── Model B (XGBoost) ──► C4H6_pred ──┘
                       (non-stuck rows only)

RESEARCH MODEL (Tier 2)
─────────────────────────────────
Process + C4 lag ──┬── Model A+ ──► C4H8_pred ──┐
(~104 features)    │                             ├──► Total C4
                   └── Model B+ ──► C4H6_pred ──┘

**Training Strategy**:
1. **Block-aware split**: Block 1+2 train, Block 3 test
2. **Time-series CV**: Expanding window (no random K-fold)
3. **Hyperparameters**: Optuna Bayesian optimization over learning_rate, max_depth, n_estimators, colsample, subsample, reg_alpha, reg_lambda

**Performance Goals**:

| Metric | Goal | Notes |
|--------|------|-------|
| **Primary** | Beat naive lag-1 baseline (R² > 0.88) | If we can't beat "C4(t) = C4(t-1)" with process vars, the model has no value |
| **Stretch** | R² > 0.85 for Production Soft Sensor | Industrial datasets are messy — 0.85 is excellent |
| **Comparison** | Show gap: Soft Sensor R² vs Research Model R² | Quantifies analyzer value for management |

> [!IMPORTANT]
> **No hard R²/MAE promises.** Report actual performance honestly. Even R² = 0.80 with a process-only model is a significant achievement for an industrial soft sensor on noisy data.

**Feature Importance Validation (Physics Sanity Check)**:

After training, the top features must make physical sense:

| ✅ Good (physically meaningful) | ❌ Bad (artifacts) |
|--------------------------------|-------------------|
| Reflux_Ratio, Steam_Feed_Ratio | month_sin |
| Column_Top_Pressure | day_of_week_cos |
| Bottom_Temp, Reboiler_Outlet_Temp | hour_sin |
| Feed_Flow, Steam_lag1..lag3 | Random feature |

If nonsense features dominate top-10 importance:
1. Model is learning temporal artifacts, not physics
2. Investigate: are time features correlating with seasonal operating changes?
3. Consider removing time features and retraining
4. Add a **random noise column** during training — if it ranks high, model is overfitting

This validation will be a dedicated section in the dashboard and final report.

---

### Phase 4: Constrained Multi-Objective Optimizer

#### [NEW] [optimizer.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/optimizer.py)

> [!WARNING]
> **Not just "minimize C4"** — that trivially recommends max steam + max reflux, which operators already know. The optimizer must balance C4 recovery against energy cost.

**Multi-objective formulation**:

python
# Objective: minimize total operating cost
total_cost = (
    C4_loss_cost                          # ₹ lost from C4 slippage above spec
    + steam_cost                          # ₹ cost of LP steam consumed
    + reflux_energy_cost                  # ₹ cost of reflux pumping + condenser duty
)

# Where:
C4_loss_cost    = max(0, predicted_C4 - 0.5) * feed_flow * 1000 * C4_price_per_kg
steam_cost      = steam_flow * steam_price_per_ton
reflux_cost     = reflux_flow * reflux_energy_per_ton

# Subject to:
18.0 <= steam <= 24.4                    # Recommended range
80.0 <= reflux <= 103.9                  # Recommended range
|steam_new - steam_current| <= 2.0       # Rate constraint
|reflux_new - reflux_current| <= 5.0     # Rate constraint
predicted_bottom_temp <= 111.5           # Safety
predicted_pressure <= 4.45               # Safety

**Outputs**:
- Recommended setpoints (steam, reflux)
- Expected C4 reduction (wt%)
- Energy cost of the change (₹/hr)
- **Net savings** = C4 recovery value − additional energy cost
- Trade-off curve: "spend X more on steam → save Y on C4"

---

### Phase 5: Drift Detection & Monitoring

#### [NEW] [drift_detection.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/drift_detection.py)

Since data spans 2023–2026, refinery conditions change (feed slate, catalyst aging, equipment fouling). A model trained on 2023 may degrade in 2026.

**Population Stability Index (PSI)** per feature across time blocks:

PSI < 0.1  → No drift (green)
PSI 0.1-0.2 → Moderate drift (yellow)  
PSI > 0.2  → Significant drift (red) → retrain recommended

**Implementation**:
1. Compute PSI for each feature: Block 1 vs Block 2, Block 2 vs Block 3
2. Flag features with PSI > 0.2
3. Track model residual distribution over time (are errors growing?)
4. Dashboard panel showing drift status per feature

**Dashboard integration**: A "Model Health" indicator showing:
- Feature drift status (per-feature PSI heatmap)
- Residual trend (is accuracy degrading over time?)
- Recommended action: "Model performing well" / "Consider retraining"

This becomes a strong **Future Scope** slide:
> *"When deployed live, PSI monitoring triggers automatic retraining when operating conditions drift beyond training distribution."*

---

### Phase 6: Streamlit Dashboard

#### [NEW] [app.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/app.py)

**Tab 1: Live Prediction**
- Process value inputs (manual or CSV upload)
- C4 prediction gauge (green < 0.3, yellow 0.3-0.5, red > 0.5)
- Confidence display
- SHAP waterfall: "why is C4 high right now?"

**Tab 2: Actual vs Predicted Trends**
- Overlay chart: actual C4 (analyzer) vs predicted C4 (soft sensor)
- Residual plot over time
- Anomaly flag: when soft sensor ≠ analyzer → "analyzer may be stuck"
- **Tier 1 vs Tier 2 comparison chart** (shows value of soft sensor)

**Tab 3: Operator Recommendations**
- Current vs optimal operating point
- Recommended adjustments with constraint limits shown visually
- **Net savings calculator**: C4 recovery savings − energy cost increase = net ₹/hr
- "What-if" simulator: bounded sliders
- Constraint violation warnings

**Tab 4: Model Health & Analytics**
- **Feature importance ranking** with physics validation (✅/❌ markers)
- **Drift detection heatmap** (PSI per feature per time block)
- Monthly C4 trends
- PDF report generation

---

### Phase 7: Supporting Files

#### [MODIFY] [requirements.txt](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/requirements.txt)
Add: scikit-learn, optuna, shap, plotly, joblib, scipy

#### [NEW] [config.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/config.py)
- Column mappings and tag IDs
- Operating constraints table (hard limits, recommended ranges, rate limits)
- Cost parameters (C4 price, steam price, reflux energy cost)
- Model hyperparameters
- PSI drift thresholds

#### [DELETE] [analyze_data.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/analyze_data.py)
#### [DELETE] [analyze_constraints.py](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/analyze_constraints.py)
Temporary analysis scripts — findings captured in this document.

---

## Exploratory Validation Steps (Before Model Training)

Before committing to the model architecture, generate these diagnostic plots:

1. **Regime scatter plot**: Reboiler_Outlet_Temp vs Column_Top_Temp, colored by Total_C4
   - If clear clusters exist → add regime feature
   - If gradient/continuous → skip clustering, let trees handle it

2. **Random noise feature test**: Add a random column during training
   - If it ranks in top-20 importance → model is overfitting
   - If bottom → model is learning real signal

3. **Block consistency check**: Train on Block 1 alone, test on Block 2 and Block 3
   - If performance drops heavily → significant drift between periods

---

## Verification Plan

### Automated Tests
1. **Preprocessing**: No NaN/inf, correct row count, dtypes
2. **Feature engineering**: Lag features aligned correctly, no look-ahead bias, no cross-block leakage
3. **Model performance**: Beat naive lag-1 baseline; report honest metrics
4. **Feature importance**: Top-10 features pass physics sanity check
5. **Constraint enforcement**: Optimizer never recommends outside hard limits
6. **Drift detection**: PSI computed correctly across all blocks

### Manual Verification
- Upload original Excel → verify predictions visually match trends
- Test what-if simulator with known scenarios
- Verify constraint warnings trigger near limits
- Review feature importance for physical plausibility
- Generate PDF report

---

## Execution Order

mermaid
graph LR
    A[Phase 1: Preprocessing] --> B[Phase 2: Feature Eng.]
    B --> V[Validation Plots]
    V --> C[Phase 3: Model Training]
    C --> D[Phase 4: Optimizer]
    C --> E[Phase 5: Drift Detection]
    D --> F[Phase 6: Dashboard]
    E --> F
    F --> G[Phase 7: Polish & Docs]

here is the updated plan

* * *

**ChatGPT:**

This version is significantly stronger. If I were reviewing this as a refinery analytics lead, I'd say you're now thinking like a process-data project rather than a generic ML project.

A few final improvements before you start coding:

* * *

# 1. Change the Primary Success Metric

Right now you still have:

```
Primary:
Beat lag-1 baseline (R² > 0.88)
```

For the **Production Soft Sensor**, I would not use that as the main success criterion.

Why?

Because the lag-1 baseline uses:

```
Previous C4
```

while your production model deliberately does not.

That's not a fair comparison.

Instead:

### Production Model Success

```
MAE
RMSE
R²
```

on Block 3.

Then separately report:

```
Naive Lag Model
Research Model
Production Model
```

Example:

| Model | R² |
| --- | --- |
| Naive Lag | 0.89 |
| Production Soft Sensor | 0.84 |
| Research Model | 0.93 |

This tells a much better story.

* * *

# 2. Add Prediction Horizon

This is the one thing missing that could make the project really valuable.

Currently:

```
Input now
↓
Predict C4 now
```

But operators already have an analyzer.

More useful:

```
Input now
↓
Predict C4 in 1 hour
Predict C4 in 2 hours
Predict C4 in 4 hours
```

Then operator can act before slippage occurs.

I would seriously consider:

### Model Targets

```
C4(t+1)
C4(t+2)
C4(t+4)
```

instead of:

```
C4(t)
```

or at least build both.

That transforms it from a soft sensor into a predictive advisory system.

* * *

# 3. Add Feature Stability Analysis

You already have PSI.

Excellent.

Now add:

```
Feature Importance
across blocks
```

Example:

| Feature | Block1 | Block2 | Block3 |
| --- | --- | --- | --- |
| Reflux Ratio | #1 | #2 | #1 |
| Pressure | #3 | #4 | #2 |

If rankings stay similar:

```
Physics-driven model
```

If rankings completely change:

```
Possible drift
```

Management likes seeing consistency.

* * *

# 4. Don't Trust Historical Limits Blindly

Current plan:

```
P1-P99
```

used as hard limits.

Be careful.

A historian contains:

```
Startup
Shutdown
Faults
Sensor spikes
```

I would rename them:

```
Historical Operating Envelope
```

not

```
Hard Limits
```

Actual hard limits should ideally come from:

```
Operating Manual
DCS Limits
Trip Settings
Engineer Validation
```

Use historian limits only if no official limits exist.

* * *

# 5. Add Uncertainty Estimation

This is a huge upgrade.

Instead of:

```
Predicted C4 = 0.63
```

show:

```
Predicted C4 = 0.63 ± 0.08
```

or:

```
95% interval:
0.55–0.71
```

Why?

Operators trust:

```
Prediction
+
confidence
```

much more than a single number.

You can get this using:

* Quantile XGBoost
* Ensemble variance
* Bootstrap models

* * *

# 6. Add Analyzer Health Score

You already detect stuck readings.

Turn it into a feature.

Example:

```
Analyzer Health

🟢 Healthy
🟡 Suspicious
🔴 Stuck 27 hours
```

Dashboard users will love this.

Especially because the whole project exists due to analyzer reliability problems.

* * *

# 7. One More Physics Feature

I would definitely add:

```
Pressure Corrected Temperature
```

or something similar.

Because in distillation:

```
Temperature alone
```

is less meaningful than:

```
Temperature + Pressure
```

Since boiling behavior depends on both.

Examples:

```
TopTemp / Pressure
BottomTemp / Pressure
```

or

```
TopTemp - f(Pressure)
```

Even simple combinations can help.

* * *

# 8. What I Would Remove

I would remove:

```
hour_of_day
day_of_week
```

initially.

Reason:

A debutanizer doesn't care if it's Tuesday.

If these become important:

```
hour
```

is probably acting as a proxy for:

```
shift changes
feed schedules
operations patterns
```

which may not generalize.

Keep:

```
month
```

if seasonal effects exist.

Test both versions.

* * *

# 9. What Will Impress IOCL Most

Not:

```
R² = 0.91
```

But:

```
Current Conditions

Predicted C4:
0.74%

Recommended:
Steam +1.2 TPH
Reflux +2.8 TPH

Expected C4:
0.48%

Expected Recovery:
₹48,000/hr

Confidence:
85%
```

That is exactly the type of output a refinery team can visualize using later APC deployment.

* * *

## Overall Assessment

Current version:

**Data Engineering:** 9/10  
**Process Understanding:** 8.5/10  
**ML Design:** 9/10  
**Industrial Deployability:** 8.5/10

The two biggest additions I'd still make are:

1. **Future C4 prediction (1–4 hour horizon)**
2. **Prediction uncertainty/confidence intervals**

Those two features move it from "good internship project" to something that resembles a real refinery decision-support system.

* * *

**You:**

2\. Add Prediction Horizon
==========================

This is the one thing missing that could make the project really valuable.

Currently:

    Input now↓Predict C4 now

But operators already have an analyzer.

More useful:

    Input now↓Predict C4 in 1 hourPredict C4 in 2 hoursPredict C4 in 4 hours

Then operator can act before slippage occurs.

I would seriously consider:

### Model Targets

    C4(t+1)C4(t+2)C4(t+4)

instead of:

    C4(t)

or at least build both.

That transforms it from a soft sensor into a predictive advisory system.the think is the analyzer itself is not as reliable, that is why we are making this model.

also this was some additonal information discussed:
 ## Model Constraints

The discussion revolves around setting up models with confirmed parameters. A key point is to ensure that while the model can change values, it should be within a limited range, for instance, plus or minus 50%. This prevents the model from making extreme recommendations that could lead to sudden, undesirable outcomes. It's crucial to incorporate limits and conditions into the model's functionality.

## Temperature Constraints

Temperature is specifically mentioned as a constraint. There should be a limit on temperature, and if this limit is exceeded, the system might shut down. The speaker suggests looking into these conditions, possibly related to a plant's operation, to understand the implications of such constraints.


so suggest plans accordingly

* * *

**ChatGPT:**

That additional information from the discussion actually changes my recommendation quite a bit.

The earlier suggestion of predicting:

```
C4(t+1)
C4(t+2)
C4(t+4)
```

was based on the assumption that the analyzer is reasonably trustworthy.

But your project statement literally says:

```
Analyzer unreliable
Analyzer cycle time 12 minutes
Lab delayed ~2 hours
```

So your PRIMARY deliverable should absolutely remain:

```
Current Process Variables
        ↓
Soft Sensor
        ↓
Current True C4 Estimate
```

because the refinery currently does **not know the actual C4 reliably**.

* * *

# What I would build instead

## Layer 1: Soft Sensor (Primary Deliverable)

This is the actual project.

```
Feed
Steam
Reflux
Pressure
Temperatures
        ↓
Soft Sensor
        ↓
Estimated Current C4
```

This replaces:

```
Analyzer
```

when analyzer is:

```
Stuck
Faulty
Delayed
Offline
```

* * *

## Layer 2: Analyzer Health Monitor

Since you already discovered:

```
889 identical readings
```

I would explicitly add:

```
Analyzer Health Score
```

Example:

```
Healthy
Warning
Stuck
```

Rules:

```
Same value > 12 hours
        ↓
Warning

Same value > 24 hours
        ↓
Likely stuck

Same value > 48 hours
        ↓
Fault alarm
```

This will impress the engineers because you're solving a real problem they already complain about.

* * *

# Layer 3: Constrained Recommender

This is where the discussion about ±50% matters.

Notice what your lead said:

> model can change values but should be within a limited range

This tells me:

### They do NOT want autonomous control.

They want:

```
Current Conditions
        ↓
AI Recommendation
        ↓
Operator Decides
```

not:

```
AI
 ↓
Automatically changes valves
```

* * *

So I'd implement:

```
Current C4 = 0.82%

Recommendation:

Steam +1.2 TPH
Reflux +2.5 TPH

Expected C4 = 0.55%
```

and stop there.

* * *

# About the ±50%

I actually would NOT hardcode:

```
±50%
```

into the optimizer.

Instead:

Use what you've already calculated:

```
Steam:
max ±2 TPH/hr

Reflux:
max ±5 TPH/hr
```

This is much better.

Because:

```
50% of steam
```

could mean:

```
20 → 30 TPH
```

which may be completely unrealistic.

* * *

I'd present it as:

```
Rate-of-change constrained optimizer
```

instead of:

```
50% constrained optimizer
```

because it sounds more professional.

* * *

# Temperature Constraints

This part is extremely important.

From the discussion:

> if temperature exceeds limit, system might shut down

That means temperature is NOT just a feature.

It's a safety constraint.

I'd classify variables:

## Manipulated Variables

Optimizer can change:

```
Steam Flow
Reflux Flow
```

* * *

## Safety Variables

Optimizer must respect:

```
Bottom Temperature
Control Tray Temperature
Pressure
```

* * *

Example:

Bad recommendation:

```
Increase steam by 4 TPH

Predicted C4 = 0.3%
Predicted Bottom Temp = 117°C
```

Rejected.

Because:

```
117°C > 115°C alarm
```

* * *

So your optimizer becomes:

```
Minimize C4
subject to:

Bottom Temp < Alarm Limit
Pressure < Trip Limit
Steam within limits
Reflux within limits
Rate-of-change limits
```

This is exactly how real industrial optimization works.

* * *

# I Would Modify Your Final Architecture

Instead of:

```
Soft Sensor
↓
Optimizer
↓
Dashboard
```

I would do:

```
Historian Data
        ↓
Data Cleaning
        ↓
Analyzer Health Detection
        ↓
Soft Sensor
        ↓
Current C4 Estimate
        ↓
Constraint Checker
        ↓
Recommendation Engine
        ↓
Expected Savings
        ↓
Dashboard
```

* * *

# What I Would NOT Build Right Now

I would postpone:

❌ Multi-step future prediction (t+4, t+12)

❌ LSTM

❌ Reinforcement Learning

❌ Closed-loop control

❌ Automatic optimization

Reason:

Your lead's discussion is heavily focused on:

```
Reliable current prediction
Safe recommendations
Operational constraints
```

not:

```
Forecasting next 12 hours
```

* * *

# If this were my IOCL project

Priority order:

### Phase 1

Reliable soft sensor

```
Current C4 prediction
```

### Phase 2

Analyzer fault detection

```
Stuck analyzer alerts
```

### Phase 3

Constrained recommendation engine

```
Recommended Steam/Reflux
Expected C4 reduction
```

### Phase 4

Savings calculator

```
₹/hr benefit
```

### Phase 5

Future prediction

```
C4 in 1h
C4 in 2h
```

only if time remains.

Given everything you've shared from the actual discussions, I think this order matches what the refinery engineers are expecting much more closely than a pure ML-research approach. The soft sensor and safe recommendation engine are the core deliverables; forecasting is a nice future enhancement.

* * *

**You:**

Pasted text.txtDocumentPasted markdown (2).mdFilehere is the next bit of execution i did, the findings and the next implementation plan:

* * *

**You:**

Pasted markdown.mdFilePasted markdown (2).mdFileokay next bit of execution is also done:

After execution, findings and analysis:

Viewed model_a_with_block.json

Listed directory notebooks

Listed directory models

Viewed default_leaderboard.csv:1-7

Viewed training_metrics.csv:1-6

Viewed experiment_features.py:1-65

Viewed experiment_regimes.py:1-62

Now I have everything I need. Let me create the comprehensive terminal outputs markdown file and the updated implementation plan.

Edited terminal_outputs.md

Now update the implementation plan with full Phase 3 findings:

Edited implementation_plan.md

Edited implementation_plan.md

Edited task.md

All done. Here is a complete summary of everything produced.

---

## Phase 3 Analysis & Findings Summary

### What Was Done

All 7 diagnostic scripts ran. Terminal outputs archived in [terminal_outputs.md](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/notebooks/terminal_outputs.md). The [implementation plan](file:///C:/Users/KIIT/.gemini/antigravity-ide/brain/1b375b7a-b03b-4bda-8c16-30f169f79138/implementation_plan.md) is fully updated with Findings F16–F22 and a Phase 4 options roadmap.

---

### The Critical Finding: Why All Tier 1 Models Failed

The models train perfectly (R² = 0.98 on training set) but completely fail on Block 4. There are **two simultaneous problems**:

#### 1. Covariate Shift (Extrapolation)

The plant ran differently in 2025–26 vs 2023–24:

- **38–54% of Block 4 Reflux_Flow rows** are below the minimum Reflux_Flow the model ever saw during training (70.4 vs 85.0 TPH minimum)

- Column_Top_Temp in Block 4 goes 11°C colder than any training row (18°C vs 29°C)

#### 2. Concept Drift (Correlation Reversal)

Even within the same hot regime, the sign of key correlations reversed:

| Feature | Train → C4H8 | Test → C4H8 |

|---------|-------------|-------------|

| Control_Tray_Temp | **−0.37** | **+0.39** |

| Reflux_Flow | +0.11 | **−0.17** |

| Column_Bottom_Temp | −0.07 | **+0.22** |

**Root cause**: Operating pressure dropped ~0.21 kg/cm²g between campaigns. At lower pressure, the same temperature corresponds to a heavier composition — all temperature-based signals carry different information without pressure normalization.

The Pearson correlation between predictions and truth = **−0.33** (inversely correlated). The model learned the wrong direction.

#### Why Tier 2 Works (R² = 0.71)

C4H8_Bottom_lag1 acts as a self-correcting campaign bias anchor — it tells the model where the baseline level currently is, absorbing the 258-day mean shift. This quantifies exactly what the analyzer is worth: **+1.74 R² points**.

---

### Your Decision is Needed

Three paths forward are documented in the [implementation plan](file:///C:/Users/KIIT/.gemini/antigravity-ide/brain/1b375b7a-b03b-4bda-8c16-30f169f79138/implementation_plan.md):

1. **Option A (Recommended)**: Build pressure-compensated temperature features + a sparse 72h campaign anchor feature — then retrain and measure if R² goes positive

2. **Option B**: Accept Tier 2 as the production model (requires IOCL to confirm at least one valid analyzer reading per day)

3. **Option C**: Get Block 4 labeled data from IOCL for retraining (cleanest fix, but requires data)

Five IOCL questions are also listed that would unblock the decision (C4H6 collapse reason, pressure setpoint change, reflux strategy change, data availability, sparse anchor acceptability).







 tell me what you thoughts you have on the current execution and what proceedings should we make next

also how did the output act on the concern of :

"The Temp_Gradient / Reboiler_Delta red flag is the one thing worth actively watching. After XGBoost trains, if your top-5 features are Data_Block, Temp_Gradient, Reboiler_Delta and two lags of themselves — that's a signal the model is memorizing campaigns, not learning physics. It won't crash your results but it weakens the production argument."

* * *

**You:**

Pasted markdown.mdFilehere is what we got

* * *

**You:**

Pasted markdown.mdFilehere is the after the changes, tell me of i  should send you any other files

* * *

**You:**

![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_000000001aec7207af62c976c37eadc8&ts=494638&p=fs&cid=1&sig=e2e346e36f5ed60d8d6fba886f624a68ad9feb4bc91558a3caa0619b106c6d1d&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000274071fa9ea2282d35fe2b82&ts=494638&p=fs&cid=1&sig=d63df609021977ae15d884b6f39e81cf7b9ef6f1751b63a67878e276711d11b1&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000461071fa9796ecb6ba653069&ts=494638&p=fs&cid=1&sig=a7f70eca233ad131b6c412754531002aa53da69046880e08e7f0342dc93a453b&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000b73871fab087760ca1ace46f&ts=494638&p=fs&cid=1&sig=00bcf875dbee57b413d9c4d472f79a53220f34564f74fea80983534ebc5fd3c6&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_000000008f247206a42266e8cc14bd57&ts=494638&p=fs&cid=1&sig=8b54e64fcda9b9fa5d3a1fbf893cd4166df2edf1f48dfeb2a505cc6de1943954&v=0)Pasted text.txtDocument

* * *

**You:**

![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000fbc87207a6814630817b2f13&ts=494638&p=fs&cid=1&sig=d6e96d186c8e6d991b53a3966eaf1b0b7195bdeccd9569bb1050b39c8b11c8f0&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000144471fa91a762e6b3fe2ce0&ts=494638&p=fs&cid=1&sig=843beed00bb9b160c9eae0a9337e65766c4455fe7831aedfc1909e7b11037a20&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000ab5c71fab2a2073e31f08dc0&ts=494638&p=fs&cid=1&sig=6d18afd87e08e84b68377f7b26dcd62e62c302f3868b654402b614d3adcb7cf3&v=0)I will write a Python script notebooks/run_physics_and_tuning.py to run all these experiments (Experiment 1: Pure Physics Model, Experiment 2: CatBoost tuning via Optuna, and Experiment 3: Diagnostic plots and Pressure_x_TopTemp analysis). This handles all required actions, runs them on your workspace, and generates the SHAP values and diagnostic plots.

Edited run_physics_and_tuning.py

I will run the comprehensive physics and tuning experiments script now. This will fit the Pure Physics model, generate the diagnostic plots (including for Pressure_x_TopTemp), run the 50-trial CatBoost Optuna CV optimization, and print out all comparisons.

Ran command: python notebooks/run_physics_and_tuning.py

I will set a 60-second timer to check the progress of the CatBoost optimization and Pure Physics diagnostic run.

Used tool: schedule

I will wait.

Viewed run_drift_experiments.py:15-47

I will check the status of our background training and optimization task.

Used tool: manage_task

I will terminate the current background task because default CatBoost training on CPU is taking over 50 seconds per trial, which would make the 50-trial search take over 40 minutes. I will modify the script to restrict the search space slightly (depth to 3–6, iterations to 100–300, and 3 CV folds instead of 5) to make the tuning finish in about 2–3 minutes, then rerun it.

Used tool: manage_task

I will view the CatBoost tuning portion of notebooks/run_physics_and_tuning.py (lines 160 to 205) to find where to modify the hyperparameters and CV splits for a faster, efficient search.

Viewed run_physics_and_tuning.py:1-262

I will edit notebooks/run_physics_and_tuning.py using the replace_file_content tool to modify the CV split count to 3, and restrict CatBoost parameters (depth to 3–6 and iterations to 50–200) to ensure the 50 Optuna search trials execute within 2 minutes.

Edited run_physics_and_tuning.py

I will now rerun run_physics_and_tuning.py with the accelerated CatBoost tuning parameters and splits. This should finish in under 2 minutes.

Ran command: python notebooks/run_physics_and_tuning.py

I will set a 20-second timer to monitor the completion of the optimization and diagnostic script.

Used tool: schedule

I will check the status of the background task.

Used tool: manage_task
Used tool: manage_task
Used tool: schedule

I've started a background timer to check on the progress of the run_physics_and_tuning.py script. The script is currently running Optuna hyperparameter tuning for the CatBoostRegressor using the Pure Physics feature set (which removes all time proxies like week sin/cos and hour sin/cos, leaving only the process-based chemistry features).

I'll wait until the execution completes to analyze the metrics, diagnostic plots (including Pressure_x_TopTemp), and SHAP feature importances.

Used tool: manage_task
Used tool: schedule

The CatBoost hyperparameter tuning is currently running trial 10 of 50. I have scheduled another 60-second timer to wait for it. I will report back as soon as it progresses or finishes!

Used tool: manage_task
Used tool: schedule

It's around 78% complete now. I have set a 45-second timer to check again. We are almost there!

Viewed comparison_opt_leaderboard.csv:1-5
Listed directory diagnostics
Viewed shap_importances_pure_physics.csv:1-83
Viewed run_physics_and_tuning.py:80-140
Viewed run_physics_and_tuning.py:141-263
Viewed default_leaderboard.csv:1-7
Viewed training_metrics.csv:1-6
Viewed shap_importances_no_month.csv:1-87
Viewed shap_importances.csv:1-69

The background task task-761 executing run_physics_and_tuning.py has completed successfully. Here is the analysis of the experiments and the next steps:

### 1. Leaderboard Results & Key Metrics

Comparing the Pure Physics model outputs to the baseline models:

| Model | Month Features? | Time Proxies? | Test $R^2$ on Block 4 | Test MAE (wt%) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Baseline)** | Yes | Yes | -1.0287 | 0.3017 |
| **XGBoost (Pure Physics)** | No | No | **-0.9353** | **0.2912** |
| **CatBoost (Default, No Month)** | No | Yes | -0.9489 | 0.2803 |
| **CatBoost (Tuned, Pure Physics)** | No | No | **-0.6908** | **0.2581** |

> [!TIP]
> **Tuning CatBoost on the Pure Physics dataset is our best process-only result yet.**
> By completely stripping month, day-of-week, and hour proxies, and tuning parameters (iterations: 115, learning_rate: 0.0746, depth: 5, l2_leaf_reg: 7.99), $R^2$ improved from the initial baseline of **-1.0287** to **-0.6908**, and MAE reduced to **0.2581 wt%**.

---

### 2. SHAP Analysis: Process Physics Discovered

Now that the crutches (month_sin, month_cos, dow_sin, dow_cos, hour_sin, hour_cos, Data_Block) are completely gone, the model uses genuine physical process interactions to explain composition.

Here are the top 10 SHAP features for the **Pure Physics** model:
1. **Pressure_x_TopTemp**: **0.0538** (Extremely dominant)
2. **Column_Bottom_Temp_Pnorm_k10**: **0.0217** (Local pressure-normalized temperature)
3. **Reboiler_Outlet_Temp_lag1**: **0.0198**
4. **Feed_Flow_roll_mean_12h**: **0.0169**
5. **Control_Tray_Temp_dev24h**: **0.0136**
6. **Feed_Flow_roll_std_6h**: **0.0113**
7. **Reboiler_Delta**: **0.0111**
8. **Reboiler_Outlet_Temp**: **0.0099**
9. **Steam_Feed_Ratio**: **0.0091**
10. **Control_Tray_Temp_Pratio**: **0.0090**

#### Focus: Pressure_x_TopTemp Dominance
Pressure_x_TopTemp is **2.5x more important** than the second-place feature. In the previous no_month run, dow_sin was becoming a time surrogate in rank #10. With all weekday/hour cyclical proxies removed, the model has fully adapted to this physical interaction.

---

### 3. Verification & Diagnostic Plots Inspection

Let's check the generated plots under [experiments/diagnostics/](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/experiments/diagnostics/):
* [pure_physics_plot_4_residual_vs_time.png](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/experiments/diagnostics/pure_physics_plot_4_residual_vs_time.png): Check if the long-term residual drift bands (e.g. August near zero, November negative, January positive) still persist.
* [pressure_xtoptemp_scatter.png](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/experiments/diagnostics/pressure_xtoptemp_scatter.png): Review relation of $C4H8$ versus Pressure_x_TopTemp.
* [pressure_xtoptemp_dependence.png](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/experiments/diagnostics/pressure_xtoptemp_dependence.png): SHAP dependence plot for Pressure_x_TopTemp.

Let's update our walkthrough.md and task.md with these results and prepare the next step. I'll read the task status and plan next steps. Let's inspect the task.md first.





"""
notebooks/run_physics_and_tuning.py
====================================
1. Trains the 'Pure Physics Model' (removing all calendar and time features).
2. Generates diagnostic plots for the Pure Physics Model (Actual vs Pred, Res vs Press, Res vs Time, SHAP).
3. Investigates Pressure_x_TopTemp (Scatter of C4H8 vs Pressure_x_TopTemp, SHAP dependence plot).
4. Performs CatBoost tuning using Optuna (50 trials, TimeSeriesSplit CV).
"""


import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap
import optuna


# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
DIAGNOSTICS_DIR = os.path.join("experiments", "diagnostics")
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)


# Monkeypatch SHAP base_score bug
import shap.explainers._tree
original_float = float
def custom_float_parser(val):
    if isinstance(val, str):
        val = val.strip('[] \t\n\r')
    elif isinstance(val, list):
        if len(val) > 0:
            val = val[0]
    try:
        return original_float(val)
    except ValueError:
        return original_float(str(val).strip('[] \t\n\r'))
shap.explainers._tree.float = custom_float_parser


def main():
    df = pd.read_parquet(FEATURES_FILE)
   
    # 1. DEFINE FEATURES FOR PURE PHYSICS MODEL
    TARGET_LAG_COLS = (
        [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
        [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
    )
    TIME_PROXIES = ["month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"]
    META_COLS = [
        "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
        "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
        "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
        "Analyzer_Health", "is_extreme_event", "Data_Block",
    ] + TARGET_LAG_COLS + TIME_PROXIES
   
    physics_feats = [c for c in df.columns if c not in META_COLS]
    print("=" * 80)
    print(f"RUNNING PURE PHYSICS MODEL EXPERIMENTS ({len(physics_feats)} features)")
    print("=" * 80)
   
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
   
    # Reconstruct train/test datasets
    train_df = df[train_mask & mA_filter].dropna(subset=physics_feats + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=physics_feats + ["C4H8_Bottom"])
   
    X_train, y_train = train_df[physics_feats], train_df["C4H8_Bottom"].values
    X_test, y_test   = test_df[physics_feats], test_df["C4H8_Bottom"].values
   
    # --- EXPERIMENT 1 & 3: PURE PHYSICS MODEL & DIAGNOSTICS ---
    print("\nTraining Pure Physics XGBoost Model...")
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
   
    preds = model.predict(X_test)
    test_r2 = r2_score(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)
    print(f"Pure Physics XGBoost | R² = {test_r2:.4f} | MAE = {test_mae:.4f} wt%")
   
    residuals = y_test - preds
   
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.5, color="darkgreen", edgecolors="w", s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual C4H8_Bottom (wt%)")
    plt.ylabel("Predicted C4H8_Bottom (wt%)")
    plt.title("Pure Physics: Actual vs. Predicted C4H8_Bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_1_actual_vs_predicted.png"))
    plt.close()
   
    # Plot 2: Residual Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="teal", bins=50)
    plt.axvline(x=0, color="red", linestyle="--", lw=2)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Pure Physics: Residual Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_2_residual_histogram.png"))
    plt.close()
   
    # Plot 3: Residual vs Top Pressure
    plt.figure(figsize=(8, 6))
    plt.scatter(test_df["Column_Top_Pressure"], residuals, alpha=0.5, color="darkorange", edgecolors="w", s=40)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("Column_Top_Pressure (kg/cm²g)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Pure Physics: Residual vs. Column_Top_Pressure")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_3_residual_vs_pressure.png"))
    plt.close()
   
    # Plot 4: Residual vs Time
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["DateTime"], residuals, alpha=0.6, color="firebrick", lw=1.5)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("DateTime")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Pure Physics: Residual vs. Time")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_4_residual_vs_time.png"))
    plt.close()
   
    # Plot 5: SHAP summary
    print("Generating Pure Physics SHAP values...")
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
   
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.title("Pure Physics: SHAP Summary Plot (Top 20 Features)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_5_shap.png"))
    plt.close()
   
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
   
    print("\n" + "=" * 50)
    print("TOP 20 SHAP FEATURES (PURE PHYSICS MODEL)")
    print("=" * 50)
    print(shap_imp_df.head(20).to_string(index=False))
    print("=" * 50)
    shap_imp_df.to_csv(os.path.join(DIAGNOSTICS_DIR, "shap_importances_pure_physics.csv"), index=False)
   
    # --- INVESTIGATE Pressure_x_TopTemp ---
    print("\nInvestigating Pressure_x_TopTemp...")
    # Scatter of C4H8 vs Pressure_x_TopTemp by Data_Block
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df[mA_filter & (~df["C4H8_Bottom"].isna())],
        x="Pressure_x_TopTemp", y="C4H8_Bottom",
        hue="Data_Block", palette="viridis", alpha=0.6, edgecolor=None
    )
    plt.xlabel("Pressure_x_TopTemp (Pressure * Column_Top_Temp)")
    plt.ylabel("C4H8_Bottom (wt%)")
    plt.title("Pressure_x_TopTemp vs. C4H8_Bottom by Data_Block")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pressure_xtoptemp_scatter.png"))
    plt.close()
   
    # SHAP dependence plot for Pressure_x_TopTemp
    plt.figure(figsize=(8, 6))
    feat_idx = physics_feats.index("Pressure_x_TopTemp")
    shap.dependence_plot(
        "Pressure_x_TopTemp", shap_values, X_sample,
        interaction_index=None, show=False
    )
    plt.title("SHAP Dependence Plot for Pressure_x_TopTemp")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pressure_xtoptemp_dependence.png"))
    plt.close()
   
    # --- EXPERIMENT 2: CATBOOST TUNING (OPTUNA) ---
    print("\nTuning CatBoostRegressor via Optuna CV (50 trials)...")
    tscv = TimeSeriesSplit(n_splits=3)
   
    def objective_cb(trial):
        params = {
            "iterations":    trial.suggest_int("iterations", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth":         trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_seed":   42,
            "thread_count":  -1,
            "verbose":       0
        }
       
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
           
            model_cb = CatBoostRegressor(**params)
            model_cb.fit(X_tr, y_tr)
           
            preds_val = model_cb.predict(X_val)
            r2 = r2_score(y_val, preds_val)
            cv_scores.append(r2)
           
        return np.mean(cv_scores)
       
    study_cb = optuna.create_study(direction="maximize")
    study_cb.optimize(objective_cb, n_trials=50, show_progress_bar=True)
   
    print("\n" + "=" * 50)
    print("CATBOOST OPTUNA COMPLETED")
    print("=" * 50)
    print(f"Best CV R² score: {study_cb.best_value:.4f}")
    print("Best CatBoost parameters:")
    for k, v in study_cb.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)
   
    # Train final tuned CatBoost model
    best_cb_params = study_cb.best_params.copy()
    best_cb_params.update({"random_seed": 42, "verbose": 0})
    tuned_cb = CatBoostRegressor(**best_cb_params)
    tuned_cb.fit(X_train, y_train)
   
    tuned_cb_preds = tuned_cb.predict(X_test)
    tuned_cb_r2 = r2_score(y_test, tuned_cb_preds)
    tuned_cb_mae = mean_absolute_error(y_test, tuned_cb_preds)
   
    print(f"\nTuned CatBoost Model on Block 4 Test Set:")
    print(f"  Test R²:  {tuned_cb_r2:.4f}")
    print(f"  Test MAE: {tuned_cb_mae:.4f} wt%")
    print("=" * 50)
   
    # Save optimized CatBoost model
    tuned_cb.save_model("models/model_A_CatBoost_opt.bin")
    print("Optimized CatBoost model saved to models/model_A_CatBoost_opt.bin")
   
    # Save comparison data
    comparison_opt = [
        {"Model": "XGBoost (Pure Physics)", "R2": float(test_r2), "MAE": float(test_mae)},
        {"Model": "CatBoost (Default, No Month)", "R2": -0.9489, "MAE": 0.2803},
        {"Model": "CatBoost (Tuned, Pure Physics)", "R2": float(tuned_cb_r2), "MAE": float(tuned_cb_mae)}
    ]
    comparison_opt_df = pd.DataFrame(comparison_opt)
    comparison_opt_df.to_csv("experiments/comparison_opt_leaderboard.csv", index=False)
    print("Leaderboard comparison saved to experiments/comparison_opt_leaderboard.csv")
    print("All experiments completed successfully.")


if __name__ == "__main__":
    main()

* * *

**You:**

![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_0000000016407208910fc45a3acdd1a8&ts=494638&p=fs&cid=1&sig=b13258838892b8c25d6de122cef4b39ee7e0c760ceb0b6a1966afe9ee4abaace&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000b3607208a915a403b2a2b598&ts=494638&p=fs&cid=1&sig=5783769272c1aae7b2e35a03ab74d1502267ca6318d70890910b9bfa5673bb3c&v=0)Pasted text.txtDocument

* * *

**You:**

![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000e92c7207be2dbd08e284fb75&ts=494638&p=fs&cid=1&sig=290269debefe949c91ce85459f9de60c03a7c56c1e9a6a3b7586e865f1ea5ad5&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000a3647209b1c72ae1435c4f99&ts=494638&p=fs&cid=1&sig=7a160dc0919769c04cb6d00148d41046300ec1a818984aeef11b08cd73c90841&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_000000002ff072078c61e101da749df3&ts=494638&p=fs&cid=1&sig=c91e1a7c4c9b007af1dd881e08b4aadd5eb1ca1fe72af5c82fbe43d206dc2b4a&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000e26c7207bd49a57fcfb937c4&ts=494638&p=fs&cid=1&sig=03966abc56b5fb28f86d06133fbf477b1c33090b1ba765586d00a6f544dd82af&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000e2d47206bad2dc1ec0feaf95&ts=494638&p=fs&cid=1&sig=8a633fb7357d1b8f184a0f5efe0c0d0debea8e2f27404a9dc0cbe706d7cb9e10&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_000000001cd872069bf867311732b25f&ts=494638&p=fs&cid=1&sig=036755ed93780894334d8fa6fac2c935ec2c36867b5f03f1c02f30c3c29fddab&v=0)Pasted markdown.mdFilePasted markdown (2).mdFilethis is what i did

* * *

**You:**

![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_000000001d1472098839b4f0ca6d7e3e&ts=494638&p=fs&cid=1&sig=4586d47d5e99fcf18ca32de7036f5243665051427096cbaf913c1e20258b7b12&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_000000005f0872079f475f7305ad78f7&ts=494638&p=fs&cid=1&sig=c2a0a3d2acf222811d571490cd999417f28f0163023e49f3c0f9be6480191b6c&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000197c7207abeb2773be8f3e7f&ts=494638&p=fs&cid=1&sig=d126620c93c1ebb75ef97cc71b64b3bd30ef8230aaa3e7e542e7c6830cfdecb1&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000bbb0720795e6755ae6d14bcb&ts=494638&p=fs&cid=1&sig=191489c2a686628c2457b37ea5112ed0a27de19ae36824dd4d441efc55951ef9&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_0000000004b07207a78731754795be33&ts=494638&p=fs&cid=1&sig=9a83101399b00881dd2cfbfc08264a4fcf33adebedaa850e3320286d514ed749&v=0)![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_000000004b7072078466687efdc81def&ts=494638&p=fs&cid=1&sig=aaf667abb1dca249a0318d235d23ac209caf509241d6b651ccc67e71ce60e6f6&v=0)Pasted markdown.mdFile

* * *

**You:**

Viewed model_b_target_audit.py:1-11

I will create the anchor audit script notebooks/model_b_anchor_audit.py to calculate the coverage, mean age, and max age of the C4H6 analyzer anchor for both the 12-hour and 72-hour limits in Block 4.

Edited model_b_anchor_audit.py

I will run the anchor audit script to check coverage and age metrics for C4H6 anchors.

Ran command: python notebooks/model_b_anchor_audit.py

I will create notebooks/anchor_only_baselines.py to run the baseline comparisons. It evaluates how much predictive power sits in the analyzer memory alone using the campaign anchor (12h and 72h limits), the 24-hour rolling mean of the anchor, and the constant target mean.

Edited anchor_only_baselines.py

I will run anchor_only_baselines.py to evaluate the predictive performance of the raw anchor, rolling anchor, and mean baselines.

Ran command: python notebooks/anchor_only_baselines.py

Here are the results of the anchor availability and baseline metrics audits:

### 1. C4H6 Anchor Coverage on Block 4 Test Set

| Target Subset | Anchor Limit | Coverage (%) | Mean Age (hours) | Max Age (hours) |
| :--- | :---: | :---: | :---: | :---: |
| **All Block 4 Rows** (incl. stuck) | **12h** | 55.28% | 1.07 | 12.0 |
| **All Block 4 Rows** (incl. stuck) | **72h** | 69.62% | 7.74 | 72.0 |
| **Healthy Analyzer Rows Only** | **12h** | 98.45% | 0.03 | 12.0 |
| **Healthy Analyzer Rows Only** | **72h** | 99.76% | 0.36 | 69.0 |

---

### 2. Model B (C4H6) Anchor-Only Baseline Performance on Block 4

| Baseline Model | R² Score | MAE (wt%) | Pearson Correlation | Available % |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline A (12h Anchor)** | **0.9606** | **0.0005** (5 ppm) | **+0.9830** | 98.45% |
| **Baseline A (72h Anchor)** | **0.6074** | **0.0007** | **+0.7813** | 99.76% |
| **Baseline B (24h Roll 12h Anchor)** | **0.9308** | **0.0009** | **+0.9672** | 99.56% |
| **Baseline B (24h Roll 72h Anchor)** | **0.1510** | **0.0011** | **+0.3898** | 99.80% |
| **Baseline C (Block 4 Mean: 0.005663)**| -0.0000 | 0.0032 | Constant | 100.00% |
| **Baseline C (Train Mean: 0.0273)** | -4.6618 | 0.0221 | Constant | 100.00% |

---

### Key Takeaways & Suspicion Confirmed
1.  **The Anchor is Breathtakingly Powerful**: Your suspicion was $100\%$ correct. Baseline A (the 12h campaign anchor alone, shifted by 1 hour) achieves an $R^2$ of **0.9606** and an $MAE$ of **0.0005 wt%** (5 ppm). The analyzer memory dominates the target variance because the concentration is tiny and slow-moving.
2.  **Age & Sticking Behavior**: Because the analyzer in Block 4 is flagged as stuck $50.9\%$ of the time, the *overall* coverage of a 12-hour limit anchor drops to $55.3\%$ for all rows. However, when the analyzer is healthy, the anchor is almost always available ($98.45\%$).
3.  **Baseline B (Rolling average) Fallback is Strong**: If we use the 24h rolling mean of the 12h anchor, we can expand coverage to $99.56\%$ while maintaining a high $R^2$ of **0.9308** and an $MAE$ of **0.0009 wt%**.
4.  **Engineering Conclusion**: We are not building a process-based separation model for Model B. We are building an **analyzer-tracking model** that uses the process variables solely for minor high-frequency corrections when the analyzer is available, and relies on the fallback baseline when it is stuck or offline.

* * *

**You:**

Pasted markdown.mdFile# Models A & B Production-Ready Implementation Plan

This document details the final engineering design and production release roadmap for the **C4H8 (Model A)** and **C4H6 (Model B)** soft-sensors in the Debutanizer Column.

---

## 1. Model A (C4H8_Bottom) ML Sensor

Model A is frozen and validated for production deployment using an optimized XGBoost regressor trained on the robust 8-feature process set:

*   **Production Feature Set**:
    1.  C4H8_campaign_anchor (72h limit, Shift-1)
    2.  Steam_Feed_Ratio (Reboiling_Steam_Flow / Feed_Flow)
    3.  Reflux_Ratio (Reflux_Flow / Feed_Flow)
    4.  Reboiling_Steam_Flow_dev24h
    5.  Reflux_Flow_dev24h
    6.  Column_Bottom_Temp_dev24h
    7.  Control_Tray_Temp_dev24h
    8.  Column_Top_Pressure_dev24h
*   **Validation Performance (Block 4 Test Set)**:
    *   **$R^2$ Score**: **0.9074**
    *   **MAE**: **0.0516 wt%**
*   **Safety Ceilings**: Safety ceilings for bottom temperature ($\le 115.0$ °C) and top pressure ($\le 5.0$ kg/cm²g) are hard-coded in the DCS system.

---

## 2. Model B (C4H6_Bottom) Deterministic Sensor

Our target audits and process delta modeling established a critical engineering discovery: **process variables do not contain additional learnable signal to improve upon analyzer memory for C4H6.**
*   **The Delta Mismatch**: Overlaying an XGBoost regressor to predict high-frequency deviations around the anchor degraded Block 4 test performance, dropping $R^2$ from **$0.9606$ to $0.9010$** and doubling $MAE$ from **$5.5$ ppm to $11.9$ ppm**.
*   **The Decision**: We have frozen Model B as a deterministic, analyzer-tracking state estimator. It has zero training maintenance overhead and zero concept drift risk.

### Target Inversion Robustness Matrix (12h Limit Anchor)
*   **Block 2 (Target Mean = 0.0314 wt%)**: $R^2 = \mathbf{0.7518}$ | $MAE = \mathbf{42.9}$ ppm | Coverage = $99.71\%$
*   **Block 3 (Target Mean = 0.0234 wt%)**: $R^2 = \mathbf{0.7651}$ | $MAE = \mathbf{48.5}$ ppm | Coverage = $99.41\%$
*   **Block 4 (Target Mean = 0.0057 wt%)**: $R^2 = \mathbf{0.9606}$ | $MAE = \mathbf{5.5}$ ppm | Coverage = $98.45\%$

---

## 3. Online DCS Prediction & Combined Total C4 Release

The online inference scripts are stored under [inference/](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/final_v1/inference/) in the final_v1 directory:

### predict_c4h8.py
Loads [model_A_final_v1.pkl](file:///c:/Users/KIIT/OneDrive/Desktop/DEBUTANIZER-model/final_v1/models/model_A_final_v1.pkl), computes the ratios and 24h rolling deviations at time $t$, and predicts C4H8 wt%.
*   **Fallback**: If the C4H8 analyzer is offline or stuck $> 72$ hours, falls back to the rolling mean of recent predictions (last 24h), then to the Block 4 baseline mean (**$0.480$ wt%**).

### predict_c4h6.py
Executes the deterministic tracking logic for C4H6:
*   **Level 1**: Shifted Campaign Anchor with a 12-hour limit (**MAE = 5.5 ppm**).
*   **Level 2**: 24-hour rolling mean of the anchor (**MAE = 9.0 ppm**).
*   **Level 3**: Constant campaign mean value of **$0.0057$ wt%** (100% coverage, **MAE = 32.0 ppm**).

### predict_total_c4.py
Unified prediction wrapper. Takes raw process inputs and analyzer readings, calls predict_c4h8 and predict_c4h6, and outputs the final combined slippage:
$$\text{Total C4} = \text{Predicted C4H8} + \text{Predicted C4H6}$$

---

## 4. Verification Plan

### Automated Validation Runs
*   python final_v1/notebooks/verify_anchor_leakage.py — Formally proves Model A leak-free status and verifies Block 3 $R^2 = 0.7694$.
*   python final_v1/notebooks/model_b_inversion_check.py — Verifies Model B anchor robustness across Blocks 2, 3, and 4.
*   python final_v1/inference/predict_total_c4.py — Runs the dummy unit-test scenarios for online execution verification.

* * *

**ChatGPT:**

Yes. Overall, I would approve moving forward.

But there are **3 small changes** I'd still make before calling this truly "frozen".

* * *

# 1. Model A fallback still has one thing I don't like

Current:

```Python
if len(valid_preds) >= 6:
    rolling_mean
else:
    last_prediction
```

For C4H8, this is much safer than it was for C4H6, but I'd still prefer:

```Python
if len(valid_preds) >= 6:
    rolling_mean
else:
    default_mean
```

Why?

Imagine:

```
Analyzer dead
Model restarted
Only 1 previous prediction available
```

Then you're using:

```
one potentially bad prediction
```

instead of:

```
historical baseline
```

For refinery deployment I'd rather be conservative.

Not mandatory.

But recommended.

* * *

# 2. Add confidence / health status

I would add this to both outputs:

```Python
"prediction_health": "GREEN"
```

Possible values:

```Python
GREEN
YELLOW
RED
```

Example:

```Python
if anchor_available:
    GREEN

elif rolling_mean_used:
    YELLOW

else:
    RED
```

Dashboard guys LOVE this.

Because operators instantly know:

```
GREEN
= analyzer available

YELLOW
= fallback active

RED
= emergency default
```

* * *

# 3. Save fallback reason

Add:

```Python
"fallback_reason"
```

Example:

```Python
{
    "model_used": "Fallback (Default)",
    "fallback_reason": "Analyzer offline >168h"
}
```

or

```Python
{
    "model_used": "Fallback (Rolling Mean)",
    "fallback_reason": "Analyzer stale >72h"
}
```

This becomes extremely useful later when:

```
Sir asks:
Why was prediction weird yesterday?
```

and you immediately know.

* * *

# Documentation

Absolutely do it now.

Actually this is the best time.

Because after optimizer/dashboard starts:

```
soft sensor
optimizer
api
dashboard
deployment
```

all get mixed together.

Right now the modelling phase is complete and cleanly separated.

I would create:

```
docs/

01_problem_statement.md
02_dataset_understanding.md
03_feature_engineering.md
04_model_a_development.md
05_model_b_development.md
06_drift_analysis.md
07_final_architecture.md
08_deployment_guide.md
09_future_optimizer.md
```

This will save you later when sir asks:

> "Why did we remove Block 1 from Model B?"

and you can point to one section instead of searching old chats.

* * *

# Message to Sir

I'd send something like this:

Hello Sir,

Quick update on the Debutanizer soft-sensor project.

Model A (C4H8_Bottom) has been finalized and frozen for production evaluation. After extensive drift analysis, feature ablation, leakage verification, and Optuna tuning, the final model uses a constrained 8-feature process set consisting of ratios, 24-hour deviations, and a leak-free shifted analyzer anchor. The final Block 4 validation performance is:

• R² = 0.9074  
• MAE = 0.0516 wt%

The model includes analyzer availability handling, rolling fallback logic, and hard timeout safeguards for prolonged analyzer outages.

For Model B (C4H6_Bottom), the investigations produced an important engineering finding. We evaluated ML-based approaches, anchor-only baselines, and delta-correction models. The results showed that the shifted analyzer anchor alone consistently outperformed machine learning corrections. The best deterministic configuration achieved:

• R² = 0.9606  
• MAE = 0.000547 wt% (5.5 ppm)

Adding ML correction layers degraded performance, so Model B has been frozen as a deterministic analyzer-tracking estimator with structured fallback logic instead of a trained ML model.

A unified Total C4 prediction pipeline has now been implemented, combining both Model A and Model B outputs for online inference.

Current status:  
• Model A finalized and frozen  
• Model B finalized and frozen  
• Leakage checks completed  
• Drift analysis completed  
• Online inference logic implemented  
• Release package prepared

Next planned phase is documentation finalization followed by optimizer/dashboard development and deployment integration.

Please let us know if you would like any additional validation runs or engineering checks before we move into the optimization and visualization phase.

* * *

# My recommendation

Before touching optimizer/dashboard:

✅ Make documentation.

✅ Add health status field.

✅ Add fallback reason field.

✅ Run one final end-to-end test.

Then freeze:

```
final_v1
```

and never touch the soft sensor code again unless a bug is found.

From everything you've shown, you're about **95–98% done with the modelling phase**. The biggest value left in this project is no longer improving R² — it's turning predictions into actionable recommendations and a usable dashboard.

* * *

**You:**

Pasted markdown.mdFilei will go and make a final documentation, also if there are questions that need to be asked to sir before moving to optimizer, do tell me that.
also this whole ficaso is also supposed to predict / recommend, input values or something from what i remember, will it able to do that or not?
as this point i dont even remember, was that something we were supposed to do bruh
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

* * *

**ChatGPT:**

Bro, after reading the original problem statement you pasted, I think you've remembered something important.

The project was **never just a soft sensor**.

The soft sensor was Phase 1.

The actual business objective was:

> "AI Based Model to Minimize C4 Slippage in Debutanizer"

Not:

> "Build a model that predicts C4."

The prediction model is just the engine.

* * *

# Where are we right now?

Current status:

✅ Data cleaning

✅ Analyzer diagnostics

✅ Feature engineering

✅ Drift analysis

✅ Model A

✅ Model B

✅ Unified Total C4

✅ Fallback logic

✅ Health monitoring

✅ Documentation

You are basically done with:

```
Soft Sensor Layer
```

* * *

# What is still left?

This section from the original requirement:

```
Operator recommendations
with loss INR/hr calculations
```

and

```
Dynamic adjustment of reflux and steam
Maintain optimal bottom temperature
Balance energy vs recovery
```

This is the

```
Optimizer Layer
```

which comes AFTER prediction.

* * *

# Very important question for sir

Before touching optimizer, ask this.

Sir, before we begin the optimization phase, could you please clarify what operational action the optimizer is expected to recommend?

Some possibilities are:

1. Reboiling Steam Flow recommendation only
2. Reflux Flow recommendation only
3. Both Steam and Reflux recommendations simultaneously
4. Bottom Temperature target recommendation
5. Full operating-window recommendation

Also, should the optimizer prioritize:

A. Minimum Total C4 slippage only  
B. Minimum steam consumption while maintaining specification  
C. Maximum economic benefit (energy vs recovery trade-off)

This will help us define the optimizer objective function correctly before implementation.

This is probably the most important question right now.

* * *

# Another question for sir

You need this too.

Sir, for calculating operator recommendations and loss INR/hr, do we have:

• LP steam cost (₹/ton steam)  
• Product value of recovered C4s  
• Estimated penalty/cost for off-spec product

If available, we can convert optimization recommendations into direct economic impact (₹/hr savings).

Without economics you're optimizing blindly.

* * *

# Can your current model recommend values?

Not directly.

Current model:

```
Inputs
   ↓
Soft Sensor
   ↓
Predicted Total C4
```

That's all.

* * *

# How optimizer works

Tomorrow you'll build:

```
Current Plant State
        ↓
Soft Sensor
        ↓
Virtual Plant
        ↓
Optimizer
        ↓
Recommended Steam
Recommended Reflux
Expected Total C4
Expected INR/hr Savings
```

For example:

Current:

```
Steam = 22.0 TPH
Reflux = 92 TPH

Predicted Total C4 = 0.44%
```

Optimizer tests:

```
Steam = 21.8
Reflux = 92

Steam = 21.5
Reflux = 92

Steam = 21.0
Reflux = 92

...
```

and asks:

```
Will C4 stay below limit?
```

using your model.

Then it finds:

```
Steam = 20.7 TPH
Reflux = 92 TPH

Predicted C4 = 0.39%
```

and tells operator:

```
Reduce Steam by 1.3 TPH
Expected Savings = ₹xxx/hr
```

That is the real optimization.

* * *

# One thing that worries me

You currently have:

```
Predict Total C4
```

but optimization needs:

```
"What happens if I change steam?"
```

That's a slightly different problem.

Because your model was trained on:

```
Historical operations
```

not

```
Counterfactual operations
```

(what-if scenarios).

This may still work because you have steam ratio and reflux ratio as inputs.

But before building optimizer I would ask sir:

> Are operator recommendations expected to be advisory only, or do we need rigorous process optimization with economic calculations?

Big difference.

* * *

# My guess of the final architecture

I think what sir actually wants is:

```
Historian Data
        ↓
Soft Sensor
        ↓
Predicted Total C4
        ↓
Optimizer
        ↓
Recommended:
   Steam Flow
   Reflux Flow
        ↓
Dashboard
```

Dashboard shows:

```
Current Total C4

Predicted Total C4

Spec Status

Recommended Steam

Recommended Reflux

Potential Savings

Health Status
```

* * *

# One more thing before optimizer

Ask sir this directly:

```
What is the control authority?

Can operators change:

1. Steam only
2. Reflux only
3. Steam + Reflux
4. Bottom Temperature setpoint
5. All of the above
```

Because optimizer cannot recommend variables that operators are not allowed to manipulate.

* * *

So if I were you, I'd:

1. Finish documentation.
2. Send update.
3. Ask the 3 optimizer questions above.
4. Wait for sir's response.

Because the modeling phase is essentially complete. The next phase depends entirely on how refinery operations want recommendations to be generated.

And yes — based on the original objective, **you are still expected to build the recommendation/optimizer layer.** The soft sensor itself is only the foundation.

* * *

**You:**

pretty sure operator can control all input variables
Top & Bottom temperature 
  Reboiler outlet temperature 
  Reboiler steam flow
  Reflux flow
  Feed flow
 Control tray temperature 
 Tray Temperature
 Bottom Analyzer
 Re-boiling steam flow


but recommendations to be in limitations as we were discussing very early on that it can't just +100 the value, its to be in a set limit, for whcih we have scoured the internet for.

* * *

**ChatGPT:**

Yep, and that's actually consistent with what I remember from your earlier work on this project.

The key thing is:

**Control variables ≠ Recommendation variables**

Just because an operator can touch a variable doesn't mean the optimizer should recommend changing it.

* * *

# What I would NOT optimize directly

I would not directly recommend:

* Top Temperature
* Bottom Temperature
* Control Tray Temperature
* Tray Temperatures
* Bottom Analyzer

Why?

Because these are mostly **responses** to the process.

Example:

```
Increase Steam
      ↓
Bottom Temperature rises
      ↓
Tray temperatures shift
      ↓
Composition changes
```

Temperature is an outcome, not the actuator.

* * *

# What I WOULD optimize

The true manipulated variables are probably:

1. Reboiler Steam Flow
2. Reflux Flow

Potentially:

3. Feed Flow (if operations allow it)

but usually production planning owns feed rate, not the board operator.

So I would ask sir whether Feed Flow is actually adjustable by operators or fixed by upstream units.

* * *

# The good news

You already have these limits.

Remember all the operating envelope analysis you guys did?

Things like:

```
Steam:
18.0 → 24.4 TPH

Reflux:
80.0 → 103.9 TPH
```

and rate limits:

```
Steam:
±2 TPH/hr

Reflux:
±5 TPH/hr
```

Those are exactly the constraints an optimizer needs.

* * *

# What optimizer should probably do

Current state:

```
Steam = 22.0
Reflux = 92.0

Predicted Total C4 = 0.44
```

Optimizer explores:

```
Steam = 21.5
Reflux = 92.0

Steam = 21.0
Reflux = 92.0

Steam = 20.5
Reflux = 92.0
```

and checks:

```
Will Total C4 remain below target?
```

using your model.

Then it finds:

```
Steam = 20.8
Reflux = 92.0

Predicted Total C4 = 0.39
```

and says:

```
Recommended Action:
Reduce steam by 1.2 TPH

Expected C4:
0.39 wt%

Steam Saved:
1.2 TPH
```

That's exactly the kind of recommendation refinery operators like.

* * *

# One thing I would change in your current plan

Your documentation says:

```
Predicted Total C4 <= 0.40 wt%
```

instead of the plant limit:

```
0.50 wt%
```

I actually like this.

Keep the optimizer target at:

```
0.40 wt%
```

and dashboard show:

```
Green  : < 0.40
Yellow : 0.40 - 0.50
Red    : > 0.50
```

because:

```
Model error exists
Process disturbances exist
Analyzer delay exists
```

You need a safety buffer.

* * *

# Questions I'd still ask sir

Only these now:

### 1. Economic objective

```
Should optimizer minimize:

A) Steam only

or

B) Steam + Reflux energy cost

or

C) INR/hr economic loss
```

Because the objective function changes completely.

* * *

### 2. Recommendation frequency

```
Every hour?

Every analyzer cycle?

Every 15 minutes?
```

Important for dashboard design.

* * *

### 3. Advisory or automatic?

```
Operator recommendation only

or

Closed-loop implementation later?
```

Your future scope mentions APC integration, so I suspect current phase is advisory only.

* * *

# My assessment of project status

Honestly:

```
Soft Sensor Phase:
95-100% complete

Documentation:
90%+

Dashboard:
Not started

Optimizer:
Not started
```

The biggest technical risk (building a reliable predictive model on drifting plant data) is already behind you.

The optimizer and dashboard are mostly engineering and productization work compared to the model-development phase.

So yes, I think making the final documentation now is the correct move, then getting sir's answers on optimization objectives before writing a single optimizer line.