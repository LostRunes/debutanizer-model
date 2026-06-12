"""Inspect models/test_predictions.parquet to understand what before/after data is available."""
import pandas as pd
import numpy as np

tp = pd.read_parquet('models/test_predictions.parquet')
print("Shape:", tp.shape)
print("Columns:", list(tp.columns))
print()
print("Sample head:")
print(tp.head(3).to_string())
print()
print("Data types:")
print(tp.dtypes)
print()
if 'Data_Block' in tp.columns:
    print("Data_Block distribution:", tp['Data_Block'].value_counts().to_dict())
