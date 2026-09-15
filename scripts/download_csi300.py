import akshare as ak
import pandas as pd
import os

# Download CSI 300 Index daily OHLCV data from Sina Finance
# Symbol: sh000300 (Sina format)
# Data goes back to 2002-01-04

print("Downloading CSI300 daily data from Sina...")
df = ak.stock_zh_index_daily(symbol="sh000300")

# Convert date to datetime and filter from 2000
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] >= '2000-01-01'].reset_index(drop=True)

print(f"Records: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# Save to data/ folder
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "csi300_daily.csv")
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
