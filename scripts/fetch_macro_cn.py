"""Fetch CSI300-adjacent cross-asset / volatility series for the C+ hybrid
sign model (China proxies for the SPY macro set, PROJECT_NOTES 7.18).

Sources (akshare, reachable in this environment — East Money blocked, so
Sina-based endpoints only):
  csi500  sh000905  CSI SmallCap 500        (small/cyclical RS: -> rs_iwm_1d)
  growth  sz399006  ChiNext (创业板)          (growth-style RS: -> rs_qqq_1d)
  ss50    sh000016  SSE 50                  (largest-cap RS:  extra slot)
  qvix    50ETF implied vol index           (China VIX analog: -> vol_term)

Skipped (no reliable China proxy in this env):
  tnx (China 10Y CGB yield — spotty history), usdcny (historical endpoint
  unreliable), credit (no clean China HY ETF verified), skew (no China SKEW).

Outputs: data/macro_cn/<name>.parquet (date index, OHLCV) + manifest.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import akshare as ak  # noqa: E402

OUT_DIR = ROOT / "data" / "macro_cn"

INDEX_SYMBOLS = {
    "csi500": "sh000905",
    "growth": "sz399006",
    "ss50": "sh000016",
}


def fetch_qvix() -> pd.DataFrame:
    df = ak.index_option_50etf_qvix()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["adj_close"] = df["close"]
    df["volume"] = 0
    return df


def fetch_index(symbol: str) -> pd.DataFrame:
    raw = ak.stock_zh_index_daily(symbol=symbol)
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["adj_close"] = df["close"]
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}

    frames: dict[str, pd.DataFrame] = {}
    for name, symbol in INDEX_SYMBOLS.items():
        df = fetch_index(symbol)
        frames[name] = df
        df.to_parquet(OUT_DIR / f"{name}.parquet")
        manifest[name] = {
            "symbol": symbol,
            "n_rows": int(len(df)),
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
        }
        print(f"{name:8s} ({symbol:9s}) {len(df):5d} rows  {df.index.min().date()} -> {df.index.max().date()}")

    qvix = fetch_qvix()
    qvix.to_parquet(OUT_DIR / "qvix.parquet")
    manifest["qvix"] = {
        "symbol": "50ETF QVIX",
        "n_rows": int(len(qvix)),
        "start": str(qvix.index.min().date()),
        "end": str(qvix.index.max().date()),
    }
    print(f"qvix    (50ETF QVIX ) {len(qvix):5d} rows  {qvix.index.min().date()} -> {qvix.index.max().date()}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nwrote data/macro_cn/ ({len(INDEX_SYMBOLS) + 1} series)")


if __name__ == "__main__":
    main()
