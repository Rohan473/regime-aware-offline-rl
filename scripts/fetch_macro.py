"""Fetch daily macro / cross-asset series from Yahoo's public chart API and
store them under data/macro/ (project: add non-correlated features for the
SPY models — PROJECT_NOTES 7.18).

Sources (Yahoo chart API, keyless, one ~2-decade window per call):
  tlt    TLT    20y+ Treasury ETF (risk-on vs flight-to-safety)
  tnx    ^TNX   10y Treasury yield level (%)
  vix    ^VIX   CBOE spot volatility
  vix3m  ^VIX3M CBOE 3-month volatility (term structure: vix3m - vix)
  dxy    DX-Y.NYB ICE US Dollar Index
  hyg    HYG    high-yield bond ETF (credit-stress proxy via price, not yield)
qqq    QQQ    Nasdaq-100 ETF (large-cap/tech relative strength)
    iwm    IWM    Russell 2000 ETF (small-cap/cyclical relative strength)
    skew   ^SKEW  CBOE Skew index (options tail-risk / put-demand sentiment)

Adjusted close is kept (dividend-adjusted returns) alongside raw OHLCV.
Outputs: data/macro/<name>.parquet (date, open, high, low, close, adj_close,
volume) + data/macro/manifest.json. Reproducible from the stored parquet;
re-fetching is only needed to extend history.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "macro"
START = pd.Timestamp("2004-12-01")
END = pd.Timestamp("2026-12-31")

TICKERS = {
    "tlt": "TLT",
    "tnx": "^TNX",
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "dxy": "DX-Y.NYB",
    "hyg": "HYG",
    "qqq": "QQQ",
    "iwm": "IWM",
    "skew": "^SKEW",
}


def fetch_daily(symbol: str) -> pd.DataFrame:
    p1 = int(START.timestamp())
    p2 = int(END.timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?interval=1d&period1={p1}&period2={p2}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read().decode())["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(ts))
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "adj_close": adj,
            "volume": q["volume"],
        }
    ).dropna(subset=["close"])
    df["date"] = df["date"].dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    df = df[(df.index >= START) & (df.index <= END)]
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    for name, symbol in TICKERS.items():
        df = fetch_daily(symbol)
        df.to_parquet(OUT_DIR / f"{name}.parquet")
        manifest[name] = {
            "symbol": symbol,
            "n_rows": int(len(df)),
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
        }
        print(f"{name:6s} ({symbol:10s}) {len(df):6d} rows  {df.index.min().date()} -> {df.index.max().date()}")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nwrote data/macro/ ({len(TICKERS)} series)")


if __name__ == "__main__":
    main()