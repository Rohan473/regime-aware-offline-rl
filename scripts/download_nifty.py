"""Fetch NIFTY 50 daily OHLCV (Yahoo ^NSEI) -> data/nifty_daily.csv.

Mirror of scripts/download_csi300.py + scripts/fetch_macro.py conventions:
the resulting CSV must satisfy the same daily loader contract used by the
csi300 pipeline (date, open, high, low, close, volume; no zero-volume rows).

Source: Yahoo public chart API (keyless) — same endpoint options used by
scripts/fetch_macro.py. Period covers the full study horizon (2005-01-01 ..
2026-12-31) so the frozen canonical C+ protocol can be transferred to NIFTY
over the SAME date windows as SPY and CSI300 (train <=2018-12-31, val
2019-2020, test 2021-2024, OOS 2025-2026).
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

START = pd.Timestamp("2004-12-01")
END = pd.Timestamp("2026-12-31")

OUT_PATH = ROOT / "data" / "nifty_daily.csv"


def fetch_chunk(p1: int, p2: int) -> pd.DataFrame:
    symbol = "%5ENSEI"  # ^NSEI
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?interval=1d&period1={p1}&period2={p2}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        res = json.loads(r.read().decode())["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }
    ).dropna(subset=["close"])
    df["date"] = df["date"].dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df


def fetch_nifty() -> pd.DataFrame:
    """Fetch year-by-year to dodge Yahoo's per-request cap, then stitch."""
    parts = []
    for y in range(START.year, END.year + 1):
        p1 = int(pd.Timestamp(f"{y}-01-01").timestamp())
        p2 = int(pd.Timestamp(f"{y}-12-31").timestamp())
        try:
            parts.append(fetch_chunk(p1, p2))
        except Exception as e:  # noqa: BLE001 - a single bad year shouldn't kill the stitch
            print(f"  warn: year {y} fetch failed: {e}", file=sys.stderr)
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index >= START) & (df.index <= END)]
    # Yahoo <^NSEI> carries zero volume before ~2013-01; keep the same
    # volume>0 drop rule the CSI300 pipeline applied (drops holiday/pre-launch
    # back-calc). Consequence (transparent): NIFTY coverage here starts
    # 2013-01-21, not 2005; the 2021-2024 test and 2025-2026 OOS windows are
    # fully covered either way.
    df = df[df["volume"] > 0].copy()
    return df.reset_index()


def main() -> None:
    df = fetch_nifty()
    print(f"NIFTY50 rows: {len(df)}")
    print(f"  range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"  columns: {list(df.columns)}")
    print(df.head())
    print(df.tail())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()