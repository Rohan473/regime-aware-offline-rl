"""OHLCV loading and caching.

Raw source: yearly ``SPY_YYYY.parquet`` files of 1-minute bars produced by an
external downloader (currently writing into the repo root). This module:

* reads every matching file, validates schema/invariants,
* resamples minute bars to daily OHLCV (UTC timestamps -> America/New_York
  trading dates),
* caches the daily frame to ``data/processed/daily_ohlcv.parquet`` guarded by
  a manifest of input files (name / size / mtime) + resample params, so the
  cache is rebuilt only when raw data actually changed.

Failures are loud: a missing/corrupt file raises; missing *years* inside the
configured range only warn (the downloader fills them in incrementally and a
later re-run picks them up).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def resolve_path(cfg: DictConfig, key: str) -> Path:
    """Resolve a config path relative to the repository root.

    ``key`` may be dotted (e.g. "data.processed_dir")."""
    node = cfg
    for part in key.split("."):
        node = node[part]
    p = Path(str(node))
    return p if p.is_absolute() else (REPO_ROOT / p)


def _validate_daily(df: pd.DataFrame, source: str) -> None:
    if df.isnull().values.any():
        bad = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"daily OHLCV from {source} contains NaN in columns: {bad}"
        )
    if (df["close"] <= 0).any():
        raise ValueError(f"daily OHLCV from {source} has non-positive close")
    if (df["open"] <= 0).any():
        raise ValueError(f"daily OHLCV from {source} has non-positive open")
    if (df["high"] < df["low"]).any():
        raise ValueError(f"daily OHLCV from {source} has high < low")
    if (df["volume"] < 0).any():
        raise ValueError(f"daily OHLCV from {source} has negative volume")


def _validate_continuity(df: pd.DataFrame, source: str, max_gap_bdays: int = 10) -> None:
    """Reject daily frames with large interior gaps (call on daily frames only).

    pct_change()/diff() bridge missing days into one fake "daily" return
    spanning the whole hole, which fabricates crisis regimes and catastrophic
    loss days downstream. Missing *years* at the range edges are tolerated
    (incremental download); holes *inside* the covered range are a defect and
    must fail loudly.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return
    days = idx.tz_localize(None).to_numpy().astype("datetime64[D]")
    bday_gaps = np.busday_count(days[:-1], days[1:]) - 1
    bad = np.flatnonzero(bday_gaps > max_gap_bdays)
    if len(bad) == 0:
        return
    i = int(bad[np.argmax(bday_gaps[bad])])
    raise ValueError(
        f"daily OHLCV from {source} has interior business-day gaps "
        f"(max {int(bday_gaps[i])} business days: "
        f"{idx[i].date()} -> {idx[i + 1].date()}). Multi-day holes make return "
        f"features bridge the gap into one fake 'daily' return; this corrupts "
        f"regime labels and the offline dataset. Re-run after the downloader "
        f"fills the missing files, or fix the corrupt raw file. "
        f"{len(bad)} gap(s) found."
    )


def _read_raw_file(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - re-raise with file context
        raise RuntimeError(
            f"failed to read raw data file {path}: {exc!r}. "
            f"If the downloader is mid-write to this file, re-run after it "
            f"finishes."
        ) from exc

    if not set(REQUIRED_COLUMNS).issubset(raw.columns):
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        raise ValueError(f"{path.name}: missing columns {missing}; got {list(raw.columns)}")

    raw = raw[REQUIRED_COLUMNS].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.dropna().drop_duplicates(subset=["timestamp"], keep="last")
    if len(raw) == 0:
        raise ValueError(f"{path.name}: no valid rows")
    raw = raw.set_index("timestamp").sort_index()
    _validate_daily(raw, path.name)
    return raw


def list_raw_files(cfg: DictConfig) -> list[Path]:
    raw_dir = resolve_path(cfg, "data.raw_dir")
    pattern = str(cfg.data.raw_pattern)
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise RuntimeError(
            f"no files matching '{pattern}' found in {raw_dir}. "
            f"Configure data.raw_dir / data.raw_pattern in configs/data.yaml."
        )
    return files


def _manifest_of(files: list[Path], resample_cfg: DictConfig) -> dict:
    entries = [
        {"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime_ns}
        for f in files
    ]
    resample_dict = OmegaConf.to_container(resample_cfg, resolve=True)
    payload = json.dumps({"files": entries, "resample": resample_dict}, sort_keys=True)
    return {
        "files": entries,
        "resample": resample_dict,
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _resample_daily(raw: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    tz = str(cfg.data.resample.timezone)
    daily = raw.tz_convert(tz).resample("1D").agg(
        open=("open", str(cfg.data.resample.open)),
        high=("high", str(cfg.data.resample.high)),
        low=("low", str(cfg.data.resample.low)),
        close=("close", str(cfg.data.resample.close)),
        volume=("volume", str(cfg.data.resample.volume)),
    )
    daily = daily.dropna(subset=["open", "high", "low", "close"])
    daily = daily[daily["volume"] > 0]
    _validate_daily(daily, "resampled minute data")
    _validate_continuity(daily, "resampled minute data")
    daily.index.name = "date"
    return daily


def load_raw_minute_bars(cfg: DictConfig) -> pd.DataFrame:
    """Read + validate + concatenate all raw yearly files (minutes)."""
    files = list_raw_files(cfg)
    frames = []
    for f in files:
        frames.append(_read_raw_file(f))
    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    if len(raw) == 0:
        raise RuntimeError("raw minute data is empty after loading")
    return raw


def load_daily_ohlcv(cfg: DictConfig, *, force_rebuild: bool = False) -> pd.DataFrame:
    """Daily OHLCV with cache. Rebuilds when raw files or resample params change.

    Raises loudly if raw data is missing/corrupt. Missing configured years
    produce a warning only (incremental download is expected).
    """
    files = list_raw_files(cfg)
    processed_dir = resolve_path(cfg, "data.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache_path = processed_dir / "daily_ohlcv.parquet"
    manifest_path = processed_dir / "daily_ohlcv.manifest.json"

    fresh_manifest = _manifest_of(files, cfg.data.resample)

    if (
        not force_rebuild
        and cache_path.exists()
        and manifest_path.exists()
        and json.loads(manifest_path.read_text()) == fresh_manifest
    ):
        daily = pd.read_parquet(cache_path)
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "date"
        _validate_daily(daily, "cached daily OHLCV")
        _validate_continuity(daily, "cached daily OHLCV")
        _warn_missing_years(daily, cfg)
        return daily

    raw = load_raw_minute_bars(cfg)
    daily = _resample_daily(raw, cfg)

    start = pd.Timestamp(str(cfg.data.start_date), tz=daily.index.tz)
    end = pd.Timestamp(str(cfg.data.end_date), tz=daily.index.tz)
    covered = daily[(daily.index >= start) & (daily.index <= end)]
    if len(covered) == 0:
        raise RuntimeError(
            f"no daily bars in requested range {start.date()}..{end.date()}; "
            f"data spans {daily.index.min().date()}..{daily.index.max().date()}"
        )

    daily.to_parquet(cache_path)
    manifest_path.write_text(json.dumps(fresh_manifest, indent=2))
    _warn_missing_years(daily, cfg)
    return daily


def _warn_missing_years(daily: pd.DataFrame, cfg: DictConfig) -> None:
    days = daily.index.tz_localize(None) if daily.index.tz is not None else daily.index
    start = pd.Timestamp(str(cfg.data.start_date))
    end = pd.Timestamp(str(cfg.data.end_date))
    have_min, have_max = days.min().normalize(), days.max().normalize()
    print(
        f"[loaders] daily OHLCV coverage: {have_min.date()} .. {have_max.date()} "
        f"({len(daily)} trading days)"
    )
    if have_min > start or have_max < end:
        print(
            f"[loaders] WARNING: configured range {start.date()}..{end.date()} "
            f"is only partially covered by raw files on disk "
            f"({have_min.date()}..{have_max.date()}). "
            f"The downloader is incremental; re-run this pipeline after it "
            f"finishes to extend the range."
        )
