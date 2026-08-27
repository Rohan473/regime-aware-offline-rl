"""Regenerate the offline dataset with the NR7 family appended.

1. Backs up the current 31-policy offline_dataset.parquet (rollback safety
   for the in-flight experiment matrix).
2. Rebuilds from the CACHED daily OHLCV (no download; deterministic).
3. Verifies the 31 existing policies' rows are bit-identical and only nr7
   rows were added, then writes the new parquet + manifest.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.offline_dataset import build_offline_dataset, save_dataset, summarize

processed = ROOT / "data" / "processed"
backup = Path(r"E:\Temp\opencode\offline_dataset_pre_nr7.parquet")

old = pd.read_parquet(processed / "offline_dataset.parquet")
old_policies = sorted(old["policy"].unique())
print(f"old dataset: {len(old)} rows, {len(old_policies)} policies")

cfg = OmegaConf.load(str(ROOT / "configs" / "data.yaml"))
cfg.regimes = OmegaConf.load(str(ROOT / "configs" / "regimes.yaml")).regimes

transitions, frame = build_offline_dataset(cfg)
print(f"new dataset: {len(transitions)} rows, {transitions['policy'].nunique()} policies")

new_policies = sorted(transitions["policy"].unique())
added = set(new_policies) - set(old_policies)
removed = set(old_policies) - set(new_policies)
print(f"added: {added} | removed: {removed}")
assert added == {"nr7"} and not removed, "unexpected policy set change"

# bit-identity of the pre-existing policies
merged = transitions[transitions["policy"].isin(old_policies)].reset_index(drop=True)
old_sorted = old.sort_values(["policy", "date"]).reset_index(drop=True)
new_sorted = merged.sort_values(["policy", "date"]).reset_index(drop=True)
assert len(old_sorted) == len(new_sorted), (
    f"row count changed for existing policies: {len(old_sorted)} -> {len(new_sorted)}"
)
for col in ("policy", "date", "action", "reward", "done"):
    same = (old_sorted[col].to_numpy() == new_sorted[col].to_numpy()).all() if col != "date" \
        else (old_sorted[col].to_numpy() == new_sorted[col].to_numpy()).all()
    assert same, f"column {col} changed for existing policies"
for col in ("z_ret_1d", "z_rsi_14", "z_bollinger_pos", "next_z_ret_1d"):
    assert (old_sorted[col].to_numpy() == new_sorted[col].to_numpy()).all(), f"{col} changed"
print("verified: all 31 existing policies bit-identical; only nr7 rows appended")

nr7 = transitions[transitions["policy"] == "nr7"]
print(f"nr7: {len(nr7)} transitions, action counts: "
      f"{nr7['action'].value_counts().to_dict()}")

# only now: backup + write
if not backup.exists():
    shutil.copy2(processed / "offline_dataset.parquet", backup)
    print(f"backup written: {backup}")
manifest = save_dataset(transitions, frame, cfg)
print(summarize(manifest))