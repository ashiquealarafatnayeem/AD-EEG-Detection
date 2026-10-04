import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import PROCESSED, participants

f = pd.read_csv(PROCESSED / "features_region.csv").merge(
    participants()[["participant_id", "MMSE"]], on="participant_id")
ad = f[f.group == "AD"]

print("Spearman correlation with MMSE, AD patients only (lower MMSE = more impaired)")
for col in ["alpha_iaf", "alpha_height", "logTAR__temporal", "logTAR__global", "rel_theta__global"]:
    d = ad[[col, "MMSE"]].dropna()
    rho, p = spearmanr(d[col], d["MMSE"])
    print(f"  {col:20s} n={len(d):2d}   rho = {rho:+.2f}   p = {p:.3f}")
