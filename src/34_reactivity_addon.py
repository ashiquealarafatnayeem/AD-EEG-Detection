import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import FIGURES, PROCESSED

CACHE = PROCESSED / "day09"
OUT = FIGURES / "day09"
REACT = ["react_log__posterior", "react_log__global", "react_rel__posterior", "react_rel__global"]

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / "features_region_all.csv")
    df = df[df.group.isin(["AD", "CN"]) & df.react_log__posterior.notna()].reset_index(drop=True)
    y = (df.group == "AD").astype(int).to_numpy()
    drop = ml.META + ["ds006036_id"] + REACT
    region_cols = [c for c in df.columns if c not in drop]
    print(f"People with reactivity and a confirmed partner: {len(df)} "
          f"({int(y.sum())} AD, {int((1 - y).sum())} CN); region features: {len(region_cols)}\n")

    sets = {"A. region only": region_cols, "B. reactivity only": REACT,
            "C. region + reactivity": region_cols + REACT}
    runs = {}
    for name, cols in sets.items():
        path = CACHE / f"react_{name[0]}.npz"
        if path.exists():
            runs[name] = np.load(path)["proba"]
        else:
            r = ml.run_cv(df[cols].to_numpy(float), y, "logreg")
            np.savez(path, proba=r["proba"], pred=r["pred"])
            runs[name] = r["proba"]

    rows = []
    for name, proba in runs.items():
        s = ml.summarise(y, proba, proba.argmax(axis=2))
        row = {"model": name, "n_features": len(sets[name]), "auc": s["auc"],
               "lo": s["auc_lo"], "hi": s["auc_hi"]}
        if not name.startswith("A"):
            d, lo, hi, p = ml.paired_difference(y, runs["A. region only"], proba)
            row.update(diff_vs_A=d, diff_lo=lo, diff_hi=hi, p=p)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day09_reactivity.csv", index=False)

    print("=== AD vs CN on the 32 people with reactivity ===")
    print(f"{'model':26s}{'n':>4s}{'AUC [95% CI]':>24s}{'minus A [95% CI]':>26s}{'p':>7s}")
    for _, r in res.iterrows():
        extra = (f"{r.diff_vs_A:>+10.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]{r.p:>7.3f}"
                 if not r.model.startswith("A") else "")
        print(f"{r.model:26s}{r.n_features:>4d}{r.auc:>9.3f} [{r.lo:.3f}, {r.hi:.3f}]{extra}")
    print("\nFor reference, the Day 8 main model on all 65 people: AUC 0.811 [0.717, 0.896].")
    print("Exploratory: 32 people only, so a difference must be large to be detectable.")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for k, (_, r) in enumerate(res.iterrows()):
        ax.errorbar(r.auc, k, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt="o", capsize=3,
                    color=["#1B2A41", "#1E8449", "#7D3C98"][k])
    ax.axvline(0.5, color="k", ls=":", lw=.8)
    ax.set_yticks(range(len(res)), res.model); ax.invert_yaxis()
    ax.set(xlim=(0.3, 1.0), xlabel="cross-validated AUC (95 % CI)",
           title=f"AD vs CN, {len(df)} people with alpha reactivity")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "04_reactivity_addon.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {OUT / '04_reactivity_addon.png'}")

if __name__ == "__main__":
    main()
