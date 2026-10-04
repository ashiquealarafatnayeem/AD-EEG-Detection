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

DAY8 = PROCESSED / "day08"
OUT = FIGURES / "day10"
SHORT = {"dummy": "Chance", "logreg": "LogReg", "linsvm": "LinSVM", "rbfsvm": "RBF SVM",
         "rf": "Forest", "knn": "kNN"}

def fold_offset(y, proba):
    if proba.shape[2] != 2:
        return np.nan
    return float(np.std([proba[r, te, 1].mean() for r, te in ml.outer_splits(y, proba.shape[0])]))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ml.TASKS:
        for fs in ["region", "channel", "age_sex"]:
            _, y, *_ = ml.load_task(task, fs)
            for model in ml.MODELS:
                path = DAY8 / f"{task}__{fs}__{model}.npz"
                if not path.exists():
                    continue
                proba = np.load(path)["proba"]
                fw = ml.foldwise_auc(y, proba)
                rows.append({"task": task, "features": fs, "model": model,
                             "pooled": ml.main_score(y, proba), "foldwise": float(fw.mean()),
                             "foldwise_sd": float(fw.std()),
                             "folds_below_half": float((fw < 0.5).mean()),
                             "fold_offset_sd": fold_offset(y, proba)})
    res = pd.DataFrame(rows)
    res["diff"] = res.foldwise - res.pooled
    res.to_csv(PROCESSED / "day10_foldwise.csv", index=False)

    for task in ml.TASKS:
        d = res[res.task == task]
        print(f"\n=== {task}: pooled (Day 8) vs fold-wise AUC ===")
        print(f"{'model':9s}{'features':>9s}{'pooled':>8s}{'fold-wise':>11s}{'diff':>8s}"
              f"{'folds < 0.5':>13s}{'offset SD':>11s}")
        for _, r in d.iterrows():
            off = "" if np.isnan(r.fold_offset_sd) else f"{r.fold_offset_sd:.3f}"
            print(f"{SHORT[r.model]:9s}{r.features:>9s}{r.pooled:>8.3f}{r.foldwise:>11.3f}"
                  f"{r['diff']:>+8.3f}{r.folds_below_half:>12.0%}{off:>11s}")

    main_row = res[(res.task == "AD_vs_CN") & (res.features == "region") & (res.model == "logreg")].iloc[0]
    real = res[res.model != "dummy"]
    print(f"\nMain result: pooled {main_row.pooled:.3f}, fold-wise {main_row.foldwise:.3f} "
          f"(SD over 50 folds {main_row.foldwise_sd:.3f})")
    print(f"Over all {len(real)} non-chance runs: fold-wise minus pooled, median {real['diff'].median():+.3f}, "
          f"range {real['diff'].min():+.3f} to {real['diff'].max():+.3f}")
    b = real.dropna(subset=["fold_offset_sd"])
    if len(b) > 2:
        rho = pd.Series(b.fold_offset_sd.values).corr(pd.Series(b["diff"].values), method="spearman")
        print(f"Binary runs: Spearman rho(offset SD, fold-wise minus pooled) = {rho:+.2f} "
              f"(positive = larger offsets, larger pooling penalty)")
    below = res[(res.pooled < 0.5) & (res.model != "dummy")]
    if len(below):
        print("\nRuns with a pooled AUC below 0.5, and their fold-wise AUC:")
        for _, r in below.iterrows():
            print(f"  {r.task:9s} {r.features:8s} {SHORT[r.model]:8s} pooled {r.pooled:.3f}  "
                  f"fold-wise {r.foldwise:.3f}  offset SD {r.fold_offset_sd:.3f}")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    colors = {"AD_vs_CN": "#C0392B", "FTD_vs_CN": "#E67E22", "AD_vs_FTD": "#7D3C98", "3class": "#2471A3"}
    for task, col in colors.items():
        d = real[real.task == task]
        ax.scatter(d.pooled, d.foldwise, color=col, s=28, label=task.replace("_", " "))
    ax.scatter([main_row.pooled], [main_row.foldwise], s=140, facecolors="none", edgecolors="k",
               label="main result")
    ax.plot([0.3, 1], [0.3, 1], color="k", lw=.8, ls="--")
    ax.axhline(0.5, color="k", lw=.6, ls=":"); ax.axvline(0.5, color="k", lw=.6, ls=":")
    ax.set(xlim=(0.35, 0.95), ylim=(0.35, 0.95), xlabel="pooled AUC (Day 8)",
           ylabel="fold-wise AUC", title="Above the dashed line = pooling lowered the AUC")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "02_foldwise_vs_pooled.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure: {OUT / '02_foldwise_vs_pooled.png'}")

if __name__ == "__main__":
    main()
