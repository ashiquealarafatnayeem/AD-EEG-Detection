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
OUT = FIGURES / "day09"
MODELS = [m for m in ml.MODELS if m != "dummy"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ml.TASKS:
        _, y, _, _, _ = ml.load_task(task, "region")
        for model in MODELS:
            pa = np.load(DAY8 / f"{task}__region__{model}.npz")["proba"]
            pb = np.load(DAY8 / f"{task}__channel__{model}.npz")["proba"]
            d, lo, hi, p = ml.paired_difference(y, pa, pb)
            rows.append({"task": task, "model": model, "auc_region": ml.main_score(y, pa),
                         "auc_channel": ml.main_score(y, pb), "diff": d, "lo": lo, "hi": hi, "p": p})
    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day09_region_vs_channel.csv", index=False)

    print("=== Channel minus region AUC, paired bootstrap over people (2000 resamples) ===")
    print(f"{'task':11s}{'model':22s}{'region':>8s}{'channel':>9s}{'difference [95% CI]':>28s}{'p':>7s}")
    for _, r in res.iterrows():
        flag = "  *" if r.lo > 0 or r.hi < 0 else ""
        print(f"{r.task:11s}{ml.MODEL_NAMES[r.model]:22s}{r.auc_region:>8.3f}{r.auc_channel:>9.3f}"
              f"{r['diff']:>+12.3f} [{r.lo:+.3f}, {r.hi:+.3f}]{r.p:>7.3f}{flag}")
    n_sig = int(((res.lo > 0) | (res.hi < 0)).sum())
    print(f"\n* = interval excludes 0. {n_sig} of {len(res)} comparisons. With {len(res)} comparisons,"
          f" about {0.05 * len(res):.0f} would do so by chance alone at the 5 % level.")
    main_row = res[(res.task == "AD_vs_CN") & (res.model == "logreg")].iloc[0]
    print(f"\nMain model, AD vs CN: channel - region = {main_row['diff']:+.3f} "
          f"[{main_row.lo:+.3f}, {main_row.hi:+.3f}], p = {main_row.p:.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, task in zip(axes, ml.TASKS):
        d = res[res.task == task].reset_index(drop=True)
        for k, r in d.iterrows():
            col = "#1E8449" if r.lo > 0 else ("#C0392B" if r.hi < 0 else "#2471A3")
            ax.errorbar(r["diff"], k, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o",
                        color=col, capsize=3)
        ax.axvline(0, color="k", lw=.8, ls=":")
        ax.set(title=task.replace("_", " "), xlabel="AUC(channel) - AUC(region)", xlim=(-0.25, 0.25))
        ax.grid(alpha=.3, axis="x")
    axes[0].set_yticks(range(len(MODELS)), [ml.MODEL_NAMES[m] for m in MODELS])
    axes[0].invert_yaxis()
    fig.suptitle("Right of 0 = channel features better; bars = 95 % CI of the paired difference", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "02_region_vs_channel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {OUT / '02_region_vs_channel.png'}")

if __name__ == "__main__":
    main()
