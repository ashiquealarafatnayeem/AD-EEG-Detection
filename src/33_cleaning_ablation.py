import json
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
TASKS = ["AD_vs_CN", "FTD_vs_CN"]
LEVELS = ["L0_raw", "L1_filtered", "L2_car", "L2r_reject", "L3_full"]
LABELS = {"L0_raw": "raw\n(no filter)", "L1_filtered": "+ Day 3\nfilters",
          "L2_car": "+ average\nreference", "L2r_reject": "+ 150 uV\nrejection",
          "L3_full": "+ ICA\n(full)"}

def run_level(task, level):
    if level == "L3_full":
        z = np.load(PROCESSED / "day08" / f"{task}__region__logreg.npz")
        return ml.load_task(task, "region"), {"proba": z["proba"]}
    data = ml.load_task(task, f"region_{level}")
    path = CACHE / f"abl_{task}__{level}.npz"
    if path.exists():
        return data, {"proba": np.load(path)["proba"]}
    X, y, *_ = data
    r = ml.run_cv(X, y, "logreg")
    np.savez(path, proba=r["proba"], pred=r["pred"], params=json.dumps(r["params"], default=str))
    return data, r

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(PROCESSED / "day09_ablation_log.csv")
    rows = []
    for task in TASKS:
        (Xf, yf, idf, _, _), full = run_level(task, "L3_full")
        for level in LEVELS:
            (X, y, ids, _, _), r = run_level(task, level)
            if not (np.array_equal(ids, idf) and np.array_equal(y, yf)):
                raise SystemExit(f"{level}: people or order differ from Day 8 - cannot pair.")
            s = ml.summarise(y, r["proba"], r["proba"].argmax(axis=2))
            d, lo, hi, p = ml.paired_difference(y, full["proba"], r["proba"])
            grp = log[log.group.isin(ml.TASKS[task])]
            rows.append({"task": task, "level": level, "auc": s["auc"], "auc_lo": s["auc_lo"],
                         "auc_hi": s["auc_hi"], "diff_vs_full": d, "lo": lo, "hi": hi, "p": p,
                         "epochs_median": float(grp[f"epochs_{level}"].median())})
            print(f"  {task:9s} {level:12s} AUC {s['auc']:.3f} [{s['auc_lo']:.3f}, {s['auc_hi']:.3f}]"
                  f"   vs full: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  p = {p:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day09_cleaning_ablation.csv", index=False)
    for task in TASKS:
        print(f"\n=== {task}: main pipeline at each cleaning level ===")
        print(f"{'level':14s}{'epochs (median)':>16s}{'AUC [95% CI]':>24s}{'minus full [95% CI]':>26s}")
        for _, r in res[res.task == task].iterrows():
            print(f"{r.level:14s}{r.epochs_median:>16.0f}{r.auc:>9.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]"
                  f"{r.diff_vs_full:>+10.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for task, col, dx in [("AD_vs_CN", "#C0392B", -.06), ("FTD_vs_CN", "#E67E22", .06)]:
        d = res[res.task == task]
        x = np.arange(len(LEVELS)) + dx
        ax.errorbar(x, d.auc, yerr=[d.auc - d.auc_lo, d.auc_hi - d.auc], fmt="-o", color=col,
                    capsize=3, label=task.replace("_", " "))
    ax.axhline(0.5, color="k", ls=":", lw=.8)
    ax.set_xticks(range(len(LEVELS)), [LABELS[l] for l in LEVELS], fontsize=8.5)
    ax.set(ylim=(0.2, 1.02), ylabel="cross-validated AUC (95 % CI)",
           title="Same features, same classifier, more and more cleaning")
    ax.grid(alpha=.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "03_cleaning_ablation.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure: {OUT / '03_cleaning_ablation.png'}")

if __name__ == "__main__":
    main()
