import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import FIGURES, PROCESSED

CACHE = PROCESSED / "day10" / "seeds"
OUT = FIGURES / "day10"
TASKS = ["AD_vs_CN", "FTD_vs_CN"]
SEEDS = list(range(1, 21))

def run(task, X, y, seed):
    if seed == ml.SEED:
        return np.load(PROCESSED / "day08" / f"{task}__region__logreg.npz")["proba"], "Day 8"
    path = CACHE / f"{task}__seed{seed}.npz"
    if path.exists():
        return np.load(path)["proba"], "cached"
    r = ml.run_cv(X, y, "logreg", seed=seed)
    np.savez(path, proba=r["proba"], pred=r["pred"])
    return r["proba"], "ok"

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for task in TASKS:
        X, y, *_ = ml.load_task(task, "region")
        for seed in [ml.SEED] + SEEDS:
            proba, status = run(task, X, y, seed)
            per_rep = ml.auc_per_repeat(y, proba)
            rows.append({"task": task, "seed": seed, "auc": float(per_rep.mean()),
                         "auc_sd_repeats": float(per_rep.std()),
                         "auc_min_repeat": float(per_rep.min()), "auc_max_repeat": float(per_rep.max())})
            print(f"  {task:9s} seed {seed:4d}  AUC {per_rep.mean():.3f}  "
                  f"(10 repeats: {per_rep.min():.3f}-{per_rep.max():.3f})  "
                  f"{status:6s} {(time.perf_counter() - t0) / 60:5.1f} min")
    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day10_seeds.csv", index=False)

    n = len(SEEDS) + 1
    print(f"\n=== Main pipeline, {n} seeds (2026 = the pre-declared one) ===")
    print(f"{'task':10s}{'seed 2026':>10s}{f'other {n - 1}: median':>18s}{'min':>7s}{'max':>7s}"
          f"{'rank of 2026':>14s}{f'all {10 * n} repeats':>17s}")
    for task in TASKS:
        d = res[res.task == task]
        main = d[d.seed == ml.SEED].auc.iloc[0]
        other = d[d.seed != ml.SEED].auc
        rank = int((d.auc > main).sum()) + 1
        print(f"{task:10s}{main:>10.3f}{other.median():>18.3f}{other.min():>7.3f}{other.max():>7.3f}"
              f"{f'{rank} of {n}':>14s}{d.auc.mean():>17.3f}")
    print(f"\nRank 1 = seed 2026 gave the highest AUC of all {n}; rank {n} = the lowest.")
    print(f"'all {10 * n} repeats' = the average over every repeat of every seed: the most stable estimate.")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    rng = np.random.default_rng(0)
    for k, task in enumerate(TASKS):
        d = res[res.task == task]
        other = d[d.seed != ml.SEED]
        ax.scatter(other.auc, k + rng.uniform(-.12, .12, len(other)), color="#8a96a3", s=26,
                   label=f"seeds {SEEDS[0]}-{SEEDS[-1]}" if k == 0 else None)
        ax.scatter(d[d.seed == ml.SEED].auc, [k], color="#C0392B", s=80, marker="D", zorder=3,
                   label="seed 2026 (pre-declared)" if k == 0 else None)
    ax.set_yticks(range(len(TASKS)), [t.replace("_", " ") for t in TASKS])
    ax.set(xlim=(0.6, 0.95), xlabel="cross-validated AUC (mean of 10 repeats)",
           title="Main pipeline: does the random seed matter?")
    ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "01_seeds.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {OUT / '01_seeds.png'}")

if __name__ == "__main__":
    main()
