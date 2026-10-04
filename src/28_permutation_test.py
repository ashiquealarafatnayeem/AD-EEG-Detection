import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import FIGURES, PROCESSED

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
REPEATS = 2
CHUNK = 20
CACHE = PROCESSED / "day08"
OUT = FIGURES / "day08"

def score(X, y):
    r = ml.run_cv(X, y, "logreg", n_repeats=REPEATS, n_jobs=1)
    return ml.main_score(y, r["proba"])

def one_permutation(X, y, i):
    y_perm = np.random.default_rng(10_000 + i).permutation(y)
    return score(X, y_perm)

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nulls = [], {}
    for task in ml.TASKS:
        X, y, *_ = ml.load_task(task, "region")
        path = CACHE / f"perm_{task}.npz"
        if path.exists():
            z = np.load(path)
            observed, null = float(z["observed"]), list(z["null"])
        else:
            observed, null = score(X, y), []
        t0 = time.perf_counter()
        while len(null) < N_PERM:
            start = len(null)
            stop = min(start + CHUNK, N_PERM)
            null += Parallel(n_jobs=-1)(delayed(one_permutation)(X, y, i) for i in range(start, stop))
            np.savez(path, observed=observed, null=np.array(null))
            print(f"  {task:9s} {len(null):3d}/{N_PERM} shuffles done   "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)")
        null = np.array(null)
        p = (1 + np.sum(null >= observed)) / (1 + len(null))
        rows.append({"task": task, "observed_auc": observed, "null_mean": null.mean(),
                     "null_95th": np.percentile(null, 95), "null_max": null.max(), "p": p})
        nulls[task] = (observed, null)

    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day08_permutation.csv", index=False)
    print(f"\n=== Permutation test, main pipeline (region, logistic regression), {N_PERM} shuffles ===")
    print(f"{'task':10s}{'real AUC':>10s}{'shuffled: mean':>16s}{'95th pct':>10s}{'max':>8s}{'p':>9s}")
    for _, r in res.iterrows():
        print(f"{r.task:10s}{r.observed_auc:>10.3f}{r.null_mean:>16.3f}{r.null_95th:>10.3f}"
              f"{r.null_max:>8.3f}{r.p:>9.3f}")
    print(f"(the smallest possible p with {N_PERM} shuffles is {1 / (N_PERM + 1):.3f})")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, task in zip(axes, ml.TASKS):
        observed, null = nulls[task]
        ax.hist(null, bins=20, color="#8a96a3", edgecolor="white")
        ax.axvline(observed, color="#C0392B", lw=2, label=f"real {observed:.2f}")
        p = res.loc[res.task == task, "p"].iloc[0]
        ax.set(title=f"{task}   p = {p:.3f}", xlabel="AUC", xlim=(0.2, 1.0))
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"number of shuffles (of {N_PERM})")
    fig.suptitle("Grey = scores with shuffled labels (chance); red = the real score", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "06_permutation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {OUT / '06_permutation.png'}")

if __name__ == "__main__":
    main()
