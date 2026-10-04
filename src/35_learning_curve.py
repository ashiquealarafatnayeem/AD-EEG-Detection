import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import FIGURES, PROCESSED

OUT = FIGURES / "day09"
SIZES = [20, 26, 32, 39, 45, 52]
N_SPLITS = 30

def one(X, y, n_train, i):
    warnings.filterwarnings("ignore")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=1000 * n_train + i)
    tr, te = next(sss.split(X, y))
    pipe, grid = ml.model_spec("logreg", X.shape[1])
    inner = StratifiedKFold(5, shuffle=True, random_state=i)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1).fit(X[tr], y[tr])
    return roc_auc_score(y[te], gs.predict_proba(X[te])[:, 1])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, *_ = ml.load_task("AD_vs_CN", "region")
    rows = []
    for n in SIZES:
        aucs = Parallel(n_jobs=-1)(delayed(one)(X, y, n, i) for i in range(N_SPLITS))
        rows.append({"n_train": n, "n_test": len(y) - n, "auc_mean": np.mean(aucs),
                     "auc_sd": np.std(aucs), "auc_q10": np.percentile(aucs, 10),
                     "auc_q90": np.percentile(aucs, 90)})
        print(f"  train on {n:2d} people, test on {len(y) - n:2d}: AUC {np.mean(aucs):.3f} "
              f"± {np.std(aucs):.3f}  (10th-90th percentile {rows[-1]['auc_q10']:.3f}-{rows[-1]['auc_q90']:.3f})")
    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day09_learning_curve.csv", index=False)
    gain = res.auc_mean.iloc[-1] - res.auc_mean.iloc[-3]
    print(f"\nGain from {SIZES[-3]} to {SIZES[-1]} training people: {gain:+.3f} AUC")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(res.n_train, res.auc_q10, res.auc_q90, color="#2471A3", alpha=.2,
                    label="10th-90th percentile of 30 splits")
    ax.plot(res.n_train, res.auc_mean, "-o", color="#1B2A41", label="mean")
    ax.axhline(0.5, color="k", ls=":", lw=.8)
    ax.set(xlabel="number of training people", ylabel="test AUC", ylim=(0.4, 1.0),
           title="AD vs CN, main pipeline: learning curve")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "05_learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {OUT / '05_learning_curve.png'}")

if __name__ == "__main__":
    main()
