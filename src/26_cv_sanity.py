import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import FIGURES, PROCESSED

OUT = FIGURES / "day08"
N_DRAWS = 10
REPEATS = 2

def nested_auc(X, y):
    r = ml.run_cv(X, y, "logreg", n_repeats=REPEATS)
    return ml.main_score(y, r["proba"])

def pooled_auc(model, X, y, cv):
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, _, names, classes = ml.load_task("AD_vs_CN", "region")
    n, d = X.shape
    print(f"Real data: {n} people ({int((y == 1).sum())} AD, {int((y == 0).sum())} CN), "
          f"{d} features\n")
    rng = np.random.default_rng(0)
    rows = []
    t0 = time.perf_counter()

    a = [nested_auc(rng.standard_normal((n, d)), y) for _ in range(N_DRAWS)]
    rows += [("A. noise features", v) for v in a]
    print(f"A. Noise features, real labels  : AUC mean {np.mean(a):.3f}, "
          f"range {min(a):.3f}-{max(a):.3f}   ({time.perf_counter() - t0:.0f} s)")

    b = [nested_auc(X, rng.permutation(y)) for _ in range(N_DRAWS)]
    rows += [("B. shuffled labels", v) for v in b]
    print(f"B. Real features, shuffled labels: AUC mean {np.mean(b):.3f}, "
          f"range {min(b):.3f}-{max(b):.3f}   ({time.perf_counter() - t0:.0f} s)")

    c = []
    for _ in range(3):
        Z = rng.standard_normal((n, d))
        Z[y == 1, :5] += 1.0
        c.append(nested_auc(Z, y))
    rows += [("C. planted signal", v) for v in c]
    print(f"C. Noise + planted difference     : AUC mean {np.mean(c):.3f}, "
          f"range {min(c):.3f}-{max(c):.3f}   ({time.perf_counter() - t0:.0f} s)")

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
                                                            max_iter=5000))
    lr_sel = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10),
                           LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    leak, clean = [], []
    for i in range(N_DRAWS):
        Z = rng.standard_normal((n, d))
        cv = StratifiedKFold(5, shuffle=True, random_state=i)
        top = SelectKBest(f_classif, k=10).fit(Z, y).get_support()
        leak.append(pooled_auc(lr, Z[:, top], y, cv))
        clean.append(pooled_auc(lr_sel, Z, y, cv))
    rows += [("D. select on everyone (WRONG)", v) for v in leak]
    rows += [("D. select inside folds", v) for v in clean]
    print(f"D. Noise, 10 features chosen on everyone first: AUC mean {np.mean(leak):.3f}   <- the trap")
    print(f"   Noise, 10 features chosen inside each fold  : AUC mean {np.mean(clean):.3f}")

    prior = DummyClassifier(strategy="prior")
    e_loo = pooled_auc(prior, X, y, LeaveOneOut())
    e_kf = np.mean([pooled_auc(prior, X, y, StratifiedKFold(5, shuffle=True, random_state=i))
                    for i in range(10)])
    lr_all = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    r_loo = pooled_auc(lr_all, X, y, LeaveOneOut())
    r_kf = [pooled_auc(lr_all, X, y, StratifiedKFold(5, shuffle=True, random_state=i))
            for i in range(10)]
    print("E. A model that only learns the class ratio (no EEG at all):")
    print(f"   leave-one-out AUC = {e_loo:.3f}    5-fold AUC = {e_kf:.3f}   (the truth is 0.5)")
    print("   Logistic regression (balanced class weights) on the real features:")
    print(f"   leave-one-out AUC = {r_loo:.3f}    5-fold AUC = {np.mean(r_kf):.3f} "
          f"(10 shuffles, range {min(r_kf):.3f}-{max(r_kf):.3f})")

    checks = [("A. noise gives chance", 0.40 <= np.mean(a) <= 0.60),
              ("B. shuffled labels give chance", 0.40 <= np.mean(b) <= 0.60),
              ("C. a real difference is found", np.mean(c) >= 0.75),
              ("D. the trap is visible and avoided", np.mean(leak) >= 0.65 and np.mean(clean) <= 0.60),
              ("E. leave-one-out trap shown", e_loo <= 0.2)]
    print("\n=== Checkpoint ===")
    for name, ok in checks:
        print(f"  {'PASS ' if ok else 'CHECK'}  {name}")
    print(f"  Chance spread: pure noise reached AUC {max(a):.2f} in one of {N_DRAWS} runs. "
          f"A single run is never enough on its own.")
    df = pd.DataFrame(rows, columns=["test", "auc"])
    df.to_csv(PROCESSED / "day08_sanity.csv", index=False)

    order = ["A. noise features", "B. shuffled labels", "C. planted signal",
             "D. select on everyone (WRONG)", "D. select inside folds"]
    colors = ["#8a96a3", "#8a96a3", "#1E8449", "#C0392B", "#2471A3"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for k, (name, col) in enumerate(zip(order, colors)):
        v = df.loc[df.test == name, "auc"].to_numpy()
        ax.scatter(v, np.full(len(v), k) + rng.uniform(-.12, .12, len(v)), color=col, s=26, alpha=.85)
        ax.plot([v.mean()] * 2, [k - .3, k + .3], color="k", lw=1.6)
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set(xlim=(0.2, 1.0), xlabel="cross-validated AUC (black bar = mean)",
           title="The testing machinery on data with a known answer")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "01_cv_sanity.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure: {OUT / '01_cv_sanity.png'}   Total time {time.perf_counter() - t0:.0f} s")

if __name__ == "__main__":
    main()
