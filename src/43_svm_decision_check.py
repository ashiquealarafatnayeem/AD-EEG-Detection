import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import PROCESSED

RUNS = [("AD_vs_FTD", "linsvm"), ("AD_vs_FTD", "rbfsvm"), ("AD_vs_CN", "linsvm"), ("AD_vs_CN", "rbfsvm")]

def refit(X, y, name):
    outer = RepeatedStratifiedKFold(n_splits=ml.N_FOLDS, n_repeats=ml.N_REPEATS, random_state=ml.SEED)
    proba = np.full((ml.N_REPEATS, len(y), 2), np.nan)
    dec = np.full((ml.N_REPEATS, len(y)), np.nan)
    flipped = []
    for f, (tr, te) in enumerate(outer.split(X, y)):
        pipe, grid = ml.model_spec(name, X.shape[1])
        inner = StratifiedKFold(ml.INNER_FOLDS, shuffle=True, random_state=ml.SEED + f)
        est = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1).fit(X[tr], y[tr]).best_estimator_
        r = f // ml.N_FOLDS
        proba[r, te] = est.predict_proba(X[te])
        dec[r, te] = est.decision_function(X[te])
        rho = spearmanr(dec[r, te], proba[r, te, 1]).correlation
        flipped.append(rho < 0)
    return proba, dec, np.array(flipped)

def main():
    warnings.filterwarnings("ignore")
    rows = []
    for task, name in RUNS:
        X, y, *_ = ml.load_task(task, "region")
        proba, dec, flipped = refit(X, y, name)
        day8 = np.load(PROCESSED / "day08" / f"{task}__region__{name}.npz")["proba"]
        same = np.allclose(proba, day8, atol=1e-8)
        dec3 = np.stack([-dec, dec], axis=2)
        row = {"task": task, "model": name, "same_as_day8": same,
               "auc_proba_pooled": ml.main_score(y, proba),
               "auc_decision_pooled": ml.main_score(y, dec3),
               "auc_proba_foldwise": float(ml.foldwise_auc(y, proba).mean()),
               "auc_decision_foldwise": float(ml.foldwise_auc(y, dec3).mean()),
               "folds_flipped": float(flipped.mean())}
        rows.append(row)
        print(f"  {task:9s} {ml.MODEL_NAMES[name]:12s} reproduces Day 8: {same}")
        if not same:
            print("    CHECK: probabilities differ from Day 8 - paste this output to me.")
    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day10_svm_decision.csv", index=False)

    print("\n=== The same fitted SVMs, ranked two ways (region features) ===")
    print(f"{'':22s}{'---- pooled AUC ----':>22s}{'-- fold-wise AUC --':>22s}")
    print(f"{'task':10s}{'model':12s}{'proba':>11s}{'decision':>11s}{'proba':>11s}{'decision':>11s}"
          f"{'flipped':>10s}")
    for _, r in res.iterrows():
        print(f"{r.task:10s}{ml.MODEL_NAMES[r.model]:12s}{r.auc_proba_pooled:>11.3f}{r.auc_decision_pooled:>11.3f}"
              f"{r.auc_proba_foldwise:>11.3f}{r.auc_decision_foldwise:>11.3f}{r.folds_flipped:>10.0%}")
    print("\n'folds flipped' = share of the 50 fold-models whose probabilities run OPPOSITE to their")
    print("own decision values (Platt sigmoid fitted the wrong way round).")

if __name__ == "__main__":
    main()
