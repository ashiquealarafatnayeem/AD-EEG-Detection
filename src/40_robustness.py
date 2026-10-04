import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import PROCESSED

CACHE = PROCESSED / "day10"
TASKS = ["AD_vs_CN", "FTD_vs_CN"]
ALTERNATIVES = ["E2", "R100", "R200", "no_delta"]
STRESS = ["logTAR_only", "posterior_only"]
METHOD = {"E2": "2 s epochs", "R100": "reject at 100 uV", "R200": "reject at 200 uV"}
SUBSETS = {
    "no_delta": ("no delta features", lambda n: not (n.startswith("rel_delta") or n.startswith("logSlowFast"))),
    "logTAR_only": ("theta/alpha ratios only", lambda n: n.startswith("logTAR")),
    "posterior_only": ("posterior features only", lambda n: n.endswith("__posterior") or n.startswith("alpha_")),
}

def cached(tag, X, y):
    path = CACHE / f"{tag}.npz"
    if path.exists():
        return np.load(path)["proba"], "cached"
    r = ml.run_cv(X, y, "logreg")
    np.savez(path, proba=r["proba"], pred=r["pred"], sel_freq=r["sel_freq"])
    return r["proba"], "ok"

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for task in TASKS:
        X0, y0, ids0, names, _ = ml.load_task(task, "region")
        main = np.load(PROCESSED / "day08" / f"{task}__region__logreg.npz")["proba"]
        s = ml.summarise(y0, main, main.argmax(axis=2))
        rows.append({"task": task, "variant": "main", "label": "Day 8 main result", "n_features": len(names),
                     "auc": s["auc"], "lo": s["auc_lo"], "hi": s["auc_hi"], "diff": 0.0, "d_lo": 0.0,
                     "d_hi": 0.0, "p": 1.0})
        runs = []
        for v, label in METHOD.items():
            if not (PROCESSED / f"features_region_{v}.csv").exists():
                print(f"  {task:9s} {v:15s} SKIPPED: run 39_robust_features.py first")
                continue
            X, y, ids, _, _ = ml.load_task(task, f"region_{v}")
            if not (np.array_equal(ids, ids0) and np.array_equal(y, y0)):
                raise SystemExit(f"{v}: people or order differ from Day 8 - cannot pair.")
            runs.append((v, label, X))
        for v, (label, keep) in SUBSETS.items():
            cols = np.array([keep(n) for n in names])
            runs.append((v, label, X0[:, cols]))
        for v, label, X in runs:
            proba, status = cached(f"{task}__{v}", X, y0)
            s = ml.summarise(y0, proba, proba.argmax(axis=2))
            d, lo, hi, p = ml.paired_difference(y0, main, proba)
            rows.append({"task": task, "variant": v, "label": label, "n_features": X.shape[1],
                         "auc": s["auc"], "lo": s["auc_lo"], "hi": s["auc_hi"],
                         "diff": d, "d_lo": lo, "d_hi": hi, "p": p})
            print(f"  {task:9s} {v:15s} ({X.shape[1]:2d} features) AUC {s['auc']:.3f} "
                  f"[{s['auc_lo']:.3f}, {s['auc_hi']:.3f}]  vs main {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"({status}, {(time.perf_counter() - t0) / 60:.1f} min)")
    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day10_robustness.csv", index=False)

    for task in TASKS:
        print(f"\n=== {task}: main pipeline with one choice changed ===")
        print(f"{'variant':15s}{'n':>4s}{'AUC [95% CI]':>23s}{'minus main [95% CI]':>26s}{'p':>7s}")
        for _, r in res[res.task == task].iterrows():
            print(f"{r.variant:15s}{r.n_features:>4d}{r.auc:>8.3f} [{r.lo:.3f}, {r.hi:.3f}]"
                  f"{r['diff']:>+10.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}]{r.p:>7.3f}")
    d = res[(res.task == "AD_vs_CN") & (res.variant != "main")]
    main_auc = res[(res.task == "AD_vs_CN") & (res.variant == "main")].auc.iloc[0]
    print(f"\nAD vs CN, all {len(d)} variants: AUC {d.auc.min():.3f} to {d.auc.max():.3f}; "
          f"lowest CI lower bound {d.lo.min():.3f}")
    print("\nPre-declared criteria (Day 10 guide, B.2), AD vs CN:")
    alt = d[d.variant.isin(ALTERNATIVES)]
    stress = d[d.variant.isin(STRESS)]
    ok_a = bool((alt.lo > 0.5).all() and ((alt.auc - main_auc).abs() <= 0.05).all())
    ok_s = bool((stress.lo > 0.5).all())
    print(f"  3. alternatives, CI > 0.5 and within 0.05 of main: {'MET' if ok_a else 'NOT MET'}"
          f"   ({', '.join(alt.variant)})")
    print(f"  4. stress tests, CI > 0.5: {'MET' if ok_s else 'NOT MET'}   ({', '.join(stress.variant)})")
    for _, r in d.iterrows():
        bad = r.lo <= 0.5 or (r.variant in ALTERNATIVES and abs(r.auc - main_auc) > 0.05)
        if bad:
            print(f"     not met by {r.variant}: AUC {r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}]")

if __name__ == "__main__":
    main()
