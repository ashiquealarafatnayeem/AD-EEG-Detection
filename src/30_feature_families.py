import json
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

CACHE = PROCESSED / "day09"
OUT = FIGURES / "day09"
TASKS = ["AD_vs_CN", "FTD_vs_CN"]
FAMILIES = {
    "band powers": ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma"],
    "ratios": ["logTAR", "logSlowFast"],
    "spectral shape": ["entropy", "SEF50", "SEF95"],
    "Hjorth": ["hjorth_logActivity", "hjorth_mobility", "hjorth_complexity"],
    "alpha peak": ["alpha_present", "alpha_iaf", "alpha_height"],
}

def family_of(name):
    stem = name.split("__")[0]
    for fam, stems in FAMILIES.items():
        if stem in stems:
            return fam
    raise ValueError(f"feature {name} belongs to no family")

def cached_run(tag, X, y):
    path = CACHE / f"{tag}.npz"
    if path.exists():
        z = np.load(path)
        return {"proba": z["proba"], "pred": z["pred"]}, "cached"
    r = ml.run_cv(X, y, "logreg")
    np.savez(path, proba=r["proba"], pred=r["pred"], sel_freq=r["sel_freq"], coef=r["coef"],
             params=json.dumps(r["params"], default=str))
    return r, "ok"

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    X, y, ids, names, _ = ml.load_task("AD_vs_CN", "region")
    day8 = np.load(PROCESSED / "day08" / "AD_vs_CN__region__logreg.npz")
    r, status = cached_run("AD_vs_CN__all", X, y)
    a_now, a_day8 = ml.main_score(y, r["proba"]), ml.main_score(y, day8["proba"])
    same = np.allclose(r["proba"], day8["proba"], atol=1e-8)
    print("=== Checkpoint: re-run the Day 8 main model ===")
    print(f"  Day 8 AUC {a_day8:.4f}   now {a_now:.4f}   probabilities identical: {same}")
    if not same:
        raise SystemExit("Not reproducible - stop and paste this output to me.")
    print("  PASS: same splits, same result, so every comparison below is paired and fair.\n")

    fams = np.array([family_of(n) for n in names])
    print("Features per family:", {f: int((fams == f).sum()) for f in FAMILIES}, "\n")

    rows = []
    t0 = time.perf_counter()
    for task in TASKS:
        X, y, ids, names, _ = ml.load_task(task, "region")
        base, _ = cached_run(f"{task}__all", X, y)
        a_all = ml.main_score(y, base["proba"])
        rows.append({"task": task, "set": "all 94", "family": "all", "n_features": X.shape[1],
                     "auc": a_all, "diff": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0})
        for fam in FAMILIES:
            for mode in ["only", "without"]:
                cols = (fams == fam) if mode == "only" else (fams != fam)
                r, status = cached_run(f"{task}__{mode}__{fam.replace(' ', '_')}", X[:, cols], y)
                d, lo, hi, p = ml.paired_difference(y, base["proba"], r["proba"])
                rows.append({"task": task, "set": f"{mode} {fam}", "family": fam, "mode": mode,
                             "n_features": int(cols.sum()), "auc": ml.main_score(y, r["proba"]),
                             "diff": d, "lo": lo, "hi": hi, "p": p})
                print(f"  {task:9s} {mode:7s} {fam:14s} ({int(cols.sum()):2d} features) "
                      f"AUC {rows[-1]['auc']:.3f}   vs all: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                      f"({status}, {(time.perf_counter() - t0) / 60:.1f} min)")

    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day09_families.csv", index=False)
    for task in TASKS:
        d = res[res.task == task]
        a_all = d[d.family == "all"].auc.iloc[0]
        print(f"\n=== {task}: all 94 features AUC {a_all:.3f} ===")
        print(f"{'family':16s}{'n':>4s}{'ONLY this family':>20s}{'WITHOUT this family':>22s}"
              f"{'change without [95% CI]':>28s}")
        for fam in FAMILIES:
            o = d[(d.family == fam) & (d["mode"] == "only")].iloc[0]
            w = d[(d.family == fam) & (d["mode"] == "without")].iloc[0]
            print(f"{fam:16s}{int(o.n_features):>4d}{o.auc:>20.3f}{w.auc:>22.3f}"
                  f"{w['diff']:>+12.3f} [{w.lo:+.3f}, {w.hi:+.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    fam_list = list(FAMILIES)
    for ax, task in zip(axes, TASKS):
        d = res[res.task == task]
        a_all = d[d.family == "all"].auc.iloc[0]
        yy = np.arange(len(fam_list))
        only = [d[(d.family == f) & (d["mode"] == "only")].iloc[0] for f in fam_list]
        wo = [d[(d.family == f) & (d["mode"] == "without")].iloc[0] for f in fam_list]
        ax.barh(yy - .18, [o.auc for o in only], height=.34, color="#2471A3", label="only this family")
        ax.barh(yy + .18, [w.auc for w in wo], height=.34, color="#8a96a3", label="all except this family")
        ax.axvline(a_all, color="#C0392B", lw=1.5, ls="--", label=f"all 94 features ({a_all:.2f})")
        ax.axvline(0.5, color="k", lw=.8, ls=":")
        ax.set(xlim=(0.3, 1.0), title=task.replace("_", " "), xlabel="cross-validated AUC")
        ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=8, loc="lower right")
    axes[0].set_yticks(range(len(fam_list)), fam_list)
    axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT / "01_feature_families.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure: {OUT / '01_feature_families.png'}")

if __name__ == "__main__":
    main()
