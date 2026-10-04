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

OUT = FIGURES / "day10"
GROUPS = [("Random seed and AUC method", "#1B2A41"), ("How the features are measured", "#2471A3"),
          ("Which features are used", "#7D3C98"), ("Age and sex", "#1E8449")]

def collect(task):
    rows = []
    rb = pd.read_csv(PROCESSED / "day10_robustness.csv")
    rb = rb[rb.task == task].set_index("variant")
    main = rb.loc["main"]
    rows.append(("Day 8 main result (pooled AUC)", main.auc, main.lo, main.hi, 0))
    sd = pd.read_csv(PROCESSED / "day10_seeds.csv")
    sd = sd[(sd.task == task) & (sd.seed != ml.SEED)]
    rows.append((f"{len(sd)} other seeds (median, range)", sd.auc.median(), sd.auc.min(), sd.auc.max(), 0))
    fw = pd.read_csv(PROCESSED / "day10_foldwise.csv")
    fw = fw[(fw.task == task) & (fw.features == "region") & (fw.model == "logreg")].iloc[0]
    rows.append(("fold-wise AUC (no CI)", fw.foldwise, np.nan, np.nan, 0))
    for v, lab, g in [("E2", "2 s epochs", 1), ("R100", "reject at 100 uV", 1), ("R200", "reject at 200 uV", 1),
                      ("no_delta", "no delta features (80)", 2), ("logTAR_only", "theta/alpha ratios only (7)", 2),
                      ("posterior_only", "posterior features only (16)", 2)]:
        if v in rb.index:
            r = rb.loc[v]
            rows.append((lab, r.auc, r.lo, r.hi, g))
    cf = pd.read_csv(PROCESSED / "day10_confounds.csv")
    cf = cf[cf.task == task].set_index("variant")
    for v, lab in [("deconfounded", "age + sex regressed out"),
                   ("matched_age_day8", "age-matched, Day 8 model"),
                   ("matched_age_retrained", "age-matched, retrained"),
                   ("matched_age_sex_day8", "age+sex-matched, Day 8 model"),
                   ("matched_age_sex_retrained", "age+sex-matched, retrained")]:
        if v in cf.index:
            r = cf.loc[v]
            n = f" (n={int(r.n_people)})" if v.startswith("matched") else ""
            rows.append((lab + n, r.auc, r.lo, r.hi, 3))
    return pd.DataFrame(rows, columns=["check", "auc", "lo", "hi", "group"]), main.auc

def verdict():
    sd = pd.read_csv(PROCESSED / "day10_seeds.csv"); sd = sd[sd.task == "AD_vs_CN"]
    rb = pd.read_csv(PROCESSED / "day10_robustness.csv"); rb = rb[rb.task == "AD_vs_CN"].set_index("variant")
    fw = pd.read_csv(PROCESSED / "day10_foldwise.csv")
    fw = fw[(fw.task == "AD_vs_CN") & (fw.features == "region") & (fw.model == "logreg")].iloc[0]
    cf = pd.read_csv(PROCESSED / "day10_confounds.csv"); cf = cf[cf.task == "AD_vs_CN"].set_index("variant")
    main_auc = rb.loc["main", "auc"]
    s2026 = sd[sd.seed == ml.SEED].auc.iloc[0]
    med = sd[sd.seed != ml.SEED].auc.median()
    alt = rb.loc[[v for v in ["E2", "R100", "R200", "no_delta"] if v in rb.index]]
    stress = rb.loc[[v for v in ["logTAR_only", "posterior_only"] if v in rb.index]]
    conf = cf.loc[[v for v in cf.index if v != "main"]]
    checks = [
        ("1. seed 2026 within 0.03 of others",
         abs(s2026 - med) <= 0.03, f"{s2026:.3f} vs {med:.3f}"),
        ("2. fold-wise >= pooled - 0.05",
         fw.foldwise >= fw.pooled - 0.05, f"{fw.foldwise:.3f} vs {fw.pooled:.3f}"),
        ("3. alternatives: CI > 0.5, |diff| <= 0.05",
         bool((alt.lo > 0.5).all() and ((alt.auc - main_auc).abs() <= 0.05).all()),
         f"AUC {alt.auc.min():.3f}-{alt.auc.max():.3f}, CI low {alt.lo.min():.3f}"),
        ("4. stress tests: CI > 0.5",
         bool((stress.lo > 0.5).all()), f"CI low {stress.lo.min():.3f}"),
        ("5. age/sex: EEG adds, all CIs > 0.5",
         bool(cf.loc["main", "lr_p"] < 0.05 and (conf.lo > 0.5).all()),
         f"LR p = {cf.loc['main', 'lr_p']:.1e}, CI low {conf.lo.min():.3f}"),
    ]
    print("\n=== Pre-declared robustness criteria (Day 10 guide, B.2), AD vs CN ===")
    for name, ok, info in checks:
        print(f"  {'MET    ' if ok else 'NOT MET'} {name:41s} {info}")
    n_ok = sum(bool(c[1]) for c in checks)
    print(f"  {n_ok} of {len(checks)} criteria met.")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = ["AD_vs_CN", "FTD_vs_CN"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=False)
    allrows = []
    for ax, task in zip(axes, tasks):
        d, main_auc = collect(task)
        d.insert(0, "task", task)
        allrows.append(d)
        print(f"\n=== {task}: every Day 10 check ===")
        print(f"{'check':40s}{'AUC':>7s}{'95% CI (seeds: range)':>24s}{'minus main':>12s}")
        for _, r in d.iterrows():
            ci = "" if np.isnan(r.lo) else f"[{r.lo:.3f}, {r.hi:.3f}]"
            print(f"{r.check:40s}{r.auc:>7.3f}{ci:>24s}{r.auc - main_auc:>+12.3f}")
        yy = np.arange(len(d))[::-1]
        for k, (_, r) in enumerate(d.iterrows()):
            col = GROUPS[int(r.group)][1]
            if not np.isnan(r.lo):
                ax.plot([r.lo, r.hi], [yy[k]] * 2, color=col, lw=2)
            ax.plot(r.auc, yy[k], "D" if k == 0 else "o", color=col, ms=7 if k == 0 else 6)
        ax.axvline(main_auc, color="#C0392B", ls="--", lw=1.2)
        ax.axvline(0.5, color="k", ls=":", lw=.8)
        ax.set_yticks(yy, d.check, fontsize=8.5)
        ax.set(xlim=(0.35, 1.0), xlabel="cross-validated AUC (95 % CI)", title=task.replace("_", " "))
        ax.grid(alpha=.3, axis="x")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=g) for g, c in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Robustness of the main result: red dashed line = Day 8, dotted = chance", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / "03_robustness_forest.png", dpi=150)
    plt.close(fig)
    pd.concat(allrows).to_csv(PROCESSED / "day10_summary.csv", index=False)
    verdict()
    print(f"\nFigure: {OUT / '03_robustness_forest.png'}")

if __name__ == "__main__":
    main()
