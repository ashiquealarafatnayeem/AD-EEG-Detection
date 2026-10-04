import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_REST, PROCESSED, participants

N_BOOT = 2000

def auc(y, s, w=None):
    w = np.ones(len(y)) if w is None else w
    pos, neg = y == 1, y == 0
    d = s[pos][:, None] - s[neg][None, :]
    ww = w[pos][:, None] * w[neg][None, :]
    wins = (d > 0) + 0.5 * (d == 0)
    return float((wins * ww).sum() / ww.sum())

def boot_ci(y, s, rng):
    idx = [np.where(y == c)[0] for c in (0, 1)]
    vals = []
    for _ in range(N_BOOT):
        b = np.concatenate([rng.choice(i, len(i), replace=True) for i in idx])
        vals.append(auc(y[b], s[b]))
    return np.percentile(vals, [2.5, 97.5])

def main():
    pred = pd.read_csv(PROCESSED / "day08_predictions_main.csv")
    meta = participants(DS_REST)[["participant_id", "Gender", "Age"]]
    df = pred.merge(meta, on="participant_id")
    df["y"] = (df.group == "AD").astype(int)
    df["female"] = (df.Gender.str.upper() == "F")
    rng = np.random.default_rng(0)

    tab = pd.crosstab(df.group, df.Gender)
    _, p_sex = fisher_exact(tab.loc[["AD", "CN"], ["F", "M"]].to_numpy())
    print("=== Sex by group ===")
    print(tab.to_string())
    print(f"Fisher exact p = {p_sex:.3f}\n")

    rows = []
    print("=== 1. The main model's AUC inside each sex (out-of-sample probabilities) ===")
    print("  (from each person's AD probability averaged over the 10 repeats, so 'everyone'")
    print("   differs slightly from 27's AUC, which is the average of 10 per-repeat AUCs)")
    for name, m in [("everyone", np.ones(len(df), bool)), ("women", df.female.to_numpy()),
                    ("men", ~df.female.to_numpy())]:
        d = df[m]
        a = auc(d.y.to_numpy(), d.p_AD_mean.to_numpy())
        lo, hi = boot_ci(d.y.to_numpy(), d.p_AD_mean.to_numpy(), rng)
        n_ad, n_cn = int(d.y.sum()), int((1 - d.y).sum())
        rows.append({"subset": name, "n_AD": n_ad, "n_CN": n_cn, "auc": a, "lo": lo, "hi": hi})
        print(f"  {name:9s} AD {n_ad:2d}, CN {n_cn:2d}   AUC {a:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n=== 2. Does the AD probability depend on sex within each group? ===")
    for g in ["AD", "CN"]:
        d = df[df.group == g]
        f, mm = d.loc[d.female, "p_AD_mean"], d.loc[~d.female, "p_AD_mean"]
        _, p = mannwhitneyu(f, mm, alternative="two-sided")
        print(f"  {g}: women median {f.median():.2f} (n={len(f)}), men median {mm.median():.2f} "
              f"(n={len(mm)}), Mann-Whitney p = {p:.2f}")

    print("\n=== 3. Sex-balanced AUC (each group weighted to 50 % women, 50 % men) ===")
    w = np.ones(len(df))
    for g in ["AD", "CN"]:
        for fem in [True, False]:
            m = ((df.group == g) & (df.female == fem)).to_numpy()
            w[m] = 0.5 / m.sum()
    a_bal = auc(df.y.to_numpy(), df.p_AD_mean.to_numpy(), w)
    print(f"  AUC = {a_bal:.3f}   (unweighted: {rows[0]['auc']:.3f})")
    rows.append({"subset": "sex-balanced weights", "auc": a_bal})

    pd.DataFrame(rows).to_csv(PROCESSED / "day08_sex_check.csv", index=False)
    print("\nIf the AUC stays well above 0.5 inside women AND inside men, and the AD")
    print("probability does not differ by sex within a group, the model is not reading sex.")

if __name__ == "__main__":
    main()
