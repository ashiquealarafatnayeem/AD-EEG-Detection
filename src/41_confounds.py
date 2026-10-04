import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, mannwhitneyu, norm

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import PROCESSED

MATCHINGS = {"age": (2.0, False), "age_sex": (3.0, True)}

def people(task):
    df = pd.read_csv(PROCESSED / "features_region.csv")
    df = df[df.group.isin(ml.TASKS[task])].reset_index(drop=True)
    df["male"] = (df.sex.astype(str).str.upper() == "M").astype(float)
    return df

def match(df, caliper, same_sex):
    ad, cn = df[df.group == "AD"], df[df.group == "CN"]
    cand = sorted((abs(a.age - c.age), a.participant_id, c.participant_id)
                  for a in ad.itertuples() for c in cn.itertuples()
                  if abs(a.age - c.age) <= caliper and (not same_sex or a.male == c.male))
    used, pairs = set(), []
    for d, a, c in cand:
        if a not in used and c not in used:
            used |= {a, c}
            pairs.append((a, c, d))
    return pairs

def describe(df, label):
    ad, cn = df[df.group == "AD"], df[df.group == "CN"]
    p_age = mannwhitneyu(ad.age, cn.age).pvalue
    tab = [[int(ad.male.sum()), int(len(ad) - ad.male.sum())], [int(cn.male.sum()), int(len(cn) - cn.male.sum())]]
    p_sex = fisher_exact(tab)[1]
    print(f"  {label}:")
    for g, d in [("AD", ad), ("CN", cn)]:
        print(f"      {g} n = {len(d):2d}   age median {d.age.median():4.1f} (range {d.age.min():.0f}-"
              f"{d.age.max():.0f})   women {100 * (1 - d.male.mean()):3.0f} %")
    print(f"      AD vs CN: age p = {p_age:.2f} (Mann-Whitney), sex p = {p_sex:.2f} (Fisher)")

def logit_fit(Z, y):
    Z = np.column_stack([np.ones(len(y)), Z])
    b = np.zeros(Z.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-Z @ b))
        H = Z.T @ (Z * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, Z.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-Z @ b)), 1e-12, 1 - 1e-12)
    H = Z.T @ (Z * (p * (1 - p))[:, None])
    return b, np.sqrt(np.diag(np.linalg.inv(H))), float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

def row(task, variant, label, y, proba, main=None, n=None):
    s = ml.summarise(y, proba, proba.argmax(axis=2))
    r = {"task": task, "variant": variant, "label": label, "n_people": n or len(y),
         "auc": s["auc"], "lo": s["auc_lo"], "hi": s["auc_hi"]}
    if main is not None:
        r["diff"], r["d_lo"], r["d_hi"], r["p"] = ml.paired_difference(y, main, proba)
    return r

def main():
    rows = []
    df = people("AD_vs_CN")
    X, y, ids, names, _ = ml.load_task("AD_vs_CN", "region")
    assert list(ids) == list(df.participant_id)
    main_p = np.load(PROCESSED / "day08" / "AD_vs_CN__region__logreg.npz")["proba"]
    rows.append(row("AD_vs_CN", "main", "Day 8 main result", y, main_p))

    print("=== A. Age and sex by group ===")
    describe(df, "all AD and CN")

    print("\n=== B. Matched subsets ===")
    all_pairs = []
    for name, (caliper, same_sex) in MATCHINGS.items():
        pairs = match(df, caliper, same_sex)
        keep = {p for a, c, _ in pairs for p in (a, c)}
        idx = np.where(df.participant_id.isin(keep))[0]
        sub = df.iloc[idx]
        all_pairs += [{"matching": name, "AD": a, "CN": c, "age_diff": d} for a, c, d in pairs]
        what = "age" if not same_sex else "age and sex"
        print(f" Matched on {what} (within {caliper:.0f} years): {len(pairs)} pairs, "
              f"median age difference {np.median([d for *_, d in pairs]):.1f} years")
        describe(sub, "   matched set")
        r1 = row("AD_vs_CN", f"matched_{name}_day8", f"{what}-matched, Day 8 model", y[idx], main_p[:, idx])
        print(f"   (i)  Day 8 out-of-sample predictions, matched people only: AUC {r1['auc']:.3f} "
              f"[{r1['lo']:.3f}, {r1['hi']:.3f}]")
        cv = ml.run_cv(X[idx], y[idx], "logreg")
        r2 = row("AD_vs_CN", f"matched_{name}_retrained", f"{what}-matched, retrained", y[idx], cv["proba"])
        print(f"   (ii) main pipeline retrained on the matched people:   AUC {r2['auc']:.3f} "
              f"[{r2['lo']:.3f}, {r2['hi']:.3f}]")
        rows += [r1, r2]
    pd.DataFrame(all_pairs).to_csv(PROCESSED / "day10_matched_pairs.csv", index=False)

    print("\n=== C (i). Does the EEG score add to age and sex? (logistic regression) ===")
    score = np.clip(main_p[:, :, 1].mean(axis=0), 1e-6, 1 - 1e-6)
    eeg = np.log(score / (1 - score))
    z = lambda v: (v - v.mean()) / v.std()
    Z0 = np.column_stack([z(df.age.to_numpy(float)), df.male.to_numpy()])
    Z1 = np.column_stack([Z0, z(eeg)])
    b0, se0, ll0 = logit_fit(Z0, y.astype(float))
    b1, se1, ll1 = logit_fit(Z1, y.astype(float))
    lr = 2 * (ll1 - ll0)
    p_lr = chi2.sf(lr, 1)
    print(f"{'term':28s}{'odds ratio':>11s}{'95% CI':>18s}{'p':>9s}")
    for k, lab in [(1, "age (per SD)"), (2, "male sex"), (3, "EEG score (per SD)")]:
        orr, lo, hi = np.exp([b1[k], b1[k] - 1.96 * se1[k], b1[k] + 1.96 * se1[k]])
        p = 2 * norm.sf(abs(b1[k] / se1[k]))
        print(f"{lab:28s}{orr:>11.2f}   [{lo:5.2f}, {hi:6.2f}]{p:>9.4f}")
    print(f"Likelihood-ratio test, EEG score added to age + sex: chi2(1) = {lr:.1f}, p = {p_lr:.1e}")
    print("(The EEG score is out-of-sample from Day 8, so this test uses no information twice.)")

    print("\n=== C (ii). Age and sex regressed out of every feature, inside each training fold ===")
    for task in ["AD_vs_CN", "FTD_vs_CN"]:
        d = people(task)
        Xt, yt, idt, _, _ = ml.load_task(task, "region")
        assert list(idt) == list(d.participant_id)
        mp = np.load(PROCESSED / "day08" / f"{task}__region__logreg.npz")["proba"]
        Xc = np.column_stack([Xt, d.age.to_numpy(float), d.male.to_numpy()])
        cv = ml.run_cv(Xc, yt, "logreg", n_confounds=2)
        r = row(task, "deconfounded", "age + sex regressed out", yt, cv["proba"], main=mp)
        rows.append(r)
        print(f"  {task:9s} AUC {r['auc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]   vs main {r['diff']:+.3f} "
              f"[{r['d_lo']:+.3f}, {r['d_hi']:+.3f}], p = {r['p']:.3f}")

    res = pd.DataFrame(rows)
    res["lr_chi2"], res["lr_p"] = np.nan, np.nan
    res.loc[res.variant == "main", ["lr_chi2", "lr_p"]] = [lr, p_lr]
    res.to_csv(PROCESSED / "day10_confounds.csv", index=False)
    print(f"\nSaved {PROCESSED / 'day10_confounds.csv'}")

if __name__ == "__main__":
    main()
