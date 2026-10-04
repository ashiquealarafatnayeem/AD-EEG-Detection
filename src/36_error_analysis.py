import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import DS_REST, FIGURES, PROCESSED, participants

OUT = FIGURES / "day09"
COLORS = {"AD": "#C0392B", "FTD": "#E67E22", "CN": "#2471A3"}

def compare(df, col, a, b, label):
    x, y = df.loc[a, col].dropna(), df.loc[b, col].dropna()
    if len(x) < 2 or len(y) < 2:
        return f"  {label:34s} too few people"
    p = mannwhitneyu(x, y, alternative="two-sided").pvalue
    return (f"  {label:34s} wrong: median {x.median():6.1f} (n={len(x):2d})   "
            f"right: median {y.median():6.1f} (n={len(y):2d})   p = {p:.2f}")

def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PROCESSED / "day08_predictions_main.csv")
    meta = participants(DS_REST)[["participant_id", "Age", "Gender", "MMSE"]]
    df = pred.merge(meta, on="participant_id", how="left")
    log_path = PROCESSED / "day05_batch_log.csv"
    if log_path.exists():
        lg = pd.read_csv(log_path).rename(columns={"subject": "participant_id"})
        keep = [c for c in ["participant_id", "dur_s", "n_ica_removed", "drop_pct"] if c in lg]
        df = df.merge(lg[keep], on="participant_id", how="left")
    df["wrong"] = np.where(df.group == "AD", df.called_AD_fraction < 0.5, df.called_AD_fraction > 0.5)

    pd.set_option("display.width", 200)
    for g, what in [("AD", "AD patients called healthy (missed)"),
                    ("CN", "healthy people called AD (false alarms)")]:
        d = df[(df.group == g) & df.wrong].sort_values("p_AD_mean")
        print(f"=== {what}: {len(d)} of {int((df.group == g).sum())} ===")
        cols = [c for c in ["participant_id", "p_AD_mean", "called_AD_fraction", "MMSE", "Age",
                            "Gender", "drop_pct", "n_ica_removed", "dur_s"] if c in d]
        print(d[cols].round(2).to_string(index=False) if len(d) else "  none")
        print()

    print("=== Are the misclassified people different? (Mann-Whitney) ===")
    for g in ["AD", "CN"]:
        d = df[df.group == g]
        print(f" {g}:")
        for col, label in [("MMSE", "MMSE (dementia severity)"), ("Age", "age"),
                           ("drop_pct", "% epochs dropped (data quality)"),
                           ("n_ica_removed", "ICA components removed"), ("dur_s", "recording length (s)")]:
            if col in d:
                print(compare(d, col, d.wrong, ~d.wrong, label))
    print("  (about 10 tests: one p below 0.05 is expected by chance alone; treat single hits as hints)")
    ad = df[df.group == "AD"].dropna(subset=["MMSE"])
    rho, p = spearmanr(ad.MMSE, ad.p_AD_mean)
    print(f"\n  Within AD: Spearman rho(MMSE, AD probability) = {rho:+.2f}, p = {p:.3f}  (n = {len(ad)})")
    print("  (negative rho = milder patients look less like AD; see the caveat in the header)")

    full = pd.read_csv(PROCESSED / "features_region.csv")
    feat_cols = [c for c in full.columns if c not in ml.META]
    train = full[full.group.isin(["AD", "CN"])]
    ftd = full[full.group == "FTD"]
    pipe, grid = ml.model_spec("logreg", len(feat_cols))
    gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                      cv=StratifiedKFold(5, shuffle=True, random_state=ml.SEED), n_jobs=-1)
    gs.fit(train[feat_cols].to_numpy(float), (train.group == "AD").astype(int).to_numpy())
    p_ftd = gs.predict_proba(ftd[feat_cols].to_numpy(float))[:, 1]
    p_ad = df.loc[df.group == "AD", "p_AD_mean"].to_numpy()
    p_cn = df.loc[df.group == "CN", "p_AD_mean"].to_numpy()
    print("\n=== FTD patients scored by the AD-vs-CN model (trained on all 65 AD + CN) ===")
    print(f"  AD probability, median: CN {np.median(p_cn):.2f}   FTD {np.median(p_ftd):.2f}   "
          f"AD {np.median(p_ad):.2f}")
    print(f"  FTD called 'AD' (probability >= 0.5): {int((p_ftd >= 0.5).sum())} of {len(p_ftd)}")
    print(f"  FTD vs CN: Mann-Whitney p = {mannwhitneyu(p_ftd, p_cn).pvalue:.3g}   "
          f"FTD vs AD: p = {mannwhitneyu(p_ftd, p_ad).pvalue:.3g}")
    print("  (AD and CN values are out-of-sample from Day 8; FTD values come from a model that")
    print("   never saw any FTD patient, so all three are fair to compare)")

    out = df.copy()
    out = pd.concat([out, pd.DataFrame({"participant_id": ftd.participant_id, "group": "FTD",
                                        "p_AD_mean": p_ftd})], ignore_index=True)
    out.to_csv(PROCESSED / "day09_errors.csv", index=False)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k, (g, vals) in enumerate([("CN", p_cn), ("FTD", p_ftd), ("AD", p_ad)]):
        ax.scatter(k + rng.uniform(-.18, .18, len(vals)), vals, color=COLORS[g], s=22, alpha=.85)
        ax.plot([k - .3, k + .3], [np.median(vals)] * 2, color="k", lw=1.6)
    ax.axhline(0.5, color="k", ls="--", lw=.8)
    ax.set_xticks([0, 1, 2], ["CN\n(out-of-sample)", "FTD\n(never seen)", "AD\n(out-of-sample)"])
    ax.set(ylabel="probability of AD from the main model", ylim=(0, 1),
           title="Where each group falls on the AD-vs-CN scale (black bar = median)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "06_errors_and_ftd.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(ad.MMSE + rng.uniform(-.2, .2, len(ad)), ad.p_AD_mean, color=COLORS["AD"], s=24)
    ax.axhline(0.5, color="k", ls="--", lw=.8)
    ax.set(xlabel="MMSE (higher = milder)", ylabel="probability of AD",
           title=f"AD patients: rho = {rho:+.2f}, p = {p:.2f}")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "07_mmse.png", dpi=150)
    plt.close(fig)
    print(f"\nFigures: {OUT / '06_errors_and_ftd.png'}, {OUT / '07_mmse.png'}")

if __name__ == "__main__":
    main()
