import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr, wilcoxon

sys.path.append(str(Path(__file__).resolve().parent))
import features as FT
import photic as PH
import spectral as S
from eegio import FIGURES, PROCESSED

OUT = FIGURES / "day07"
ALPHA = (8.0, 13.0)
COLORS = {"AD": "#C0392B", "FTD": "#E67E22", "CN": "#2471A3"}
GROUPS = ["CN", "FTD", "AD"]
REGIONS = ["posterior", "global"]
NEW_FEATURES = [f"{m}__{r}" for m in ["react_log", "react_rel"] for r in REGIONS]

def region_idx(ch_names, region):
    return [ch_names.index(c) for c in FT.REGIONS[region]]

def alpha_measures(freqs, psd, idx):
    p = psd[idx]
    absolute = S.band_power(freqs, p, *ALPHA)
    relative = absolute / S.band_power(freqs, p, FT.FMIN, FT.FMAX)
    return np.log10(absolute.mean()), relative.mean()

def auc_test(a, b):
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    s = u / (len(a) * len(b))
    return max(s, 1 - s), ("AD higher" if s >= 0.5 else "AD lower"), p

def load_pairs():
    path = PROCESSED / "day07_id_mapping.csv"
    if not path.exists():
        raise SystemExit("day07_id_mapping.csv not found: run src\\24_id_mapping.py first")
    m = pd.read_csv(path)
    keep = m[m.mutual & (m.group_006036 == m.group_004504)]
    return dict(zip(keep.ds006036, keep.match_004504))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rest = np.load(PROCESSED / "psd_clean.npz")
    eoec = np.load(PROCESSED / "psd_eoec.npz")
    ch = [str(c) for c in rest["ch_names"]]
    if [str(c) for c in eoec["ch_names"]] != ch:
        raise SystemExit("channel order differs between the two datasets")
    f_rest, f_eoec = rest["freqs"], eoec["freqs"]
    rest_subs = [str(s) for s in rest["subjects"]]
    eoec_subs = [str(s) for s in eoec["subjects"]]
    pairs = load_pairs()
    print(f"EEG-matched pairs used to join the datasets: {len(pairs)}")

    rows = []
    for j, sid in enumerate(eoec_subs):
        var_eo, n_ec = float(eoec["var_eo"][j]), int(eoec["n_ec_near"][j])
        partner = pairs.get(sid)
        row = {"ds006036_id": sid, "group": str(eoec["groups"][j]),
               "participant_id": partner,
               "n_eo": int(eoec["n_eo"][j]), "var_eo": var_eo, "n_ec_near": n_ec,
               "has_reactivity": (var_eo <= PH.MAX_EO_VAR_RATIO) and (n_ec >= PH.MIN_EC_WINDOWS)}
        for r in REGIONS:
            idx = region_idx(ch, r)
            la_eo, rel_eo = alpha_measures(f_eoec, eoec["psd_eo"][j], idx)
            la_ec, rel_ec = alpha_measures(f_eoec, eoec["psd_ec_near"][j], idx)
            la_all, _ = alpha_measures(f_eoec, eoec["psd_ec_all"][j], idx)
            la_rest = (alpha_measures(f_rest, rest["psd"][rest_subs.index(partner)], idx)[0]
                       if partner in rest_subs else np.nan)
            row.update({f"alphaEO_log__{r}": la_eo, f"alphaEC_log__{r}": la_ec,
                        f"alphaECall_log__{r}": la_all, f"alphaREST_log__{r}": la_rest,
                        f"react_log__{r}": la_ec - la_eo, f"react_rel__{r}": rel_ec - rel_eo,
                        f"xreact_log__{r}": la_rest - la_eo})
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.startswith(("alphaEO", "alphaEC_", "react", "xreact"))]:
        df.loc[~df.has_reactivity, c] = np.nan
    df.to_csv(PROCESSED / "alpha_reactivity.csv", index=False)
    ok = df[df.has_reactivity]
    print(f"Reactivity available for {len(ok)} of {len(df)} ds006036 recordings")

    print("\n=== Check A: does posterior alpha drop when the eyes open? (paired Wilcoxon) ===")
    for g in GROUPS:
        d = ok.loc[ok.group == g, "react_log__posterior"]
        if len(d) < 5:
            print(f"  {g:3s} n={len(d):2d}  too few people")
            continue
        _, p = wilcoxon(d)
        print(f"  {g:3s} n={len(d):2d}  median react_log = {d.median():+.2f} "
              f"(alpha x{10 ** d.median():.1f} with eyes closed)   "
              f"drops in {(d > 0).mean() * 100:5.1f} % of people   p = {p:.1e}")

    b = df.dropna(subset=["alphaECall_log__posterior", "alphaREST_log__posterior"])
    rho, p_rho = spearmanr(b["alphaECall_log__posterior"], b["alphaREST_log__posterior"])
    print("\n=== Check B: eyes-closed alpha, ds006036 vs ds004504, EEG-matched pairs ===")
    print(f"  n = {len(b)}   Spearman rho = {rho:.2f}   p = {p_rho:.1e}")
    print("  (the pairs were found from the spectra, so a high rho here is expected;")
    print("   what proves the pairing is the fixed ID shifts in 24, not this number)")

    print("\n=== Check C: who has no reactivity value (ds006036) ===")
    miss = df.groupby("group").has_reactivity.agg(lambda s: int((~s).sum()))
    tot = df.groupby("group").size()
    for g in GROUPS:
        print(f"  {g:3s} missing {miss[g]:2d} of {tot[g]:2d}")
    _, p_f = fisher_exact([[miss["AD"], tot["AD"] - miss["AD"]],
                           [miss["CN"], tot["CN"] - miss["CN"]]])
    print(f"  AD vs CN, Fisher exact p = {p_f:.3f}")

    cols = NEW_FEATURES + ["alphaEO_log__posterior", "alphaEC_log__posterior"]
    print("\n=== AD vs CN, inside ds006036 (everyone with reactivity) ===")
    print(f"{'feature':24s}{'AD median':>11s}{'CN median':>11s}{'direction':>12s}{'AUC':>7s}{'p':>10s}")
    for c in cols:
        a = ok.loc[ok.group == "AD", c].to_numpy()
        bb = ok.loc[ok.group == "CN", c].to_numpy()
        auc, dirn, p = auc_test(a, bb)
        print(f"{c:24s}{np.median(a):>11.3f}{np.median(bb):>11.3f}{dirn:>12s}{auc:>7.3f}{p:>10.1e}")
    okm = ok.dropna(subset=["participant_id"])
    a = okm.loc[okm.group == "AD", "react_log__posterior"]
    bb = okm.loc[okm.group == "CN", "react_log__posterior"]
    auc, dirn, p = auc_test(a, bb)
    print(f"  Only recordings with a confirmed ds004504 partner (AD {len(a)}, CN {len(bb)}): "
          f"react_log__posterior AUC {auc:.3f} ({dirn}), p = {p:.1e}")

    d_ = ok.dropna(subset=["xreact_log__posterior"])
    rho_d, p_d = spearmanr(d_["react_log__posterior"], d_["xreact_log__posterior"])
    print("\n=== Check D: within-ds006036 vs cross-dataset reactivity, matched pairs ===")
    print(f"  n = {len(d_)}   Spearman rho = {rho_d:.2f}, p = {p_d:.1e}")

    add = ok.dropna(subset=["participant_id"])[["participant_id", "ds006036_id"] + NEW_FEATURES]
    for name in ["features_region", "features_channel"]:
        f = pd.read_csv(PROCESSED / f"{name}.csv").merge(add, on="participant_id", how="left")
        f.to_csv(PROCESSED / f"{name}_all.csv", index=False)
        have = f["react_log__posterior"].notna()
        print(f"\n{name}_all.csv: {len(f)} subjects x {f.shape[1] - 6} features; "
              f"reactivity for {int(have.sum())} ("
              + ", ".join(f"{g} {int((have & (f.group == g)).sum())}" for g in GROUPS) + ")")

    post = region_idx(ch, "posterior")
    band = (f_eoec >= 1) & (f_eoec <= 30)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, g in zip(axes, GROUPS):
        jj = [eoec_subs.index(s) for s in ok[ok.group == g].ds006036_id]
        for key, ls, lab in [("psd_ec_near", "-", "eyes closed"), ("psd_eo", "--", "eyes open")]:
            m = 10 ** np.log10(eoec[key][jj][:, post].mean(axis=1)).mean(axis=0)
            ax.semilogy(f_eoec[band], m[band], color=COLORS[g], lw=1.7, ls=ls, label=lab)
        ax.axvspan(*ALPHA, color="#8a96a3", alpha=.12)
        ax.set(title=f"{g} (n = {len(jj)})", xlabel="Frequency (Hz)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("posterior PSD (V²/Hz), geometric mean")
    fig.suptitle("Same recording (ds006036): opening the eyes suppresses alpha "
                 "(grey band = 8-13 Hz)", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "02_ec_vs_eo_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, c, lab in [(axes[0], "react_log__posterior", "log10(alpha closed / alpha open)"),
                       (axes[1], "react_rel__posterior", "relative alpha: closed - open")]:
        data = [ok.loc[ok.group == g, c].to_numpy() for g in GROUPS]
        ax.boxplot(data, widths=.55, showfliers=False)
        for k, (g, d) in enumerate(zip(GROUPS, data), 1):
            ax.scatter(k + rng.uniform(-.15, .15, len(d)), d, s=14, color=COLORS[g], alpha=.8)
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.set_xticks([1, 2, 3], GROUPS)
        ax.set(ylabel=lab, title=c)
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "03_reactivity_by_group.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for g in GROUPS:
        d = b[b.group == g]
        axes[0].scatter(d["alphaREST_log__posterior"], d["alphaECall_log__posterior"],
                        color=COLORS[g], s=20, label=g)
        d = d_[d_.group == g]
        axes[1].scatter(d["xreact_log__posterior"], d["react_log__posterior"],
                        color=COLORS[g], s=20, label=g)
    axes[0].set(xlabel="log10 alpha, eyes closed, ds004504",
                ylabel="log10 alpha, eyes closed, ds006036",
                title=f"Check B: EEG-matched pairs (rho = {rho:.2f})")
    axes[1].axhline(0, color="k", lw=.7, ls=":"); axes[1].axvline(0, color="k", lw=.7, ls=":")
    axes[1].set(xlabel="cross-dataset react_log", ylabel="within-recording react_log (main)",
                title=f"Check D: two ways to measure reactivity (rho = {rho_d:.2f})")
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "04_checks.png", dpi=150)
    plt.close(fig)
    print(f"\nFigures saved to {OUT}")

if __name__ == "__main__":
    main()
