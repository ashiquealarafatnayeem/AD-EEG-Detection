import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.append(str(Path(__file__).resolve().parent))
import features as FT
from eegio import FIGURES, PROCESSED

OUT = FIGURES / "day06"
META = ["participant_id", "group", "age", "sex", "n_epochs"]
COLORS = {"AD": "#C0392B", "FTD": "#E67E22", "CN": "#2471A3"}
TOPO_FEATURES = ["logTAR", "rel_alpha", "rel_theta", "SEF50"]

def auc_signed(ad, cn):
    u, p = mannwhitneyu(ad, cn, alternative="two-sided")
    return u / (len(ad) * len(cn)), p

def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out

def rank_features(df, cols):
    ad = df[df.group == "AD"]
    cn = df[df.group == "CN"]
    rows = []
    for c in cols:
        a, b = ad[c].dropna().to_numpy(), cn[c].dropna().to_numpy()
        s, p = auc_signed(a, b)
        rows.append(dict(feature=c, AD_median=np.median(a), CN_median=np.median(b),
                         AUC=max(s, 1 - s), direction="AD higher" if s >= 0.5 else "AD lower",
                         p=p))
    res = pd.DataFrame(rows)
    res["q"] = bh_fdr(res["p"])
    return res.sort_values("AUC", ascending=False).reset_index(drop=True)

def main():
    mne.set_log_level("ERROR")
    OUT.mkdir(parents=True, exist_ok=True)
    rg = pd.read_csv(PROCESSED / "features_region.csv")
    ch = pd.read_csv(PROCESSED / "features_channel.csv")
    z = np.load(PROCESSED / "psd_clean.npz")
    freqs, psds = z["freqs"], z["psd"]
    ch_names = [str(c) for c in z["ch_names"]]

    cols = [c for c in rg.columns if c not in META + ["alpha_iaf"]]
    res = rank_features(rg, cols)
    res.to_csv(PROCESSED / "feature_ranking.csv", index=False)
    n_sig = int((res.q < 0.05).sum())
    print(f"=== AD vs CN: {len(res)} region features tested, "
          f"{n_sig} significant after FDR correction (q < 0.05) ===\n")
    show = res.head(20).copy()
    for c in ["AD_median", "CN_median"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    show["AUC"] = show["AUC"].map(lambda v: f"{v:.3f}")
    show["q"] = show["q"].map(lambda v: f"{v:.1e}")
    pd.set_option("display.width", 200)
    print(show[["feature", "AD_median", "CN_median", "direction", "AUC", "q"]]
          .to_string(index=False))

    grid = res[res.feature.str.contains("__")].copy()
    grid[["name", "region"]] = grid.feature.str.split("__", expand=True)
    table = grid.pivot(index="name", columns="region", values="AUC")
    table = table[list(FT.REGIONS)]
    table = table.loc[table.max(axis=1).sort_values(ascending=False).index]
    print("\n=== AUC of every feature in every region (AD vs CN) ===")
    print(table.round(3).to_string())

    print("\n=== Posterior alpha peak (after removing the 1/f background) ===")
    counts = rg.groupby("group")["alpha_present"].agg(["sum", "count"])
    for g in ["AD", "CN", "FTD"]:
        s, n = int(counts.loc[g, "sum"]), int(counts.loc[g, "count"])
        iaf = rg[(rg.group == g) & (rg.alpha_present == 1)]["alpha_iaf"]
        print(f"  {g:3s}: peak found in {s:2d} of {n:2d} ({100 * s / n:5.1f} %)   "
              f"median IAF {iaf.median():.2f} Hz (range {iaf.min():.2f}-{iaf.max():.2f})")
    ad_c = counts.loc["AD"]; cn_c = counts.loc["CN"]
    _, p_f = fisher_exact([[ad_c["count"] - ad_c["sum"], ad_c["sum"]],
                           [cn_c["count"] - cn_c["sum"], cn_c["sum"]]])
    print(f"  No peak, AD vs CN: Fisher's exact test p = {p_f:.4f}")
    iaf_ad = rg[(rg.group == "AD") & (rg.alpha_present == 1)]["alpha_iaf"]
    iaf_cn = rg[(rg.group == "CN") & (rg.alpha_present == 1)]["alpha_iaf"]
    if len(iaf_ad) >= 3 and len(iaf_cn) >= 3:
        _, p_i = mannwhitneyu(iaf_ad, iaf_cn, alternative="two-sided")
        print(f"  IAF where a peak exists, AD vs CN: Mann-Whitney p = {p_i:.2e}")

    print("\n  Sensitivity to the peak threshold (peaks found):")
    post = [ch_names.index(c) for c in FT.REGIONS["posterior"]]
    groups = np.array([str(g) for g in z["groups"]])
    keep = FT.PEAK_THRESHOLD
    for thr in [0.05, 0.10, 0.20]:
        FT.PEAK_THRESHOLD = thr
        found = np.array([FT.alpha_peak(freqs, p[post].mean(axis=0))["present"] for p in psds])
        txt = "   ".join(f"{g} {found[groups == g].sum()}/{(groups == g).sum()}"
                         for g in ["AD", "CN", "FTD"])
        print(f"    threshold {thr:.2f}:  {txt}")
    FT.PEAK_THRESHOLD = keep

    top = res.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    col = ["#C0392B" if d == "AD higher" else "#2471A3" for d in top.direction]
    bars = ax.barh(top.feature, top.AUC, color=col)
    for b, q in zip(bars, top.q):
        if q >= 0.05:
            b.set_hatch("//"); b.set_alpha(.5)
    ax.axvline(0.5, color="k", lw=.8, ls=":")
    ax.set(xlim=(0.5, 1.0), xlabel="AUC, AD vs CN (0.5 = chance)",
           title="Top 20 features   (red: higher in AD · blue: lower in AD)")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "02_feature_ranking.png", dpi=150)
    plt.close(fig)

    info = mne.read_epochs(PROCESSED / "epochs" / f"{ch.participant_id.iloc[0]}_epo.fif",
                           preload=False, verbose="ERROR").info
    fig, axes = plt.subplots(1, len(TOPO_FEATURES), figsize=(3.2 * len(TOPO_FEATURES), 3.4))
    ad = ch[ch.group == "AD"]; cn = ch[ch.group == "CN"]
    for a, name in zip(axes, TOPO_FEATURES):
        vals = np.array([auc_signed(ad[f"{name}__{c}"], cn[f"{name}__{c}"])[0]
                         for c in info.ch_names])
        im, _ = mne.viz.plot_topomap(vals, info, axes=a, show=False, cmap="RdBu_r",
                                     vlim=(0.0, 1.0), contours=0)
        a.set_title(name, fontsize=11)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="P(AD value > CN value)")
    fig.suptitle("Where on the scalp each feature separates the groups (0.5 = no difference)",
                 fontsize=10)
    fig.savefig(OUT / "03_auc_topomaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    six = [f for f in res.feature if "__" in f][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    rng = np.random.default_rng(0)
    for a, name in zip(axes.ravel(), six):
        data = [rg.loc[rg.group == g, name].to_numpy() for g in ["CN", "FTD", "AD"]]
        a.boxplot(data, widths=.55, showfliers=False)
        for k, (g, d) in enumerate(zip(["CN", "FTD", "AD"], data), 1):
            a.scatter(k + rng.uniform(-.15, .15, len(d)), d, s=12, color=COLORS[g], alpha=.75)
        a.set_xticks([1, 2, 3], ["CN", "FTD", "AD"])
        auc = res.set_index("feature").loc[name, "AUC"]
        a.set_title(f"{name}\nAUC {auc:.3f}", fontsize=9.5)
        a.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "04_top_features_by_group.png", dpi=150)
    plt.close(fig)

    subs = [str(s) for s in z["subjects"]]
    ex_cn = rg[rg.group == "CN"].sort_values("alpha_height").participant_id.iloc[-1]
    ex_ad = rg[rg.group == "AD"].sort_values("alpha_height").participant_id.iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, sid in zip(axes[:2], [ex_cn, ex_ad]):
        spec = psds[subs.index(sid)][post].mean(axis=0)
        pk = FT.alpha_peak(freqs, spec)
        f = freqs[1:]; m = (f >= 1) & (f <= 40)
        a.semilogy(f[m], spec[1:][m], color="k", lw=1.3, label="posterior spectrum")
        a.semilogy(f[m], 10 ** (pk["intercept"] + pk["slope"] * np.log10(f[m])),
                   "--", color="#6b7a8c", label="1/f background fit")
        a.axvspan(*FT.PEAK_WINDOW, color="#2471A3", alpha=.08, label="search window")
        if pk["present"]:
            a.axvline(pk["iaf"], color="#C0392B", lw=1, label=f"peak {pk['iaf']:.2f} Hz")
        g = rg.set_index("participant_id").loc[sid, "group"]
        a.set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
              title=f"{sid} ({g}): {'peak found' if pk['present'] else 'no peak'}, "
                    f"height {pk['height']:.2f}")
        a.grid(alpha=.3); a.legend(fontsize=7.5)
    for k, g in enumerate(["CN", "FTD", "AD"]):
        d = rg[(rg.group == g) & (rg.alpha_present == 1)]["alpha_iaf"]
        axes[2].scatter(np.full(len(d), k) + rng.uniform(-.12, .12, len(d)), d,
                        color=COLORS[g], s=18)
        n_no = int(((rg.group == g) & (rg.alpha_present == 0)).sum())
        axes[2].text(k, FT.PEAK_WINDOW[0] - 0.4, f"no peak: {n_no}", ha="center", fontsize=8.5)
    axes[2].set_xticks([0, 1, 2], ["CN", "FTD", "AD"])
    axes[2].set(ylabel="alpha peak frequency (Hz)", ylim=(FT.PEAK_WINDOW[0] - 0.8, FT.PEAK_WINDOW[1]),
                title="Peak frequency, where a peak exists")
    axes[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "05_alpha_peak.png", dpi=150)
    plt.close(fig)

    print(f"\nRanking saved to {PROCESSED / 'feature_ranking.csv'}")
    print(f"Figures saved to {OUT}")

if __name__ == "__main__":
    main()
