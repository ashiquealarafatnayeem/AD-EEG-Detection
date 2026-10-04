import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_REST, FIGURES, PROCESSED, participants

TRAPZ = getattr(np, "trapezoid", None) or np.trapz

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 25), "gamma": (25, 45)}
OUT = FIGURES / "day02"

def band_power(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return TRAPZ(psd[..., m], freqs[m], axis=-1)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    d = np.load(PROCESSED / "psd_rest.npz", allow_pickle=True)
    psds, freqs = d["psds"], d["freqs"]
    subjects, ch_names = list(d["subjects"]), list(d["ch_names"])

    meta = participants(DS_REST).set_index("participant_id").loc[subjects]
    groups = meta["group"].to_numpy()
    print("psds:", psds.shape, " groups:", pd.Series(groups).value_counts().to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"AD": "#C0392B", "FTD": "#E67E22", "CN": "#2471A3"}
    for g in ["CN", "FTD", "AD"]:
        sel = psds[groups == g].mean(axis=1)
        m, se = sel.mean(0), sel.std(0) / np.sqrt(len(sel))
        for ax, scale in zip(axes, ["log", "linear"]):
            ax.plot(freqs, m, color=colors[g], lw=2, label=f"{g} (n={len(sel)})")
            ax.fill_between(freqs, m - se, m + se, color=colors[g], alpha=.2)
            ax.set_yscale(scale)
    for ax, ttl in zip(axes, ["log power", "linear power (0–20 Hz)"]):
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.set_title(f"Grand-average spectrum — {ttl}")
        ax.legend()
        ax.grid(alpha=.3)
    axes[1].set_xlim(0, 20)
    fig.tight_layout()
    fig.savefig(OUT / "01_grand_average_psd.png", dpi=150)
    plt.close(fig)

    total = band_power(psds, freqs, 0.5, 45)
    rows = []
    for i, sid in enumerate(subjects):
        rec = {"participant_id": sid, "group": groups[i]}
        rel = {}
        for b, (lo, hi) in BANDS.items():
            rel[b] = band_power(psds[i], freqs, lo, hi) / total[i]
            rec[f"rel_{b}"] = rel[b].mean()
        rec["TAR"] = (rel["theta"] / rel["alpha"]).mean()
        rec["slow_fast"] = ((rel["delta"] + rel["theta"]) /
                            (rel["alpha"] + rel["beta"])).mean()
        m = (freqs >= 6) & (freqs <= 13)
        rec["IAF"] = freqs[m][np.argmax(psds[i].mean(0)[m])]
        rows.append(rec)

    feat = pd.DataFrame(rows)
    feat.to_csv(PROCESSED / "day02_quicklook_features.csv", index=False)

    print("\n=== AD vs CN, channel-averaged features ===")
    print(f"{'feature':<12}{'AD mean':>10}{'CN mean':>10}{'p (MWU)':>12}{'AUC':>8}")
    a_mask, c_mask = feat.group == "AD", feat.group == "CN"
    stats_rows = []
    for col in ["rel_delta", "rel_theta", "rel_alpha", "rel_beta",
                "rel_gamma", "TAR", "slow_fast", "IAF"]:
        a, c = feat.loc[a_mask, col], feat.loc[c_mask, col]
        u, p = stats.mannwhitneyu(a, c, alternative="two-sided")
        auc = u / (len(a) * len(c))
        auc = max(auc, 1 - auc)
        print(f"{col:<12}{a.mean():>10.3f}{c.mean():>10.3f}{p:>12.2e}{auc:>8.3f}")
        stats_rows.append({"feature": col, "AD_mean": a.mean(),
                           "CN_mean": c.mean(), "p": p, "AUC": auc})
    pd.DataFrame(stats_rows).to_csv(PROCESSED / "day02_stats.csv", index=False)

    show = ["rel_delta", "rel_theta", "rel_alpha", "TAR", "slow_fast", "IAF"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    order = ["CN", "FTD", "AD"]
    for ax, col in zip(axes.ravel(), show):
        data = [feat.loc[feat.group == g, col] for g in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=.6)
        for patch, g in zip(bp["boxes"], order):
            patch.set_facecolor(colors[g]); patch.set_alpha(.55)
        for j, vals in enumerate(data, start=1):
            ax.scatter(np.random.normal(j, .06, len(vals)), vals,
                       s=12, color="k", alpha=.5, zorder=3)
        ax.set_title(col); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "02_feature_boxplots.png", dpi=150)
    plt.close(fig)

    import mne
    info = mne.create_info(ch_names, sfreq=500., ch_types="eeg")
    info.set_montage("standard_1020", match_case=False)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    for r, band in enumerate(["theta", "alpha"]):
        lo, hi = BANDS[band]
        rel = band_power(psds, freqs, lo, hi) / total
        maps = {g: rel[groups == g].mean(0) for g in ["AD", "CN"]}
        vmax = max(v.max() for v in maps.values())
        vmin = min(v.min() for v in maps.values())
        for c, g in enumerate(["CN", "AD"]):
            im, _ = mne.viz.plot_topomap(maps[g], info, axes=axes[r, c],
                                         show=False, cmap="RdBu_r",
                                         vlim=(vmin, vmax), contours=4)
            axes[r, c].set_title(f"{g} — relative {band}")
        fig.colorbar(im, ax=axes[r, :].tolist(), shrink=.7)
    fig.savefig(OUT / "03_topomaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved 3 figures to {OUT}")

if __name__ == "__main__":
    main()
