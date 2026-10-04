import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.append(str(Path(__file__).resolve().parent))
import spectral as S
from eegio import DS_REST, EXCLUDED, FIGURES, PROCESSED, participants

EPO_DIR = PROCESSED / "epochs"
OUT = FIGURES / "day05"
FMIN, FMAX = 0.5, 45.0
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 25), "gamma": (25, 45)}
POSTERIOR = ["P3", "P4", "Pz", "O1", "O2", "P7", "P8"]
COLORS = {"AD": "#C0392B", "FTD": "#E67E22", "CN": "#2471A3"}

DAY2_AUC = {"rel_delta": 0.614, "rel_theta": 0.804, "rel_alpha": 0.756,
            "rel_beta": 0.693, "rel_gamma": 0.533, "TAR": 0.824,
            "slow_fast": 0.715}

def features(freqs, psd, ch_names, picks):
    idx = [ch_names.index(c) for c in picks]
    p = psd[idx]
    total = S.band_power(freqs, p, FMIN, FMAX)
    rel = {b: S.band_power(freqs, p, lo, hi) / total for b, (lo, hi) in BANDS.items()}
    out = {f"rel_{b}": rel[b].mean() for b in BANDS}
    out["TAR"] = (rel["theta"] / rel["alpha"]).mean()
    out["slow_fast"] = ((rel["delta"] + rel["theta"]) /
                        (rel["alpha"] + rel["beta"])).mean()
    return out

def auc_ad_vs_cn(ad, cn):
    u, p = mannwhitneyu(ad, cn, alternative="two-sided")
    auc = u / (len(ad) * len(cn))
    return max(auc, 1 - auc), ("AD higher" if auc >= 0.5 else "AD lower"), p

def main():
    mne.set_log_level("ERROR")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = participants(DS_REST)

    subjects, groups, psds, n_epochs, ch_names, freqs = [], [], [], [], None, None
    missing = []
    for sid, grp in zip(meta["participant_id"], meta["group"]):
        if sid in EXCLUDED:
            print(f"Excluded {sid} ({grp}): {EXCLUDED[sid]}")
            continue
        path = EPO_DIR / f"{sid}_epo.fif"
        if not path.exists():
            missing.append(sid)
            continue
        ep = mne.read_epochs(path, preload=True, verbose="ERROR")
        if ch_names is None:
            ch_names = list(ep.ch_names)
        if list(ep.ch_names) != ch_names:
            raise SystemExit(f"channel order differs at {sid}")
        f, p = S.welch_epochs(ep.get_data(), ep.info["sfreq"])
        keep = (f >= 0.0) & (f <= FMAX)
        freqs = f[keep]
        subjects.append(sid); groups.append(grp)
        psds.append(p[:, keep]); n_epochs.append(len(ep))

    if missing:
        print(f"WARNING: no epochs file for {len(missing)} subject(s): {missing}")
        print("         Run 12_batch_preprocess.py again before using these results.\n")

    psds = np.stack(psds)
    groups = np.array(groups)
    np.savez(PROCESSED / "psd_clean.npz", psd=psds, freqs=freqs,
             subjects=np.array(subjects), groups=groups,
             ch_names=np.array(ch_names), n_epochs=np.array(n_epochs))
    print(f"Saved psd_clean.npz: {psds.shape[0]} subjects x {psds.shape[1]} channels"
          f" x {psds.shape[2]} frequencies (0-{FMAX:.0f} Hz, "
          f"Δf = {freqs[1] - freqs[0]:.2f} Hz)")
    print("Subjects per group:", pd.Series(groups).value_counts().to_dict())
    print(f"Epochs per subject: median {np.median(n_epochs):.0f}, "
          f"min {np.min(n_epochs)}, max {np.max(n_epochs)}")

    band = (freqs >= FMIN) & (freqs <= FMAX)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for g in ["CN", "FTD", "AD"]:
        m = psds[groups == g][:, :, band].mean(axis=1)
        mu = m.mean(axis=0)
        se = m.std(axis=0, ddof=1) / np.sqrt(len(m))
        for a in ax:
            a.plot(freqs[band], mu, color=COLORS[g], lw=1.6, label=f"{g} (n={len(m)})")
            a.fill_between(freqs[band], mu - se, mu + se, color=COLORS[g], alpha=.2)
    ax[0].set(yscale="log", xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
              title="Clean data, our Welch — log scale")
    ax[1].set(xlim=(2, 20), xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
              title="Linear scale, 2–20 Hz")
    for a in ax:
        a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "05_grand_average_clean.png", dpi=150)
    plt.close(fig)

    rows = []
    for region, picks in [("all 19", ch_names), ("posterior 7", POSTERIOR)]:
        feats = pd.DataFrame([features(freqs, p, ch_names, picks) for p in psds])
        feats["group"] = groups
        for name in DAY2_AUC:
            ad = feats.loc[feats.group == "AD", name].to_numpy()
            cn = feats.loc[feats.group == "CN", name].to_numpy()
            auc, direction, pval = auc_ad_vs_cn(ad, cn)
            rows.append(dict(feature=name, region=region,
                             AD_median=np.median(ad), CN_median=np.median(cn),
                             direction=direction, p=pval, AUC=auc,
                             AUC_day2=DAY2_AUC[name]))
    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day05_clean_vs_day2.csv", index=False)

    wide = res.pivot(index="feature", columns="region", values="AUC")
    wide = wide[["all 19", "posterior 7"]]
    wide.insert(0, "Day 2", pd.Series(DAY2_AUC))
    wide = wide.loc[list(DAY2_AUC)]
    print("\n=== AD vs CN, AUC: Day 2 (placeholder filters) vs clean data ===")
    print(wide.round(3).to_string())

    detail = res[res.region == "all 19"].set_index("feature")
    print("\n=== Clean data, all 19 channels: medians and p values ===")
    fmt = detail[["AD_median", "CN_median", "direction", "p"]].copy()
    fmt["AD_median"] = fmt["AD_median"].map(lambda v: f"{v:.3f}")
    fmt["CN_median"] = fmt["CN_median"].map(lambda v: f"{v:.3f}")
    fmt["p"] = fmt["p"].map(lambda v: f"{v:.1e}")
    print(fmt.to_string())
    print(f"\nFigure: {OUT / '05_grand_average_clean.png'}")

if __name__ == "__main__":
    main()
