import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import features as FT
from eegio import DS_REST, FIGURES, PROCESSED, participants

EPO_DIR = PROCESSED / "epochs"
OUT = FIGURES / "day06"

def main():
    mne.set_log_level("ERROR")
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load(PROCESSED / "psd_clean.npz")
    psds, freqs = z["psd"], z["freqs"]
    subjects = [str(s) for s in z["subjects"]]
    groups = [str(g) for g in z["groups"]]
    ch_names = [str(c) for c in z["ch_names"]]

    meta = participants(DS_REST).set_index("participant_id")
    post = [ch_names.index(c) for c in FT.REGIONS["posterior"]]

    ch_rows, rg_rows, check = [], [], []
    for i, sid in enumerate(subjects):
        ep = mne.read_epochs(EPO_DIR / f"{sid}_epo.fif", preload=True, verbose="ERROR")
        if list(ep.ch_names) != ch_names:
            raise SystemExit(f"channel order differs at {sid}")
        fs = ep.info["sfreq"]
        data = ep.get_data()

        feats = FT.channel_features(freqs, psds[i], data, fs)
        peak = FT.alpha_peak(freqs, psds[i][post].mean(axis=0))
        base = {"participant_id": sid, "group": groups[i],
                "age": meta.loc[sid, "Age"], "sex": meta.loc[sid, "Gender"],
                "n_epochs": len(ep)}
        alpha = {"alpha_present": peak["present"], "alpha_iaf": peak["iaf"],
                 "alpha_height": peak["height"]}

        ch_rows.append({**base,
                        **{f"{k}__{c}": v[j] for k, v in feats.items()
                           for j, c in enumerate(ch_names)},
                        **alpha})
        rg_rows.append({**base,
                        **{f"{k}__{r}": FT.region_average(v, ch_names, r)
                           for k, v in feats.items() for r in FT.REGIONS},
                        **alpha})

        pred = FT.rms_frequency(freqs, psds[i], fs)
        check.append((feats["hjorth_mobility"].mean(), pred.mean()))
        where = f"yes, {peak['iaf']:.2f} Hz" if peak["present"] else "no"
        print(f"[{i + 1:2d}/{len(subjects)}] {sid} ({groups[i]:3s})  alpha peak: {where}")

    ch_df, rg_df = pd.DataFrame(ch_rows), pd.DataFrame(rg_rows)
    ch_df.to_csv(PROCESSED / "features_channel.csv", index=False)
    rg_df.to_csv(PROCESSED / "features_region.csv", index=False)

    n_meta = 5
    print(f"\nfeatures_channel.csv: {len(ch_df)} subjects x {ch_df.shape[1] - n_meta} features")
    print(f"features_region.csv : {len(rg_df)} subjects x {rg_df.shape[1] - n_meta} features")
    print("Subjects per group:", rg_df["group"].value_counts().to_dict())
    bad = rg_df.drop(columns=["alpha_iaf"]).isna().sum().sum()
    print(f"Missing values (excluding IAF, which is empty when no peak): {bad}")

    c = np.array(check)
    r = np.corrcoef(c[:, 0], c[:, 1])[0, 1]
    ratio = c[:, 0] / c[:, 1]
    print("\n=== Check: Hjorth mobility (time domain) vs spectrum (frequency domain) ===")
    print(f"  correlation r = {r:.4f}")
    print(f"  mobility / prediction: mean {ratio.mean():.4f}, "
          f"range {ratio.min():.4f} - {ratio.max():.4f}")
    print(f"  {'PASS' if r > 0.99 and abs(ratio.mean() - 1) < 0.05 else 'CHECK THIS'}")

    fig, ax = plt.subplots(figsize=(5.6, 5))
    colors = {"AD": "#C0392B", "FTD": "#E67E22", "CN": "#2471A3"}
    for g in ["CN", "FTD", "AD"]:
        m = np.array(groups) == g
        ax.scatter(c[m, 1], c[m, 0], s=22, color=colors[g], label=g, alpha=.85)
    lo, hi = c.min() * 0.95, c.max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=.8, label="equal")
    ax.set(xlabel="predicted from the spectrum (Hz)", ylabel="Hjorth mobility, time domain (Hz)",
           title=f"Two domains, one answer (r = {r:.3f})")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "01_hjorth_vs_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure: {OUT / '01_hjorth_vs_spectrum.png'}")

if __name__ == "__main__":
    main()
