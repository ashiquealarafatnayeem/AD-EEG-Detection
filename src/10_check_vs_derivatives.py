import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_REST, FIGURES, PROCESSED, OLD_TO_NEW

OUT = FIGURES / "day04"
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 25), "gamma": (25, 45)}
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

def load_derivative(subject):
    d = DS_REST / "derivatives" / subject / "eeg"
    hits = sorted(d.glob("*_eeg.set"))
    if not hits:
        raise FileNotFoundError(f"no derivative .set in {d}")
    raw = mne.io.read_raw_eeglab(hits[0], preload=True, verbose="ERROR")
    raw.rename_channels({k: v for k, v in OLD_TO_NEW.items()
                         if k in raw.ch_names})
    return raw

def avg_psd(raw, fmin=0.5, fmax=45.0):
    fs = raw.info["sfreq"]
    nper = int(round(4 * fs))
    f, p = signal.welch(raw.get_data(), fs=fs, nperseg=nper,
                        noverlap=nper // 2, axis=-1)
    m = (f >= fmin) & (f <= fmax)
    return f[m], p[:, m].mean(axis=0)

def rel_bands(f, p):
    total = TRAPZ(p, f)
    return {b: TRAPZ(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) / total
            for b, (lo, hi) in BANDS.items()}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.name.split("_")[0]
                      for p in PROCESSED.glob("*_clean_raw.fif"))
    if not subjects:
        raise SystemExit("Run 09_preprocess_demo.py first.")

    rows = []
    fig, axes = plt.subplots(1, len(subjects), figsize=(5.2 * len(subjects), 4.4),
                             squeeze=False)
    for ax, sid in zip(axes[0], subjects):
        ours = mne.io.read_raw_fif(PROCESSED / f"{sid}_clean_raw.fif",
                                   preload=True, verbose="ERROR")
        theirs = load_derivative(sid)

        f_o, p_o = avg_psd(ours)
        f_t, p_t = avg_psd(theirs)
        assert np.allclose(f_o, f_t), "frequency grids differ"

        r = np.corrcoef(np.log10(p_o), np.log10(p_t))[0, 1]
        bo, bt = rel_bands(f_o, p_o), rel_bands(f_t, p_t)

        ax.semilogy(f_o, p_o, lw=1.4, label="ours")
        ax.semilogy(f_t, p_t, lw=1.4, ls="--", label="authors' derivative")
        ax.set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
               title=f"{sid}   log-PSD r = {r:.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=.3)

        row = {"subject": sid, "logpsd_r": round(r, 4)}
        for b in BANDS:
            row[f"{b}_ours"] = round(bo[b], 4)
            row[f"{b}_theirs"] = round(bt[b], 4)
            row[f"{b}_diff"] = round(bo[b] - bt[b], 4)
        rows.append(row)

    fig.tight_layout()
    fig.savefig(OUT / "06_vs_derivatives.png", dpi=150)
    plt.close(fig)

    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== Our pipeline vs the authors' derivatives ===")
    print(tab[["subject", "logpsd_r"]].to_string(index=False))
    print()
    print(tab[["subject"] + [c for c in tab.columns if c.endswith("_diff")]]
          .to_string(index=False))
    tab.to_csv(PROCESSED / "day04_derivative_check.csv", index=False)
    print(f"\nSaved figure to {OUT / '06_vs_derivatives.png'}")

if __name__ == "__main__":
    main()
