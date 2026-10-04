import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parent))
import filters as F
import preprocess as P
from eegio import DS_REST, PROCESSED, OLD_TO_NEW, load_raw

SUBJECTS = {"sub-001": "AD", "sub-037": "CN", "sub-066": "FTD"}
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 25), "gamma": (25, 45)}
TRAPZ = getattr(np, "trapezoid", None) or np.trapz

VARIANTS = {
    "full pipeline":          (True,  True,  True),
    "no ICA":                 (True,  True,  False),
    "no average reference":   (True,  False, True),
    "no ICA, no average ref": (True,  False, False),
    "gentle filter":          (False, True,  True),
}

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

def load_derivative(subject):
    d = DS_REST / "derivatives" / subject / "eeg"
    hits = sorted(d.glob("*_eeg.set"))
    if not hits:
        raise FileNotFoundError(f"no derivative .set in {d}")
    raw = mne.io.read_raw_eeglab(hits[0], preload=True, verbose="ERROR")
    raw.rename_channels({k: v for k, v in OLD_TO_NEW.items()
                         if k in raw.ch_names})
    return raw

def run_variant(subject, our_filter=True, avg_ref=True, use_ica=True):
    raw = load_raw(subject, DS_REST)
    fs = raw.info["sfreq"]
    dur = raw.n_times / fs

    if our_filter:
        P.apply_own_filters(raw)
    else:
        sos_n = F.design_notch(fs=fs)
        sos_b = signal.butter(4, [0.5, 45.0], btype="bandpass",
                              output="sos", fs=fs)

        def _gentle(d):
            return signal.sosfiltfilt(sos_b, signal.sosfiltfilt(sos_n, d, axis=-1),
                                      axis=-1)
        raw.apply_function(_gentle, picks="eeg", channel_wise=False)

    raw.crop(tmin=P.CROP_EDGE_S, tmax=dur - P.CROP_EDGE_S)
    raw.resample(P.TARGET_SFREQ, verbose="ERROR")

    if avg_ref:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    n_removed = 0
    if use_ica:
        fit = raw.copy().filter(l_freq=P.ICA_HIGHPASS, h_freq=None,
                                verbose="ERROR")
        ica = mne.preprocessing.ICA(
            n_components=P.ICA_N_COMPONENTS,
            method=P.ICA_METHOD,
            fit_params=dict(extended=True) if P.ICA_METHOD == "infomax" else None,
            max_iter=500,
            random_state=P.ICA_RANDOM_STATE,
        )
        ica.fit(fit, decim=P.ICA_DECIM, verbose="ERROR")
        bad, _ = P.detect_blink_components(ica, fit, P.PreprocReport(subject=subject))
        ica.exclude = bad
        ica.apply(raw, verbose="ERROR")
        n_removed = len(bad)

    return rel_bands(*avg_psd(raw)), n_removed

def main():
    rows = []
    for sid, grp in SUBJECTS.items():
        print(f"\n--- {sid} ({grp}) ---")
        theirs = rel_bands(*avg_psd(load_derivative(sid)))
        rows.append({"subject": sid, "group": grp, "variant": "AUTHORS",
                     "ica_removed": "", **theirs})
        for name, (flt, ref, ica) in VARIANTS.items():
            bands, k = run_variant(sid, our_filter=flt, avg_ref=ref, use_ica=ica)
            rows.append({"subject": sid, "group": grp, "variant": name,
                         "ica_removed": k if ica else "", **bands})
            print(f"  {name:24s} delta = {bands['delta']:.3f}"
                  f"   (authors {theirs['delta']:.3f})")

    tab = pd.DataFrame(rows)
    ref = tab[tab.variant == "AUTHORS"].set_index("subject")["delta"]
    tab["delta_gap"] = tab.apply(
        lambda r: r["delta"] - ref[r["subject"]], axis=1)
    tab = tab.round(3)

    pd.set_option("display.width", 200)
    print("\n=== Relative band power, every variant ===")
    print(tab[["subject", "group", "variant", "ica_removed", "delta", "theta",
               "alpha", "beta", "gamma", "delta_gap"]].to_string(index=False))

    print("\n=== How much each change closes the delta gap ===")
    print(f"{'subject':9s}{'variant':26s}{'gap':>9s}{'closed':>10s}")
    for sid in SUBJECTS:
        sub = tab[(tab.subject == sid) & (tab.variant != "AUTHORS")]
        full = sub.loc[sub.variant == "full pipeline", "delta_gap"].iloc[0]
        for _, r in sub.iterrows():
            closed = (1 - abs(r["delta_gap"]) / abs(full)) * 100 if full else 0.0
            print(f"{sid:9s}{r['variant']:26s}{r['delta_gap']:>9.3f}{closed:>9.0f}%")
        best = sub.loc[sub.delta_gap.abs().idxmin(), "variant"]
        print(f"{'':9s}-> closest to the authors: {best}\n")

    tab.to_csv(PROCESSED / "day04_delta_ablation.csv", index=False)
    print(f"Saved table to {PROCESSED / 'day04_delta_ablation.csv'}")

if __name__ == "__main__":
    main()
