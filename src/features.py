import numpy as np

import spectral as S

FMIN, FMAX = 0.5, 45.0
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 25), "gamma": (25, 45)}

REGIONS = {
    "global":    ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                  "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"],
    "frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central":   ["C3", "Cz", "C4"],
    "temporal":  ["T7", "T8", "P7", "P8"],
    "parietal":  ["P3", "Pz", "P4"],
    "occipital": ["O1", "O2"],
    "posterior": ["P3", "P4", "Pz", "O1", "O2", "P7", "P8"],
}

FIT_RANGE = (2.0, 40.0)
PEAK_WINDOW = (6.0, 14.0)
PEAK_THRESHOLD = 0.1

def relative_band_powers(freqs, psd):
    total = S.band_power(freqs, psd, FMIN, FMAX)
    return {b: S.band_power(freqs, psd, lo, hi) / total
            for b, (lo, hi) in BANDS.items()}

def spectral_entropy(freqs, psd):
    m = (freqs >= FMIN) & (freqs <= FMAX)
    p = psd[:, m] / psd[:, m].sum(axis=1, keepdims=True)
    h = -(p * np.log(p + 1e-300)).sum(axis=1)
    return h / np.log(m.sum())

def spectral_edge(freqs, psd, q):
    m = (freqs >= FMIN) & (freqs <= FMAX)
    f = freqs[m]
    out = np.empty(psd.shape[0])
    for c in range(psd.shape[0]):
        cum = np.cumsum(psd[c, m])
        out[c] = np.interp(q, cum / cum[-1], f)
    return out

def rms_frequency(freqs, psd, fs):
    gain2 = (2 * np.sin(np.pi * freqs / fs)) ** 2
    ratio = (psd * gain2).sum(axis=1) / psd.sum(axis=1)
    return fs / (2 * np.pi) * np.sqrt(ratio)

def hjorth(data, fs):
    d1 = np.diff(data, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = data.var(axis=-1).mean(axis=0)
    v1 = d1.var(axis=-1).mean(axis=0)
    v2 = d2.var(axis=-1).mean(axis=0)
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return {"activity": v0,
            "mobility_hz": mobility * fs / (2 * np.pi),
            "complexity": complexity}

def alpha_peak(freqs, spectrum):
    lf, lp = np.log10(freqs[1:]), np.log10(spectrum[1:])
    f = freqs[1:]
    fit_m = (f >= FIT_RANGE[0]) & (f <= FIT_RANGE[1]) & \
            ~((f >= PEAK_WINDOW[0]) & (f <= PEAK_WINDOW[1]))
    slope, intercept = np.polyfit(lf[fit_m], lp[fit_m], 1)
    resid = lp - (intercept + slope * lf)

    win = np.where((f >= PEAK_WINDOW[0]) & (f <= PEAK_WINDOW[1]))[0]
    k = win[np.argmax(resid[win])]
    height = float(resid[k])
    on_edge = k in (win[0], win[-1])
    present = (not on_edge) and height >= PEAK_THRESHOLD

    iaf = np.nan
    if present:
        y0, y1, y2 = resid[k - 1], resid[k], resid[k + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        iaf = float(f[k] + shift * (f[1] - f[0]))

    return {"present": int(present), "iaf": iaf, "height": height,
            "slope": float(slope), "intercept": float(intercept)}

def channel_features(freqs, psd, epochs_data, fs):
    rel = relative_band_powers(freqs, psd)
    hj = hjorth(epochs_data, fs)
    feats = {f"rel_{b}": rel[b] for b in BANDS}
    feats["logTAR"] = np.log10(rel["theta"] / rel["alpha"])
    feats["logSlowFast"] = np.log10((rel["delta"] + rel["theta"]) /
                                    (rel["alpha"] + rel["beta"]))
    feats["entropy"] = spectral_entropy(freqs, psd)
    feats["SEF50"] = spectral_edge(freqs, psd, 0.50)
    feats["SEF95"] = spectral_edge(freqs, psd, 0.95)
    feats["hjorth_logActivity"] = np.log10(hj["activity"])
    feats["hjorth_mobility"] = hj["mobility_hz"]
    feats["hjorth_complexity"] = hj["complexity"]
    return feats

def region_average(values, ch_names, region):
    idx = [ch_names.index(c) for c in REGIONS[region]]
    return float(np.mean(values[idx]))
