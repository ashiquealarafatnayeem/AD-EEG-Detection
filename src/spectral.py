import numpy as np

TRAPZ = getattr(np, "trapezoid", None) or np.trapz

def hann(n):
    k = np.arange(n)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)

def n_segments(n_samples, nperseg, noverlap):
    step = nperseg - noverlap
    return (n_samples - noverlap) // step

def segment(x, nperseg, noverlap):
    step = nperseg - noverlap
    k = n_segments(x.shape[-1], nperseg, noverlap)
    if k < 1:
        raise ValueError("signal is shorter than one segment")
    idx = np.arange(nperseg)[None, :] + step * np.arange(k)[:, None]
    return x[..., idx]

def _segment_spectra(seg, fs, window, detrend=True):
    n = seg.shape[-1]
    if detrend:
        seg = seg - seg.mean(axis=-1, keepdims=True)

    spec = np.fft.rfft(seg * window, axis=-1)
    p = (spec.real ** 2 + spec.imag ** 2) / (fs * np.sum(window ** 2))

    if n % 2 == 0:
        p[..., 1:-1] *= 2.0
    else:
        p[..., 1:] *= 2.0

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, p

def welch(x, fs, nperseg, noverlap=None, window=None, detrend=True):
    x = np.asarray(x, dtype=float)
    if noverlap is None:
        noverlap = nperseg // 2

    w = hann(nperseg) if window is None else np.asarray(window, dtype=float)
    seg = segment(x, nperseg, noverlap)
    freqs, p = _segment_spectra(seg, fs, w, detrend)

    return freqs, p.mean(axis=-2)

def welch_epochs(data, fs, window=None, detrend=True):
    data = np.asarray(data, dtype=float)
    n = data.shape[-1]
    w = hann(n) if window is None else np.asarray(window, dtype=float)
    freqs, p = _segment_spectra(data, fs, w, detrend)
    return freqs, p.mean(axis=0)

def welch_variance_ratio(nperseg, noverlap, k, window=None):
    w = hann(nperseg) if window is None else np.asarray(window, dtype=float)
    d = nperseg - noverlap
    energy = np.sum(w ** 2)
    total = 1.0
    j = 1
    while j < k and j * d < nperseg:
        rho = (np.sum(w[: nperseg - j * d] * w[j * d:]) / energy) ** 2
        total += 2.0 * (k - j) / k * rho
        j += 1
    return total / k

def band_power(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    return TRAPZ(psd[..., m], freqs[m], axis=-1)
