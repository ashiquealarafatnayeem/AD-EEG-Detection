import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parent))
import spectral as S
from eegio import FIGURES, PROCESSED

OUT = FIGURES / "day05"
REL_TOL = 1e-10
CH = "Pz"
SEED = 5

def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

def load_clean(subject):
    path = PROCESSED / f"{subject}_clean_raw.fif"
    if not path.exists():
        raise SystemExit(f"{path} not found - run 09_preprocess_demo.py first.")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    return raw.get_data(), raw.info["sfreq"], raw.ch_names

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = []

    print("=== Check 1: our Welch vs scipy.signal.welch, random data ===")
    x = rng.standard_normal((19, 60_000)) * 1e-5
    for nper, nover in [(1000, 500), (250, 125), (999, 499), (512, 0), (1000, 750)]:
        f1, p1 = S.welch(x, 250.0, nper, nover)
        f2, p2 = signal.welch(x, fs=250.0, nperseg=nper, noverlap=nover)
        e = rel_err(p1, p2)
        ok = e < REL_TOL and np.allclose(f1, f2)
        results.append(ok)
        print(f"  nperseg={nper:5d} noverlap={nover:4d}   relative error = {e:.2e}"
              f"   {'PASS' if ok else 'FAIL'}")

    print("\n=== Check 2: our Welch vs SciPy, real cleaned EEG (sub-001) ===")
    eeg, fs, chs = load_clean("sub-001")
    nper = int(4 * fs)
    f_o, p_o = S.welch(eeg, fs, nper, nper // 2)
    f_s, p_s = signal.welch(eeg, fs=fs, nperseg=nper, noverlap=nper // 2)
    e = rel_err(p_o, p_s)
    ok = e < REL_TOL
    results.append(ok)
    print(f"  {len(chs)} channels, fs = {fs:.0f} Hz, "
          f"{S.n_segments(eeg.shape[-1], nper, nper // 2)} segments of 4 s")
    print(f"  relative error = {e:.2e}   {'PASS' if ok else 'FAIL'}")

    print("\n=== Check 3: welch_epochs() on the same segments ===")
    epochs = np.moveaxis(S.segment(eeg, nper, nper // 2), -2, 0)
    _, p_ep = S.welch_epochs(epochs, fs)
    e = rel_err(p_ep, p_o)
    ok = e < REL_TOL
    results.append(ok)
    print(f"  {epochs.shape[0]} epochs x {epochs.shape[1]} channels x "
          f"{epochs.shape[2]} samples   relative error = {e:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")

    print("\n=== Check 4: white noise, variance 1, fs = 250 Hz ===")
    wn = rng.standard_normal(150_000)
    f_w, p_w = S.welch(wn, 250.0, 1000)
    level = p_w[1:-1].mean()
    total = S.band_power(f_w, p_w, 0.0, 125.0)
    ok = abs(level / (2 / 250.0) - 1) < 0.02 and abs(total - 1) < 0.02
    results.append(ok)
    print(f"  mean density = {level:.5f}  (theory 2/fs = {2 / 250.0:.5f})")
    print(f"  total power  = {total:.4f}   (theory 1)   {'PASS' if ok else 'FAIL'}")

    print("\n=== Check 5: 10 Hz sine, amplitude 10 uV ===")
    t = np.arange(150_000) / 250.0
    amp = 10e-6
    f_s5, p_s5 = S.welch(amp * np.sin(2 * np.pi * 10 * t), 250.0, 1000)
    power = S.band_power(f_s5, p_s5, 9.0, 11.0)
    ok = abs(power / (amp ** 2 / 2) - 1) < 1e-6
    results.append(ok)
    print(f"  power in 9-11 Hz = {power:.4e} V^2  (theory A^2/2 = {amp ** 2 / 2:.4e})"
          f"   {'PASS' if ok else 'FAIL'}")

    ci = chs.index(CH)
    band = (f_o >= 0.5) & (f_o <= 45)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].semilogy(f_s[band], p_s[ci, band], lw=4, alpha=.35, label="scipy.signal.welch")
    ax[0].semilogy(f_o[band], p_o[ci, band], lw=1.2, color="k", label="our welch()")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
              title=f"sub-001 {CH}: the two estimates overlap exactly")
    ax[0].grid(alpha=.3); ax[0].legend()
    diff = np.abs(p_o[:, band] - p_s[:, band]) / p_s[:, band]
    ax[1].semilogy(f_o[band], diff.max(axis=0) + 1e-20, color="C3")
    ax[1].axhline(REL_TOL, color="k", ls="--", lw=.8, label=f"tolerance {REL_TOL:.0e}")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="|ours − SciPy| / SciPy",
              title="Relative difference, worst of 19 channels", ylim=(1e-18, 1e-8))
    ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "01_welch_vs_scipy.png", dpi=150)
    plt.close(fig)

    eeg_cn, fs_cn, chs_cn = load_clean("sub-037")
    xp = eeg_cn[chs_cn.index(CH)]
    configs = [("Periodogram, whole recording", len(xp), 0),
               ("Welch, 16 s segments", int(16 * fs_cn), int(8 * fs_cn)),
               ("Welch, 4 s segments", int(4 * fs_cn), int(2 * fs_cn)),
               ("Welch, 1 s segments", int(1 * fs_cn), int(0.5 * fs_cn))]
    print("\n=== Resolution versus variance (sub-037, Pz) ===")
    print(f"  {'method':32s}{'segment':>10s}{'K':>6s}{'Δf (Hz)':>10s}")
    fig, ax = plt.subplots(len(configs), 1, figsize=(11, 10), sharex=True)
    for a, (name, nper_c, nover_c) in zip(ax, configs):
        f_c, p_c = S.welch(xp, fs_cn, nper_c, nover_c)
        k = S.n_segments(len(xp), nper_c, nover_c)
        df = fs_cn / nper_c
        m = (f_c >= 0.5) & (f_c <= 30)
        a.semilogy(f_c[m], p_c[m], lw=.8)
        a.set(ylabel="V²/Hz", title=f"{name}   —   K = {k} segments,  Δf = {df:.4f} Hz")
        a.grid(alpha=.3)
        print(f"  {name:32s}{nper_c / fs_cn:>9.1f}s{k:>6d}{df:>10.4f}")
    ax[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Longer segments: finer resolution but a noisier estimate", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "02_resolution_tradeoff.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig)

    fs_l, n_l = 250.0, 1000
    t_l = np.arange(n_l) / fs_l
    strong_f, weak_f, weak_amp = 10.125, 16.0, 1e-3
    tone = np.sin(2 * np.pi * strong_f * t_l) + weak_amp * np.sin(2 * np.pi * weak_f * t_l)
    print("\n=== Spectral leakage: strong tone at 10.125 Hz, weak tone at 16 Hz (-60 dB) ===")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for a, (name, w) in zip(ax, [("Rectangular window", np.ones(n_l)),
                                 ("Hann window", S.hann(n_l))]):
        f_l, p_l = S.welch(tone, fs_l, n_l, 0, window=w, detrend=False)
        db = 10 * np.log10(p_l / p_l.max() + 1e-30)
        a.plot(f_l, db, lw=.9)
        a.axvline(weak_f, color="C3", ls=":", lw=1, label="weak tone, 16 Hz")
        a.set(xlim=(0, 30), ylim=(-140, 5), xlabel="Frequency (Hz)", title=name)
        a.grid(alpha=.3); a.legend(loc="upper right")
        at_weak = db[np.argmin(np.abs(f_l - weak_f))]
        floor = np.median(db[(f_l > 18) & (f_l < 22)])
        print(f"  {name:20s} level at 16 Hz = {at_weak:7.1f} dB,"
              f"  leakage floor near 20 Hz = {floor:7.1f} dB")
    ax[0].set_ylabel("dB relative to peak")
    fig.tight_layout(); fig.savefig(OUT / "03_window_leakage.png", dpi=150)
    plt.close(fig)

    print("\n=== Estimator scatter on white noise: measured vs Welch (1967) theory ===")
    wn2 = rng.standard_normal(150_000)
    rows = []
    for nper_v in [150_000, 50_000, 15_000, 5_000, 2_000, 1_000, 500, 250]:
        nover_v = 0 if nper_v == len(wn2) else nper_v // 2
        f_v, p_v = S.welch(wn2, 250.0, nper_v, nover_v)
        inner = p_v[2:-1]
        k = S.n_segments(len(wn2), nper_v, nover_v)
        measured = inner.std() / inner.mean()
        theory = np.sqrt(S.welch_variance_ratio(nper_v, nover_v, k))
        rows.append((nper_v, k, 250.0 / nper_v, measured, theory))
        print(f"  segment {nper_v / 250:7.1f} s  K = {k:4d}  Δf = {250.0 / nper_v:8.4f} Hz"
              f"   std/mean measured = {measured:.3f}   theory = {theory:.3f}")
    r = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(r[:, 1], r[:, 3], "o", ms=7, label="measured (white noise)")
    ax.loglog(r[:, 1], r[:, 4], "-", lw=1.2, label="Welch (1967) theory")
    ax.set(xlabel="number of averaged segments K",
           ylabel="std / mean of the PSD estimate",
           title="Averaging K segments shrinks the scatter as 1/√K")
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "04_variance_vs_segments.png", dpi=150)
    plt.close(fig)

    print("\n=== Verdict ===")
    print(f"  {sum(results)} of {len(results)} checks passed")
    print(f"  Figures saved to {OUT}")
    if not all(results):
        raise SystemExit("VALIDATION FAILED - do not use spectral.py until fixed.")

if __name__ == "__main__":
    main()
