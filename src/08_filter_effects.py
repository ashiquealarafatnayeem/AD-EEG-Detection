import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parent))
import filters as F
from eegio import DS_REST, FIGURES, load_raw

OUT = FIGURES / "day03"
FS = F.FS
SUBJECT = "sub-001"
CH = "Pz"

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    raw = load_raw(SUBJECT, DS_REST)
    x = raw.get_data(picks=[CH])[0]
    x = x[int(60 * FS):int(120 * FS)]
    t = np.arange(len(x)) / FS

    sos_bp, N_proto = F.design_iir_bandpass()
    sos_n = F.design_notch()
    taps, _ = F.design_fir_bandpass()

    y_ff = signal.sosfiltfilt(sos_bp, x)
    y_1p = signal.sosfilt(sos_bp, x)
    y_fir = F.apply_fir_zerophase(x, taps)[0]

    seg = slice(int(10 * FS), int(13 * FS))
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(t[seg], x[seg] * 1e6, color="grey", lw=.8, label="raw")
    ax[0].plot(t[seg], y_ff[seg] * 1e6, lw=1.3, label="IIR, sosfiltfilt (zero phase)")
    ax[0].plot(t[seg], y_fir[seg] * 1e6, lw=1.1, ls="--",
               label="FIR, delay-compensated (zero phase)")
    ax[0].set(ylabel="µV", title=f"{SUBJECT} {CH} — two zero-phase routes agree")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(t[seg], y_ff[seg] * 1e6, lw=1.3, label="sosfiltfilt (zero phase)")
    ax[1].plot(t[seg], y_1p[seg] * 1e6, lw=1.3, color="C3",
               label="sosfilt (single pass — SHIFTED)")
    ax[1].set(xlabel="time (s)", ylabel="µV",
              title="Single-pass filtering shifts the signal in time")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "05_phase_distortion.png", dpi=150)
    plt.close(fig)

    tt = np.arange(0, 0.2, 1 / FS)
    tone = np.sin(2 * np.pi * 200 * tt)
    naive = tone[::2]
    proper = signal.decimate(tone, 2, ftype="fir", zero_phase=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for lbl, sig_, fs_ in [("original, fs=500", tone, 500),
                           ("naive ::2, fs=250", naive, 250),
                           ("decimate(), fs=250", proper, 250)]:
        f_, P_ = signal.welch(sig_, fs=fs_, nperseg=min(128, len(sig_)))
        ax[0].semilogy(f_, P_, label=lbl)
    ax[0].axvline(200, color="k", ls=":", lw=.8)
    ax[0].axvline(50, color="r", ls=":", lw=.8)
    ax[0].set(xlabel="Frequency (Hz)", ylabel="PSD",
              title="200 Hz tone: naive ::2 aliases it to 50 Hz (red)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(tt[:100], tone[:100], lw=.9, label="original")
    ax[1].plot(tt[:100:2], naive[:50], "o-", ms=3, lw=.9, label="naive ::2")
    ax[1].set(xlabel="time (s)", title="Same thing in the time domain")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "06_aliasing.png", dpi=150)
    plt.close(fig)

    y_full = signal.sosfiltfilt(sos_bp, signal.sosfiltfilt(sos_n, x))
    fr, Pr = signal.welch(x, fs=FS, nperseg=2000, noverlap=1000)
    ff, Pf = signal.welch(y_full, fs=FS, nperseg=2000, noverlap=1000)

    fig, ax = plt.subplots(2, 1, figsize=(13, 8))
    ax[0].plot(t[seg], x[seg] * 1e6, color="grey", lw=.8, label="raw")
    ax[0].plot(t[seg], y_full[seg] * 1e6, lw=1.2, label="notch + band-pass")
    ax[0].set(xlabel="time (s)", ylabel="µV", title=f"{SUBJECT} {CH} — time domain")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].semilogy(fr, Pr, color="grey", lw=1, label="raw")
    ax[1].semilogy(ff, Pf, lw=1.2, label="filtered")
    ax[1].axvline(50, color="r", ls=":", lw=.8, label="50 Hz mains")
    ax[1].set(xlim=(0, 120), xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
              title="Power spectrum — mains and drift removed")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "07_before_after.png", dpi=150)
    plt.close(fig)

    big = raw.get_data()[:, :int(300 * FS)]
    print("\n=== Timing on 19 channels x 300 s ===")
    t0 = time.perf_counter(); signal.sosfiltfilt(sos_bp, big, axis=-1)
    print(f"  IIR sosfiltfilt        : {time.perf_counter()-t0:6.2f} s")
    t0 = time.perf_counter(); F.apply_fir_zerophase(big, taps)
    print(f"  FIR overlap-add + shift: {time.perf_counter()-t0:6.2f} s")

    print(f"\nSaved 3 figures to {OUT}")

if __name__ == "__main__":
    main()
