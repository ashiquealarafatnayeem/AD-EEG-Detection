import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parent))
import filters as F
from eegio import FIGURES

OUT = FIGURES / "day03"
FS = F.FS

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    taps_k, beta = F.design_fir_bandpass()
    taps_h = F.design_fir_hamming()
    sos_bp, N_proto = F.design_iir_bandpass()
    sos_n = F.design_notch()

    print(f"FIR (Kaiser) : {len(taps_k)} taps, beta = {beta:.3f}, "
          f"group delay = {(len(taps_k)-1)//2} samples "
          f"= {(len(taps_k)-1)/2/FS:.3f} s")
    print(f"FIR (Hamming): {len(taps_h)} taps")
    print(f"IIR Butterworth: prototype order {N_proto}, realised order "
          f"{2*N_proto}, {sos_bp.shape[0]} biquad sections")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    n = np.arange(len(taps_k))
    ax[0].plot(n, taps_k, lw=.8)
    ax[0].set(title=f"FIR impulse response h[n] ({len(taps_k)} taps)",
              xlabel="n (samples)", ylabel="h[n]")
    ax[0].grid(alpha=.3)
    c = len(taps_k) // 2
    ax[1].plot(n[c-200:c+200], taps_k[c-200:c+200], lw=1.2)
    ax[1].set(title="Centre of h[n] — the windowed sinc",
              xlabel="n (samples)")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "01_fir_impulse.png", dpi=150)
    plt.close(fig)

    wk, hk = signal.freqz(taps_k, worN=16384, fs=FS)
    wh, hh = signal.freqz(taps_h, worN=16384, fs=FS)
    wi, hi = F.sos_response(sos_bp)
    wn, hn = F.sos_response(sos_n)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    for a in [ax[0, 0], ax[0, 1]]:
        a.plot(wk, F.db(hk), label=f"FIR Kaiser ({len(taps_k)} taps)")
        a.plot(wh, F.db(hh), label=f"FIR Hamming ({len(taps_h)} taps)", ls="--")
        a.plot(wi, F.db(hi), label=f"IIR Butterworth (order {2*N_proto})")
        a.axhline(-3, color="k", lw=.6, ls=":")
        a.set(xlabel="Frequency (Hz)", ylabel="|H(f)| (dB)")
        a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0, 0].set(xlim=(0, 80), ylim=(-100, 5), title="Band-pass, 0–80 Hz")
    ax[0, 1].set(xlim=(0, 3), ylim=(-60, 5), title="Low edge detail, 0–3 Hz")

    ax[1, 0].plot(wn, F.db(hn), color="C3")
    ax[1, 0].set(xlim=(40, 60), ylim=(-60, 5), xlabel="Frequency (Hz)",
                 ylabel="|H(f)| (dB)", title="50 Hz notch (Q = 30)")
    ax[1, 0].grid(alpha=.3)

    wc, hc = F.sos_response(np.vstack([sos_n, sos_bp]))
    ax[1, 1].plot(wc, F.db(hc), label="cascade, single pass")
    ax[1, 1].plot(wc, 2 * F.db(hc), label="cascade, filtfilt (|H|²)", ls="--")
    ax[1, 1].axvline(50, color="r", lw=.8, ls=":")
    ax[1, 1].set(xlim=(0, 80), ylim=(-160, 5), xlabel="Frequency (Hz)",
                 ylabel="|H(f)| (dB)", title="Full chain — note filtfilt doubles dB")
    ax[1, 1].grid(alpha=.3); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(OUT / "02_magnitude.png", dpi=150)
    plt.close(fig)

    gd_k = F.group_delay_samples(wk, hk)
    gd_i = F.group_delay_samples(wi, hi)
    band = (wk >= 0.5) & (wk <= 45)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(wk, np.unwrap(np.angle(hk)), label="FIR (linear phase)")
    ax[0].plot(wi, np.unwrap(np.angle(hi)), label="IIR (nonlinear phase)")
    ax[0].set(xlim=(0, 60), xlabel="Frequency (Hz)", ylabel="phase (rad)",
              title="Unwrapped phase response")
    ax[0].grid(alpha=.3); ax[0].legend()

    ax[1].plot(wk[band], gd_k[band], label="FIR")
    ax[1].plot(wi[(wi >= .5) & (wi <= 45)],
               gd_i[(wi >= .5) & (wi <= 45)], label="IIR")
    ax[1].axhline((len(taps_k) - 1) / 2, color="k", ls=":", lw=.8,
                  label="(N-1)/2")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="group delay (samples)",
              title="Group delay in the passband", yscale="symlog")
    ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "03_phase_groupdelay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    for a, sos, ttl in [(ax[0], sos_bp, f"Butterworth band-pass (order {2*N_proto})"),
                        (ax[1], sos_n, "50 Hz notch (order 2)")]:
        z, p, k = signal.sos2zpk(sos)
        th = np.linspace(0, 2 * np.pi, 512)
        a.plot(np.cos(th), np.sin(th), "k", lw=.8)
        a.scatter(z.real, z.imag, marker="o", s=70, facecolors="none",
                  edgecolors="C0", label="zeros")
        a.scatter(p.real, p.imag, marker="x", s=70, color="C3", label="poles")
        a.set(title=ttl, xlabel="Re", ylabel="Im")
        a.axhline(0, lw=.5, color="grey"); a.axvline(0, lw=.5, color="grey")
        a.set_aspect("equal"); a.grid(alpha=.3); a.legend(fontsize=8)
        print(f"{ttl}: max |pole| = {np.max(np.abs(p)):.6f}  (stable if < 1)")
    fig.tight_layout(); fig.savefig(OUT / "04_polezero.png", dpi=150)
    plt.close(fig)

    def at(w, h, f):
        return F.db(h)[np.argmin(np.abs(w - f))]

    print("\n=== Measured filter characteristics (dB) ===")
    print(f"{'freq (Hz)':>10}{'FIR':>10}{'IIR':>10}{'IIR filtfilt':>14}")
    for f in [0.1, 0.25, 0.5, 1.0, 10.0, 45.0, 50.0, 60.0]:
        print(f"{f:>10.2f}{at(wk, hk, f):>10.1f}{at(wi, hi, f):>10.1f}"
              f"{2*at(wi, hi, f):>14.1f}")

    print(f"\nStopband attenuation at 50 Hz, band-pass alone: "
          f"{at(wi, hi, 50):.1f} dB single pass, "
          f"{2*at(wi, hi, 50):.1f} dB with filtfilt")
    print(f"With the notch cascaded: {at(wc, hc, 50):.1f} dB single pass")

    print("\n=== True IIR Group Delay (from derivative of phase) ===")
    for f0 in [2.0, 6.0, 10.5, 19.0]:
        i = np.argmin(np.abs(wi - f0))
        print(f"  IIR group delay at {f0:5.1f} Hz: {gd_i[i]:8.1f} samples = {gd_i[i]/FS*1000:7.1f} ms")

    print("\n=== IIR design ===")
    print(f"Prototype order {n_bp}, band-pass order {2 * n_bp}, "
          f"{sos_bp.shape[0]} second-order sections")

    print(f"\nSaved 4 figures to {OUT}")

if __name__ == "__main__":
    main()
