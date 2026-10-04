import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_REST, FIGURES, load_raw, participants

mne.viz.set_browser_backend("matplotlib")

SUBJECT = "sub-001"
OUT = FIGURES / "day01"

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    info_row = participants(DS_REST).query("participant_id == @SUBJECT")
    print(info_row.to_string(index=False))

    raw = load_raw(SUBJECT, DS_REST)
    print(raw)
    print(raw.info)
    print("Channels:", raw.ch_names)
    print("Sampling rate:", raw.info["sfreq"], "Hz")
    print("Duration:", round(raw.n_times / raw.info["sfreq"], 1), "s")

    fig = raw.plot_sensors(show_names=True, show=False)
    fig.savefig(OUT / "01_sensors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = raw.plot(duration=10.0, start=60.0, n_channels=19,
                   scalings=dict(eeg=75e-6), show=False)
    fig.savefig(OUT / "02_raw_10s.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    psd = raw.compute_psd(method="welch", fmin=0.5, fmax=60.0,
                          n_fft=2048, verbose="ERROR")
    fig = psd.plot(show=False)
    fig.axes[0].axvline(50, color="red", linestyle="--", linewidth=1)
    fig.axes[0].set_title(f"{SUBJECT} — PSD (red line = 50 Hz mains)")
    fig.savefig(OUT / "03_psd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved 3 figures to {OUT}")

if __name__ == "__main__":
    main()
