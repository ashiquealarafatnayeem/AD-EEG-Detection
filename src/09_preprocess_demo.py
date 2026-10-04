import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parent))
import preprocess as P
from eegio import DS_REST, FIGURES, PROCESSED, load_raw, participants

OUT = FIGURES / "day04"
CH = "Pz"
SEG = (30.0, 35.0)

def pick_subjects():
    df = participants(DS_REST)
    return [df[df.group == g].participant_id.iloc[0] for g in ["AD", "CN", "FTD"]]

def stage_figure(stages, subject, group, path):
    order = ["1_raw", "2_filtered", "3_referenced", "4_ica_cleaned"]
    titles = ["Raw", "After notch + band-pass",
              "After common-average reference", "After ICA blink removal"]
    fig, axes = plt.subplots(len(order), 1, figsize=(13, 9), sharex=True)
    for ax, key, ttl in zip(axes, order, titles):
        t, x = stages[key]
        m = (t >= SEG[0]) & (t <= SEG[1])
        ax.plot(t[m], x[m] * 1e6, lw=.9)
        ax.set_ylabel("µV")
        ax.set_title(ttl, loc="left", fontsize=10)
        ax.grid(alpha=.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{subject} ({group}) — channel {CH}, pipeline stages")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def psd_figure(subject, raw_clean, path):
    raw0 = load_raw(subject, DS_REST)
    x0 = raw0.get_data(picks=[CH])[0]
    x1 = raw_clean.get_data(picks=[CH])[0]
    f0, p0 = signal.welch(x0, fs=raw0.info["sfreq"], nperseg=2000, noverlap=1000)
    f1, p1 = signal.welch(x1, fs=raw_clean.info["sfreq"],
                          nperseg=int(4 * raw_clean.info["sfreq"]),
                          noverlap=int(2 * raw_clean.info["sfreq"]))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.semilogy(f0, p0, color="grey", lw=1, label="raw (500 Hz)")
    ax.semilogy(f1, p1, lw=1.3, label="preprocessed (250 Hz)")
    ax.axvline(50, color="r", ls=":", lw=.8, label="50 Hz mains")
    ax.set(xlim=(0, 120), xlabel="Frequency (Hz)", ylabel="PSD (V²/Hz)",
           title=f"{subject} {CH} — spectrum before and after preprocessing")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    meta = participants(DS_REST).set_index("participant_id")

    rows = []
    for sid in pick_subjects():
        group = meta.loc[sid, "group"]
        print(f"\n=== {sid} ({group}) ===")
        t0 = time.perf_counter()
        raw, epochs, ica, rep, stages = P.preprocess_subject(
            sid, snapshot_channel=CH)
        elapsed = time.perf_counter() - t0

        print(f"  duration      : {rep.dur_raw_s:.1f} s -> "
              f"{rep.dur_cropped_s:.1f} s after cropping")
        print(f"  sampling rate : {rep.sfreq_out:.0f} Hz")
        print(f"  ICA method    : {rep.ica_method_used}")
        print(f"  ICA excluded  : {rep.ica_excluded} "
              f"of {P.ICA_N_COMPONENTS} components")
        print(f"  epochs        : {rep.n_epochs_kept} kept of "
              f"{rep.n_epochs_made} ({rep.drop_pct:.1f} % dropped)")
        print(f"  wall time     : {elapsed:.1f} s")

        stage_figure(stages, sid, group, OUT / f"01_stages_{sid}.png")
        psd_figure(sid, raw, OUT / f"02_psd_{sid}.png")

        try:
            figs = ica.plot_components(show=False)
            figs = figs if isinstance(figs, list) else [figs]
            for i, f in enumerate(figs):
                f.savefig(OUT / f"03_ica_topo_{sid}_{i}.png",
                          dpi=150, bbox_inches="tight")
                plt.close(f)
        except Exception as e:
            print(f"    (topography plot skipped: {e})")

        try:
            f = ica.plot_sources(raw, show=False)
            f.savefig(OUT / f"04_ica_sources_{sid}.png",
                      dpi=150, bbox_inches="tight")
            plt.close(f)
        except Exception as e:
            print(f"    (sources plot skipped: {e})")

        if rep.ica_excluded:
            try:
                f = ica.plot_overlay(raw, exclude=rep.ica_excluded,
                                     picks="eeg", show=False)
                f.savefig(OUT / f"05_ica_overlay_{sid}.png",
                          dpi=150, bbox_inches="tight")
                plt.close(f)
            except Exception as e:
                print(f"    (overlay plot skipped: {e})")

        raw.save(PROCESSED / f"{sid}_clean_raw.fif", overwrite=True,
                 verbose="ERROR")

        rows.append(dict(subject=sid, group=group,
                         dur_s=round(rep.dur_cropped_s, 1),
                         sfreq=rep.sfreq_out,
                         ica_excluded=str(rep.ica_excluded),
                         n_made=rep.n_epochs_made, n_kept=rep.n_epochs_kept,
                         drop_pct=round(rep.drop_pct, 1),
                         seconds=round(elapsed, 1)))

    tab = pd.DataFrame(rows)
    print("\n=== Day 4 summary ===")
    print(tab.to_string(index=False))
    tab.to_csv(PROCESSED / "day04_preproc_summary.csv", index=False)
    print(f"\nEstimated time for all 88 subjects: "
          f"{tab.seconds.mean() * 88 / 60:.0f} minutes")
    print(f"Figures in {OUT}")

if __name__ == "__main__":
    main()
