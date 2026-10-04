import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import photic as PH
import preprocess as P
from eegio import DS_PHOTO, EXCLUDED, FIGURES, PROCESSED, load_raw, participants

OUT = FIGURES / "day07"
EXAMPLE = "sub-037"
KIND_COLORS = {"photic": "#E67E22", "open": "#1E8449", "closed": "#2471A3",
               "artefact": "#C0392B", "equipment": "#7D3C98", "eeg_mode": "#7D3C98",
               "calib": "#1B2A41", "other": "#8a96a3"}

def summarise(sid):
    ev = PH.read_events(sid)
    raw = load_raw(sid, DS_PHOTO, preload=False)
    dur = raw.n_times / raw.info["sfreq"]
    return ev, dur, PH.plan(ev, dur, P.CROP_EDGE_S)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    meta = participants(DS_PHOTO)
    rows, all_values, blocks = [], [], []
    for sid, grp in zip(meta.participant_id, meta.group):
        if sid in EXCLUDED:
            continue
        ev, dur, pl = summarise(sid)
        all_values += [(v, k, sid) for v, k in zip(ev.value, ev.kind)]
        blocks += [b - a for a, b in pl["eo_stretches"]]
        photic_t = ev.loc[ev.kind == "photic", "onset"]
        after = (dur - P.CROP_EDGE_S - (photic_t.max() + PH.ZONE_PAD_AFTER_S)
                 if len(photic_t) else dur - 2 * P.CROP_EDGE_S)
        rows.append(dict(subject=sid, group=grp, dur_s=round(dur, 1),
                         after_photic_s=round(max(after, 0.0), 1),
                         n_open_markers=int((ev.kind == "open").sum()),
                         eo_stretches=len(pl["eo_stretches"]),
                         eo_s=round(sum(b - a for a, b in pl["eo_stretches"]), 1),
                         eo_windows=len(pl["eo_starts"]),
                         eo_var=round(PH.variance_ratio(pl["eo_starts"]), 2),
                         ec_near_windows=len(pl["ec_near_starts"]),
                         ec_all_windows=len(pl["ec_all_starts"])))

    av = pd.DataFrame(all_values, columns=["value", "kind", "subject"])
    av["value"] = av["value"].map(repr)
    vals = (av.groupby(["kind", "value"]).subject.nunique().rename("n_recordings")
            .reset_index().sort_values(["kind", "n_recordings"], ascending=[True, False]))
    print("=== Every distinct marker text, its classification, and in how many recordings ===")
    print(vals.to_string(index=False))
    print("  -> 'other' is ignored. If an eye, movement or equipment marker is there, tell me.")

    tab = pd.DataFrame(rows)
    tab["qualifies"] = ((tab.eo_var <= PH.MAX_EO_VAR_RATIO)
                        & (tab.ec_near_windows >= PH.MIN_EC_WINDOWS))
    tab.to_csv(PROCESSED / "day07_event_summary.csv", index=False)
    print("\n=== Per group (median over people; windows are 2 s long, 0.5 s apart) ===")
    print(tab.groupby("group")[["dur_s", "after_photic_s", "eo_stretches", "eo_s",
                                "eo_windows", "ec_near_windows", "ec_all_windows"]]
          .median().round(1).to_string())
    b = np.array(blocks)
    if len(b):
        print(f"\nEyes-open stretches: {len(b)} in total, length median {np.median(b):.1f} s "
              f"(10th-90th percentile {np.percentile(b, 10):.1f}-{np.percentile(b, 90):.1f} s)")

    print(f"\n=== Who has enough data (eyes-open variance ratio <= {PH.MAX_EO_VAR_RATIO} and "
          f">= {PH.MIN_EC_WINDOWS} nearby eyes-closed windows, BEFORE amplitude rejection) ===")
    q = tab.groupby("group").qualifies.agg(["sum", "count"])
    q.columns = ["qualify", "of"]
    print(q.to_string())
    print(f"Total: {int(tab.qualifies.sum())} of {len(tab)}")
    no = tab[~tab.qualifies]
    if len(no):
        print("\nNot enough data:")
        print(no[["subject", "group", "dur_s", "after_photic_s", "eo_stretches",
                  "eo_windows", "eo_var", "ec_near_windows"]].to_string(index=False))

    ev, dur, pl = summarise(EXAMPLE)
    t = pl["times"]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    for a, bb in PH.photic_zones(ev):
        ax.axvspan(a, bb, color="#E67E22", alpha=.15)
    for a, bb in PH.calibration_zones(ev):
        ax.axvspan(a, min(bb, dur), color="#1B2A41", alpha=.15)
    ax.axvspan(0, P.CROP_EDGE_S, color="#8a96a3", alpha=.25)
    ax.axvspan(dur - P.CROP_EDGE_S, dur, color="#8a96a3", alpha=.25)
    ax.fill_between(t, 0.0, 0.35, where=pl["open"], color="#1E8449", alpha=.9, step="mid",
                    label="usable eyes-open")
    ax.fill_between(t, 0.0, 0.35, where=pl["near"], color="#2471A3", alpha=.9, step="mid",
                    label="eyes-closed, used")
    ax.fill_between(t, 0.0, 0.35, where=pl["closed"] & ~pl["near"], color="#2471A3",
                    alpha=.3, step="mid", label="eyes-closed, check only")
    for _, r in ev.iterrows():
        ax.vlines(r.onset, 0.45, 1.0, color=KIND_COLORS[r.kind], lw=1)
    for k, c in KIND_COLORS.items():
        ax.plot([], [], color=c, lw=2, label=f"marker: {k}")
    ax.set(xlim=(0, dur), ylim=(0, 1.05), yticks=[], xlabel="time in the recording (s)",
           title=f"{EXAMPLE}: markers (top), photic zones (orange), cropped ends (grey), used time (bottom)")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(OUT / f"01_event_timeline_{EXAMPLE}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {OUT / f'01_event_timeline_{EXAMPLE}.png'}")

if __name__ == "__main__":
    main()
