import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import features as FT
import preprocess as P
import spectral as S
from eegio import DS_REST, PROCESSED, load_raw

CACHE = PROCESSED / "ablation"
EPO_DIR = PROCESSED / "epochs"
LEVELS = ["L0_raw", "L1_filtered", "L2_car", "L2r_reject"]
FMAX = 45.0
CHECK_SUBJECTS = ["sub-001", "sub-037", "sub-066"]

def epochs_of(raw):
    return mne.make_fixed_length_epochs(raw, duration=P.EPOCH_LEN_S, overlap=P.EPOCH_OVERLAP_S,
                                        preload=True, verbose="ERROR")

def region_features(data, fs, ch_names):
    f, p = S.welch_epochs(data, fs)
    keep = f <= FMAX
    freqs, psd = f[keep], p[:, keep]
    feats = FT.channel_features(freqs, psd, data, fs)
    post = [ch_names.index(c) for c in FT.REGIONS["posterior"]]
    peak = FT.alpha_peak(freqs, psd[post].mean(axis=0))
    row = {f"{k}__{r}": FT.region_average(v, ch_names, r) for k, v in feats.items()
           for r in FT.REGIONS}
    row.update({"alpha_present": peak["present"], "alpha_iaf": peak["iaf"],
                "alpha_height": peak["height"]})
    return row

def self_test(reference):
    print("=== Checkpoint: rebuild Day 6 features from Day 5 epochs ===")
    worst = 0.0
    check = [s for s in CHECK_SUBJECTS if s in reference.index] or list(reference.index[:3])
    for sid in check:
        ep = mne.read_epochs(EPO_DIR / f"{sid}_epo.fif", preload=True, verbose="ERROR")
        row = region_features(ep.get_data(), ep.info["sfreq"], list(ep.ch_names))
        ref = reference.loc[sid]
        diffs = [abs(row[c] - ref[c]) / (abs(ref[c]) + 1e-12) for c in row
                 if np.isfinite(row[c]) and np.isfinite(ref[c])]
        worst = max(worst, max(diffs))
        print(f"  {sid}: {len(diffs)} features compared, largest relative difference {max(diffs):.1e}")
    if worst > 1e-6:
        raise SystemExit("  CHECK: the feature code does not reproduce Day 6. Stop and paste this.")
    print("  PASS: identical to Day 6, so differences between levels come from the cleaning only.\n")

def process(sid):
    raw0 = load_raw(sid, DS_REST)
    dur = raw0.n_times / raw0.info["sfreq"]
    out, counts = {}, {}

    r0 = raw0.copy().crop(tmin=P.CROP_EDGE_S, tmax=dur - P.CROP_EDGE_S)
    r0.resample(P.TARGET_SFREQ, verbose="ERROR")
    e = epochs_of(r0)
    out["L0_raw"] = region_features(e.get_data(), e.info["sfreq"], list(e.ch_names))
    counts["L0_raw"] = len(e)

    r1 = P.apply_own_filters(raw0.copy())
    r1.crop(tmin=P.CROP_EDGE_S, tmax=dur - P.CROP_EDGE_S)
    r1.resample(P.TARGET_SFREQ, verbose="ERROR")
    e = epochs_of(r1)
    out["L1_filtered"] = region_features(e.get_data(), e.info["sfreq"], list(e.ch_names))
    counts["L1_filtered"] = len(e)

    r2 = r1.copy()
    bads = P.BAD_CHANNELS.get(sid, [])
    if bads:
        r2.info["bads"] = list(bads)
        r2.interpolate_bads(reset_bads=True, verbose="ERROR")
    r2.set_eeg_reference("average", projection=False, verbose="ERROR")
    e = epochs_of(r2)
    out["L2_car"] = region_features(e.get_data(), e.info["sfreq"], list(e.ch_names))
    counts["L2_car"] = len(e)

    e.drop_bad(reject=dict(eeg=P.REJECT_PTP), verbose="ERROR")
    counts["L2r_reject"] = len(e)
    if len(e) >= 1:
        out["L2r_reject"] = region_features(e.get_data(), e.info["sfreq"], list(e.ch_names))
    else:
        out["L2r_reject"] = {k: np.nan for k in out["L2_car"]}
    return out, counts

def main():
    mne.set_log_level("ERROR")
    CACHE.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(PROCESSED / "features_region.csv").set_index("participant_id")
    self_test(reference)

    subjects = list(reference.index)
    rows = {lv: [] for lv in LEVELS}
    log = []
    t_start = time.perf_counter()
    for i, sid in enumerate(subjects, 1):
        path = CACHE / f"{sid}.npz"
        t0 = time.perf_counter()
        if path.exists():
            z = np.load(path, allow_pickle=True)
            out, counts, status = z["out"].item(), z["counts"].item(), "cached"
        else:
            out, counts = process(sid)
            np.savez(path, out=np.array(out, dtype=object), counts=np.array(counts, dtype=object))
            status = "ok"
        for lv in LEVELS:
            rows[lv].append({"participant_id": sid, **out[lv]})
        log.append({"participant_id": sid, "group": reference.loc[sid, "group"],
                    **{f"epochs_{lv}": counts[lv] for lv in LEVELS},
                    "epochs_L3_full": int(reference.loc[sid, "n_epochs"])})
        print(f"[{i:2d}/{len(subjects)}] {sid} ({reference.loc[sid, 'group']:3s}) {status:6s} "
              f"epochs kept without ICA {counts['L2r_reject']:3d}/{counts['L2_car']:3d}   "
              f"{time.perf_counter() - t0:5.1f} s  (total {(time.perf_counter() - t_start) / 60:.1f} min)")

    meta_cols = ["participant_id", "group", "age", "sex", "n_epochs"]
    meta = reference.reset_index()[meta_cols]
    for lv in LEVELS:
        df = meta.drop(columns="n_epochs").merge(pd.DataFrame(rows[lv]), on="participant_id")
        df.insert(4, "n_epochs", [r[f"epochs_{lv}"] for r in log])
        df = df[reference.reset_index().columns]
        df.to_csv(PROCESSED / f"features_region_{lv}.csv", index=False)
    lg = pd.DataFrame(log)
    lg.to_csv(PROCESSED / "day09_ablation_log.csv", index=False)

    print("\n=== Epochs per person (median) at each level ===")
    print(lg.groupby("group")[[c for c in lg.columns if c.startswith("epochs_")]].median().to_string())
    few = lg[lg.epochs_L2r_reject < 20]
    print(f"\nPeople with fewer than 20 epochs left after rejection WITHOUT ICA: {len(few)}")
    if len(few):
        print(few[["participant_id", "group", "epochs_L2r_reject", "epochs_L3_full"]].to_string(index=False))
    print(f"\nSaved features_region_<level>.csv for {len(LEVELS)} levels "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)")

if __name__ == "__main__":
    main()
