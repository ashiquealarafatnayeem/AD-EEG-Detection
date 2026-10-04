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
from eegio import PROCESSED

CACHE = PROCESSED / "robust"
FMAX = 45.0
CHECK_SUBJECTS = ["sub-001", "sub-037", "sub-066"]
VARIANTS = {
    "base": (4.0, 2.0, 150e-6),
    "E2":   (2.0, 1.0, 150e-6),
    "R100": (4.0, 2.0, 100e-6),
    "R200": (4.0, 2.0, 200e-6),
}
TOL = 1e-3

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

def process(sid):
    raw = P.preprocess_subject(sid, make_epochs=False)[0]
    out, counts = {}, {}
    for name, (length, overlap, limit) in VARIANTS.items():
        ep = mne.make_fixed_length_epochs(raw, duration=length, overlap=overlap,
                                          preload=True, verbose="ERROR")
        made = len(ep)
        ep.drop_bad(reject=dict(eeg=limit), verbose="ERROR")
        counts[name] = (made, len(ep))
        if len(ep) == 0:
            out[name] = None
            continue
        data = ep.get_data().astype(np.float32).astype(np.float64)
        out[name] = region_features(data, ep.info["sfreq"], list(ep.ch_names))
    return out, counts

def compare(row, ref):
    d = [abs(row[c] - ref[c]) / (abs(ref[c]) + 1e-12) for c in row
         if np.isfinite(row[c]) and np.isfinite(ref[c])]
    return max(d) if d else np.nan

def main():
    mne.set_log_level("ERROR")
    CACHE.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(PROCESSED / "features_region.csv").set_index("participant_id")
    subjects = list(reference.index)
    check = [s for s in CHECK_SUBJECTS if s in reference.index] or subjects[:3]
    order = check + [s for s in subjects if s not in check]

    results, log = {}, []
    t_start = time.perf_counter()
    for i, sid in enumerate(order, 1):
        path = CACHE / f"{sid}.npz"
        t0 = time.perf_counter()
        if path.exists():
            z = np.load(path, allow_pickle=True)
            out, counts, status = z["out"].item(), z["counts"].item(), "cached"
        else:
            out, counts = process(sid)
            np.savez(path, out=np.array(out, dtype=object), counts=np.array(counts, dtype=object))
            status = "ok"
        results[sid] = out
        ref = reference.loc[sid]
        diff = compare(out["base"], ref) if out["base"] is not None else np.nan
        same_n = counts["base"][1] == int(ref["n_epochs"])
        log.append({"participant_id": sid, "group": ref["group"], "base_max_rel_diff": diff,
                    "base_same_epochs": same_n,
                    **{f"made_{v}": counts[v][0] for v in VARIANTS},
                    **{f"kept_{v}": counts[v][1] for v in VARIANTS}})
        print(f"[{i:2d}/{len(order)}] {sid} ({ref['group']:3s}) {status:6s} kept: "
              f"base {counts['base'][1]:3d}  E2 {counts['E2'][1]:3d}  R100 {counts['R100'][1]:3d}  "
              f"R200 {counts['R200'][1]:3d}  {time.perf_counter() - t0:4.0f} s "
              f"({(time.perf_counter() - t_start) / 60:.1f} min)")

        if i == len(check):
            print("\n=== Checkpoint: does the rerun reproduce Day 5/6? ===")
            ok = True
            for r in log:
                print(f"  {r['participant_id']}: same epochs as Day 5: {r['base_same_epochs']}   "
                      f"largest relative difference {r['base_max_rel_diff']:.1e}")
                ok &= bool(r["base_same_epochs"]) and r["base_max_rel_diff"] <= TOL
            if not ok:
                raise SystemExit("  CHECK: the rerun does not reproduce Day 5/6. Stop and paste this output.")
            print("  PASS: same cleaned data as Day 5, so the variants differ only in epoching/rejection.\n")

    lg = pd.DataFrame(log).set_index("participant_id").loc[subjects].reset_index()
    lg.to_csv(PROCESSED / "day10_robust_log.csv", index=False)
    cols = reference.reset_index().columns
    for v in [v for v in VARIANTS if v != "base"]:
        rows = []
        for sid in subjects:
            feats = results[sid][v] or {c: np.nan for c in results[sid]["base"]}
            rows.append({"participant_id": sid, **feats})
        df = reference.reset_index()[["participant_id", "group", "age", "sex"]].merge(
            pd.DataFrame(rows), on="participant_id")
        df.insert(4, "n_epochs", lg[f"kept_{v}"].to_numpy())
        df[cols].to_csv(PROCESSED / f"features_region_{v}.csv", index=False)

    bad = lg[(~lg.base_same_epochs) | (lg.base_max_rel_diff > TOL)]
    print(f"\n=== Whole cohort: rerun vs Day 6 ===")
    print(f"  largest relative difference over all {len(lg)} people: {lg.base_max_rel_diff.max():.1e}")
    print(f"  people not reproduced (epoch count or features): {len(bad)}")
    if len(bad):
        print(bad[["participant_id", "group", "kept_base", "base_max_rel_diff"]].to_string(index=False))
    print("\n=== Epochs kept per person (median) ===")
    show = lg.groupby("group")[[f"kept_{v}" for v in VARIANTS]].median()
    show.columns = ["4 s, 150 uV", "2 s, 150 uV", "4 s, 100 uV", "4 s, 200 uV"]
    print(show.to_string())
    lg["pct_kept_R100"] = 100 * lg.kept_R100 / lg.made_R100
    print(f"\nWith 100 uV: median {lg.pct_kept_R100.median():.0f} % of epochs kept; "
          f"people with fewer than 20 epochs: {int((lg.kept_R100 < 20).sum())}")
    print(f"Saved features_region_E2.csv, _R100.csv, _R200.csv "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)")

if __name__ == "__main__":
    main()
