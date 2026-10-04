import sys
import time
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import photic as PH
import preprocess as P
import spectral as S
from eegio import DS_PHOTO, EXCLUDED, PROCESSED, participants

CACHE = PROCESSED / "eoec"
FMAX = 45.0
N_FREQS = int(FMAX * PH.WIN_S) + 1
STATES = ["eo", "ec_near", "ec_all"]

def cut(data, fs, starts_s):
    n = int(round(PH.WIN_S * fs))
    starts_s = np.asarray(starts_s)
    idx = np.round(starts_s * fs).astype(int)
    ok = (idx >= 0) & (idx + n <= data.shape[1])
    if not ok.any():
        return np.empty((0, data.shape[0], n)), starts_s[ok]
    return np.stack([data[:, i:i + n] for i in idx[ok]]), starts_s[ok]

def spectrum(windows, starts_s, fs, state):
    if len(windows):
        keep = (windows.max(axis=-1) - windows.min(axis=-1) <= P.REJECT_PTP).all(axis=1)
        windows, starts_s = windows[keep], starts_s[keep]
    ratio = PH.variance_ratio(starts_s, fs) if state == "eo" else np.nan
    enough = (ratio <= PH.MAX_EO_VAR_RATIO) if state == "eo" else len(windows) >= PH.MIN_EC_WINDOWS
    if not enough:
        return np.full((windows.shape[1], N_FREQS), np.nan), len(windows), ratio
    f, psd = S.welch_epochs(windows, fs)
    return psd[:, :N_FREQS], len(windows), ratio

def process(sid):
    raw, _, _, rep, _ = P.preprocess_subject(sid, dataset=DS_PHOTO, make_epochs=False)
    fs = raw.info["sfreq"]
    data = raw.get_data()
    pl = PH.plan(PH.read_events(sid), rep.dur_raw_s, P.CROP_EDGE_S)

    out, info = {}, {"n_eo_stretches": len(pl["eo_stretches"])}
    for s in STATES:
        w, starts = cut(data, fs, pl[s + "_starts"] - P.CROP_EDGE_S)
        out["psd_" + s], kept, ratio = spectrum(w, starts, fs, s)
        info[f"n_{s}_windows"], info[f"n_{s}"] = len(w), kept
        if s == "eo":
            out["var_eo"] = ratio
    np.savez(CACHE / f"{sid}.npz", **out, **info,
             freqs=np.arange(N_FREQS) / PH.WIN_S, ch_names=np.array(raw.ch_names),
             ica_excluded=np.array(rep.ica_excluded),
             interpolated=np.array(rep.interpolated, dtype=str))
    info["n_ica_removed"] = len(rep.ica_excluded)
    info["var_eo"] = float(out["var_eo"])
    return info

def load_info(path):
    z = np.load(path)
    info = {k: int(z[k]) for k in z.files if k.startswith("n_")}
    info["var_eo"] = float(z["var_eo"])
    info["n_ica_removed"] = len(z["ica_excluded"])
    return info

def main():
    mne.set_log_level("ERROR")
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = participants(DS_PHOTO)
    todo = [(s, g) for s, g in zip(meta.participant_id, meta.group) if s not in EXCLUDED]
    rows = []
    for i, (sid, grp) in enumerate(todo, 1):
        path = CACHE / f"{sid}.npz"
        t0 = time.perf_counter()
        try:
            if path.exists():
                info, status = load_info(path), "cached"
            else:
                info, status = process(sid), "ok"
        except Exception as exc:
            print(f"[{i:2d}/{len(todo)}] {sid} ({grp:3s}) FAILED - {type(exc).__name__}: {exc}")
            with open(PROCESSED / "day07_eoec_errors.txt", "a", encoding="utf-8") as fh:
                fh.write(f"===== {sid} =====\n{traceback.format_exc()}\n")
            rows.append(dict(subject=sid, group=grp, status="FAILED"))
            continue
        rows.append(dict(subject=sid, group=grp, status=status, **info))
        print(f"[{i:2d}/{len(todo)}] {sid} ({grp:3s}) {status:6s} "
              f"open {info['n_eo']:3d}/{info['n_eo_windows']:3d} (ratio {info['var_eo']:4.2f})   "
              f"closed-near {info['n_ec_near']:4d}/{info['n_ec_near_windows']:4d}   "
              f"closed-all {info['n_ec_all']:4d}   {time.perf_counter() - t0:5.1f} s")

    log = pd.DataFrame(rows)
    log.to_csv(PROCESSED / "day07_eoec_log.csv", index=False)
    ok = log[log.status != "FAILED"].copy()

    zs = [np.load(CACHE / f"{s}.npz") for s in ok.subject]
    np.savez(PROCESSED / "psd_eoec.npz",
             subjects=ok.subject.to_numpy(dtype=str), groups=ok.group.to_numpy(dtype=str),
             freqs=zs[0]["freqs"], ch_names=zs[0]["ch_names"],
             **{"psd_" + s: np.stack([z["psd_" + s] for z in zs]) for s in STATES},
             **{"n_" + s: np.array([int(z["n_" + s]) for z in zs]) for s in STATES},
             var_eo=np.array([float(z["var_eo"]) for z in zs]))

    ok["qualifies"] = (ok.var_eo <= PH.MAX_EO_VAR_RATIO) & (ok.n_ec_near >= PH.MIN_EC_WINDOWS)
    print("\n" + "=" * 72)
    print(f"Finished: {len(ok)} ok, {int((log.status == 'FAILED').sum())} failed")
    print(f"\nClean windows per person AFTER the 150 uV rejection (median):")
    print(ok.groupby("group")[["n_eo", "n_ec_near", "n_ec_all"]].median().to_string())
    print(f"\nEnough data (eyes-open variance ratio <= {PH.MAX_EO_VAR_RATIO} and "
          f">= {PH.MIN_EC_WINDOWS} nearby closed clean windows):")
    q = ok.groupby("group").qualifies.agg(["sum", "count"])
    q.columns = ["qualify", "of"]
    print(q.to_string())
    lost = ok[~ok.qualifies]
    if len(lost):
        print("\nReactivity will be left empty for:")
        print(lost[["subject", "group", "n_eo_stretches", "n_eo_windows", "n_eo",
                    "var_eo", "n_ec_near"]].round(2).to_string(index=False))
    print(f"\nSaved {PROCESSED / 'psd_eoec.npz'}")

if __name__ == "__main__":
    main()
