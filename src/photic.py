import re

import numpy as np
import pandas as pd

import spectral as S
from eegio import DS_PHOTO

PHOTIC_RE = re.compile(r"\d+(\.\d+)?\s*hz|photo|\bhv\b", re.IGNORECASE)
ARTEFACT_WORDS = ["saccade", "gaze", "eye movement", "swallow", "speech",
                  "head movement", "muscle", "mouth", "tinagma"]
ZONE_GAP_S = 15.0
ZONE_PAD_BEFORE_S = 1.0
ZONE_PAD_AFTER_S = 3.0
EYE_CHANGE_PAD_S = 0.5
ARTEFACT_PAD_S = 1.0
EQUIP_PAD_S = 2.0
GRID_STEP_S = 0.05
WIN_S = 2.0
STEP_S = 0.5
EC_NEAR_S = 30.0
MAX_EO_VAR_RATIO = 0.5
MIN_EC_WINDOWS = 20

def classify(value):
    v = str(value).lower()
    if "eye" in v and "open" in v:
        return "open"
    if "eye" in v and "clos" in v:
        return "closed"
    if re.search(r"\bcal\b", v):
        return "calib"
    if v.startswith("pat") and "eeg" in v:
        return "eeg_mode"
    if "reset" in v:
        return "equipment"
    if any(w in v for w in ARTEFACT_WORDS):
        return "artefact"
    if PHOTIC_RE.search(str(value)):
        return "photic"
    return "other"

def read_events(subject):
    hits = sorted((DS_PHOTO / subject / "eeg").glob("*_events.tsv"))
    if not hits:
        raise FileNotFoundError(f"no events.tsv for {subject}")
    df = pd.read_csv(hits[0], sep="\t")
    df.columns = df.columns.str.strip()
    col = next((c for c in ["value", "trial_type"] if c in df.columns), df.columns[-1])
    ev = pd.DataFrame({"onset": pd.to_numeric(df["onset"], errors="coerce"),
                       "value": df[col].astype(str).str.replace("\x1a", "").str.strip()})
    ev = ev.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    ev["kind"] = ev["value"].map(classify)
    return ev

def photic_zones(ev):
    t = ev.loc[ev.kind == "photic", "onset"].to_numpy()
    if len(t) == 0:
        return []
    zones, start, prev = [], t[0], t[0]
    for x in t[1:]:
        if x - prev > ZONE_GAP_S:
            zones.append((start, prev))
            start = x
        prev = x
    zones.append((start, prev))
    return [(a - ZONE_PAD_BEFORE_S, b + ZONE_PAD_AFTER_S) for a, b in zones]

def calibration_zones(ev):
    zones = []
    back = ev.loc[ev.kind == "eeg_mode", "onset"].to_numpy()
    for t0 in ev.loc[ev.kind == "calib", "onset"]:
        later = back[back > t0]
        zones.append((t0, later[0] if len(later) else np.inf))
    return zones

def eye_state(times, ev):
    eye = ev[ev.kind.isin(["open", "closed"])]
    if len(eye) == 0:
        return np.full(len(times), "unknown", dtype=object)
    idx = np.searchsorted(eye.onset.to_numpy(), times, side="right") - 1
    kinds = eye.kind.to_numpy()
    return np.where(idx >= 0, kinds[np.clip(idx, 0, None)], "unknown").astype(object)

def usable_masks(times, ev):
    clean = np.ones(len(times), dtype=bool)
    for a, b in photic_zones(ev):
        clean &= ~((times >= a) & (times <= b))
    for t0 in ev.loc[ev.kind.isin(["open", "closed"]), "onset"]:
        clean &= ~(np.abs(times - t0) <= EYE_CHANGE_PAD_S)
    for t0 in ev.loc[ev.kind == "artefact", "onset"]:
        clean &= ~(np.abs(times - t0) <= ARTEFACT_PAD_S)
    for t0 in ev.loc[ev.kind.isin(["equipment", "eeg_mode"]), "onset"]:
        clean &= ~(np.abs(times - t0) <= EQUIP_PAD_S)
    for a, b in calibration_zones(ev):
        clean &= ~((times >= a - EQUIP_PAD_S) & (times <= b + EQUIP_PAD_S))
    state = eye_state(times, ev)
    return clean & (state == "open"), clean & (state == "closed"), state

def intervals(times, mask):
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(times[s], times[e] + GRID_STEP_S) for s, e in zip(starts, ends)]

def window_starts(stretches, win_s=WIN_S, step_s=STEP_S):
    out = []
    for a, b in stretches:
        s = a
        while s + win_s <= b + 1e-9:
            out.append(s)
            s += step_s
    return np.array(out)

def time_grid(duration_s, crop_s):
    return np.arange(crop_s, duration_s - crop_s + 1e-9, GRID_STEP_S)

def near(times, stretches, dist_s):
    m = np.zeros(len(times), dtype=bool)
    for a, b in stretches:
        m |= (times >= a - dist_s) & (times <= b + dist_s)
    return m

def plan(ev, duration_s, crop_s):
    t = time_grid(duration_s, crop_s)
    open_m, closed_m, state = usable_masks(t, ev)
    eo = intervals(t, open_m)
    eo = [(a, b) for a, b in eo if b - a >= WIN_S]
    near_m = closed_m & near(t, eo, EC_NEAR_S)
    out = {"times": t, "open": open_m, "closed": closed_m, "state": state,
           "near": near_m, "eo_stretches": eo}
    for name, m in [("ec_near", near_m), ("ec_all", closed_m)]:
        out[name + "_stretches"] = intervals(t, m)
    for name in ["eo", "ec_near", "ec_all"]:
        out[name + "_starts"] = window_starts(out[name + "_stretches"])
    return out

def variance_ratio(starts_s, fs=250.0):
    k = len(starts_s)
    if k == 0:
        return np.inf
    n = int(round(WIN_S * fs))
    w = S.hann(n)
    denom = np.sum(w ** 2) ** 2
    idx = np.round(np.asarray(starts_s) * fs).astype(int)
    lags = np.abs(idx[:, None] - idx[None, :])
    rho = {lag: np.dot(w[:n - lag], w[lag:]) ** 2 / denom for lag in np.unique(lags[lags < n])}
    total = sum(rho[lag] for lag in lags[lags < n])
    return round(total / k ** 2, 6)
