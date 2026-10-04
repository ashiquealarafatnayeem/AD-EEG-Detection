from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parent))
import filters as F
from eegio import DS_REST, load_raw

CROP_EDGE_S = 10.0
TARGET_SFREQ = 250.0
ICA_METHOD = "infomax"
ICA_N_COMPONENTS = 15
ICA_HIGHPASS = 1.0
ICA_RANDOM_STATE = 97
ICA_DECIM = 3
ICA_MAX_BAD = 3
EOG_PROXY = ["Fp1", "Fp2"]
EPOCH_LEN_S = 4.0
EPOCH_OVERLAP_S = 2.0
REJECT_PTP = 150e-6

BAD_CHANNELS = {
    "sub-067": ["T8"],
}

@dataclass
class PreprocReport:
    subject: str
    n_channels: int = 0
    dur_raw_s: float = 0.0
    dur_cropped_s: float = 0.0
    sfreq_out: float = 0.0
    ica_method_used: str = ""
    ica_excluded: list = field(default_factory=list)
    n_epochs_made: int = 0
    n_epochs_kept: int = 0
    interpolated: list = field(default_factory=list)

    @property
    def drop_pct(self) -> float:
        if not self.n_epochs_made:
            return float("nan")
        return 100.0 * (1.0 - self.n_epochs_kept / self.n_epochs_made)

def _set_filter_info(raw, l_freq, h_freq):
    try:
        with raw.info._unlock():
            raw.info["highpass"] = float(l_freq)
            raw.info["lowpass"] = float(h_freq)
    except Exception:
        pass

def apply_own_filters(raw):
    fs = raw.info["sfreq"]
    sos_notch = F.design_notch(fs=fs)
    sos_bp, _ = F.design_iir_bandpass(fs=fs)

    def _fun(data):
        y = signal.sosfiltfilt(sos_notch, data, axis=-1)
        return signal.sosfiltfilt(sos_bp, y, axis=-1)

    raw.apply_function(_fun, picks="eeg", channel_wise=False)
    _set_filter_info(raw, F.L_FREQ, F.H_FREQ)
    return raw

def _normalise_scores(scores):
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 2:
        arr = np.max(np.abs(arr), axis=0)
    return np.abs(arr)

def detect_blink_components(ica, inst, report):
    inds, scores = [], None
    try:
        inds, raw_scores = ica.find_bads_eog(
            inst, ch_name=EOG_PROXY, threshold=3.0, verbose="ERROR")
        scores = _normalise_scores(raw_scores)
        report.ica_method_used = "find_bads_eog"
    except Exception as exc:
        print(f"    find_bads_eog failed ({type(exc).__name__}); "
              f"using correlation fallback")

    if not inds:
        src = ica.get_sources(inst).get_data()
        proxy = inst.get_data(picks=EOG_PROXY).mean(axis=0)
        sos = signal.butter(4, [1.0, 10.0], btype="band",
                            output="sos", fs=inst.info["sfreq"])
        proxy = signal.sosfiltfilt(sos, proxy)
        r = np.array([abs(np.corrcoef(s, proxy)[0, 1]) for s in src])
        scores = r
        z = (r - r.mean()) / (r.std() + 1e-12)
        inds = list(np.where((z > 3.0) | (r > 0.5))[0])
        report.ica_method_used = "correlation fallback"

    inds = [int(i) for i in inds]
    if scores is not None and len(inds) > ICA_MAX_BAD:
        inds = sorted(inds, key=lambda i: -scores[i])[:ICA_MAX_BAD]
    return sorted(inds), scores

def _snap(raw, ch, store, name):
    if store is None:
        return
    x = raw.get_data(picks=[ch])[0]
    store[name] = (np.arange(len(x)) / raw.info["sfreq"], x)

def preprocess_subject(subject, dataset=DS_REST, make_epochs=True,
                       snapshot_channel=None):
    rep = PreprocReport(subject=subject)
    stages = {} if snapshot_channel else None

    raw = load_raw(subject, dataset)
    rep.n_channels = len(raw.ch_names)
    rep.dur_raw_s = raw.n_times / raw.info["sfreq"]
    _snap(raw, snapshot_channel, stages, "1_raw")

    apply_own_filters(raw)
    _snap(raw, snapshot_channel, stages, "2_filtered")

    raw.crop(tmin=CROP_EDGE_S, tmax=rep.dur_raw_s - CROP_EDGE_S)
    rep.dur_cropped_s = raw.n_times / raw.info["sfreq"]

    raw.resample(TARGET_SFREQ, verbose="ERROR")
    rep.sfreq_out = raw.info["sfreq"]

    bads = BAD_CHANNELS.get(subject, [])
    if bads:
        raw.info["bads"] = list(bads)
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        rep.interpolated = list(bads)
        print(f"    {subject}: interpolated {bads}")

    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    _snap(raw, snapshot_channel, stages, "3_referenced")

    raw_fit = raw.copy().filter(l_freq=ICA_HIGHPASS, h_freq=None,
                                verbose="ERROR")
    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        fit_params=dict(extended=True) if ICA_METHOD == "infomax" else None,
        max_iter=500,
        random_state=ICA_RANDOM_STATE,
    )
    ica.fit(raw_fit, decim=ICA_DECIM, verbose="ERROR")

    bad, _ = detect_blink_components(ica, raw_fit, rep)
    ica.exclude = bad
    rep.ica_excluded = bad
    ica.apply(raw, verbose="ERROR")
    _snap(raw, snapshot_channel, stages, "4_ica_cleaned")

    epochs = None
    if make_epochs:
        epochs = mne.make_fixed_length_epochs(
            raw, duration=EPOCH_LEN_S, overlap=EPOCH_OVERLAP_S,
            preload=True, verbose="ERROR")
        rep.n_epochs_made = len(epochs)
        epochs.drop_bad(reject=dict(eeg=REJECT_PTP), verbose="ERROR")
        rep.n_epochs_kept = len(epochs)

    return raw, epochs, ica, rep, stages
