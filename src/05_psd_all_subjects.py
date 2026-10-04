import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_REST, PROCESSED, load_raw, participants

CROP_EDGE_S = 10.0
N_FFT = 2000
FMIN, FMAX = 0.5, 45.0

def main() -> None:
    df = participants(DS_REST)
    psds, freqs, kept = [], None, []

    for sid in tqdm(df.participant_id, desc="PSD"):
        raw = load_raw(sid, DS_REST)

        dur = raw.n_times / raw.info["sfreq"]
        raw.crop(tmin=CROP_EDGE_S, tmax=dur - CROP_EDGE_S)

        raw.notch_filter(freqs=[50.0], verbose="ERROR")
        raw.filter(l_freq=FMIN, h_freq=FMAX, verbose="ERROR")
        raw.set_eeg_reference("average", verbose="ERROR")

        spec = raw.compute_psd(method="welch", fmin=FMIN, fmax=FMAX,
                               n_fft=N_FFT, n_per_seg=N_FFT,
                               n_overlap=N_FFT // 2, verbose="ERROR")
        data = spec.get_data()

        if freqs is None:
            freqs = spec.freqs
            ch_names = raw.ch_names
        else:
            assert np.allclose(freqs, spec.freqs), f"freq mismatch at {sid}"
            assert raw.ch_names == ch_names, f"channel order differs at {sid}"

        psds.append(data)
        kept.append(sid)

    psds = np.stack(psds)
    print("PSD array shape:", psds.shape)
    assert np.isfinite(psds).all(), "non-finite values in PSD"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "psd_rest.npz"
    np.savez_compressed(out, psds=psds, freqs=freqs,
                        subjects=np.array(kept),
                        ch_names=np.array(ch_names))
    print("Saved", out)

if __name__ == "__main__":
    main()
