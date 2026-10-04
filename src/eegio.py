from pathlib import Path

import mne
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
FIGURES = PROJECT_ROOT / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"

DS_REST = RAW / "ds004504"
DS_PHOTO = RAW / "ds006036"

OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

GROUP_NAMES = {"A": "AD", "F": "FTD", "C": "CN"}

EXCLUDED = {
    "sub-086": "only 16 of 316 epochs artefact-free (under 2 min); 5 channels faulty",
}

def participants(dataset: Path = DS_REST) -> pd.DataFrame:
    df = pd.read_csv(dataset / "participants.tsv", sep="\t")
    df.columns = df.columns.str.strip()
    df["group"] = df["Group"].map(GROUP_NAMES)
    if df["group"].isna().any():
        bad = df.loc[df["group"].isna(), "Group"].unique()
        raise ValueError(f"Unmapped group code(s): {bad}")
    return df

def load_raw(subject: str, dataset: Path = DS_REST,
             preload: bool = True) -> mne.io.BaseRaw:
    eeg_dir = dataset / subject / "eeg"
    matches = sorted(eeg_dir.glob("*_eeg.set"))
    if not matches:
        raise FileNotFoundError(f"No *_eeg.set file found in {eeg_dir}")

    raw = mne.io.read_raw_eeglab(matches[0], preload=preload,
                                 verbose="ERROR")
    raw.rename_channels(
        {old: new for old, new in OLD_TO_NEW.items() if old in raw.ch_names}
    )
    raw.set_montage("standard_1020", match_case=False)
    return raw
