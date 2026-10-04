import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_REST, PROCESSED, load_raw, participants

EXPECTED = {"AD": 36, "FTD": 23, "CN": 29}

def main() -> None:
    df = participants(DS_REST)
    print(f"participants.tsv rows: {len(df)}")

    counts = df["group"].value_counts().to_dict()
    print("Group counts:", counts)
    assert counts == EXPECTED, f"Expected {EXPECTED}, got {counts}"

    table = df.groupby("group").agg(
        n=("participant_id", "count"),
        age_mean=("Age", "mean"),
        age_sd=("Age", "std"),
        mmse_mean=("MMSE", "mean"),
        mmse_sd=("MMSE", "std"),
        n_female=("Gender", lambda s: (s == "F").sum()),
    ).round(2)
    print("\n=== Table 1: Demographics ===")
    print(table)

    rows = []
    missing = []
    for sid in df["participant_id"]:
        try:
            raw = load_raw(sid, DS_REST, preload=False)
        except FileNotFoundError:
            missing.append(sid)
            continue
        rows.append({
            "participant_id": sid,
            "sfreq": raw.info["sfreq"],
            "n_channels": len(raw.ch_names),
            "duration_s": round(raw.n_times / raw.info["sfreq"], 1),
        })

    if missing:
        print(f"\n!! {len(missing)} subjects not downloaded yet: "
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")

    manifest = df.merge(pd.DataFrame(rows), on="participant_id", how="left")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "manifest.csv"
    manifest.to_csv(out, index=False)
    print(f"\nWrote {out}")

    if not missing:
        print("\nRecording length by group (minutes):")
        desc = manifest.groupby("group")["duration_s"].describe()
        desc.loc[:, "mean":] = desc.loc[:, "mean":] / 60
        print(desc.round(2))

if __name__ == "__main__":
    main()
