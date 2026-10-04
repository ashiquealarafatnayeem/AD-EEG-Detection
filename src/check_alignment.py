import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_PHOTO, DS_REST, participants

rest = participants(DS_REST)
photo = participants(DS_PHOTO)

print("rest  columns:", list(rest.columns))
print("photo columns:", list(photo.columns))
print("rest  n:", len(rest), " photo n:", len(photo))

merged = rest.merge(photo, on="participant_id", suffixes=("_rest", "_photo"))
print("matched participants:", len(merged))

for col in ["Age", "Gender", "Group"]:
    a, b = f"{col}_rest", f"{col}_photo"
    if a in merged and b in merged:
        mismatch = merged.loc[merged[a] != merged[b], ["participant_id", a, b]]
        print(f"{col}: {len(mismatch)} mismatches")
        if len(mismatch):
            print(mismatch.to_string(index=False))

sub = "sub-001"
files = sorted((DS_PHOTO / sub / "eeg").iterdir())
print(f"\n{sub} files in ds006036:")
for f in files:
    print("  ", f.name, f"({f.stat().st_size / 1e6:.1f} MB)")
