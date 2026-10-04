import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import DS_PHOTO, DS_REST

def audit(root: Path, label: str) -> set:
    print(f"\n=== {label}  ({root}) ===")
    if not root.exists():
        print("  !! FOLDER DOES NOT EXIST")
        return set()

    subs = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    print(f"  subject folders: {len(subs)}")

    ok, bad = set(), []
    total_bytes = 0
    for s in subs:
        sets = list((root / s / "eeg").glob("*_eeg.set"))
        if len(sets) == 1 and sets[0].stat().st_size > 1_000_000:
            ok.add(s)
            total_bytes += sets[0].stat().st_size
        else:
            sz = sets[0].stat().st_size if sets else 0
            bad.append((s, len(sets), sz))

    print(f"  usable .set files: {len(ok)}")
    print(f"  total size: {total_bytes / 1e9:.2f} GB")
    if bad:
        print(f"  !! {len(bad)} problem subjects (first 5): {bad[:5]}")
    deriv = root / "derivatives"
    print(f"  derivatives present: {deriv.exists()}")
    return ok

rest = audit(DS_REST, "ds004504  (eyes-closed rest)")
photo = audit(DS_PHOTO, "ds006036  (eyes-open photic)")

print("\n=== Cross-dataset alignment ===")
print(f"  in rest only : {sorted(rest - photo)}")
print(f"  in photo only: {sorted(photo - rest)}")
print(f"  in both      : {len(rest & photo)}")
