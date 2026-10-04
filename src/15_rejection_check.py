import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import preprocess as P

SUBJECTS = ["sub-086", "sub-067", "sub-026", "sub-030", "sub-085", "sub-048"]

def main():
    mne.set_log_level("ERROR")
    for sid in SUBJECTS:
        raw, _, _, _, _ = P.preprocess_subject(sid, make_epochs=False)
        ep = mne.make_fixed_length_epochs(raw, duration=P.EPOCH_LEN_S,
                                          overlap=P.EPOCH_OVERLAP_S,
                                          preload=True, verbose="ERROR")
        x = ep.get_data()
        p2p = x.max(axis=-1) - x.min(axis=-1)
        over = p2p > P.REJECT_PTP

        t = pd.DataFrame({
            "channel": ep.ch_names,
            "median_p2p_uV": np.median(p2p, axis=0) * 1e6,
            "pct_epochs_over": over.mean(axis=0) * 100,
        }).sort_values("pct_epochs_over", ascending=False)

        worst = t["channel"].iloc[0]
        keep_without = ~np.delete(over, ep.ch_names.index(worst), axis=1).any(axis=1)

        print(f"\n=== {sid}: {over.any(axis=1).sum()} of {len(over)} epochs "
              f"over {P.REJECT_PTP * 1e6:.0f} uV ===")
        print(t.head(5).round(1).to_string(index=False))
        print(f"  ignoring {worst}: {keep_without.sum()} of {len(keep_without)} "
              f"epochs would be kept")

if __name__ == "__main__":
    main()
