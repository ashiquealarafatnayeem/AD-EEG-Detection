import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import FIGURES, PROCESSED

FREQS = np.arange(2.0, 30.01, 0.5)

def fingerprints(freqs, psd):
    idx = [int(np.argmin(np.abs(freqs - f))) for f in FREQS]
    x = np.log10(psd[:, :, idx]).reshape(len(psd), -1)
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)

def identify(a, b, ids_a, ids_b):
    sim = np.corrcoef(a, b)[:len(a), len(a):]
    rows = []
    for i, sid in enumerate(ids_a):
        order = np.argsort(-sim[i])
        own = ids_b.index(sid) if sid in ids_b else None
        rows.append({"subject": sid, "best_match": ids_b[order[0]],
                     "own_rank": int(np.where(order == own)[0][0]) + 1 if own is not None else None,
                     "sim_own": sim[i, own] if own is not None else np.nan,
                     "sim_best": sim[i, order[0]]})
    return pd.DataFrame(rows), sim

def report(title, res, n):
    top1 = (res.own_rank == 1).mean() * 100
    top5 = (res.own_rank <= 5).mean() * 100
    print(f"\n=== {title} (n = {n}) ===")
    print(f"  same ID is the best match : {top1:5.1f} %   (chance {100 / n:.1f} %)")
    print(f"  same ID in the top 5      : {top5:5.1f} %   (chance {500 / n:.1f} %)")
    print(f"  median rank of the same ID: {res.own_rank.median():.0f} of {n}"
          f"   (chance about {n / 2:.0f})")

def main():
    rest = np.load(PROCESSED / "psd_clean.npz")
    eoec = np.load(PROCESSED / "psd_eoec.npz")
    r_ids = [str(s) for s in rest["subjects"]]
    e_ids = [str(s) for s in eoec["subjects"]]

    ok_eo = [i for i in range(len(e_ids)) if np.all(np.isfinite(eoec["psd_eo"][i]))
             and np.all(np.isfinite(eoec["psd_ec_all"][i]))]
    ids = [e_ids[i] for i in ok_eo]
    a = fingerprints(eoec["freqs"], eoec["psd_eo"][ok_eo])
    b = fingerprints(eoec["freqs"], eoec["psd_ec_all"][ok_eo])
    ctrl, _ = identify(a, b, ids, ids)
    report("CONTROL: ds006036 eyes open vs ds006036 eyes closed (same recording)", ctrl, len(ids))

    ok = [i for i in range(len(e_ids)) if np.all(np.isfinite(eoec["psd_ec_all"][i]))
          and e_ids[i] in r_ids]
    ids = [e_ids[i] for i in ok]
    a = fingerprints(eoec["freqs"], eoec["psd_ec_all"][ok])
    b = fingerprints(rest["freqs"], rest["psd"][[r_ids.index(s) for s in ids]])
    res, sim = identify(a, b, ids, ids)
    report("TEST: ds006036 eyes closed vs ds004504 eyes closed (same ID = same person?)",
           res, len(ids))

    num = lambda s: int(s.split("-")[1])
    res["offset"] = [num(m) - num(s) for s, m in zip(res.subject, res.best_match)]
    print("\n  Most common (best match ID - own ID) differences:")
    print("  " + res.offset.value_counts().head(6).to_string().replace("\n", "\n  "))
    pd.set_option("display.width", 200)
    print("\n  Every person (own_rank 1 = recognised):")
    print(res.round(2).to_string(index=False))
    res.to_csv(PROCESSED / "day07_fingerprint.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-.6, vmax=.6)
    ax.set(xlabel="ds004504 person (sorted by ID)", ylabel="ds006036 person (sorted by ID)",
           title="Spectral similarity; a bright diagonal = same people")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(FIGURES / "day07" / "05_fingerprint.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure: {FIGURES / 'day07' / '05_fingerprint.png'}")

if __name__ == "__main__":
    main()
