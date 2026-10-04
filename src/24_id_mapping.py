import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent))
import features as FT
import spectral as S
from eegio import DS_REST, PROCESSED, participants

FREQS = np.arange(2.0, 30.01, 0.5)

def fingerprints(freqs, psd):
    idx = [int(np.argmin(np.abs(freqs - f))) for f in FREQS]
    x = np.log10(psd[:, :, idx]).reshape(len(psd), -1)
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)

def traits(freqs, psd, post):
    alpha = S.band_power(freqs, psd[post], 8.0, 13.0)
    total = S.band_power(freqs, psd[post], FT.FMIN, FT.FMAX)
    theta = S.band_power(freqs, psd[post], 4.0, 8.0)
    p = psd[post].mean(axis=0)
    band = (freqs >= 7.0) & (freqs <= 13.0)
    return {"alpha_freq": np.sum(freqs[band] * p[band]) / np.sum(p[band]),
            "log_total": np.log10(total.mean()),
            "log_alpha": np.log10(alpha.mean()),
            "rel_alpha": (alpha / total).mean(),
            "log_theta_alpha": np.log10(theta.mean() / alpha.mean())}

def main():
    warnings.simplefilter("ignore")
    rest = np.load(PROCESSED / "psd_clean.npz")
    eoec = np.load(PROCESSED / "psd_eoec.npz")
    ch = [str(c) for c in rest["ch_names"]]
    post = [ch.index(c) for c in FT.REGIONS["posterior"]]
    r_ids = [str(s) for s in rest["subjects"]]
    r_grp = dict(zip(r_ids, [str(g) for g in rest["groups"]]))
    e_all = [str(s) for s in eoec["subjects"]]
    e_grp = dict(zip(e_all, [str(g) for g in eoec["groups"]]))
    ok = [i for i in range(len(e_all)) if np.all(np.isfinite(eoec["psd_ec_all"][i]))]
    e_ids = [e_all[i] for i in ok]

    a = fingerprints(eoec["freqs"], eoec["psd_ec_all"][ok])
    b = fingerprints(rest["freqs"], rest["psd"])
    sim = np.corrcoef(a, b)[:len(a), len(a):]
    best_r = sim.argmax(axis=1)
    best_e = sim.argmax(axis=0)
    rows = []
    for i, sid in enumerate(e_ids):
        j = best_r[i]
        s_sorted = np.sort(sim[i])[::-1]
        rows.append({"ds006036": sid, "group_006036": e_grp[sid],
                     "match_004504": r_ids[j], "group_004504": r_grp[r_ids[j]],
                     "offset": int(r_ids[j][4:]) - int(sid[4:]),
                     "sim": sim[i, j], "margin": s_sorted[0] - s_sorted[1],
                     "mutual": bool(best_e[j] == i)})
    m = pd.DataFrame(rows)
    mut = m[m.mutual]
    pd.set_option("display.width", 200)
    print(f"=== 1. Mutual best matches: {len(mut)} of {len(m)} ds006036 recordings ===")
    print(mut.round(2).to_string(index=False))
    print(f"\n  same group in both datasets: {int((mut.group_006036 == mut.group_004504).sum())}"
          f" of {len(mut)}")
    print(f"  same ID (offset 0)         : {int((mut.offset == 0).sum())} of {len(mut)}")

    def trait_table(pairs):
        t6 = pd.DataFrame([traits(eoec["freqs"], eoec["psd_ec_all"][e_all.index(e)], post)
                           for e, _ in pairs])
        t4 = pd.DataFrame([traits(rest["freqs"], rest["psd"][r_ids.index(r)], post)
                           for _, r in pairs])
        return {c: spearmanr(t6[c], t4[c])[0] for c in t6.columns}

    same_id = [(e, e) for e in mut.ds006036 if e in r_ids]
    matched = list(zip(mut.ds006036, mut.match_004504))
    t_same, t_match = trait_table(same_id), trait_table(matched)
    print(f"\n=== 2. Do personal traits agree? (Spearman rho, same {len(matched)} "
          f"ds006036 recordings) ===")
    print(f"{'trait':18s}{'paired by same ID':>20s}{'paired by EEG match':>22s}")
    for c in t_same:
        print(f"{c:18s}{t_same[c]:>20.2f}{t_match[c]:>22.2f}")
    print("  For the same person, expect rho of about 0.7 or more, above all for alpha_freq.")

    meta = participants(DS_REST).set_index("participant_id")
    print("\n=== 3. Does each dataset's EEG fit the age and MMSE listed for its IDs? ===")
    for name, ids, get in [
            ("ds004504", [r for r in r_ids if r in meta.index],
             lambda s: traits(rest["freqs"], rest["psd"][r_ids.index(s)], post)),
            ("ds006036", [e for e in e_ids if e in meta.index],
             lambda s: traits(eoec["freqs"], eoec["psd_ec_all"][e_all.index(s)], post))]:
        t = pd.DataFrame([get(s) for s in ids], index=ids).join(meta[["Age", "MMSE"]])
        ad = t[meta.loc[t.index, "group"] == "AD"]
        print(f"  {name}: alpha_freq vs Age (all, n={len(t)}) rho = "
              f"{spearmanr(t.alpha_freq, t.Age)[0]:+.2f};  "
              f"theta/alpha vs MMSE (AD only, n={len(ad)}) rho = "
              f"{spearmanr(ad.log_theta_alpha, ad.MMSE)[0]:+.2f}")
    print("  Expected if the labels fit: alpha_freq falls with age (rho < 0);")
    print("  theta/alpha falls as MMSE rises (rho < 0). With n this small, a hint only.")

    m.to_csv(PROCESSED / "day07_id_mapping.csv", index=False)
    print(f"\nSaved {PROCESSED / 'day07_id_mapping.csv'}")

if __name__ == "__main__":
    main()
