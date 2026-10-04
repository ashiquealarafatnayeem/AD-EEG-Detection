import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent))
import features as FT
import spectral as S
from eegio import PROCESSED

ALPHA = (8.0, 13.0)
IAF_BAND = (7.0, 13.0)

def measures(freqs, psd, post):
    if not np.all(np.isfinite(psd)):
        return None
    alpha_ch = S.band_power(freqs, psd, *ALPHA)
    total_ch = S.band_power(freqs, psd, FT.FMIN, FT.FMAX)
    theta_ch = S.band_power(freqs, psd, 4.0, 8.0)
    p = psd[post].mean(axis=0)
    band = (freqs >= IAF_BAND[0]) & (freqs <= IAF_BAND[1])
    return {"log_total": np.log10(total_ch[post].mean()),
            "log_alpha": np.log10(alpha_ch[post].mean()),
            "rel_alpha": (alpha_ch[post] / total_ch[post]).mean(),
            "log_theta_alpha": np.log10(theta_ch[post].mean() / alpha_ch[post].mean()),
            "alpha_freq": np.sum(freqs[band] * p[band]) / np.sum(p[band]),
            "alpha_map": np.log10(alpha_ch),
            "post_share": alpha_ch[post].sum() / alpha_ch.sum()}

def main():
    warnings.simplefilter("ignore")
    rest = np.load(PROCESSED / "psd_clean.npz")
    eoec = np.load(PROCESSED / "psd_eoec.npz")
    ch = [str(c) for c in rest["ch_names"]]
    post = [ch.index(c) for c in FT.REGIONS["posterior"]]
    rest_subs = [str(s) for s in rest["subjects"]]
    eoec_subs = [str(s) for s in eoec["subjects"]]

    rows = []
    for i, sid in enumerate(rest_subs):
        if sid not in eoec_subs:
            continue
        j = eoec_subs.index(sid)
        r = measures(rest["freqs"], rest["psd"][i], post)
        for key in ["ec_all", "ec_near"]:
            e = measures(eoec["freqs"], eoec["psd_" + key][j], post)
            if r is None or e is None:
                continue
            rows.append({"subject": sid, "group": str(rest["groups"][i]), "which": key,
                         **{f"{m}_rest": r[m] for m in r if m != "alpha_map"},
                         **{f"{m}_photo": e[m] for m in e if m != "alpha_map"},
                         "map_corr": np.corrcoef(r["alpha_map"], e["alpha_map"])[0, 1]})
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "day07_check_b_diagnosis.csv", index=False)
    pd.set_option("display.width", 200)

    names = [("log_total", "1. total power, log (signal size)"),
             ("log_alpha", "   alpha power, log (the Check B measure)"),
             ("rel_alpha", "1. relative alpha (no signal size)"),
             ("log_theta_alpha", "1. theta/alpha ratio (no signal size)"),
             ("alpha_freq", "4. alpha frequency (personal trait)")]
    for key in ["ec_all", "ec_near"]:
        d = df[df.which == key]
        print(f"\n=== ds004504 eyes closed vs ds006036 eyes closed ({key}), n = {len(d)} ===")
        print(f"{'measure':42s}{'rho all':>9s}{'rho AD':>9s}{'rho CN':>9s}"
              f"{'median ds004504':>17s}{'median ds006036':>17s}")
        for m, label in names:
            out = [spearmanr(x[f"{m}_rest"], x[f"{m}_photo"])[0]
                   for x in [d, d[d.group == "AD"], d[d.group == "CN"]]]
            print(f"{label:42s}{out[0]:>9.2f}{out[1]:>9.2f}{out[2]:>9.2f}"
                  f"{d[f'{m}_rest'].median():>17.3f}{d[f'{m}_photo'].median():>17.3f}")

    d = df[df.which == "ec_all"]
    diff = d["log_total_photo"] - d["log_total_rest"]
    print("\n=== 1. Signal size: log10(total power ds006036 / ds004504), per person ===")
    print(f"  median {diff.median():+.2f}   10th-90th percentile {diff.quantile(.1):+.2f} to "
          f"{diff.quantile(.9):+.2f}   (0 = same size; +1 = ten times more power)")
    print("\n=== 2. Montage: does alpha have the same map over the head? ===")
    print(f"  correlation of the 19-channel alpha map, per person: median {d.map_corr.median():.2f} "
          f"(10th percentile {d.map_corr.quantile(.1):.2f})")
    print(f"  share of alpha at the back of the head: ds004504 {d.post_share_rest.median():.2f}, "
          f"ds006036 {d.post_share_photo.median():.2f}")
    worst = d.nsmallest(8, "map_corr")[["subject", "group", "map_corr", "post_share_rest",
                                        "post_share_photo"]]
    print("  lowest map correlations:")
    print(worst.round(2).to_string(index=False))
    print(f"\nSaved {PROCESSED / 'day07_check_b_diagnosis.csv'}")

if __name__ == "__main__":
    main()
