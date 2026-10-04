import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent))
import features as FT
import ml
import spectral as S
from eegio import FIGURES, PROCESSED

EPO_DIR = PROCESSED / "epochs"
CACHE = PROCESSED / "demo_cache.npz"
OUT = FIGURES / "demo"
FMAX = 45.0
COLORS = {"CN": "#2471A3", "AD": "#C0392B", "FTD": "#E67E22"}

def spectrum_and_features(sid):
    ep = mne.read_epochs(EPO_DIR / f"{sid}_epo.fif", preload=True, verbose="ERROR")
    data, fs, ch = ep.get_data(), ep.info["sfreq"], list(ep.ch_names)
    f, p = S.welch_epochs(data, fs)
    keep = f <= FMAX
    freqs, psd = f[keep], p[:, keep]
    feats = FT.channel_features(freqs, psd, data, fs)
    post = [ch.index(c) for c in FT.REGIONS["posterior"]]
    peak = FT.alpha_peak(freqs, psd[post].mean(axis=0))
    row = {f"{k}__{r}": FT.region_average(v, ch, r) for k, v in feats.items() for r in FT.REGIONS}
    row.update({"alpha_present": peak["present"], "alpha_iaf": peak["iaf"], "alpha_height": peak["height"]})
    return ep, data, fs, ch, freqs, psd[post].mean(axis=0), row, peak

def prepare():
    table = pd.read_csv(PROCESSED / "features_region.csv")
    spectra, groups, ids = [], [], []
    t0 = time.perf_counter()
    for i, (sid, g) in enumerate(zip(table.participant_id, table.group), 1):
        _, _, _, _, freqs, post, _, _ = spectrum_and_features(sid)
        spectra.append(np.log10(post)); groups.append(g); ids.append(sid)
        print(f"\r  {i}/{len(table)} people   {time.perf_counter() - t0:4.0f} s", end="")
    np.savez(CACHE, freqs=freqs, log_psd=np.array(spectra), groups=np.array(groups), ids=np.array(ids))
    print(f"\nSaved {CACHE}. The demo is ready.")

def suggest():
    pred = pd.read_csv(PROCESSED / "day08_predictions_main.csv")
    print("People the Day 8 cross-validation scored most confidently (out-of-sample):")
    for g, asc in [("CN", True), ("AD", False)]:
        d = pred[pred.group == g].sort_values("p_AD_mean", ascending=asc).head(4)
        print(f"  {g}: " + ", ".join(f"{r.participant_id} (P(AD) {r.p_AD_mean:.2f})" for r in d.itertuples()))
    ftd = pd.read_csv(PROCESSED / "features_region.csv")
    print("  FTD (never used for training the AD-vs-CN model): " +
          ", ".join(ftd[ftd.group == "FTD"].participant_id.head(4)))

def main(sid):
    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
    t_all = time.perf_counter()
    table = pd.read_csv(PROCESSED / "features_region.csv")
    if sid not in set(table.participant_id):
        raise SystemExit(f"{sid} is not in features_region.csv")
    truth = table.set_index("participant_id").loc[sid, "group"]
    print(f"\n=== LIVE DEMO: {sid} (diagnosis hidden until the end) ===")

    t = time.perf_counter()
    ep, data, fs, ch, freqs, post, row, peak = spectrum_and_features(sid)
    print(f"1. Cleaned EEG loaded: {len(ep)} clean 4 s epochs ({len(ep) * 2 / 60 + 2 / 60:.1f} min), "
          f"{len(ch)} channels, {fs:.0f} Hz")
    print(f"2. Welch spectrum (our own code): resolution {freqs[1] - freqs[0]:.2f} Hz, "
          f"averaged over {len(ep)} segments")
    iaf = f"{peak['iaf']:.1f} Hz" if peak["present"] else "no alpha peak found"
    print(f"   posterior alpha peak: {iaf};  theta/alpha (posterior) = "
          f"{10 ** row['logTAR__posterior']:.2f}")
    feat_cols = [c for c in table.columns if c not in ml.META]
    ref = table.set_index("participant_id").loc[sid, feat_cols].astype(float)
    same = np.allclose([row[c] for c in feat_cols], ref.to_numpy(), rtol=1e-6, equal_nan=True)
    print(f"3. 94 features computed ({time.perf_counter() - t:.1f} s); identical to the Day 6 table: {same}")

    t = time.perf_counter()
    train = table[table.group.isin(["AD", "CN"]) & (table.participant_id != sid)]
    pipe, grid = ml.model_spec("logreg", len(feat_cols))
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", n_jobs=-1,
                      cv=StratifiedKFold(5, shuffle=True, random_state=ml.SEED))
    gs.fit(train[feat_cols].to_numpy(float), (train.group == "AD").astype(int).to_numpy())
    x = np.array([[row[c] for c in feat_cols]], dtype=float)
    p_ad = float(gs.predict_proba(x)[0, 1])
    print(f"4. Main model trained on the other {len(train)} AD/CN people, never on {sid} "
          f"({time.perf_counter() - t:.1f} s)")
    verdict = "looks like ALZHEIMER'S" if p_ad >= 0.5 else "looks HEALTHY"
    print(f"\n   >>> P(AD) = {p_ad:.2f}  ->  the EEG {verdict}")
    pred_path = PROCESSED / "day08_predictions_main.csv"
    if pred_path.exists():
        pr = pd.read_csv(pred_path).set_index("participant_id")
        if sid in pr.index:
            print(f"   (Day 8 cross-validation gave this person {pr.loc[sid, 'p_AD_mean']:.2f} on average)")
    print(f"   True diagnosis: {truth}" + ("   (FTD: the model was trained on AD vs CN only)" if truth == "FTD" else ""))
    print(f"   Total time {time.perf_counter() - t_all:.1f} s")

    fig = plt.figure(figsize=(14, 7.2))
    gsp = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], width_ratios=[1.35, 1, 0.55])
    ax = fig.add_subplot(gsp[0, :2])
    k = len(ep) // 2
    for j, c in enumerate(["Fz", "Cz", "Pz", "O1", "O2"]):
        y = data[k, ch.index(c)] * 1e6
        tt = np.arange(len(y)) / fs
        ax.plot(tt, y - 45 * j, color="#1B2A41", lw=0.8)
        ax.text(-0.08, -45 * j, c, ha="right", va="center", fontsize=10)
    ax.set(xlim=(0, len(y) / fs), yticks=[], xlabel="time (s)",
           title=f"{sid}: one clean 4 s epoch after our filters, average reference and ICA")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gsp[1, 0])
    if CACHE.exists():
        z = np.load(CACHE)
        for g in ["CN", "AD"]:
            m = (z["groups"] == g) & (z["ids"] != sid)
            ax.semilogy(z["freqs"], 10 ** z["log_psd"][m].mean(axis=0), color=COLORS[g], lw=2,
                        alpha=.55, label=f"{g} average (n={m.sum()})")
    ax.semilogy(freqs, post, color="k", lw=2.2, label=f"{sid}")
    ax.axvspan(4, 8, color="#E67E22", alpha=.08); ax.axvspan(8, 13, color="#2471A3", alpha=.08)
    ax.text(6, ax.get_ylim()[1] * 0.5, "θ", ha="center", fontsize=13)
    ax.text(10.5, ax.get_ylim()[1] * 0.5, "α", ha="center", fontsize=13)
    ax.set(xlim=(1, 30), xlabel="frequency (Hz)", ylabel="posterior PSD (V²/Hz)",
           title="Our Welch spectrum vs the group averages")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = fig.add_subplot(gsp[1, 1])
    rng = np.random.default_rng(0)
    for i, g in enumerate(["CN", "FTD", "AD"]):
        v = table.loc[(table.group == g) & (table.participant_id != sid), "logTAR__posterior"]
        ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, color=COLORS[g], s=16, alpha=.6)
    ax.scatter([["CN", "FTD", "AD"].index(truth)], [row["logTAR__posterior"]], s=220, marker="*",
               color="k", zorder=5, label=sid)
    ax.set_xticks([0, 1, 2], ["CN", "FTD", "AD"])
    ax.set(ylabel="log10(theta / alpha), posterior", title="EEG slowing marker")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=.3, axis="y")

    ax = fig.add_subplot(gsp[:, 2])
    ax.bar([0], [p_ad], width=.6, color=COLORS["AD"] if p_ad >= .5 else COLORS["CN"])
    ax.axhline(.5, color="k", ls="--", lw=1)
    ax.text(0, p_ad + .03, f"{p_ad:.2f}", ha="center", fontsize=20, fontweight="bold")
    ax.set(ylim=(0, 1.08), xlim=(-.6, .6), xticks=[], ylabel="probability of Alzheimer's",
           title="Model output")
    ax.text(0, -.07, f"true: {truth}", ha="center", fontsize=11, transform=ax.get_xaxis_transform())
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{sid}.png", dpi=150)
    print(f"   Figure saved: {OUT / (sid + '.png')}")
    plt.show()

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--suggest"
    if arg == "--prepare":
        prepare()
    elif arg == "--suggest":
        suggest()
    else:
        main(arg)
