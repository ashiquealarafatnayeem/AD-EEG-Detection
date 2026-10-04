import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import ml
from eegio import FIGURES, PROCESSED

CACHE = PROCESSED / "day08"
OUT = FIGURES / "day08"
TASK_TITLES = {"AD_vs_CN": "AD vs CN (main)", "FTD_vs_CN": "FTD vs CN",
               "AD_vs_FTD": "AD vs FTD", "3class": "AD / FTD / CN (3 classes)"}
MODEL_COLORS = {"dummy": "#8a96a3", "logreg": "#1B2A41", "linsvm": "#2471A3",
                "rbfsvm": "#7D3C98", "rf": "#1E8449", "knn": "#E67E22"}

def plan():
    runs = [ml.PRIMARY]
    for task in ml.TASKS:
        for fs in ml.FEATURE_SETS:
            for model in ml.MODELS:
                if (task, fs, model) != ml.PRIMARY:
                    runs.append((task, fs, model))
        runs.append((task, "age_sex", "logreg"))
    return runs

def run(task, fs, model):
    X, y, ids, names, classes = ml.load_task(task, fs)
    path = CACHE / f"{task}__{fs}__{model}.npz"
    if path.exists():
        z = np.load(path)
        r = {k: z[k] for k in ["proba", "pred", "sel_freq", "coef"]}
        r["params"] = json.loads(str(z["params"]))
        status = "cached"
    else:
        r = ml.run_cv(X, y, model)
        np.savez(path, proba=r["proba"], pred=r["pred"], sel_freq=r["sel_freq"],
                 coef=r["coef"], params=json.dumps(r["params"], default=str))
        status = "ok"
    return X, y, ids, names, classes, r, status

def fmt(row, key):
    return f"{row[key]:.3f} [{row[key + '_lo']:.3f}, {row[key + '_hi']:.3f}]"

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = plan()
    rows, keep = [], {}
    t_start = time.perf_counter()
    for i, (task, fs, model) in enumerate(runs, 1):
        t0 = time.perf_counter()
        X, y, ids, names, classes, r, status = run(task, fs, model)
        s = ml.summarise(y, r["proba"], r["pred"])
        rows.append({"task": task, "feature_set": fs, "model": model,
                     "n": len(y), "n_features": X.shape[1], **s,
                     "main": (task, fs, model) == ml.PRIMARY})
        keep[(task, fs, model)] = (y, ids, names, classes, r)
        print(f"[{i:2d}/{len(runs)}] {task:9s} {fs:7s} {model:6s} {status:6s} "
              f"AUC {fmt(rows[-1], 'auc')}  bal.acc {rows[-1]['bacc']:.3f}  "
              f"{time.perf_counter() - t0:6.1f} s  (total {(time.perf_counter() - t_start) / 60:.1f} min)")

    res = pd.DataFrame(rows)
    res.to_csv(PROCESSED / "day08_results.csv", index=False)

    pd.set_option("display.width", 220)
    for task in ml.TASKS:
        d = res[res.task == task]
        classes = ml.TASKS[task]
        chance_bacc = 1 / len(classes)
        print(f"\n=== {TASK_TITLES[task]}   ({d.n.iloc[0]} people: "
              + ", ".join(f"{g}" for g in classes) + f"; chance AUC 0.5, "
              f"chance balanced accuracy {chance_bacc:.2f}) ===")
        head = f"{'model':22s}{'features':9s}{'AUC [95% CI]':>26s}{'bal. acc [95% CI]':>26s}"
        if len(classes) == 2:
            head += f"{'sens':>7s}{'spec':>7s}"
        print(head)
        for fs in ml.FEATURE_SETS + ["age_sex"]:
            for model in ml.MODELS:
                row = d[(d.feature_set == fs) & (d.model == model)]
                if row.empty:
                    continue
                row = row.iloc[0]
                line = (f"{ml.MODEL_NAMES[model]:22s}{fs:9s}{fmt(row, 'auc'):>26s}"
                        f"{fmt(row, 'bacc'):>26s}")
                if len(classes) == 2:
                    line += f"{row.sens:>7.2f}{row.spec:>7.2f}"
                print(line + ("   <- MAIN RESULT" if row.main else ""))

    y, ids, names, classes, r = keep[ml.PRIMARY]
    cs = Counter(p["clf__C"] for p in r["params"])
    ks = Counter(str(p["select__k"]) for p in r["params"])
    print("\n=== Main model: settings chosen by the inner cross-validation (50 outer folds) ===")
    print("  C (1/regularisation):", dict(sorted(cs.items())))
    print("  number of features  :", dict(ks))
    order = np.argsort(-r["sel_freq"])[:15]
    print("  most often chosen features (fraction of folds, mean coefficient; + = pushes towards AD):")
    for j in order:
        print(f"    {names[j]:32s} {r['sel_freq'][j]:5.2f}  {r['coef'][j]:+.3f}")

    p_ad = r["proba"][:, :, 1]
    pd.DataFrame({"participant_id": ids, "group": [classes[v] for v in y],
                  "p_AD_mean": p_ad.mean(axis=0),
                  "called_AD_fraction": (r["pred"] == 1).mean(axis=0)}
                 ).to_csv(PROCESSED / "day08_predictions_main.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for model in ml.MODELS:
        y2, _, _, _, r2 = keep[("AD_vs_CN", "region", model)]
        fpr, tpr = ml.mean_roc(y2, r2["proba"])
        auc = res[(res.task == "AD_vs_CN") & (res.feature_set == "region") & (res.model == model)].auc.iloc[0]
        ax.plot(fpr, tpr, color=MODEL_COLORS[model], lw=2.4 if model == "logreg" else 1.4,
                ls=":" if model == "dummy" else "-", label=f"{ml.MODEL_NAMES[model]} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], color="k", lw=.6)
    ax.set(xlabel="false-positive rate (1 - specificity)", ylabel="true-positive rate (sensitivity)",
           title="AD vs CN, region features (mean of 10 repeats)")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "02_roc_ad_vs_cn.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, task in zip(axes, ml.TASKS):
        d = res[res.task == task]
        for k, model in enumerate(ml.MODELS):
            for fs, dy, mk, fill in [("region", -.13, "o", True), ("channel", .13, "s", False)]:
                row = d[(d.feature_set == fs) & (d.model == model)].iloc[0]
                col = MODEL_COLORS[model]
                ax.errorbar(row.auc, k + dy, xerr=[[row.auc - row.auc_lo], [row.auc_hi - row.auc]],
                            fmt=mk, color=col, mfc=col if fill else "white", ms=6, capsize=2, lw=1.2)
        base = d[d.feature_set == "age_sex"].auc.iloc[0]
        ax.axvline(base, color="#C0392B", ls="--", lw=1, label=f"age + sex only ({base:.2f})")
        ax.axvline(0.5, color="k", ls=":", lw=1)
        ax.set(title=TASK_TITLES[task], xlim=(0.3, 1.0), xlabel="AUC with 95% CI")
        ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_yticks(range(len(ml.MODELS)), [ml.MODEL_NAMES[m] for m in ml.MODELS])
    axes[0].invert_yaxis()
    fig.suptitle("Filled circle = region features (94), open square = channel features (250)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "03_auc_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    y3, _, _, cl3, r3 = keep[("3class", "region", "logreg")]
    cm = ml.mean_confusion(y3, r3["pred"], len(cl3))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for a in range(len(cl3)):
        for b in range(len(cl3)):
            ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center",
                    color="white" if cm[a, b] > .5 else "#1B2A41", fontsize=11)
    ax.set_xticks(range(len(cl3)), cl3); ax.set_yticks(range(len(cl3)), cl3)
    bacc = res[(res.task == "3class") & (res.feature_set == "region") & (res.model == "logreg")].bacc.iloc[0]
    ax.set(xlabel="predicted", ylabel="true group",
           title=f"3 classes, logistic regression\nbalanced accuracy {bacc:.2f} (chance 0.33)")
    fig.tight_layout(); fig.savefig(OUT / "04_confusion_3class.png", dpi=150)
    plt.close(fig)

    top = np.argsort(-r["sel_freq"])[:20][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(range(len(top)), r["sel_freq"][top],
            color=["#C0392B" if r["coef"][j] > 0 else "#2471A3" for j in top])
    ax.set_yticks(range(len(top)), [names[j] for j in top], fontsize=8)
    ax.set(xlabel="fraction of the 50 outer folds in which the feature was chosen", xlim=(0, 1),
           title="Main model (AD vs CN): red = higher in AD, blue = lower in AD")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "05_main_model_features.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {PROCESSED / 'day08_results.csv'} and figures in {OUT}")
    print(f"Total time {(time.perf_counter() - t_start) / 60:.1f} min")

if __name__ == "__main__":
    main()
