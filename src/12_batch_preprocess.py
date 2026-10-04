import sys
import time
import traceback
from pathlib import Path
import mne
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import preprocess as P
from eegio import DS_REST, PROCESSED, participants

EPO_DIR = PROCESSED / "epochs"
LOG_CSV = PROCESSED / "day05_batch_log.csv"
ERR_LOG = PROCESSED / "day05_batch_errors.txt"

MIN_EPOCHS = 60
MAX_DROP_PCT = 20.0

def load_log():
    if LOG_CSV.exists():
        return pd.read_csv(LOG_CSV).to_dict("records")
    return []

def main():
    mne.set_log_level("ERROR")
    EPO_DIR.mkdir(parents=True, exist_ok=True)
    df = participants(DS_REST)
    rows = load_log()
    done = {r["subject"] for r in rows if r.get("status") == "ok"}
    n_total = len(df)
    t_start = time.perf_counter()

    for i, (sid, grp) in enumerate(zip(df["participant_id"], df["group"]), 1):
        out = EPO_DIR / f"{sid}_epo.fif"
        if sid in done and out.exists():
            print(f"[{i:2d}/{n_total}] {sid} ({grp:3s}) already done - skipped")
            continue

        t0 = time.perf_counter()
        try:
            _, epochs, _, rep, _ = P.preprocess_subject(sid)
            epochs.save(out, fmt="single", overwrite=True)

            row = dict(subject=sid, group=grp, status="ok",
                       dur_s=round(rep.dur_cropped_s, 1),
                       ica_method=rep.ica_method_used,
                       n_ica_removed=len(rep.ica_excluded),
                       ica_excluded=str(rep.ica_excluded),
                       n_made=rep.n_epochs_made, n_kept=rep.n_epochs_kept,
                       drop_pct=round(rep.drop_pct, 1),
                       seconds=round(time.perf_counter() - t0, 1))

            print(f"[{i:2d}/{n_total}] {sid} ({grp:3s}) ok  "
                  f"ICA removed {row['n_ica_removed']}  "
                  f"epochs {row['n_kept']:3d}/{row['n_made']:3d}  "
                  f"{row['seconds']:5.1f} s")
        except Exception as exc:
            row = dict(subject=sid, group=grp,
                       status=f"FAILED: {type(exc).__name__}: {exc}")
            print(f"[{i:2d}/{n_total}] {sid} ({grp:3s}) FAILED - {type(exc).__name__}: {exc}")
            with open(ERR_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"===== {sid} =====\n{traceback.format_exc()}\n")

        rows = [r for r in rows if r["subject"] != sid] + [row]
        pd.DataFrame(rows).to_csv(LOG_CSV, index=False)

    log = pd.DataFrame(rows)
    ok = log[log["status"] == "ok"].copy()
    for col in ["n_ica_removed", "n_made", "n_kept"]:
        if col in ok:
            ok[col] = ok[col].astype(int)

    failed = log[log["status"] != "ok"]
    minutes = (time.perf_counter() - t_start) / 60

    print("\n" + "=" * 70)
    print(f"Finished: {len(ok)} ok, {len(failed)} failed, this run took {minutes:.1f} min")

    if len(failed):
        print("\nFAILED subjects (details in day05_batch_errors.txt):")
        print(failed[["subject", "group", "status"]].to_string(index=False))

    if len(ok):
        print("\n=== Per group ===")
        summ = ok.groupby("group").agg(
            subjects=("subject", "count"),
            epochs_median=("n_kept", "median"),
            epochs_min=("n_kept", "min"),
            drop_pct_median=("drop_pct", "median"),
            drop_pct_max=("drop_pct", "max")
        )
        print(summ.to_string())

        print("\n=== ICA components removed (number of subjects) ===")
        print(pd.crosstab(ok["group"], ok["n_ica_removed"]).to_string())

        print("\n=== ICA detection route ===")
        print(ok["ica_method"].value_counts().to_string())

        flag = ok[(ok["n_kept"] < MIN_EPOCHS) |
                  (ok["drop_pct"] > MAX_DROP_PCT) |
                  (ok["n_ica_removed"] == 0) |
                  (ok["n_ica_removed"] == P.ICA_MAX_BAD)]
        print(f"\n=== Flagged for a visual check: {len(flag)} subject(s) ===")
        if len(flag):
            print(flag[["subject", "group", "n_ica_removed", "n_kept",
                        "n_made", "drop_pct"]].to_string(index=False))

        print(f"\nTotal clean data: {ok['n_kept'].sum()} epochs from {len(ok)} subjects")
        print(f"Log: {LOG_CSV}")

if __name__ == "__main__":
    main()
