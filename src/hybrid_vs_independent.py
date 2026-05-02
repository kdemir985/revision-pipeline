from __future__ import annotations
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.paths import PREDICTIONS_DIR, TABLES_DIR
from src.io_utils import load_predictions
from src.labels import TASKS


def joint_label(func_pred: str, prio_pred: str) -> str:
    return f"{func_pred}-{prio_pred}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    rows = []
    for m in args.models:
        for s in args.seeds:
            try:
                func_df = load_predictions(m, "func", s)
                prio_df = load_predictions(m, "prio", s)
                hyb_df = load_predictions(m, "hybrid", s)
            except FileNotFoundError as e:
                print(f"[skip] {m} seed={s}: {e}")
                continue

            j = func_df.merge(
                prio_df,
                on=["project", "req_id", "text", "source"],
                suffixes=("_func", "_prio"),
            )
            j = j.merge(
                hyb_df.rename(
                    columns={"y_true": "y_true_hyb", "y_pred": "y_pred_hyb",
                             "correct": "correct_hyb"}
                )[["project", "req_id", "y_true_hyb", "y_pred_hyb", "correct_hyb"]],
                on=["project", "req_id"],
            )
            j["pred_independent"] = [
                joint_label(f, p) for f, p in zip(j["y_pred_func"], j["y_pred_prio"])
            ]
            valid = set(TASKS["hybrid"]["labels"])
            j["pred_independent"] = j["pred_independent"].where(
                j["pred_independent"].isin(valid), other="<INVALID>"
            )

            acc_indep = accuracy_score(j["y_true_hyb"], j["pred_independent"])
            f1_indep = f1_score(
                j["y_true_hyb"], j["pred_independent"], average="macro", zero_division=0
            )
            acc_hyb = accuracy_score(j["y_true_hyb"], j["y_pred_hyb"])
            f1_hyb = f1_score(
                j["y_true_hyb"], j["y_pred_hyb"], average="macro", zero_division=0
            )

            rows.append(
                dict(
                    model=m,
                    seed=s,
                    n=len(j),
                    acc_independent=acc_indep,
                    f1_independent=f1_indep,
                    acc_hybrid=acc_hyb,
                    f1_hybrid=f1_hyb,
                    delta_acc=acc_hyb - acc_indep,
                    delta_f1=f1_hyb - f1_indep,
                )
            )
            print(
                f"{m} seed={s}: indep_acc={acc_indep*100:5.2f}  "
                f"hyb_acc={acc_hyb*100:5.2f}  delta={100*(acc_hyb-acc_indep):+5.2f}"
            )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        agg = out_df.groupby("model")[
            ["acc_independent", "acc_hybrid", "delta_acc", "f1_independent", "f1_hybrid", "delta_f1"]
        ].agg(["mean", "std"])
        print("\n=== AGGREGATE (mean ± std across seeds) ===")
        print(agg.round(4))
        out_df.to_csv(TABLES_DIR / "hybrid_vs_independent.csv", index=False)
        agg.to_csv(TABLES_DIR / "hybrid_vs_independent_agg.csv")
        print(f"\nWrote {TABLES_DIR / 'hybrid_vs_independent.csv'}")
        print(f"Wrote {TABLES_DIR / 'hybrid_vs_independent_agg.csv'}")


if __name__ == "__main__":
    main()
