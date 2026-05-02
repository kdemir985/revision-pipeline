from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.paths import PREDICTIONS_DIR, TABLES_DIR
from src.labels import TASKS


def load_all() -> pd.DataFrame:
    rows = []
    for run in sorted(PREDICTIONS_DIR.glob("*__*__seed*")):
        meta = run / "meta.json"
        preds = run / "predictions.csv"
        if not meta.exists() or not preds.exists():
            continue
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        rows.append(
            dict(
                model=m.get("model_id"),
                model_label=m.get("model_label", m.get("model_id")),
                family=m.get("family", m.get("feature", "")),
                task=m.get("task"),
                seed=m.get("seed"),
                accuracy=m.get("accuracy"),
                f1_macro=m.get("f1_macro"),
                n_test=m.get("n_test"),
            )
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="main_table")
    args = ap.parse_args()

    df = load_all()
    if df.empty:
        print("No prediction artifacts found. Run training first.")
        return
    print(f"Loaded {len(df)} runs covering "
          f"{df['model'].nunique()} models, "
          f"{df['task'].nunique()} tasks, "
          f"{df['seed'].nunique()} seeds.")

    agg = (
        df.groupby(["model", "model_label", "family", "task"])
        .agg(
            n_seeds=("seed", "count"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
        )
        .round(4)
        .reset_index()
    )
    agg.to_csv(TABLES_DIR / f"{args.out_prefix}_combined.csv", index=False)
    print(f"Wrote {TABLES_DIR / (args.out_prefix + '_combined.csv')}")

    for task in TASKS.keys():
        sub = agg[agg["task"] == task].sort_values("acc_mean", ascending=False)
        out = TABLES_DIR / f"{args.out_prefix}_{task}.csv"
        sub.to_csv(out, index=False)
        print(f"  Top 5 for task={task}:")
        print(sub.head(5).to_string(index=False))
        print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
