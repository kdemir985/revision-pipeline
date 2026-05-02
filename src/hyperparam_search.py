from __future__ import annotations
import argparse
import itertools
import pandas as pd

from src.paths import TABLES_DIR, DATASET_CSV, PREDICTIONS_DIR
from src.transformer_train import load_config, find_model_spec, train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[1e-5, 2e-5, 3e-5, 5e-5],
    )
    ap.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[8, 16, 32],
    )
    args = ap.parse_args()

    cfg = load_config()
    spec = find_model_spec(cfg, args.model)
    df = pd.read_csv(DATASET_CSV)

    rows = []
    for lr, bs in itertools.product(args.lrs, args.batch_sizes):
        out = train_one(
            spec,
            args.task,
            args.seed,
            cfg,
            df,
            PREDICTIONS_DIR / "_hpsearch" / f"lr{lr}_bs{bs}",
            overrides={
                "learning_rate": lr,
                "per_device_train_batch_size": bs,
            },
        )
        out["lr"] = lr
        out["batch_size"] = bs
        rows.append(out)

    df_out = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    out_path = TABLES_DIR / f"hpsearch_{spec['id']}_{args.task}_seed{args.seed}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n=== Hyperparameter search ({spec['label']} / {args.task} / seed={args.seed}) ===")
    print(df_out[["lr", "batch_size", "accuracy", "f1_macro"]].to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
