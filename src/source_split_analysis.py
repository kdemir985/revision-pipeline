from __future__ import annotations
import argparse
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import accuracy_score, f1_score

from src.paths import DATASET_CSV, TABLES_DIR
from src.io_utils import load_predictions
from src.labels import TASKS


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()
    rows = []
    for src, sub in df.groupby("source"):
        rows.append(
            dict(
                source=src,
                n=len(sub),
                mean_chars=sub["n_chars"].mean(),
                std_chars=sub["n_chars"].std(),
                mean_words=sub["n_words"].mean(),
                std_words=sub["n_words"].std(),
                pct_FR=(sub["type"] == "FR").mean(),
                pct_Must=(sub["priority"] == "Must").mean(),
                pct_Should=(sub["priority"] == "Should").mean(),
                pct_Could=(sub["priority"] == "Could").mean(),
            )
        )
    return pd.DataFrame(rows)


def distributional_tests(df: pd.DataFrame) -> dict:
    a = df[df["source"] == "ETU"]
    b = df[df["source"] == "Translated"]
    ks_chars = ks_2samp(a["text"].str.len(), b["text"].str.len())
    ks_words = ks_2samp(
        a["text"].str.split().str.len(), b["text"].str.split().str.len()
    )
    ct_prio = pd.crosstab(df["source"], df["priority"])
    chi_prio = chi2_contingency(ct_prio)
    ct_type = pd.crosstab(df["source"], df["type"])
    chi_type = chi2_contingency(ct_type)
    return {
        "ks_chars_stat": float(ks_chars.statistic),
        "ks_chars_p": float(ks_chars.pvalue),
        "ks_words_stat": float(ks_words.statistic),
        "ks_words_p": float(ks_words.pvalue),
        "chi2_priority_stat": float(chi_prio.statistic),
        "chi2_priority_p": float(chi_prio.pvalue),
        "chi2_type_stat": float(chi_type.statistic),
        "chi2_type_p": float(chi_type.pvalue),
    }


def per_source_performance(models: list[str], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for m in models:
        for task in TASKS.keys():
            for s in seeds:
                try:
                    pred = load_predictions(m, task, s)
                except FileNotFoundError:
                    continue
                for src, sub in pred.groupby("source"):
                    if len(sub) == 0:
                        continue
                    acc = accuracy_score(sub["y_true"], sub["y_pred"])
                    f1m = f1_score(
                        sub["y_true"],
                        sub["y_pred"],
                        average="macro",
                        zero_division=0,
                    )
                    rows.append(
                        dict(
                            model=m,
                            task=task,
                            seed=s,
                            source=src,
                            n=len(sub),
                            accuracy=acc,
                            f1_macro=f1m,
                        )
                    )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    df = pd.read_csv(DATASET_CSV)
    desc = descriptive_stats(df)
    dist = distributional_tests(df)
    print("=== Descriptive stats by source ===")
    print(desc.to_string(index=False))
    print("\n=== Distributional tests (ETU vs Translated) ===")
    for k, v in dist.items():
        print(f"  {k}: {v:.4g}")

    perf = per_source_performance(args.models, args.seeds)
    if perf.empty:
        print("[warn] No predictions found for the requested models/seeds.")
        return
    agg = (
        perf.groupby(["model", "task", "source"])[["accuracy", "f1_macro"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print("\n=== Per-source performance (mean ± std across seeds) ===")
    print(agg)

    desc.to_csv(TABLES_DIR / "source_descriptive.csv", index=False)
    pd.DataFrame([dist]).to_csv(
        TABLES_DIR / "source_distributional_tests.csv", index=False
    )
    perf.to_csv(TABLES_DIR / "per_source_performance.csv", index=False)
    agg.to_csv(TABLES_DIR / "per_source_performance_agg.csv")
    print(f"\nWrote tables under {TABLES_DIR}")


if __name__ == "__main__":
    main()
