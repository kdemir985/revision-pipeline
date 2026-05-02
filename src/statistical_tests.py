from __future__ import annotations
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.paths import TABLES_DIR
from src.io_utils import load_predictions
from src.labels import TASKS


def mcnemar_with_or(c1: np.ndarray, c2: np.ndarray) -> dict:
    c1 = np.asarray(c1).astype(int)
    c2 = np.asarray(c2).astype(int)
    a = int(((c1 == 1) & (c2 == 1)).sum())
    b = int(((c1 == 1) & (c2 == 0)).sum())
    c = int(((c1 == 0) & (c2 == 1)).sum())
    d = int(((c1 == 0) & (c2 == 0)).sum())
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        p = binomtest(min(b, c), n=n_disc, p=0.5).pvalue
    or_ratio = (b + 0.5) / (c + 0.5)
    return {
        "n": int(c1.size),
        "both_correct": a,
        "only1_correct": b,
        "only2_correct": c,
        "both_wrong": d,
        "mcnemar_p": float(p),
        "odds_ratio": float(or_ratio),
        "log_odds_ratio": float(np.log(or_ratio)),
    }


def bootstrap_ci_diff(
    c1: np.ndarray, c2: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    c1 = np.asarray(c1).astype(int)
    c2 = np.asarray(c2).astype(int)
    n = c1.size
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = c1[idx].mean() - c2[idx].mean()
    return {
        "diff_mean": float(diffs.mean()),
        "diff_lo": float(np.percentile(diffs, 2.5)),
        "diff_hi": float(np.percentile(diffs, 97.5)),
    }


def permutation_test(
    c1: np.ndarray, c2: np.ndarray, n_perm: int = 1000, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    c1 = np.asarray(c1).astype(int)
    c2 = np.asarray(c2).astype(int)
    obs = abs(c1.mean() - c2.mean())
    stacked = np.stack([c1, c2])
    cnt = 0
    for _ in range(n_perm):
        flip = rng.integers(0, 2, c1.size).astype(bool)
        a = np.where(flip, stacked[1], stacked[0])
        b = np.where(flip, stacked[0], stacked[1])
        if abs(a.mean() - b.mean()) >= obs:
            cnt += 1
    return {"perm_p": float((cnt + 1) / (n_perm + 1))}


def bh_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        v = ranked[i] * n / (i + 1)
        cummin = min(cummin, v)
        adj[order[i]] = min(cummin, 1.0)
    return adj.tolist()


def bonf_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    return np.minimum(p * len(p), 1.0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()

    rows = []
    for task in TASKS.keys():
        for seed in args.seeds:
            preds = {}
            for m in args.models:
                try:
                    preds[m] = load_predictions(m, task, seed)
                except FileNotFoundError:
                    pass
            if len(preds) < 2:
                continue

            pair_rows = []
            for m1, m2 in itertools.combinations(sorted(preds.keys()), 2):
                p1 = preds[m1].sort_values(["project", "req_id"]).reset_index(drop=True)
                p2 = preds[m2].sort_values(["project", "req_id"]).reset_index(drop=True)
                if not (p1[["project", "req_id"]].equals(p2[["project", "req_id"]])):
                    print(f"[warn] misaligned: {m1} vs {m2} task={task} seed={seed}")
                    continue
                c1, c2 = p1["correct"].values, p2["correct"].values
                acc1, acc2 = c1.mean(), c2.mean()
                mc = mcnemar_with_or(c1, c2)
                bs = bootstrap_ci_diff(c1, c2, n_boot=args.n_boot, seed=seed)
                pm = permutation_test(c1, c2, n_perm=args.n_perm, seed=seed)
                pair_rows.append(
                    dict(
                        task=task,
                        seed=seed,
                        model_1=m1,
                        model_2=m2,
                        n=mc["n"],
                        acc_1=acc1,
                        acc_2=acc2,
                        delta=acc1 - acc2,
                        mcnemar_p=mc["mcnemar_p"],
                        odds_ratio=mc["odds_ratio"],
                        log_odds_ratio=mc["log_odds_ratio"],
                        b_only1=mc["only1_correct"],
                        c_only2=mc["only2_correct"],
                        bootstrap_diff_lo=bs["diff_lo"],
                        bootstrap_diff_hi=bs["diff_hi"],
                        perm_p=pm["perm_p"],
                    )
                )

            if pair_rows:
                pdf = pd.DataFrame(pair_rows)
                pdf["mcnemar_p_bh"] = bh_adjust(pdf["mcnemar_p"].tolist())
                pdf["mcnemar_p_bonf"] = bonf_adjust(pdf["mcnemar_p"].tolist())
                rows.extend(pdf.to_dict("records"))

    if not rows:
        print("No pairwise rows produced (need >=2 models per task+seed).")
        return

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "pairwise_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out)} rows)")
    agg = (
        out.groupby(["task", "model_1", "model_2"])
        .agg(
            mean_acc_1=("acc_1", "mean"),
            mean_acc_2=("acc_2", "mean"),
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
            median_mcnemar_p=("mcnemar_p", "median"),
            median_mcnemar_p_bh=("mcnemar_p_bh", "median"),
            median_mcnemar_p_bonf=("mcnemar_p_bonf", "median"),
            mean_odds_ratio=("odds_ratio", "mean"),
        )
        .round(4)
        .reset_index()
    )
    agg.to_csv(TABLES_DIR / "pairwise_stats_agg.csv", index=False)
    print(f"Wrote {TABLES_DIR / 'pairwise_stats_agg.csv'}")


if __name__ == "__main__":
    main()
