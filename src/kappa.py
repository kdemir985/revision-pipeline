from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.paths import DATASET_CSV, DATA_DIR, TABLES_DIR
from src.seeding import set_global_seed


def export(args):
    df = pd.read_csv(DATASET_CSV)
    set_global_seed(args.seed)
    parts = []
    per_class = max(1, args.n // df["hybrid_label"].nunique())
    for lbl, sub in df.groupby("hybrid_label"):
        parts.append(sub.sample(n=min(per_class, len(sub)), random_state=args.seed))
    sample = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sample = sample[["project", "req_id", "text", "type", "priority", "hybrid_label"]]
    sample["type_2nd"] = ""
    sample["priority_2nd"] = ""
    out = args.out
    sample.to_csv(out, index=False)
    print(f"Wrote {out}  (n={len(sample)})")
    print(
        "\nSend ONLY the columns [project, req_id, text, type_2nd, priority_2nd]\n"
        "to the second annotator. Do NOT show them the original labels."
    )


def score(args):
    df = pd.read_csv(args.inp)
    df = df[df["type_2nd"].notna() & df["priority_2nd"].notna()]
    df = df[df["type_2nd"].astype(str).str.strip() != ""]
    df = df[df["priority_2nd"].astype(str).str.strip() != ""]
    print(f"Scoring on {len(df)} doubly-annotated rows.")
    if len(df) < 50:
        print("[warn] sample is small; Kappa may be unstable.")

    df["hybrid_2nd"] = df["type_2nd"].str.strip() + "-" + df["priority_2nd"].str.strip()

    type_labels = ["FR", "NFR"]
    prio_labels = ["Must", "Should", "Could"]
    hyb_labels = [f"{t}-{p}" for t in type_labels for p in prio_labels]

    out_rows = []
    for axis, gold, second, labels in [
        ("type", "type", "type_2nd", type_labels),
        ("priority", "priority", "priority_2nd", prio_labels),
        ("hybrid", "hybrid_label", "hybrid_2nd", hyb_labels),
    ]:
        a = df[gold].astype(str).str.strip()
        b = df[second].astype(str).str.strip()
        kappa = cohen_kappa_score(a, b, labels=labels)
        agree = (a == b).mean()
        cm = confusion_matrix(a, b, labels=labels)
        out_rows.append(
            dict(
                axis=axis,
                n=int(len(df)),
                cohen_kappa=float(kappa),
                pct_agreement=float(agree),
            )
        )
        print(f"\n=== Axis: {axis} ===")
        print(f"  N={len(df)}  Kappa={kappa:.3f}  pct_agreement={agree*100:.1f}%")
        print("  Confusion matrix (rows=consensus, cols=2nd annotator):")
        print(pd.DataFrame(cm, index=labels, columns=labels))
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            TABLES_DIR / f"kappa_confusion_{axis}.csv"
        )

    pd.DataFrame(out_rows).to_csv(TABLES_DIR / "kappa_summary.csv", index=False)
    print(f"\nWrote {TABLES_DIR / 'kappa_summary.csv'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    e = sub.add_parser("export")
    e.add_argument("--n", type=int, default=500)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=str(DATA_DIR / "kappa_subset_to_relabel.csv"))
    e.set_defaults(func=export)

    s = sub.add_parser("score")
    s.add_argument("--in", dest="inp", required=True)
    s.set_defaults(func=score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
