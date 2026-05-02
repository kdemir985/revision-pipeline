from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.paths import DATASET_CSV, PREDICTIONS_DIR
from src.splits import project_level_split
from src.labels import TASKS
from src.seeding import set_global_seed
from src.io_utils import save_predictions

SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def get_classifiers(seed: int) -> dict:
    return {
        "SVM": SVC(kernel="rbf", C=1.0, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "DT": DecisionTreeClassifier(random_state=seed),
        "NB": GaussianNB(),
        "LR": LogisticRegression(C=1.0, random_state=seed, max_iter=2000, n_jobs=-1),
    }


def encode_sbert(texts: list[str], device: str | None = None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT_MODEL, device=device)
    return model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    ap.add_argument("--device", default=None, help="cuda / mps / cpu")
    args = ap.parse_args()

    df = pd.read_csv(DATASET_CSV)
    print(f"Encoding {len(df)} sentences once with SBERT (device={args.device})…")
    set_global_seed(0)
    embeddings = encode_sbert(df["text"].tolist(), device=args.device)
    df = df.reset_index(drop=True)

    summary_rows = []
    for seed in args.seeds:
        set_global_seed(seed)
        train_df, test_df, _, _ = project_level_split(df, seed=seed)
        idx_tr = train_df.index.values
        idx_te = test_df.index.values

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(embeddings[idx_tr])
        X_te = scaler.transform(embeddings[idx_te])

        for task in args.tasks:
            label_col = TASKS[task]["column"]
            y_tr = train_df[label_col].tolist()
            y_te = test_df[label_col].tolist()

            for name, clf in get_classifiers(seed).items():
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
                f1m = f1_score(y_te, y_pred, average="macro")
                model_id = f"SBERT-{name}"
                save_predictions(
                    model_id=model_id,
                    task=task,
                    seed=seed,
                    test_df=test_df,
                    y_true=y_te,
                    y_pred=y_pred,
                    extra={"accuracy": acc, "f1_macro": f1m, "feature": "SBERT"},
                )
                summary_rows.append(
                    dict(
                        model=model_id,
                        task=task,
                        seed=seed,
                        accuracy=acc,
                        f1_macro=f1m,
                        n_test=len(y_te),
                    )
                )
                print(
                    f"seed={seed} task={task:<7} {model_id:<10} "
                    f"acc={acc*100:5.2f} f1m={f1m*100:5.2f}"
                )

    out = PREDICTIONS_DIR.parent / "sbert_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
