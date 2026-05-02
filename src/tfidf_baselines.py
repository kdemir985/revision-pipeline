from __future__ import annotations
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.paths import DATASET_CSV, PREDICTIONS_DIR
from src.splits import project_level_split
from src.labels import TASKS
from src.seeding import set_global_seed
from src.io_utils import save_predictions


def get_classifiers(seed: int) -> dict:
    return {
        "SVM": LinearSVC(C=1.0, random_state=seed, max_iter=5000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "DT": DecisionTreeClassifier(random_state=seed),
        "NB": MultinomialNB(),
        "LR": LogisticRegression(C=1.0, random_state=seed, max_iter=2000, n_jobs=-1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    args = ap.parse_args()

    df = pd.read_csv(DATASET_CSV)
    summary_rows = []

    for seed in args.seeds:
        set_global_seed(seed)
        train_df, test_df, _, _ = project_level_split(df, seed=seed)

        for task in args.tasks:
            label_col = TASKS[task]["column"]
            vec = TfidfVectorizer(
                ngram_range=(1, 2), max_features=10000, sublinear_tf=True
            )
            X_tr = vec.fit_transform(train_df["text"].tolist())
            X_te = vec.transform(test_df["text"].tolist())
            y_tr = train_df[label_col].tolist()
            y_te = test_df[label_col].tolist()

            for name, clf in get_classifiers(seed).items():
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
                f1m = f1_score(y_te, y_pred, average="macro")
                model_id = f"TFIDF-{name}"
                save_predictions(
                    model_id=model_id,
                    task=task,
                    seed=seed,
                    test_df=test_df,
                    y_true=y_te,
                    y_pred=y_pred,
                    extra={"accuracy": acc, "f1_macro": f1m, "feature": "TFIDF"},
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

    out = PREDICTIONS_DIR.parent / "tfidf_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
