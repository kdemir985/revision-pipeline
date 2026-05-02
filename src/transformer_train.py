from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from src.paths import CONFIG_DIR, DATASET_CSV, PREDICTIONS_DIR
from src.splits import project_level_split
from src.labels import TASKS, label2id, id2label
from src.seeding import set_global_seed
from src.io_utils import save_predictions


def load_config() -> dict:
    with open(CONFIG_DIR / "transformers.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_model_spec(cfg: dict, key: str) -> dict:
    for m in cfg["models"]:
        if m["id"] == key or m["hf_name"] == key:
            return m
    raise KeyError(
        f"Model '{key}' not found in configs/transformers.yaml; "
        f"available ids: {[m['id'] for m in cfg['models']]}"
    )


def stratified_internal_val(train_df: pd.DataFrame, label_col: str, frac: float, seed: int):
    rng = np.random.RandomState(seed)
    val_rows = []
    for label, group in train_df.groupby(label_col):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * frac)))
        val_rows.extend(idx[:n_val])
    val_set = set(val_rows)
    train_part = train_df[~train_df.index.isin(val_set)].reset_index(drop=True)
    val_part = train_df[train_df.index.isin(val_set)].reset_index(drop=True)
    return train_part, val_part


def train_one(
    model_spec: dict,
    task: str,
    seed: int,
    cfg: dict,
    df: pd.DataFrame,
    output_root: Path,
    overrides: dict | None = None,
) -> dict:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from datasets import Dataset

    set_global_seed(seed)
    label_col = TASKS[task]["column"]
    l2i = label2id(task)
    i2l = id2label(task)
    train_df, test_df, _, _ = project_level_split(df, seed=seed)

    tcfg = dict(cfg["training"])
    if overrides:
        tcfg.update(overrides)
    train_part, val_part = stratified_internal_val(
        train_df, label_col, frac=tcfg["internal_val_fraction"], seed=seed
    )
    print(
        f"[{model_spec['label']}|{task}|seed={seed}] "
        f"train={len(train_part)} val={len(val_part)} test={len(test_df)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_spec["hf_name"])

    def encode(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=tcfg["max_seq_length"],
        )
        enc["labels"] = [l2i[x] for x in batch[label_col]]
        return enc

    ds_train = Dataset.from_pandas(train_part[["text", label_col]]).map(
        encode, batched=True, remove_columns=["text", label_col]
    )
    ds_val = Dataset.from_pandas(val_part[["text", label_col]]).map(
        encode, batched=True, remove_columns=["text", label_col]
    )
    ds_test = Dataset.from_pandas(test_df[["text", label_col]]).map(
        encode, batched=True, remove_columns=["text", label_col]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_spec["hf_name"], num_labels=len(l2i), id2label=i2l, label2id=l2i
    )

    safe_id = model_spec["id"]
    run_dir = output_root / f"{safe_id}__{task}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=tcfg["num_train_epochs"],
        learning_rate=tcfg["learning_rate"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        weight_decay=tcfg["weight_decay"],
        warmup_ratio=tcfg["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
        report_to=[],
        logging_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    pred_out = trainer.predict(ds_test)
    y_pred_ids = np.argmax(pred_out.predictions, axis=-1)
    y_pred_labels = [i2l[i] for i in y_pred_ids]
    y_true_labels = test_df[label_col].tolist()
    acc = accuracy_score(y_true_labels, y_pred_labels)
    f1m = f1_score(y_true_labels, y_pred_labels, average="macro")

    save_predictions(
        model_id=safe_id,
        task=task,
        seed=seed,
        test_df=test_df,
        y_true=y_true_labels,
        y_pred=y_pred_labels,
        extra={
            "accuracy": acc,
            "f1_macro": f1m,
            "model_label": model_spec["label"],
            "hf_name": model_spec["hf_name"],
            "family": model_spec["family"],
            "training_seconds": round(train_seconds, 1),
            "training_args": tcfg,
        },
    )
    try:
        import shutil

        shutil.rmtree(run_dir / "trainer", ignore_errors=True)
    except Exception:
        pass

    print(
        f"[{model_spec['label']}|{task}|seed={seed}] "
        f"DONE acc={acc*100:.2f} f1m={f1m*100:.2f} "
        f"train_time={train_seconds:.1f}s"
    )
    return {
        "model": safe_id,
        "model_label": model_spec["label"],
        "task": task,
        "seed": seed,
        "accuracy": acc,
        "f1_macro": f1m,
        "training_seconds": train_seconds,
    }


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="model ids from configs/transformers.yaml; default = all",
    )
    ap.add_argument(
        "--task",
        nargs="+",
        default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(DATASET_CSV)
    if args.all or args.model is None:
        model_keys = [m["id"] for m in cfg["models"]]
    else:
        model_keys = args.model

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Models: {model_keys}")
    print(f"Tasks : {args.task}")
    print(f"Seeds : {args.seeds}")

    summary_rows = []
    for model_key in model_keys:
        spec = find_model_spec(cfg, model_key)
        for task in args.task:
            for seed in args.seeds:
                row = train_one(spec, task, seed, cfg, df, PREDICTIONS_DIR)
                summary_rows.append(row)

    out = PREDICTIONS_DIR.parent / "transformer_summary.csv"
    if out.exists():
        prev = pd.read_csv(out)
        out_df = pd.concat([prev, pd.DataFrame(summary_rows)], ignore_index=True)
    else:
        out_df = pd.DataFrame(summary_rows)
    out_df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
