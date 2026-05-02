from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from src.paths import PREDICTIONS_DIR


def run_dir(model_id: str, task: str, seed: int) -> Path:
    safe = model_id.replace("/", "__")
    d = PREDICTIONS_DIR / f"{safe}__{task}__seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_predictions(
    model_id: str,
    task: str,
    seed: int,
    test_df: pd.DataFrame,
    y_true,
    y_pred,
    extra: dict | None = None,
) -> Path:
    d = run_dir(model_id, task, seed)
    out = test_df[["project", "req_id", "text", "source"]].copy()
    out["y_true"] = list(y_true)
    out["y_pred"] = list(y_pred)
    out["correct"] = (out["y_true"] == out["y_pred"]).astype(int)
    out_path = d / "predictions.csv"
    out.to_csv(out_path, index=False)
    meta = {
        "model_id": model_id,
        "task": task,
        "seed": seed,
        "n_test": len(out),
        **(extra or {}),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return out_path


def load_predictions(model_id: str, task: str, seed: int) -> pd.DataFrame:
    d = run_dir(model_id, task, seed)
    return pd.read_csv(d / "predictions.csv")
