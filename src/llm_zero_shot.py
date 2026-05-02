from __future__ import annotations
import argparse
import json
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.paths import DATASET_CSV, PREDICTIONS_DIR
from src.splits import project_level_split
from src.labels import TASKS
from src.seeding import set_global_seed
from src.io_utils import save_predictions

PROMPTS = {
    "func": (
        "You will see a software requirement written in Turkish.\n"
        "Classify it as one of:\n"
        "  FR  = Functional Requirement (what the system DOES)\n"
        "  NFR = Non-Functional Requirement (quality attribute: "
        "performance, security, scalability, usability, etc.)\n"
        "Reply with EXACTLY one token: FR or NFR. No other text.\n\n"
        "Requirement: {text}\n"
        "Label:"
    ),
    "prio": (
        "You will see a software requirement written in Turkish.\n"
        "Assign a MoSCoW priority based on the textual cues alone:\n"
        "  Must   = critical, system cannot ship without it\n"
        "  Should = important but not vital for launch\n"
        "  Could  = desirable, deferrable improvement\n"
        "Reply with EXACTLY one token: Must, Should, or Could. No other text.\n\n"
        "Requirement: {text}\n"
        "Label:"
    ),
    "hybrid": (
        "You will see a software requirement written in Turkish.\n"
        "Combine functionality type AND MoSCoW priority into one label.\n"
        "Allowed labels (output exactly one, no other text):\n"
        "  FR-Must, FR-Should, FR-Could, NFR-Must, NFR-Should, NFR-Could\n\n"
        "Requirement: {text}\n"
        "Label:"
    ),
}


def normalize(raw: str, task: str) -> str:
    raw = raw.strip().split("\n")[0].strip().strip(".").strip()
    valid = TASKS[task]["labels"]
    for v in valid:
        if raw == v:
            return v
    for v in valid:
        if raw.lower() == v.lower():
            return v
    for v in valid:
        if v.lower() in raw.lower():
            return v
    return raw


def call_anthropic(model: str, prompt: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    ap.add_argument(
        "--model",
        required=True,
        help=(
            "Anthropic example: claude-sonnet-4-6 / claude-opus-4-7 / claude-haiku-4-5; "
            "OpenAI example: gpt-4o / gpt-4o-mini"
        ),
    )
    ap.add_argument("--seed", type=int, default=0, help="Project split seed")
    ap.add_argument("--n", type=int, default=200, help="Test subset size")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between calls")
    args = ap.parse_args()

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY env var first.")
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY env var first.")

    df = pd.read_csv(DATASET_CSV)
    set_global_seed(args.seed)
    _, test_df, _, _ = project_level_split(df, seed=args.seed)
    sample = test_df.sample(n=min(args.n, len(test_df)), random_state=args.seed)
    print(f"Sampled {len(sample)} of {len(test_df)} test instances (seed={args.seed})")

    model_id = f"LLM-{args.provider}-{args.model}"
    summary = []
    for task in args.tasks:
        label_col = TASKS[task]["column"]
        y_true, y_pred, raw = [], [], []
        for i, row in enumerate(sample.itertuples(), start=1):
            prompt = PROMPTS[task].format(text=row.text)
            try:
                if args.provider == "anthropic":
                    out = call_anthropic(args.model, prompt)
                else:
                    out = call_openai(args.model, prompt)
            except Exception as e:
                print(f"  [err i={i}] {e}")
                out = ""
            pred = normalize(out, task)
            y_true.append(getattr(row, label_col))
            y_pred.append(pred)
            raw.append(out)
            if i % 25 == 0:
                acc_so_far = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)
                print(f"  {task} {i}/{len(sample)}  running acc={acc_so_far*100:.1f}")
            time.sleep(args.sleep)

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        save_predictions(
            model_id=model_id,
            task=task,
            seed=args.seed,
            test_df=sample,
            y_true=y_true,
            y_pred=y_pred,
            extra={
                "accuracy": acc,
                "f1_macro": f1m,
                "provider": args.provider,
                "model": args.model,
                "n": len(sample),
                "raw_outputs_sample": raw[:10],
            },
        )
        summary.append(
            dict(
                model=model_id,
                task=task,
                seed=args.seed,
                accuracy=acc,
                f1_macro=f1m,
                n_test=len(sample),
            )
        )
        print(f"  {task}: acc={acc*100:.2f} f1m={f1m*100:.2f}")

    out = PREDICTIONS_DIR.parent / "llm_summary.csv"
    if out.exists():
        prev = pd.read_csv(out)
        out_df = pd.concat([prev, pd.DataFrame(summary)], ignore_index=True)
    else:
        out_df = pd.DataFrame(summary)
    out_df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
