# Revision Pipeline — Software Requirements Classification (Turkish)

This repository contains the experimental code that addresses the
Reviewer #5 concerns on the manuscript *"Software Requirements
Classification using Functionality Information and MoSCoW Priority
Levels: A Comparative Study of Machine Learning and Transformer Models
on a Turkish Dataset"*.

The dataset is included in `data/requirements_dedup.csv` (2,981 unique
requirements from 100 software projects, already deduplicated). It is
also published on Kaggle:
<https://www.kaggle.com/datasets/simazehrakorkulu/software-requirements>.

## 1. Setup (Windows + NVIDIA GPU)

```powershell
# Clone or copy this folder to the lab machine.
cd revision_pipeline

# One-time install (creates .venv, installs CUDA torch + everything else)
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1

# Each subsequent session:
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
```

If your driver reports a different CUDA version, edit the `cu121` tag
inside `scripts/install_windows.ps1` (e.g., `cu124`, `cu126`).

## 2. Run the full pipeline

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```

Default budget on a single 24 GB GPU:
- TF-IDF + SBERT baselines: ~5 min total
- Transformers (5 models × 3 tasks × 5 seeds = 75 fine-tunes): ~3-5 h
- Statistical tests + aggregation: <2 min

## 3. Or run stages individually

```powershell
# TF-IDF + 6 ML across 5 seeds
python -m src.tfidf_baselines --seeds 0 1 2 3 4

# SBERT (frozen) + 6 ML across 5 seeds
python -m src.sbert_baselines --seeds 0 1 2 3 4 --device cuda

# Single transformer x single task x single seed (smoke test)
python -m src.transformer_train --model mbert --task func --seeds 0

# Full transformer matrix (5 models x 3 tasks x 5 seeds)
python -m src.transformer_train --all

# Reviewer Major #6 - hybrid vs independent
python -m src.hybrid_vs_independent --models mbert berturk xlmr albert electra

# Reviewer Major #8 - source-split analysis
python -m src.source_split_analysis --models mbert berturk xlmr albert electra

# Reviewer Major #5 - statistical tests with BH correction + McNemar OR
python -m src.statistical_tests --models TFIDF-SVM SBERT-LR mbert berturk xlmr albert electra

# Reviewer Mod #12 - hyperparameter grid (run on top-2 transformers)
python -m src.hyperparam_search --model mbert --task func --seed 0

# Reviewer Major #7 - LLM zero-shot (needs API key)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python -m src.llm_zero_shot --provider anthropic --model claude-sonnet-4-6 --n 200

# Reviewer Major #3 - Cohen's Kappa (two-step workflow, see below)
python -m src.kappa export --n 500 --out data\kappa_subset_to_relabel.csv
# ... second annotator fills CSV ...
python -m src.kappa score --in data\kappa_subset_relabeled.csv

# Final aggregation -> manuscript tables
python -m src.aggregate_results
```

## 4. Outputs

Every transformer and every classical ML configuration writes the
per-instance prediction vector (so the entire statistical analysis is
fully reproducible from saved CSVs without re-running inference):

```
results/
├── predictions/
│   └── <model>__<task>__seed<N>/
│       ├── predictions.csv   # row-per-test-instance, has y_true, y_pred, correct, source
│       └── meta.json         # accuracy, f1_macro, training args, etc.
├── tables/
│   ├── main_table_combined.csv          # mean ± std across seeds
│   ├── main_table_func.csv
│   ├── main_table_prio.csv
│   ├── main_table_hybrid.csv
│   ├── pairwise_stats.csv               # raw pairwise McNemar / bootstrap / OR
│   ├── pairwise_stats_agg.csv           # aggregated with BH-corrected p-values
│   ├── hybrid_vs_independent.csv        # Major #6
│   ├── per_source_performance.csv       # Major #8
│   ├── source_descriptive.csv           # Major #8 distributional descriptors
│   ├── source_distributional_tests.csv  # KS + chi-square
│   └── kappa_summary.csv                # Major #3 (after re-labeling pass)
├── tfidf_summary.csv
├── sbert_summary.csv
└── transformer_summary.csv
```



## 6. Reproducibility

- Every script sets `torch`, `numpy`, `random`, and `CUDA` seeds via
  `src/seeding.py:set_global_seed`.
- Project-level 80/20 split is deterministic given the seed.
- `transformers.trainer.TrainingArguments(seed=…, data_seed=…)` is set
  per run.
- The exact same prediction vectors are saved and used by the
  statistical-test stage so the analysis is fully reproducible.
