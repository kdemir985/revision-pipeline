from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
TABLES_DIR = RESULTS_DIR / "tables"

DATASET_CSV = DATA_DIR / "requirements_dedup.csv"

for d in [RESULTS_DIR, PREDICTIONS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
