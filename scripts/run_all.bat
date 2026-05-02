@echo off
REM Plain-CMD orchestrator (alternative to run_all.ps1).
REM Run from the repo root:
REM   scripts\run_all.bat

setlocal
set PYTHONPATH=.
set SEEDS=0 1 2 3 4
set MODELS=mbert berturk xlmr albert electra

echo === STAGE 1: TF-IDF baselines ===
python -m src.tfidf_baselines --seeds %SEEDS%

echo === STAGE 2: SBERT baselines ===
python -m src.sbert_baselines --seeds %SEEDS% --device cuda

echo === STAGE 3: Transformer fine-tuning ===
for %%M in (%MODELS%) do (
    for %%T in (func prio hybrid) do (
        python -m src.transformer_train --model %%M --task %%T --seeds %SEEDS%
    )
)

echo === STAGE 4: Hybrid vs. Independent (Major #6) ===
python -m src.hybrid_vs_independent --models %MODELS% --seeds %SEEDS%

echo === STAGE 5: Source-split (Major #8) ===
python -m src.source_split_analysis --models %MODELS% --seeds %SEEDS%

echo === STAGE 6: Statistical tests (Major #5) ===
python -m src.statistical_tests --models TFIDF-SVM TFIDF-LR SBERT-LR SBERT-SVM mbert berturk xlmr albert electra --seeds %SEEDS%

echo === STAGE 7: Aggregate ===
python -m src.aggregate_results

echo Done.
endlocal
