# PowerShell orchestrator for the full revision pipeline.
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
#
# To skip stages, comment them out.

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."

$Seeds = @("0","1","2","3","4")
$Models = @("mbert","berturk","xlmr","albert","electra")

Write-Host "=== STAGE 1: TF-IDF baselines (fast, ~1 min total) ==="
python -m src.tfidf_baselines --seeds @Seeds

Write-Host "`n=== STAGE 2: SBERT baselines (encodes once, ~2-3 min) ==="
python -m src.sbert_baselines --seeds @Seeds --device cuda

Write-Host "`n=== STAGE 3: Transformer fine-tuning (5 models x 3 tasks x 5 seeds = 75 runs) ==="
foreach ($model in $Models) {
    foreach ($task in @("func","prio","hybrid")) {
        python -m src.transformer_train --model $model --task $task --seeds @Seeds
    }
}

Write-Host "`n=== STAGE 4: Hybrid vs. Independent (Reviewer Major #6) ==="
python -m src.hybrid_vs_independent --models @Models --seeds @Seeds

Write-Host "`n=== STAGE 5: Source-split analysis (Reviewer Major #8) ==="
python -m src.source_split_analysis --models @Models --seeds @Seeds

Write-Host "`n=== STAGE 6: Statistical tests (Reviewer Major #5) ==="
$AllModels = @(
    "TFIDF-SVM","TFIDF-LR","SBERT-LR","SBERT-SVM",
    "mbert","berturk","xlmr","albert","electra"
)
python -m src.statistical_tests --models @AllModels --seeds @Seeds

Write-Host "`n=== STAGE 7: Aggregate manuscript tables ==="
python -m src.aggregate_results

Write-Host "`n=== OPTIONAL: LLM zero-shot (Reviewer Major #7) ==="
Write-Host "Skipped by default. To run, set the API key first:"
Write-Host "  `$env:ANTHROPIC_API_KEY = 'sk-ant-...'"
Write-Host "  python -m src.llm_zero_shot --provider anthropic --model claude-sonnet-4-6"

Write-Host "`n=== OPTIONAL: Hyperparameter grid (Reviewer Mod #12) ==="
Write-Host "Run, for example:"
Write-Host "  python -m src.hyperparam_search --model mbert --task func --seed 0"

Write-Host "`nALL DONE. See results/ for tables and predictions/."
