# Reviewer #5 → Script & Output Map

This file maps every numbered concern from Reviewer #5 to (a) the
script that produces the evidence, (b) the artifact path, and (c) the
manuscript section that should be edited.

## Major Concerns

### Major #1 — Unfair monolingual comparison

**What we do:**
- Add **BERTurk** (Turkish-native) and **XLM-RoBERTa** (multilingual rival).
- Keep ALBERT and ELECTRA but **reframe** them as English-monolingual
  *control* baselines that demonstrate what happens without Turkish
  pretraining (rather than as architecture comparisons).
- Delete the speculative "architectural" rationalizations from §4.4 / §5.

**Run:**
```powershell
python -m src.transformer_train --all   # 5 models × 3 tasks × 5 seeds
python -m src.aggregate_results
```

**Artifacts:**
- `results/tables/main_table_func.csv`
- `results/tables/main_table_prio.csv`
- `results/tables/main_table_hybrid.csv`

**Manuscript edits:** Tables 4–6 (replace), §4.4, §5 (rewrite the
discussion to talk about pretraining-language match rather than
architecture).

---

### Major #2 — Theoretical justification for predicting MoSCoW from text

**What we do (text-only, no code):** Add a paragraph in §1 and §6
acknowledging that the task is "predict the labels two annotators
extracted from the text", not "predict business priority". Reframe as
*surface-text proxy* for downstream triage assistance.

**Manuscript edits:** §1 (introduction caveat paragraph) and §6
(limitations).

---

### Major #3 — Annotator agreement (Cohen's Kappa)

**What we do:**
- Export a stratified 500-sample subset.
- A *third* annotator (independent of the original two) re-labels it
  blindly.
- Compute Cohen's Kappa for type, priority, and the 6-class hybrid.

**Run:**
```powershell
python -m src.kappa export --n 500 --out data\kappa_subset_to_relabel.csv
# Hand the CSV (without 'type' and 'priority' columns visible) to the
# second annotator. Save their version as kappa_subset_relabeled.csv.
python -m src.kappa score --in data\kappa_subset_relabeled.csv
```

**Artifacts:**
- `results/tables/kappa_summary.csv`
- `results/tables/kappa_confusion_type.csv`
- `results/tables/kappa_confusion_priority.csv`
- `results/tables/kappa_confusion_hybrid.csv`

**Manuscript edits:** §3.2 (annotation procedure) — add Kappa values;
§6 (limitations).

> **NB:** if the second-coding pass is not feasible, document this as
> an unfilled limitation in §6 and respond honestly that Kappa
> reporting is deferred to future work.

---

### Major #4 — Single-split sensitivity

**What we do:** Run every model on **5 independent project-level
80/20 splits** (seeds 0–4). Report mean ± std on every metric.

**Run:** part of `run_all.ps1`. All `--seeds 0 1 2 3 4` flags.

**Artifacts:**
- `results/tables/main_table_*.csv` (mean ± std across seeds)
- `results/tables/pairwise_stats.csv` (per-seed and aggregated)

**Manuscript edits:** Tables 4–6 (add std), §4 (replace
"single-split" disclaimer with multi-split protocol description), §6
(remove this limitation).

---

### Major #5 — Statistical analysis problems

**What we do:**
- (a) Apply **Benjamini-Hochberg FDR** correction across all pairwise
  tests within a (task, seed) family (also report Bonferroni for
  transparency).
- (b) Replace **Cohen's d on 0/1 vectors** with **McNemar odds ratio**
  (b/c with Haldane–Anscombe correction) and log-OR.
- (c) **Bootstrap CI for the 6-class transformer** by resampling the
  saved prediction vector (no need to re-run inference).

**Run:**
```powershell
python -m src.statistical_tests --models TFIDF-SVM SBERT-LR mbert berturk xlmr albert electra --seeds 0 1 2 3 4
```

**Artifacts:**
- `results/tables/pairwise_stats.csv` (raw, BH-adjusted, Bonferroni)
- `results/tables/pairwise_stats_agg.csv`

**Manuscript edits:** §4.5 (rewrite stats subsection); replace
Cohen's-d columns in Tables 10+ with OR; add BH-adjusted p column;
delete footnote (a) of Table 10.

---

### Major #6 — Hybrid 6-class vs. two-independent-classifier baseline

**What we do:** Compute the joint-prediction baseline that the
reviewer points at (`pred_FR/NFR ⊕ pred_priority`) and compare to the
6-class hybrid model on the same instances.

**Run:**
```powershell
python -m src.hybrid_vs_independent --models mbert berturk xlmr albert electra
```

**Artifacts:**
- `results/tables/hybrid_vs_independent.csv`
- `results/tables/hybrid_vs_independent_agg.csv`

**Manuscript edits:** §4.3 (insert the comparison) and Contributions
(§1, last bullet) — either add this evidence or, if hybrid is worse,
withdraw the claim.

---

### Major #7 — LLM zero-shot baseline missing

**What we do:** Run Claude (or GPT-4o) zero-shot on a 200-sample
subset of the held-out test fold for all three tasks.

**Run:**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python -m src.llm_zero_shot --provider anthropic --model claude-sonnet-4-6 --n 200
# or:
$env:OPENAI_API_KEY = "sk-..."
python -m src.llm_zero_shot --provider openai --model gpt-4o --n 200
```

**Artifacts:**
- `results/llm_summary.csv`
- `results/predictions/LLM-<provider>-<model>__*__seed0/`

**Manuscript edits:** Add a short subsection in §4 (or new §4.6) with
the LLM numbers and a paragraph in §5 contrasting them with the
fine-tuned baselines.

---

### Major #8 — Translation-source confound

**What we do:**
- Per-source descriptive stats (length, label balance).
- KS tests for length distribution, χ² tests for label distribution
  between TR-native (ETU) and translated (Nevon/HKU) subsets.
- Per-source accuracy/F1 breakdown for every model.

**Run:**
```powershell
python -m src.source_split_analysis --models mbert berturk xlmr albert electra
```

**Artifacts:**
- `results/tables/source_descriptive.csv`
- `results/tables/source_distributional_tests.csv`
- `results/tables/per_source_performance.csv`
- `results/tables/per_source_performance_agg.csv`

**Manuscript edits:** Add §3.3 ("Source heterogeneity check") with the
descriptive comparison and a new sub-paragraph in §4 reporting
per-source performance. Add a sentence in §6 about residual confound.

---

## Moderate Concerns

### Mod #9 — Related work is descriptive, not critical

**Text-only edit.** No script. Rewrite §2.2 paragraphs to end each
study summary with one sentence stating the unresolved question /
limitation that the present study addresses.

### Mod #10 — Remove QFD / VOP equations

**Text-only edit.** Delete equations and definitions in §2.1 for QFD
and VOP. Keep one paragraph stating *why* MoSCoW was chosen.

### Mod #11 — Native-English proofreading

Out of scope for code. Send the revision to a professional editor
after numerical results are finalized. Specific issues flagged:
- "project-level split" (Introduction)
- inconsistent capitalization of MoSCoW categories
- inconsistent "annotation"/"labeling" usage
- Abstract / Intro / Discussion show LLM rhythm (rewrite by hand).

### Mod #12 — Hyperparameter grid

**Run:**
```powershell
python -m src.hyperparam_search --model mbert --task func --seed 0
python -m src.hyperparam_search --model berturk --task func --seed 0
```

**Artifacts:** `results/tables/hpsearch_<model>_<task>_seed<N>.csv`

Also: fix Table 2 — choose **either** `warmup_ratio=0.1` or
`warmup_steps=500` and report only the chosen value (the YAML defaults
to `warmup_ratio=0.1`).

### Mod #13 — Architectural speculation

**Text-only edit.** Delete the "parameter sharing", "replaced token
detection less effective for fine-grained semantics", and similar
sentences from §4.4 and §5. Replace with the corrected discussion
based on Major #1 evidence.

---

## Minor Concerns

| # | Issue | Action |
|---|---|---|
| Min1 | TF-IDF baselines for all 3 tasks | Already covered: `python -m src.tfidf_baselines` runs all 3. |
| Min2 | Figures 6 & 7 redundant | Manuscript-only edit. Replace with a single grouped bar chart or delete. |
| Min3 | Figure 1 Phase 1 / Phase 2 | Manuscript-only edit. Add legend. |
| Min4 | Reference 10 is arXiv only | Manuscript-only edit. Replace with peer-reviewed version if it exists. |
| Min5 | Code repository | Push this folder to GitHub or Zenodo, cite DOI. |
| Min6 | Seed setting (torch+numpy+random+CUDA) | Already implemented in `src/seeding.py`. Confirm in §3.5. |
| Min7 | LLM-flavored prose | Hand-edit Abstract, Intro, Discussion. |

---

## Quick checklist for the rebuttal cover letter

- [ ] Tables 4–6 replaced with multi-seed mean ± std numbers
- [ ] BERTurk and XLM-RoBERTa added to all 3 tables
- [ ] Architectural-speculation paragraphs deleted from §4.4 and §5
- [ ] §4.5 rewritten to use BH-adjusted p, McNemar OR, prediction-vector bootstrap
- [ ] §3.2 augmented with Cohen's Kappa from re-labeling pass
- [ ] §3.3 (NEW) source-heterogeneity check
- [ ] §4.x or new §4.6 with LLM zero-shot numbers
- [ ] §4.3 augmented with hybrid-vs-independent comparison
- [ ] §1 and §6 reframe MoSCoW prediction as surface-text proxy
- [ ] §2.1 QFD / VOP equations removed
- [ ] §2.2 each related-work paragraph ends with the unresolved question
- [ ] Code repository URL/DOI added to §3 ("Code availability")
- [ ] Native-English proofreading pass after numerical content is final
