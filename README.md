# GPF semantic response audit

A matched-prompt analysis of **7,056 recorded responses**, covering **12 model streams**, **42 Y identity labels in six dimensions**, **seven question families**, and **two news articles**. Every model has a response for each of the same 588 prompt cells. The canonical CSV preserves the requested 28 columns, full prompts, exact responses, and source links.

**Authors:** Mohd Ariful Haque and Kishor Datta Gupta.

This is a research snapshot. The dataset and dashboard contain publisher article text; see [RIGHTS.md](RIGHTS.md). Recorded model labels are not independently verified backend checkpoints. Distances describe text variation, not factual accuracy, discrimination, or a model quality ranking.

## Read and explore

- [Kaggle dataset and detailed data card](https://www.kaggle.com/datasets/kishor1123/gpf-matched-news-7056-responses-from-12-models)
- [Run the public starter notebook on Kaggle](https://www.kaggle.com/code/kishor1123/gpf-matched-news-starter-analysis-and-validation): explore the attached dataset, validate its checksum and coverage, and generate summary tables.
- [ResearchGate preprint](https://www.researchgate.net/publication/414852735_GPF_Matched-News_Dataset_Semantic_Variation_Across_Models_and_Identity_Prompts)
- [Starter Jupyter notebook](notebooks/GPF_Starter.ipynb): load and validate the CSV, summarize models and Y groups, filter records, compare responses to the same prompt, and export simple summaries. See [starter requirements](requirements-starter.txt).
- The editable report is maintained separately in Overleaf.
- [Full analysis report PDF](report/main.pdf)
- [Offline interactive dashboard](dashboard/index.html): download the HTML and open it in a browser. GitHub's file viewer does not execute HTML. No server or external service is needed.
- [Dataset CSV, gzip-compressed](data/responses.csv.gz): decompress to obtain `responses.csv`; its exact SHA-256 is below.
- [All 28 column definitions](DATA_DICTIONARY.md) and [dataset card](DATASET_CARD.md)
- [Data quality audit](results/data_quality.md), [flagged rows](results/flagged_rows.csv), and [validation record](results/validation.json)
- [Complete downloadable analysis package](GPF_Analysis_Complete.zip)

## Please cite the preprint

If you use this dataset, notebook, code, dashboard, or analysis in research, teaching, or another project, please cite the following ResearchGate preprint and link to the Kaggle dataset. This helps the community find the original data, study design, and interpretation limits.

**Haque, Mohd Ariful, and Gupta, Kishor Datta. (2026). _GPF Matched-News Dataset: Semantic Variation Across Models and Identity Prompts_. ResearchGate preprint. [Read and cite the paper](https://www.researchgate.net/publication/414852735_GPF_Matched-News_Dataset_Semantic_Variation_Across_Models_and_Identity_Prompts).**

[Citation text and BibTeX](CITATION.md) are provided for reuse. The paper is a preprint, and no DOI has been assigned. Please identify the dataset version and any filtering or cleaning applied in your work. Citation does not replace the third-party rights conditions in [RIGHTS.md](RIGHTS.md).

## Main findings

The primary MPNet representation places the two Meta models closest together (mean cosine distance **0.0903**), followed by the Mistral pair (**0.1051**), Anthropic pair (**0.1145**), and OpenAI pair (**0.1206**). GPT 5.4 and Meta Llama 3.3 Turbo are the most distant pair (**0.1833**). These are averages over 588 identical prompts for each pair.

Across the seven within-company model pairs, the mean distance is **0.1249**; across 59 between-company pairs it is **0.1502**. The paired difference is **0.02525**, with a descriptive 95% identity-cluster bootstrap interval of **0.02278–0.02762**, conditional on these two articles. Google contributes three within-company pairs, the other four companies with multiple models contribute one each, and xAI has no within-company pair. The company aggregates therefore describe this model inventory, not a balanced comparison of companies.

The two semantic encoders agree moderately on the ranking of all 66 model pairs (Spearman **0.770**). Exact identity-label masking and a lexical TF–IDF baseline provide additional checks; encoder scales are not interchangeable. Masking replaces literal supplied identity labels, not their synonyms or all demographic meaning.

Five Gemini responses contain strong content/target inconsistencies. They remain in the dataset. A conservative sensitivity analysis excludes their entire prompt cells across all models, leaving **583 cells / 6,996 responses**. The primary pair ranking remains very similar (Spearman **0.994**); the largest change in a pair's mean distance is **0.00347**. This check does not prove that every remaining row is free of collection error.

When model, article and question family are held fixed, identity-response distances vary across the six dimensions. In the screened MPNet analysis, mean sensitivity is largest for religion (**0.2741**) and smallest for gender/sexuality (**0.1776**), averaging the twelve model-specific estimates. These values depend on the supplied labels and their heterogeneous definitions. They do not measure the importance of people, protected groups, or population-level effects.

Same-company version and tier contrasts are cross-sectional. In particular, the Google web product was collected on a different platform and date from its sandbox counterparts, with an unknown backend model ID. The data cannot establish a temporal improvement or deterioration in fairness.

## Reproduce

The recorded environment uses Python 3.12 on CPU. Encoder revisions are pinned in [encoder_lock.json](encoder_lock.json). SentenceTransformer model weights are downloaded from Hugging Face the first time; no response is submitted to a model-generation API. All response embedding inference runs locally.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` decompresses the canonical CSV if necessary, audits it, computes the semantic and lexical analyses, runs the conservative alignment sensitivity, verifies chunking and numerical results, and rebuilds the standalone dashboard. Full recomputation downloads approximately hundreds of megabytes of encoder weights and uses local embedding caches. These caches are omitted from version control; all source data, encoder revisions, code, result tables, and matched-distance arrays are included.

The lexical baseline also pins its original 50,000 features in [`results/tfidf_vocabulary.json`](results/tfidf_vocabulary.json), with checksums for both the cleaned corpus and the ordered vocabulary. At the feature cutoff, 19,802 candidate terms have the same frequency, and only 7,867 can be retained. An unconstrained `max_features=50000` refit can choose different tied terms on different NumPy/platform implementations; this caused a cross-platform validation discrepancy. Reproduction therefore uses the exact originally selected vocabulary and refits its IDF weights on the canonical corpus. This preserves the published numerical results and avoids silently changing the lexical baseline. The independent validator checks the vocabulary and corpus checksums before recomputing its distances.

To rebuild only the offline dashboard after extracting the CSV:

```bash
python build_dashboard.py
```

Read the report as a [PDF](report/main.pdf). The editable report is maintained separately in Overleaf. The full report includes per-model profiles, all six dimensions and all 42 labels, every pair's numerical comparison, article and question-family contrasts, and qualitative examples with exact response evidence. The repository provides the dataset, analysis code, result tables, dashboard, and report PDF.

## Measurement and uncertainty

The primary and secondary encoders are [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Each response is split into conservative 220-token chunks, embedded locally, combined using token-count weights, then L2-normalized. Independent tokenizer checks confirm that every decoded chunk fits the encoder's sequence limit; no response tail is discarded. Chunk boundaries can still alter tokenization and representation. [SentenceTransformers documentation](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) describes the underlying similarity comparison.

Cosine distance is `1 − cosine similarity`. The primary analysis removes leading captured UI headers while retaining exact response text in the CSV and dashboard. For between-model comparisons, both responses have exactly the same full prompt. For within-model identity sensitivity, article and question family are fixed and supplied labels are varied within their recorded dimension.

Bootstrap intervals use 2,000 draws with seed 20260927. Between-model comparisons resample 42 identity clusters; the screened analysis uses a ratio-of-sums estimator because deleting five cells creates unequal cluster sizes. Identity-sensitivity intervals resample the 14 fixed article-by-family block means and are descriptive conditional uncertainty. There are no repeated stochastic responses, a neutral-persona control, randomized backend assignment, or independent samples of news sources. BBC versus Reuters is completely confounded with article/topic.

## Files and provenance

| Path | Contents |
|---|---|
| `data/responses.csv.gz` | Lossless gzip of the selected 28-column input |
| `audit_data.py` | Schema, coverage, identifiers, metadata, length, and collection-quality checks |
| `analyze.py` | Embeddings, 66 matched-model comparisons, all Y contrasts, company summaries, uncertainty, and figures |
| `screen_analysis.py` | Conservative 583-cell alignment sensitivity and nonflagged case candidates |
| `lexical_features.py`, `results/tfidf_vocabulary.json` | Original TF–IDF feature indices and checked corpus/configuration lock for cross-platform reproduction |
| `validate_chunking.py` | Independent tokenizer and cache-metadata validation |
| `validate_results.py` | Independent recomputation and data/result integrity checks |
| `build_dashboard.py`, `dashboard/template.html` | Standalone HTML builder and UI source |
| `results/` | Machine-readable tables, audit findings, intervals, and exact case evidence |
| `figures/` | Static scientific figures in PDF/PNG |
| `report/` | Dataset description and full analysis PDF, with report metadata and case evidence |

The unique analysis key is `(model_name, cell_id)`. `run_id` is not globally unique: Gemini reused 588 identifiers found in the AUC rows. Recorded `model_provider` values include both developers and hosting routers; the derived developer-company mapping is explicit in the code and report.

Canonical CSV SHA-256:

```text
c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da
```

Capture dates in this frozen CSV span September 19–27, 2026. Collection labels and timestamps are observations preserved from the experiment, not independently attested model-release metadata. This project analyzes the supplied completed snapshot; it does not alter the earlier GPF manuscript or collect new model responses.
