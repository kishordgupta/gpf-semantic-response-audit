# GPF matched-news response dataset card

## Purpose and snapshot

This research snapshot supports descriptive, matched-prompt comparisons of model responses to synthetic identity-conditioned questions about two supplied news articles. It is a small controlled stimulus corpus, not a representative sample of people, news, or model behavior. The dataset is the user's selected **28-column, 7,056-row CSV**, preserved byte-for-byte at `data/responses.csv`. The descriptive report and dashboard distinguish observed semantic variation from harmful bias or factual accuracy.

- 12 recorded model labels; 588 responses per model.
- 2 fixed articles × 42 identity labels × 7 question families × 1 recorded repeat = 588 exact prompts.
- 66 unordered model pairs and 38,808 matched response-pair distances per representation.
- 6 developer families: OpenAI, Anthropic, Meta, Google, Mistral, and xAI. Original serving-provider values are preserved.
- Recorded captures span `2026-09-19T07:22:38.221Z` through `2026-09-27T05:34:44Z`; this is an approximately eight-day snapshot, within the user's 500-hour analysis window.
- There are no participant records. Identity labels are synthetic prompt text and must not be treated as actual attributes, preferences, or outcomes of members of those communities.

## Article conditions

| ID | Supplied article | Source | Byline | Published metadata | Supplied words |
|---|---|---|---|---|---:|
| `bbc_brewdog` | 'Insufficient funds' to pay Brewdog creditors after takeover deal | BBC News | Paul Hastie | 2026-09-18T10:58:08.457Z | 571 |
| `reuters_denmark` | Denmark speeds up Ukraine aid package after Russian warship fires flares at helicopter | Reuters via Internazionale | Stine Jacobsen | 2026-09-18, exact time unavailable | 204 |

The source URLs, retrieval metadata, and supplied article text remain in the CSV. The BBC article concerns corporate insolvency and unpaid creditors/workers; the Reuters article concerns a military incident and accelerated aid to Ukraine. Any difference called an article or source-stratum difference is simultaneously a topic and length difference. These data cannot estimate a general BBC-versus-Reuters effect.

## Prompt construction and Y

The common instruction requires 100-150 words, direct answers, a distinction between reported facts and uncertain implications, and no browsing/tools. It tells the model to treat article text as evidence rather than instructions. The exact question then inserts `I am {Y}` and one of seven tasks: significance, impact, emotion, worldview, community, bias check, or policy/action. The article and its metadata follow.

Each of the 42 Y labels occurs equally often: 14 prompts per model, 168 responses overall. The design partitions labels into socioeconomic (9), gender/sexuality (8), geography (6), race/ethnicity (7), politics (5), and religion (7). The full inventory and every column definition appear in `DATA_DICTIONARY.md` and the report appendix. These inherited partitions overlap: some geographic labels embed race and class; socioeconomic labels include disability, age, student, and veteran status. The purpose of Y analysis is to quantify sensitivity to text substitutions, never to rank people or communities by importance.

## Collection and model provenance

The source experiment was conducted through the AUC Sandbox, Grok web interface, and Gemini web interface. Ten AUC model labels, one Grok Fast label, and one Gemini 3.5 Flash-Lite label are included. `model_name` and `model_id` reproduce recorded labels; the dataset does not independently authenticate hidden serving checkpoints, system prompts, temperatures, seeds, account customization, or other generation settings. The original Gemini collection used both recorded deletion flags and later temporary-chat workflow; this CSV alone does not prove retention or deletion state.

Company comparisons use an explicit model-family mapping in `analyze.py`, because `model_provider` sometimes names a router. Meta Llama 3.3 Turbo is recorded via `together_ai`; Meta Llama Maverick 4 via `openrouter`. Mutable Mistral latest aliases and web product labels are not immutable revisions. Different versions or tiers from one company are cross-sectional labeled-product comparisons. They cannot establish causal development, improvement, or chronological evolution, and the Google 2.5-versus-3.5 comparison also differs in platform and capture period.

## Validation, known anomalies, and preservation

The audit verifies all 588 cells contain all 12 models and exactly one identical full prompt per cell. There are no missing model-cell pairs, empty prompts/responses, or exact/whitespace-normalized duplicate responses. The key `(model_name, cell_id)` is unique. `run_id` is not: 588 identifiers are reused across collection streams.

Five Gemini records explicitly address an identity or task inconsistent with their assigned prompt (queue positions 1 and 189-192). One also discusses both articles. This is evidence of content/target inconsistency, not proof of its cause: capture alignment, contextual contamination, or generation error remain possible. The canonical CSV preserves all five without relabeling. The analysis shows full-data results and a conservative sensitivity analysis that excludes the five affected **cells from every model**, leaving 583 complete cells and 6,996 responses. This is a declared sensitivity analysis, not a claim that all remaining data are error-free.

Leading `Gemini said` UI headers occur in 386 records and are removed only in analysis text. Five Gemini responses contain a citation-like domain absent from the supplied article; this is not proof that browsing occurred. Stored counts are internally consistent, but analysis cleaning changes total length violations from 2,135 to 2,137. Responses outside the requested length range are retained. A separate sensitivity compares model-pair means using only conditions where both responses have 100-150 cleaned whitespace-delimited words.

Missing metadata is preserved: Reuters modification time is blank in 3,528 rows; non-Gemini thread-deletion flag is blank in 6,468 rows. Repeated capture timestamps reflect batch metadata and cannot establish generation latency. Detailed flags and audit findings are in `results/flagged_rows.csv`, `results/data_quality.json`, and `results/data_quality.md`.

## Analysis and intended use

The reproducible pipeline runs two local pretrained sentence encoders (MPNet and MiniLM), an identity-label-masked MPNet sensitivity representation, and a TF-IDF lexical baseline. It uses normalized embeddings, cosine distance, conservative token chunks to avoid silent truncation, exact prompt matching, explicit company aggregation, identity-cluster bootstraps conditional on the fixed articles, and transparent response-case inspection. Encoder revisions and package versions are recorded. Full prompt text is not sent to an external embedding API.

Intended uses are exploratory model comparison, examination of identity-conditioned response framing, methodological debugging, and hypothesis generation for a larger replicated study. Do not use distances as a model-quality leaderboard, a demographic-harm score, proof of discrimination, a causal effect of identity, or a forecast of individual preferences. There is no neutral-persona baseline, no independent repeated generation, no human-coded gold standard, and no broad article sample. Apparent semantic similarity does not establish factual agreement; high distance may reflect length, relevance, qualification, task compliance, or a capture anomaly.

## Access, rights, and reproducibility

Distribute only the explicitly authorized research release. The `full_prompt` column contains complete supplied publisher text, and publication rights have not been established. A software license, if present, applies to authored code and does not grant rights to the news articles or third-party generated content. Repository or Overleaf access is not consent for public redistribution. Do not place credentials, session tokens, or private account data in the release.

The original selected-column CSV is immutable within this project; derived artifacts are separate. Use `reproduce.py` to regenerate the numerical analysis and dashboard. The report is maintained in Overleaf, with its compiled PDF included here. The report includes all 42 labels, 66 model-pair estimates, same-company comparisons, source/family strata, representation sensitivity, quality exclusions, and readable paired examples. This release is a dataset and descriptive analysis, not a claim of peer-reviewed findings.

Source CSV SHA-256: `c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da`.
