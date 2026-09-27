# GPF dataset quality audit

Input: `data/responses.csv`. SHA-256: `c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da`.

The dataset has 7,056 rows and exactly 28 columns. It is a complete 12-model × 2-article × 42-identity × 7-question-family design, with 588 exact prompt cells. Every cell is present once per model. The full prompt is byte-identical across models in each cell; the Y metadata matches the prompt question in every row.

## Identifiers and completeness

Use `(model_name, cell_id)` as the observation key. `run_id` is not globally unique: 588 run IDs are shared between streams, giving only 6,468 unique run IDs. There are no duplicate model/cell pairs, empty responses, empty prompts, or missing source links. No exact or casefolded/whitespace/header-normalized response duplicates were detected.

Missing `news_modified_utc` values occur in the 3,528 Reuters rows. Missing `thread_deleted` occurs in the 6,468 non-Gemini rows; blank must be interpreted as unrecorded, not false. All stored response word counts equal whitespace-split counts of the raw response.

## Recorded text issues

386 of 588 Gemini 3.5 Flash-Lite responses begin with the browser label `Gemini said`. Remove only this leading label in derived analysis text and retain the original response. This changes the full-dataset count outside the requested 100–150-word range from 2,135 to 2,137 (Gemini 77 to 79). Variation in answer length can influence semantic distances and should be reported.

Five Gemini rows contain strong prompt/response target inconsistencies. Four consecutive entries (Gemini positions 189–192) answer the identities/families of positions 193–196, although they are not exact duplicate texts. This is suggestive of collection misalignment or conversation contamination, but cannot distinguish capture mistakes from generated behavior. One response discusses both articles despite a single-article prompt. These observations must not be silently corrected or attributed to model bias.

| CSV row index (zero-based) | Cell | Requested target/task | Response target/task |
| --- | --- | --- | --- |
| 6469 | `bbc_brewdog__gender_sexuality__07__significance` | Nonbinary / significance | Disabled / significance |
| 6657 | `reuters_denmark__religion__06__worldview` | Atheist / worldview | Nonbinary / policy_action |
| 6658 | `bbc_brewdog__political__00__bias_check` | Conservative / bias_check | Coastal Elite / worldview |
| 6659 | `bbc_brewdog__geography__05__community` | Coastal Elite / community | Transgender / Nonbinary / emotion |
| 6660 | `reuters_denmark__gender_sexuality__00__emotion` | Straight Male / emotion | Native American / bias_check |

Five later Gemini answers include the nonsupplied citation domain `www.pravda.com.ua`. This is evidence of unsupported citation text, not proof that a browser/tool was used. Three responses contain fewer than 50 words; shortness alone is not evidence of truncation.

## Recommended analysis masks

Retain all 7,056 rows as the primary descriptive dataset. Analyze header-cleaned text. Report an alignment sensitivity excluding the five flagged observations (7,051 rows); for balanced comparisons across every model, drop the five corresponding cells for every model, leaving 583 cells and 6,996 rows. Do not drop all length outliers or nonsupplied citations by default; those can be meaningful generated behavior. No exact-duplicate exclusion is needed.

## Coverage of Y labels

| Dimension | Labels | Responses |
| --- | ---: | ---: |
| socioeconomic | 9 | 1512 |
| gender_sexuality | 8 | 1344 |
| geography | 6 | 1008 |
| race_ethnicity | 7 | 1176 |
| political | 5 | 840 |
| religion | 7 | 1176 |

Every identity has 168 rows (12 models × 2 articles × 7 families). Labels overlap across and within dimensions, and dimension sizes differ; compare dimensions with equal-weighted summaries rather than raw totals.

## Model text length and capture range

| Model | Clean mean words | Raw range violations | Clean range violations | First capture | Last capture |
| --- | ---: | ---: | ---: | --- | --- |
| GPT 5.4 | 143.3 | 67 | 67 | 2026-09-19T07:22:38.221Z | 2026-09-25T01:55:09Z |
| GPT 5.4 Mini | 118.4 | 59 | 59 | 2026-09-19T07:23:05.453Z | 2026-09-25T01:56:36Z |
| Claude Sonnet 4.5 | 151.1 | 333 | 333 | 2026-09-19T07:25:33.322Z | 2026-09-25T01:56:36Z |
| Claude Haiku 4.5 | 139.5 | 112 | 112 | 2026-09-19T07:26:19.447Z | 2026-09-25T01:58:11Z |
| Meta Llama 3.3 Turbo | 94.8 | 346 | 346 | 2026-09-19T07:26:52.363Z | 2026-09-25T01:58:11Z |
| Meta Llama Maverick 4 | 118.6 | 78 | 78 | 2026-09-19T07:27:30.133Z | 2026-09-25T02:03:31Z |
| Gemini 2.5 Flash | 144.9 | 211 | 211 | 2026-09-19T07:28:51.745Z | 2026-09-25T02:03:31Z |
| Gemini 2.5 Pro | 128.3 | 10 | 10 | 2026-09-19T07:30:04.575Z | 2026-09-25T01:53:36Z |
| Mistral Large 3 | 179.8 | 440 | 440 | 2026-09-19T07:31:09.661Z | 2026-09-25T01:53:36Z |
| Mistral Medium 3 | 126.1 | 138 | 138 | 2026-09-19T07:34:28.135Z | 2026-09-25T01:55:09Z |
| Grok Fast | 149.4 | 264 | 264 | 2026-09-23T05:18:03Z | 2026-09-24T05:57:43Z |
| 3.5 Flash-Lite | 124.2 | 77 | 79 | 2026-09-25T03:19:07Z | 2026-09-27T05:34:44Z |

## Limits on inference

- One recorded generation per model and prompt; no empirical estimate of sampling variability or stable personality.
- Two articles only; news source is completely confounded with article/topic. Do not generalize to BBC versus Reuters.
- 42 identity labels in six dimensions are overlapping/non-exhaustive, including geographic labels that incorporate race/class; no causal or population-representative identity effect.
- Semantic distance measures text variation, not discrimination, factual correctness, fairness, or causal bias.
- Family/question wording changes the task. Match the exact cell_id for between-model comparisons, and hold article/family fixed for identity contrasts.
- Provider labels mix model developer, router/provider, and web product. Normalize company explicitly with provenance; preserve the recorded label.
- Model labels are UI/reported names; backend checkpoint and generation settings are not identified by this selected CSV.
- The names of different versions/tier labels do not establish longitudinal model evolution; compare same-company labeled products descriptively.
- AUC versus web platform and collection time may affect responses; Google 2.5 versus 3.5 comparisons are confounded by platform and date.
- Repeated capture timestamps indicate batch-recording metadata and cannot establish response duration or independent generations.
- Full prompts contain publisher article text. A repository should be private unless redistribution rights are established.

Audit outputs contain flags and evidence; canonical input is never modified. Run `python3 audit_data.py` from the repository root to regenerate. Paths default relative to the script, so invoking it from another working directory also works.
