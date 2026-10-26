# answerfold-collector

Open-source tool for measuring how AI search engines answer buyer questions: which companies they name and which pages they cite.

It asks AI search engines the same buyer questions many times over several days, stores every raw answer, then codes the answers (which companies are named, which pages are cited, who owns each page) and analyses the result with deterministic code.

**Current state.** The tool is set up for one study: AI receptionist software, 7 buyer questions, 4 engines (ChatGPT, Claude, Gemini, Google AI Mode). You can run it with your own prompts and keys, but the coding prompts, the inclusion frame and the figures are written for that market. Configurable markets come next.

## The pipeline

```
collect (paid API calls)  ->  code (paid model calls)  ->  analyse (deterministic, no calls)
```

1. **Collect.** One pass per day: every engine, every prompt, every repetition. Raw API responses are appended to `data/01_raw/runs/YYYY-MM-DD.jsonl`, untouched. Auxiliary inputs (cited-page fetches, robots.txt, Google top 10, site inventory) are collected separately, once per window.
2. **Code.** The raw answers go through the model-based coding pipeline: three extractors list the companies named in each answer, an arbiter settles disagreements, then entity resolution, verification, inclusion-frame classification and cited-URL classification. Output lands in SQLite and is frozen with a manifest.
3. **Analyse.** Reads the frozen raw window, auxiliary inputs and coded tables, checks every manifest hash, and computes every figure into one file. The same inputs always give the same file.

## Setup

```
uv sync
cp .env.example .env                                 # add your API keys
cp config/experiment.example.yaml config/experiment.yaml
```

API keys are read from `.env` only (see `.env.example`): OpenAI, Gemini, Anthropic and SerpApi (Google AI Mode and the Google top 10). `docs/INSTRUMENT.md` describes the design: the sampling structure, the engine adapter contract, the data model and the integrity rules.

## Run a collection

Put your prompts in `config/prompts/prompts.csv` (the file shipped there is the AI receptionists prompt set, as an example) and set engines, models and repetitions in `config/experiment.yaml`. Run one pass per day:

```
uv run python -m scripts.collect.run_collection \
    --day-index 1 \
    --prompts config/prompts/prompts.csv \
    --experiment-config config/experiment.yaml \
    --prompt-order-seed 20260820
```

Collection must run from inside a git checkout: every row records the commit of the code that produced it. Every run is a paid API call; the cost depends on your prompt count, repetitions, days and engines.

Auxiliary collection and the window freeze are in `scripts/aux/` (`collect_cited_urls`, `collect_robots`, `collect_google_serp`, `collect_site_inventory`, `build_roster`, `freeze_window`). Each prints an estimate and asks before it makes network calls; `--dry-run` shows the estimate only.

## Code the answers

```
uv run python -m scripts.coding.run_coding_pipeline \
    --runs-dir data/01_raw/runs \
    --db-path data/03_coded/coding.sqlite3 \
    --unbranded-intents 1,2
```

This makes paid model calls (OpenAI, Anthropic and Gemini). `scripts/coding/submit_coding_batches.py` and `collect_coding_batches.py` run the same stages through the batch APIs at lower cost; `report_coding_pipeline_cost.py` reports what a run cost. `freeze_phase_a.py` freezes the coded tables with a manifest.

Inputs you provide for coding, because they define your market:

- `config/inclusion_frame.yaml`: what counts as "in the study" (read as text and embedded in the classification prompt).
- `config/seed_vendor_list.csv`: the known vendors in your market (used to check extractor recall).
- `config/alias_list.csv` and `config/alt_domains_seed.csv`: known name variants and extra domains of those vendors.

Shipped as used in the study: `config/url_classification_prompt.yaml` (cited-URL classes) and `config/source_domain_classes.csv`.

## Analyse

```
uv run python -m scripts.results.run_phase_b --data-root data
```

No network and no model calls. The data path has no default; every manifest is checked before anything is computed.

## Personal-data cleaning

Before sharing answer text, run the cleaning in `cleaning/`. It works on a folder laid out like the study package (`data/raw/`, `data/coding/`, `data/ui/`, `data/aux/`; pass it with `--root`): e-mail addresses, phone numbers, US street addresses, person names (spaCy) and personal profile URLs are replaced with tokens such as `[PERSON]`. Company names and domains on the keep-list are never removed; add your own terms to `cleaning/remove_list.txt`. `cleaning/keep_list.txt` is regenerated from the data on every run; the shipped copy is the example from the AI receptionists study. It runs locally with no model or API calls, and its dependencies are an optional group:

```
uv sync --group cleaning
uv run --group cleaning python cleaning/clean_personal_data.py \
    --root <package folder> --private-dir <a folder outside the repo>
```

The log of removed values goes to the private folder, never into the repo.

## Reproduce the AI receptionists study (August 2026)

The study's data, figures and the exact code used for it are in the Zenodo package: https://doi.org/10.5281/zenodo.23559036

1. Download and unpack the package.
2. Inside it, run `uv run python scripts/build_publish_package.py`. It rebuilds every figure from the shipped tables and compares the result byte for byte with the shipped figures file.

To run this repository's tests against the package, set `ANSWERFOLD_STUDY_DATA` to the unpacked folder; without it, tests that need study data are skipped.

Comments in the analysis code that cite "Notion Part 3.x" point to the private planning document of that study; its frozen content is `docs/EXPERIMENT.md` in the Zenodo package.

## Tests

```
uv run --group dev python -m pytest
```

## Licence

MIT (see `LICENSE`).
