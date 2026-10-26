# INSTRUMENT

Specification for the collection instrument. Vertical-agnostic and experiment-agnostic.

This document is sufficient to rebuild the instrument from scratch. It contains no prompt strings, no company names, no dates, no thresholds and no findings. Everything of that kind lives in the `EXPERIMENT.md` of an experiment branch.

---

## 1. What the instrument is

A harness that sends fixed prompt strings to AI search engines on a fixed schedule and writes raw responses to durable storage.

It has one job and does not exceed it. It does not classify, score, aggregate, normalize or measure. Every quantity a study reports is derived afterwards, in a separate pass, from stored text.

Section 3 explains why that separation is the central design decision rather than a stylistic preference.

---

## 2. Branch model

`main` carries the instrument and nothing else. The prompt directory is empty. There is no `EXPERIMENT.md`. The repository on `main` produces no results and is not expected to.

Each experiment lives on its own long-lived branch, named `exp/<category>-<yyyy-mm>`. An experiment branch adds `EXPERIMENT.md`, its prompt set, and its results.

**Merges are one-way.** `main` merges forward into experiment branches. Experiment branches never merge back. Instrument fixes flow outward; prompts and results never flow inward.

**Every experiment records the instrument commit it ran against.** Without that, a later instrument change silently invalidates the claim that an old experiment is reproducible. The identifier is stored on every run row, not only in the document.

**Where the instrument changes mid-window, the experiment records both commits and which runs used which.** Preferably this does not happen. If it does, hiding it is worse than reporting it.

If two experiments ever need to run simultaneously on one machine, branches stop being the right mechanism and the prompt set becomes configuration passed in at run time. That is a later problem and the layout in section 11 already permits it.

---

## 3. Raw-only collection

**The rule.** During a collection window the instrument writes raw rows and nothing else. No pattern matching, no classification, no URL normalization, no aggregation while collection is live.

**The reason.** A coding or normalization error committed during the window is unrecoverable if it overwrites the original. Storing raw text and deriving afterwards turns any such error into a rerun of a script rather than a lost window. It also lets the coding taxonomy change after the data exists, which it will, without invalidating anything already collected.

**The exception set.** A small number of fields are captured at run time. Each qualifies on one criterion only: it cannot be reconstructed from stored text later.

| Field group | Why it cannot be reconstructed |
| --- | --- |
| Response metadata returned by the API | Not present in the answer text |
| Source position within the answer | Ordering is lost once sources are stored as a set |
| Daily domain redirect state | A redirect that existed during the window leaves no later trace |
| Third-party ranking snapshot | Describes the state of an external system during the window only |

Anything not on that list is derived afterwards. This includes whether a company was mentioned, which is pattern matching over stored text and therefore recomputable at will.

**Consequence for implementation.** The runner is a writer of rows. Any function that judges, classifies or normalizes belongs to the post-collection pass, not to the instrument.

---

## 4. Sampling structure

The instrument does not choose an experiment's counts. It requires a structure, and an experiment declares its numbers inside that structure.

### Required structure

```
intents
  └── paraphrases          multiple distinct wordings of one intent
        └── repetitions    identical string, same day, back to back
              └── days     repeated across the collection window
```

### Constraints any experiment must satisfy

**At least two repetitions per string per day, fired back to back within the same hour.** Repetitions of an identical string seconds apart measure decoding noise alone. The same string on different days measures noise plus change over time. Without both, the two are not separable and no claim about change is defensible. Two repetitions is the floor. Three is the point at which the noise estimate stops being one number with no spread.

**A fixed hour of day, identical every day.** Time of day is a plausible confound on retrieval behaviour. Holding it constant removes it rather than modelling it.

**At least three paraphrases per intent.** With fewer, the intent is one wording wearing a label.

**At least four days.** Fewer than four days gives fewer than three change points, which does not support any statement about movement.

### Clustering, and what it permits

Runs are nested inside paraphrases. If variance between paraphrases dominates variance between repetitions, a large run count across few strings behaves closer to the number of strings than to the number of runs.

This does not invalidate any design. It sets the level at which results may be reported.

| Level | Independent clusters | Reporting status |
| --- | --- | --- |
| Entity, pooled across intents | intents x paraphrases | Reportable as a proportion |
| Entity within a single intent | paraphrases | Directional only, raw counts not percentages |

An experiment that reports a tight percentage at the second level is overstating its data. The instrument cannot prevent that. The document can name it.

---

## 5. Prompt set requirements

The instrument treats prompt strings as data. It does not generate them. What it requires:

**The set is closed before collection begins and cannot grow afterwards.** A set that grows after results are visible is no longer a fixed instrument.

**Every string carries an intent identifier and a paraphrase identifier as separate fields**, not encoded only in a name that must be parsed.

**Paraphrases within an intent vary along declared axes, one dominant per string.** Typical axes: the noun used for the category, length from fragment to full sentence, framing as direct question against situation description, presence of a single qualifier, speaker persona. An experiment declares which axes it used.

**All paraphrases of an intent must be answerable by the same list.** If they are not, the set has drifted across two intents and the intent is no longer the unit of analysis.

**Rejection criteria an experiment should apply and record.** A draft naming a specific vendor inside an unbranded intent. A draft that has drifted into a neighbouring intent. A draft that reads as marketing copy rather than a buyer's own words. Where a qualifier defines the intent, a draft omitting it, since the intent then collapses into the general case.

**Branded intents are reported separately from unbranded ones.** An intent that names a vendor guarantees that vendor appears. Its mention rate is not comparable to an unbranded intent and must never enter a pooled primary metric.

---

## 6. Engines

The instrument supports any number of engine cells. Each cell is one access path to one surface.

### Model tier selection principle

Where a model identifier can be chosen, **pin the model the consumer surface serves by default, not the frontier model on the top paid tier.** These studies measure what a prospective buyer sees. A power user's model is not that.

Pinned identifiers are re-verified on the morning collection begins. A vendor changing its default tier mid-study is a live risk and a silent one.

Both the requested identifier and the identifier the response reports are stored on every row. They can differ, and that difference is data.

### Search fan-out

Where an engine exposes the sub-queries it actually issued, they are captured verbatim and stored in whatever shape the vendor returns them. Normalizing across vendors happens after collection, where it is visible and revisable.

Note that some vendors bill per issued sub-query rather than per prompt.

### Surfaces without an API

A cell may have no API. Such a cell is accessed by third-party scraper, its vendor and endpoint are recorded per run in place of a model identifier, and its results are reported as not directly comparable to API cells.

### Browser automation is out of scope

Driving consumer web interfaces through browser automation is excluded. It carries account termination risk, requires residential proxy infrastructure, and its output is not more faithful than the API path where an API exists. Section 9 describes the limited manual procedure that substitutes for it.

---

## 7. Engine adapter contract

Every engine is reached through one boundary, and nothing above the boundary knows which engine is in use.

An adapter accepts a prompt string and a pinned model identifier. It returns the run fields in section 8.1 that the response can supply, plus an ordered list of sources.

Three properties are required of every adapter:

1. **It does not raise.** Transport failures, rate limits and malformed responses are returned as a populated error field on an otherwise complete row. A run that failed is recorded as a run that failed, never as a run that returned nothing.
2. **It does not interpret.** Answer text passes through unmodified.
3. **It reports whether search fired.** An answer produced without retrieval is a categorically different observation from one produced with it. Conflating them corrupts every citation figure downstream.

Adding an engine means adding one adapter. It does not mean touching storage, scheduling or the prompt set.

---

## 8. Data model

Two tables are written by the instrument during collection. A third is fixed before collection. A fourth is written by a separate daily job.

### 8.1 Runs

One row per run.

| Field | Type | Note |
| --- | --- | --- |
| run identifier | text, primary key | encodes intent, paraphrase, engine, day, repetition |
| instrument commit | text | which build produced this row |
| timestamp | text | full ISO 8601 with timezone offset, never date only |
| engine | text | |
| model requested | text | the pinned identifier sent |
| model returned | text | the identifier the response reports |
| intent identifier | integer | |
| paraphrase identifier | integer | |
| prompt text | text | stored on the row, not joined from elsewhere |
| repetition index | integer | |
| day index | integer | |
| raw response text | text | the whole answer, verbatim, unmodified |
| search fired | integer | whether the engine invoked its search tool |
| fan-out queries | text | as returned, unreshaped |
| latency | integer | milliseconds |
| finish reason | text | catches truncation, which would otherwise read as genuine absence |
| error | text | null on success |
| cost or token counts | text | whichever the vendor returns |
| scraper vendor, scraper endpoint | text | populated only for cells with no API |
| raw response envelope | text | the complete unparsed response object |

**Timestamps carry timezone offsets.** Within-day repetition spacing is part of the analysis and a naive timestamp destroys it.

**Prompt text is stored on every row.** Joining to a prompt table would make historical rows depend on a table that might later be edited. This denormalization is deliberate.

**Errors are stored, not dropped.** A failed call is not an observation of absence. Rows carrying an error are excluded from denominators and the exclusion count is reported.

**The raw envelope is stored** because every other field on the row is a parse of it. If the parse is wrong, only the envelope permits recovery. This is the section 3 rule applied one level up.

### 8.2 Sources

One row per cited source per run.

| Field | Type | Note |
| --- | --- | --- |
| run identifier | text | foreign key |
| position | integer | order of appearance within the answer |
| raw URL | text | exactly as returned |
| domain | text | parsed host |
| title | text | as returned |

**Position is captured at write time** because it is the one field here that cannot be recovered later.

**No normalized URL column is populated during collection.** Stripping tracking parameters, dropping trailing slashes, lowercasing hosts and removing subdomain prefixes happen in the post-collection pass. Storing only the raw URL means a normalization error stays fixable.

### 8.3 Population

One row per entity, fixed before collection begins. The instrument defines the shape. An experiment supplies the rows and the inclusion frame.

| Field | Note |
| --- | --- |
| entity identifier | stable across the study |
| canonical name | |
| aliases | every spelling that counts as a mention |
| canonical domain | |
| alternate domains | resolved after collection, see below |
| descriptive attributes | whatever cuts the experiment intends to report |

**Neither the alias list nor the alternate domain list gates the start of collection.** Both are applied in the post-collection pass. An experiment blocked on either has misread this document.

**The population list is frozen before collection and never grows**, including for an entity the engines name repeatedly. Entities named by an engine but absent from the population receive no row and are counted only as a reported total, printed beside any concentration figure so the denominator stays honest.

### 8.4 Alternate domains, split in two halves

**Live half, automated, daily.** A job resolves every canonical domain once per day during the window, follows redirects, and records the final URL and status code. This half cannot be reconstructed afterwards.

**Deferred half, after collection.** Cited domains not already matching a canonical domain are extracted from the sources table, deduplicated and resolved then.

**Why this ordering is better and not merely cheaper.** A pre-built list of alternate domains is a guess about which alternates might exist. A post-hoc pass is grounded in which alternates were actually cited, so it cannot miss one that was never imagined. That omission is the real failure mode, because a missed alternate manufactures false observations of an entity being named without its own site being cited.

**Limitation any experiment inherits.** An alternate domain cited but never in the population list is resolved after the window rather than during it. The risk that it changed hands in the interval is small and not zero.

### 8.5 Once-per-window collection

| Item | Cadence | Note |
| --- | --- | --- |
| Third-party ranking presence | once | cannot be collected retrospectively |
| Crawler access policy per domain | once | treated as unverified until checked directly |
| Domain redirect snapshot | daily | see 8.4 |

---

## 9. Interface calibration

The API path and the consumer interface may not behave identically. Where an experiment includes cells with both, a small manual sample is run through the consumer interface for the same prompts in the same window, with a screenshot captured for each.

Results are reported as calibration and as a limitation, never as a headline finding. A sample of this size supports a flag, not a claim.

---

## 10. Integrity rules

**A missed day is recorded as missed and never backfilled on another calendar day.** The day index is the axis along which change over time is measured. Two calendar days on one index silently corrupts every drift figure.

**Repetitions run within the same hour, back to back.** A repetition drifting into another hour is no longer measuring decoding noise and is recorded as such.

**Failed runs are recorded, not retried into the same slot.** Where retries are used, they are logged as additional attempts with their own timestamps, and the analysis states which attempt was used.

**Nothing is edited in place.** Corrections are appended and the original row is retained.

**No aggregation runs while collection is live.** Not for monitoring, not for curiosity. Looking at partial results during a pre-registered window creates pressure to adjust the instrument, which is exactly what pre-registration exists to prevent.

---

## 11. What is derived after collection

Listed here so a reader can confirm none of it is present in the raw tables.

**Per unique URL, computed once.** Source class, whether the publisher is itself an entity in the category under study, and a classifier confidence score used to draw a manual spot-check sample.

**Per run and entity pair.** Whether the entity was mentioned, whether it was recommended, whether its own domain was cited, whether a link to its own domain was present, whether it was mentioned without its own citation, and the character offset of first mention.

Mention is pattern matching over stored text against the alias list. Recommendation applies a stated edge rule: an entity named as someone else's competitor, named in a caveat, or named in an unrelated aside counts as mentioned but not recommended. An entity not mentioned cannot be recommended.

**Per run.** Count of entities named, count of sources, and vendor share of sources for that run.

**Aggregates.** Whatever the experiment pre-specifies. The instrument imposes one constraint on all of them, below.

### Eligibility floor

**The rule is instrument-level. The threshold is experiment-level.**

An entity enters a per-entity statistic only if it reaches a declared minimum mention count within an engine. The count of qualifying entities is reported per engine before any other number.

If fewer than a declared minimum number of entities qualify within an engine, per-entity statistics for that engine are reported as **untestable rather than as a result. Untestable is not falsified.**

This floor exists because in sparse categories most entities may not appear at all, in which case most per-entity ratios are undefined for most of the population and a ratio of zero over zero gets quietly reported as a finding.

---

## 12. Repository layout

Identical on `main` and on every experiment branch. Directories marked empty are empty on `main`.

```
.
├── INSTRUMENT.md           this document, identical on every branch
├── EXPERIMENT.md           absent on main, present on experiment branches
├── README.md               how to run the instrument, and observed
│                           vendor API behaviour, dated
├── config/                 experiment configs and inputs
│   └── prompts/            prompt strings as data          [empty on main]
├── collector/
│   ├── paths.py            every directory of this layout, in one place
│   ├── storage             connection, schema init (schema.sql), row writes
│   ├── runner              orchestration, one prompt to one row
│   └── engines/            one adapter per engine, identical contract
├── analysis/               post-collection pass, strictly separate
│   └── aggregate           token usage and cost, computed from stored runs
└── data/                   stage-numbered, not published in this form:
    ├── 01_raw/             collector output (runs/, database)
    ├── 02_aux/             auxiliary collection
    ├── 03_coded/           coding pipeline
    └── 04_results/         figures and charts
```

Three properties are load-bearing.

**`analysis/` may import from `collector/`. The reverse is never permitted.** A one-way dependency makes the section 3 separation checkable rather than aspirational. If the arrow ever needs to point the other way, judgement has leaked into collection.

**`config/prompts/` holds data, not source.** It is an input to a study and a reader must be able to diff it without reading code. It is the one directory that differs between experiment branches, which is why it is a directory and not a constant.

**`main` is runnable and produces nothing.** A clone of `main` should initialize, connect to an engine, and fail cleanly on an empty prompt set. That is the correct behaviour and it is the fastest way to verify a fresh setup.

---

## 13. Reproducibility

The dependency lockfile is committed. Model identifiers are pinned and stored per row. Prompt strings are fixed, versioned, and stored as data. The schema is a single definition applied at initialization rather than a migration sequence, because the instrument has a defined operational life per experiment and a migration history would describe changes no published number depends on.

The working database is not the publication artifact. Published data is exported in a documented tabular format and that export is the citable object.

Observed vendor API behaviour, including response shapes and any deviation from vendor documentation, is recorded in the README with the date of observation. Vendor surfaces change and an undated note is worth little.

---

## 14. Conformance checklist for an experiment

An `EXPERIMENT.md` must declare all of the following. The instrument supplies none of them.

- [ ] Category under study and the entity inclusion frame, with in, out, and edge rules
- [ ] Population list, its source, and its freeze date
- [ ] Engine cells and the access path for each
- [ ] Pinned model identifier per cell, and the date each was verified
- [ ] Intent list, with each intent marked branded or unbranded
- [ ] Full prompt set, every string, with intent and paraphrase identifiers
- [ ] Paraphrase axes used, and rejection criteria applied
- [ ] Counts: intents, paraphrases, repetitions per day, days, resulting totals
- [ ] Collection window dates, fire hour, and its timezone
- [ ] Eligibility floor: minimum mentions per entity, minimum qualifying entities per engine
- [ ] Primary metric, its formula, its denominator, and its unit of analysis
- [ ] Confirmatory claims, each with a falsifier stated before collection
- [ ] Pre-specified descriptive tables, each with a formula and a comparison number, and no prediction
- [ ] Instrument commit the collection ran against
- [ ] Limitations specific to this experiment's choices

An experiment missing any of these is not reproducible from its branch, whatever its results say.
