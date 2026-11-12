# LLM financial intelligence engine (Phase 3)

Module documentation for `ml/llm`, in the format required by PRD section 24 (Rule 6).

## Purpose

Turn news articles into **structured, validated, source-attributed financial events**
(PRD 8.3) and into **point-in-time news features per stock** that the RL observation
can join on `(symbol, trade_date)`. The model's output is never trusted: every event is
either accepted after deterministic checks or rejected with a recorded reason.

```text
NewsSource ─► consolidate (dedup) ─► entity resolution ─► prompt ─► LLM ─► validate
                                          │ none                            │
                                          ▼                                 ▼
                                  rejected.jsonl ◄──────────────── rejected / flagged
                                                                            │
              features.csv ◄─ aggregate (point-in-time) ◄─ events.jsonl ◄───┘
```

| File | Responsibility |
|---|---|
| `schema.py` | `NewsArticle`, `FinancialEvent` (PRD 8.3), rejection reasons, flags |
| `config.py` | Typed configuration; secrets only from the environment |
| `providers.py` | `LLMClient` protocol and `OpenAICompatibleClient` (plain HTTP, no SDK) |
| `securities.py` | Securities master and deterministic entity resolution |
| `ingestion/` | `NewsSource` interface, Yahoo headlines, JSONL files, de-duplication |
| `extraction/prompts.py` | Prompt construction and prompt-injection defences |
| `extraction/validation.py` | Hallucination controls |
| `extraction/analyzer.py` | Article → validated events |
| `sentiment/aggregation.py` | Events → per-stock daily features |
| `retrieval/store.py` | Local document store (embeddings with lexical fallback) |
| `summarize.py` | Source-grounded summaries with validated citations |
| `evaluation/` | Labelled fictional cases and the metrics computed on them |
| `pipeline.py`, `runs.py` | Orchestration and versioned run directories |

Backend: `backend/app/models/news.py`, migration `0004_news.py`,
`services/news_loader.py`, `services/news.py`, `api/v1/news.py`.

## Input schema

**`NewsArticle`** (PRD 5.3), from any `NewsSource`:

| Field | Meaning |
|---|---|
| `article_id` | Unique id (`yahoo_<uuid>`, or a hash for files without ids) |
| `headline` | Required |
| `text` | Full text or permitted excerpt; **empty for headline-only sources** |
| `source` | Publisher |
| `url` | Optional |
| `published_at`, `retrieved_at` | Timezone-aware, stored in UTC; naive timestamps are rejected |
| `symbols` | Tickers the source attached — hints for entity resolution, not trusted labels |
| `duplicate_ids`, `duplicate_sources` | Articles consolidated into this one |

Sources:

| Source | What it really provides |
|---|---|
| `YahooNewsSource` | Yahoo Finance search endpoint; free, unofficial, no key. **Headlines only** (title, publisher, link, time, related tickers), **recent items only, no archive**. Personal research use; not for redistribution. Responses are cached per query and UTC day under `data/cache/news`. |
| `JsonlNewsSource` | One JSON object per line with `headline`, `source`, `published_at` (+ optional `text`, `url`, `symbols`, `article_id`). The route for licensed or archived news with full text. A malformed line is an error, never skipped. |

Yahoo's search is fuzzy and is not a per-company feed. In the live runs below a query
for an NSE ticker (`INFY.NS`) returned **no** news at all, while the company name
returned up to 20 items, many about other companies. The queried symbol is therefore
never recorded as a hint; an item is linked to a company only if Yahoo tagged it with a
related ticker or the text names the company.

**De-duplication** (`ingestion/dedup.py`): same id, or identical normalised headline,
or headline token-set (Jaccard) similarity ≥ `similarity_threshold`, within
`window_hours`. A group is consolidated into its **earliest** article; ids and
publishers of the copies are kept in `duplicate_ids` / `duplicate_sources`. Text of a
later copy is not merged in, because it may contain information that did not exist at
the earliest publication time.

## Output schema

**`FinancialEvent`** — exactly PRD 8.3:

| Field | Values |
|---|---|
| `event_id` | `evt_<hash(article_id, symbol)>` — stable across re-analysis |
| `company`, `symbol` | Canonical name and symbol from the securities master |
| `event_type` | `earnings`, `merger_acquisition`, `management_change`, `regulatory`, `litigation`, `debt`, `new_contract`, `product_launch`, `corporate_governance`, `dividend`, `fundraising`, `other`, `unknown` |
| `sentiment` | `positive`, `negative`, `neutral`, `uncertain`, `mixed` |
| `sentiment_score` | −1 … 1 |
| `impact_direction` | `potentially_positive`, `potentially_negative`, `neutral`, `uncertain` — **separate from sentiment** (PRD 8.2-D) |
| `impact_horizon` | `short_term`, `medium_term`, `long_term`, `unknown` |
| `severity`, `confidence` | 0 … 1 |
| `key_facts`, `risks` | Lists of short statements |
| `source_ids` | The article id and the ids of its consolidated duplicates |
| `publication_timestamp` | The article's publication time (UTC) |

`unknown` means "the article does not say enough"; such events are stored but carry no
signal.

**Run directory** `artifacts/news/<run_id>/` (`run_id = news-<UTC time>-<config hash>`):

| File | Contents |
|---|---|
| `articles.jsonl` | De-duplicated articles that entered the run |
| `events.jsonl` | Accepted events with flags, publisher, model, provider, prompt version |
| `rejected.jsonl` | Every refused article or model output with reason, detail and the raw output |
| `features.csv` | News features per symbol and date |
| `manifest.json` | Model, provider, base URL, prompt version, config and its hash, source and what each query returned, counts, rejections by reason, SHA-256 of every file |

A run is hash-verified before it is read or loaded into the database.

**News feature table** — one row per `(symbol, trade_date)`; `trade_date` is a
normalised `datetime64[ns]`:

| Column | Type | Definition |
|---|---|---|
| `news_available` | int64 | 1 if at least one usable event is in the window, else 0 |
| `news_event_count` | int64 | Events in the window |
| `news_event_frequency` | float64 | Events in the last `frequency_window_days`, per day |
| `news_sentiment` | float64 | Σ w·s / Σ w — confidence-weighted sentiment, −1 … 1 |
| `news_impact` | float64 | Σ w·m / Σ w, m = +1 / −1 / 0 by impact direction |
| `news_severity_max` | float64 | max(d·v) — recency-decayed peak severity |
| `news_severity_weighted` | float64 | Σ w·v / Σ w |
| `news_event_score` | float64 | Σ w·v·s — signed strength; decays to 0 |
| `news_confidence` | float64 | Σ d·r·c / Σ d·r |
| `news_source_reliability` | float64 | Σ d·c·r / Σ d·c |
| `news_freshness` | float64 | max(d); 1 = an event at the close |
| `news_days_since_last` | float64 | Age of the newest event in days |

with `d = 0.5^(age / half_life[event_type])`, `r` = publisher reliability, `c` =
confidence, `s` = sentiment score, `v` = severity, `w = d·r·c`.

Where there is no usable news **every column is 0 and `news_available = 0`** — never
NaN and never an invented value. `news_days_since_last = 0` therefore only means "just
now" when `news_available = 1`.

```python
from ml.llm.sentiment.aggregation import news_feature_table

features = news_feature_table(symbols, "2026-11-01", "2026-11-12",
                              sessions=trading_dates)   # optional calendar
```

reads every verified run under `artifacts/news`. `aggregate_news_features(events,
symbols, dates, config, publishers)` is the pure function underneath.

### Point-in-time rule

For decision date `T`, only events published **at or before the NSE close of `T`**
(15:30 Asia/Kolkata) and at most `max_age_days` old are used. News published after the
close first appears on the next date. Later events can never change an earlier row;
`test_future_news_cannot_change_past_features` checks this bit-for-bit.

## Hallucination controls

Entity resolution is deterministic and happens **before** the model is called
(`securities.py`): a company counts as referenced if its name, a configured alias or
its upper-case symbol occurs in the article, or the source tagged the article with its
ticker. Articles that reference no known security never reach the model.

| Check | Outcome when it fails |
|---|---|
| Reply / event matches the JSON schema and ranges | rejected `schema_invalid` |
| Symbol is in the securities master | rejected `unknown_symbol` |
| Symbol is referenced by the article | rejected `entity_not_in_article` |
| One event per company | rejected `duplicate_event` |
| `source_ids` non-empty subset of the supplied ids | rejected `source_attribution_invalid` |
| `publication_timestamp` equals the article's | rejected `timestamp_mismatch` |
| Article not dated in the future | rejected `article_timestamp_in_future` (no model call) |
| Sign of `sentiment_score` agrees with `sentiment` | rejected `sentiment_inconsistent` |
| Every number in `key_facts` / `risks` occurs in the article | rejected `ungrounded_number` |
| Key fact shares ≥ `min_fact_overlap` of its content words with the article | fact removed, flagged `ungrounded_fact_removed` |
| Only a headline was available | confidence capped, flagged `confidence_capped_headline_only` |
| Company only tagged by the source, not named in the text | confidence capped, flagged `entity_from_hint_only` |
| Confidence ≥ `min_confidence` | kept but reset to `unknown` / `uncertain`, flagged `low_confidence_fallback_unknown` |
| Reply is not usable JSON | rejected `llm_error` |

The JSON schema sent to the model restricts `symbol` to the candidate companies and
`source_ids` to the supplied ids, so a server that enforces the schema cannot emit
anything else; validation checks both again regardless.

**Prompt injection (PRD 17).** Instructions live only in the system message, which is
built from constants. The article appears only inside a delimited block
(`<<<ARTICLE_DATA nonce>>> … <<<END_ARTICLE_DATA nonce>>>`) that the system message
declares to be untrusted data; delimiter look-alikes and control characters are removed
from the article and the text is truncated. Articles containing instruction-like
phrases are flagged `injection_suspected`. This lowers the risk but cannot remove it —
which is why the output still has to pass every check above.

**Summaries** (`summarize.py`) follow the same pattern: a point is kept only if it
cites supplied document ids, its numbers occur in the cited documents and it shares
enough words with them. With no documents the model is not asked.

## Configuration

[configs/llm.yaml](../configs/llm.yaml), validated by
[config.py](../ml/llm/config.py). Unknown keys are rejected.

| Key | Meaning |
|---|---|
| `provider` | `name` (label), `base_url`, `model`, `embedding_model`, `temperature`, `max_tokens`, `max_tokens_param`, `seed`, timeouts, retries, `structured_output` |
| `universe_file`, `company_aliases`, `alias_exclusions`, `ticker_aliases` | Securities master and entity resolution |
| `yahoo_news` | Cache directory, items per query, pauses, retries |
| `dedup` | Similarity threshold and time window |
| `extraction` | Article length limit, `min_confidence`, confidence caps, `min_fact_overlap` |
| `aggregation` | Market close and time zone, `max_age_days`, half-lives per event type, publisher reliability |
| `retrieval`, `summary` | Documents per query, points per summary |

Environment (`.env`, see `.env.example`):

| Variable | Meaning |
|---|---|
| `AETHER_LLM_API_KEY` | Secret. **Only** read from the environment; held as a `SecretStr`, sent only in the `Authorization` header, never logged |
| `AETHER_LLM_BASE_URL`, `AETHER_LLM_MODEL` | Override the provider in the YAML for the scripts; **required** for the API |
| `AETHER_LLM_PROVIDER`, `AETHER_LLM_EMBEDDING_MODEL` | Label and embedding model |
| `AETHER_LLM_CONFIG` | Path of the engine config used by the API |

No model id is hardcoded in any code path. The half-lives (dividend 2 days … corporate
governance 30 days) and reliability weights are **assumptions you should review**, not
fitted values.

## Running it

```bash
uv run python scripts/analyze_news.py --symbols INFY WIPRO --max-articles 20   # Yahoo
uv run python scripts/analyze_news.py --jsonl data/licensed/news.jsonl         # your file
uv run python scripts/evaluate_llm.py                                          # evaluation
uv run alembic -c backend/alembic.ini upgrade head
uv run python scripts/load_news.py --latest                                    # into the DB
```

API (PRD 15.2); every response names the provider, model and prompt version:

| Endpoint | Returns |
|---|---|
| `GET /api/v1/news` | Analysed articles with their events; filters `symbol`, `sector`, `event_type`, `start`, `end`, `limit` |
| `GET /api/v1/news/{news_id}` | One article with text and events |
| `POST /api/v1/news/analyze` | Analyses one supplied article synchronously; returns events, rejections and candidates. Not stored. 503 when no LLM is configured or it is unreachable; 422 for oversized or invalid input; 429 when too many analyses are running |
| `GET /api/v1/news/sentiment/{symbol}` | Point-in-time news features per day |
| `GET /api/v1/news/events/{symbol}` | Validated events for a symbol |

<<LIVE>>

## Known limitations

- **Headlines only.** The free Yahoo source has no article text, so events from it are
  low-detail by construction; most are capped in confidence or fall back to `unknown`.
- **No historical news archive.** Yahoo returns recent items only, so news features
  exist only for the days news was collected. **Backtests over past years have no news
  features** (`news_available = 0` throughout) until a licensed archive is supplied
  through `JsonlNewsSource`. No historical news signal has been evaluated.
- **OpenAI is not verified.** The client implements the OpenAI chat-completions and
  embeddings API and was verified against Ollama's OpenAI-compatible endpoint. No API
  key was available, so it has not been run against OpenAI or the model the project
  intends to use. Hosted models may need `max_tokens_param: max_completion_tokens` or
  `temperature: null`; both are configurable for that reason.
- **Verification used a small local model** (`qwen2.5:3b`, CPU). Its quality is
  reported above and is not representative of a larger model.
- **The evaluation set is small, fictional and hand-labelled** — a smoke test of a
  model + prompt, not a benchmark, and not evidence about real news.
- **Grounding is lexical.** Numbers are matched as digits and facts by word overlap. A
  correctly copied number attached to the wrong quantity passes; numbers written as
  words are not checked; meaning is not verified.
- **Entity resolution is literal alias matching.** It only knows the configured
  aliases and exclusions and does not understand subsidiaries, pronouns or context.
- **Prompt injection is mitigated, not solved.** A model can still be influenced by
  article text; validation limits what a successful injection can produce.
- **Half-lives and source reliability are configured assumptions.** They have not been
  estimated from data.
- Feature dates default to Monday–Friday, not the exchange calendar; pass `sessions`.
- De-duplication compares headlines pairwise (O(n²)) and only by wording.
- Rejected outputs are kept in the run directory, not in the database. An event stored
  by an earlier run is not removed when a later run rejects it.
- `POST /news/analyze` is synchronous and only has a simple concurrency cap; real rate
  limiting and authentication are Phase 8.
- Annual reports, transcripts and filings (PRD 8.2-E) are not ingested; summaries work
  on whatever documents are in the store, which today means news items.
- The document store is a brute-force in-memory index, not a vector database.
