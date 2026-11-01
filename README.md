# 📈 Financial Assistant

An **agentic, multi-agent** assistant for **SEC filings** (10-K, 10-Q, 8-K) of tech companies. It answers questions with sources,
draws **charts inline**, measures **financial performance** (scorecards, year-over-year and quarterly trends, peer benchmarks),
explains **why the numbers moved**, shows **what changed since the previous filing**, and writes a **financial report**
(in chat and as a downloadable Word file, with investment considerations). Every filing is kept in a **permanent local database**.
Built around NVIDIA and Tesla, extended to similar tech peers (AMD, Intel, Broadcom, Microsoft, Alphabet, Meta, Apple, ...).

Design rule used everywhere: **the AI reads, explains and writes; code finds, calculates and verifies.**
No number reaches the user unless it comes from a stored filing, the metrics engine or the calculator.

## What it can do

| Feature | How it works |
|---|---|
| **Multi-agent routing** | A LangGraph system: a *planner* picks a specialist (question answering, analyst, chart, report); specialist agents choose their own tools step by step |
| **Answers from filings** | Retrieval-augmented generation (RAG): hybrid search finds the best passages; a *writer* model answers from them with sources |
| **Question intents** | Risk / business / strategy / legal questions search only the matching Items of the newest filing and get prose answers; numeric questions use tables + the calculator |
| **Calculator tool** | Exact maths (growth, margins, ratios), no `eval`; agents must use it instead of calculating in their head |
| **Chart tool** | "Plot NVIDIA and Tesla revenue trend": trend, comparison, growth, margins, waterfall charts inside the chat |
| **Performance review** | Deterministic scorecard, YoY / quarterly KPI tables and peer ranking, computed in code |
| **Why numbers moved** | `drivers.py`: a dedicated model explains changes from the MD&A in its own words; every sentence is checked against the filing text |
| **What changed since the last filing** | `changes.py`: code finds new / removed / reworded risk-factor and business passages; the AI explains them; each explanation is verified |
| **Financial report** | Full report in chat and `.docx`: summary, commentary, business overview, risks, drivers, peer tables, 4 charts, investment view, sources |
| **AI extraction fallback** | `llm_extract.py`: if pattern rules find nothing in a financial statement, an AI reads it and code verifies every value against the text |
| **Reviewer** | `reviewer.py` / the graph's reviewer: figures in AI text must appear in the tables or tool results; unsupported sentences are fixed or removed |
| **Live SEC ingestion** | Checks SEC EDGAR for newer filings (at most every 12 h), splits into Items, embeds and saves atomically |
| **Vector database** | SQLite is the source of truth; vectors are mirrored into a pluggable index (SQLite default, optional Qdrant) |
| **Admin panel** | Password-protected: library, fetch from SEC, inspect sections, delete, re-embed, live read-only database view |

## Architecture

### The multi-agent graph (`agents/`, built with LangGraph)

```
 User ──> app.py (Gradio) ──> agents.answer()
                                   │
                              ┌────▼─────┐   LLM decides route + companies (keyword rules if the model fails)
                              │ planner  │
                              └────┬─────┘
        ┌───────────┬──────────────┼───────────────┬────────────┐
        ▼           ▼              ▼               ▼            ▼
   qa agent    analyst agent    chart          report        clarify
   (ReAct)      (ReAct)       specialist     specialist   ("which company?")
   tools:       tools:        draw_chart     build_financial_report
   identify,    identify,                         │
   stored,      ensure, performance_review,       │
   ensure,      explain_drivers, what_changed,    │
   search,      search, calculator                │
   calculator        │                            │
        └────┬───────┘                            │
             ▼                                    │
        ┌─────────┐   writes the final answer     │
        │ writer  │   from the evidence only      │
        └────┬────┘                               │
             ▼                                    ▼
        ┌──────────────────────────────────────────────┐
        │ reviewer: every figure must be in a tool result │──> one rewrite, then unsupported sentences are removed
        └──────────────────────┬───────────────────────┘
                               ▼
                         answer + charts + Word file
```

| Node | Job | Model role (`.env`) |
|---|---|---|
| planner | picks the specialist and the companies | `PLANNER_MODEL` |
| qa / analyst agents | decide which tools to call, in what order; collect evidence | `AGENT_MODEL` (must support tool calling) |
| chart / report specialists | call the existing chart and report code directly (no model call of their own) | none |
| writer | writes the final chat answer from the evidence only | `WRITER_MODEL` (default `LLM_MODEL`) |
| reviewer | code audit of figures; one repair round | none (code) |

Guardrails (code, not model): per-agent tool allow-lists, per-tool call limits, result size cap, only covered companies can be fetched,
step / recursion limits, model timeouts and retries, per-session memory. If the graph fails (for example a rate limit) the user sees a
clear message instead of a crash. There is no separate "classic" pipeline any more.

### Models: each role can use its own

| Role | Variable | Used for |
|---|---|---|
| Main | `LLM_MODEL` | report commentary, investment view, business overview, risk summary, writer default |
| Writer | `WRITER_MODEL` | final chat answers (falls back to `LLM_MODEL`) |
| Planner | `PLANNER_MODEL` | routing (short JSON answer) |
| Agents | `AGENT_MODEL` | tool-calling loops of the qa and analyst agents |
| Drivers | `DRIVERS_MODEL` | "why numbers moved" from the MD&A |
| Changes | `CHANGES_MODEL` | "what changed" explanations (falls back to `DRIVERS_MODEL`, then `LLM_MODEL`) |
| Extraction | `EXTRACT_MODEL` | AI fallback for reading financial statements |
| Reviewer | `REVIEWER_MODEL` | semantic check of report text (a different model than the writer is better) |

A model name with the prefix `groq:` is served by Groq (`GROQ_API_KEY`); any other name goes to OpenRouter (`OPENROUTER_API_KEY`).
`llm_provider.py` decides this in one place.

### The report path (inside the `report` specialist)

| Step | Module | What happens | AI? |
|---|---|---|---|
| 1 | `model.py` ingestion | Fetches any missing 10-K / 10-Q from SEC and stores them | no |
| 2 | `analytics.py` (+ `llm_extract.py`) | Reads revenue, income, cash flow and balance-sheet lines from each stored filing into the `metrics` table; AI fallback only when patterns find nothing, and only verified values are kept | fallback only |
| 3 | `metrics_engine.py` | Margins, growth, FCF, ROE / ROA, current ratio, liabilities/equity, net cash; scorecard and peer ranking | no |
| 4 | `report.py` | Tables, charts and "who leads on what" lines | no |
| 5 | `report.py` | Commentary, business overview, risk summary and investment view written from the tables and retrieved text only (risk notes are labelled per company) | yes |
| 6 | `drivers.py` | Reasons for the changes, from the MD&A, with evidence quotes checked against the filing | yes + code check |
| 7 | `changes.py` (optional, `REPORT_CHANGES=true`) | What changed versus the previous 10-K | yes + code check |
| 8 | `reviewer.py` | Figures in AI text must match the tables; unsupported statements are removed; a note says so | code (+ optional AI) |
| 9 | `report.py` | Markdown for chat and a Word file with sources, disclaimer and a WARNING if any AI step failed | no |

### How the AI output is verified

| Where | Check |
|---|---|
| Drivers | each explanation needs 1-2 exact sentences from the MD&A; its numbers must appear in that evidence (rounding tolerated); most key words must come from it |
| Changes | the quote must occur in the passage the sentence is about (old text for removed passages); numbers and key words must come from the passage |
| AI extraction | the line label must exist in the text, every value must appear in the statement, sanity checks (revenue - cost = gross profit, assets bound lines) |
| Reviewer / graph | any sentence with a figure that is not in the tables or tool results is rewritten once, then removed |

## Project layout

```
agents/            LangGraph multi-agent system
  __init__.py        entry point: agents.answer(query, company, session_id)
  graph.py           planner, specialists, writer, reviewer, graph assembly
  tools.py           the tools (with guardrails) used by the agents
  llm.py             one chat model per role
  check.py           python -m agents.check : tests your models (JSON + tool calling)
llm_provider.py  decides OpenRouter or Groq for a model name ("groq:" prefix)
model.py         engine: SQLite store, SEC ingestion, hybrid retrieval, LLM client, calculator, prompts
intents.py       question intents -> which Items to search and how broadly
vectorstore.py   vector index layer (SQLite default, optional Qdrant, auto-fallback)
analytics.py     chart tool and financial-statement extraction -> `metrics` table
llm_extract.py   AI fallback for statement extraction, verified by code
metrics_engine.py  ratios, YoY/QoQ, scorecard, peer ranking (no AI)
drivers.py       "why numbers moved" from the MD&A
changes.py       "what changed since the previous filing"
reviewer.py      figure audit and clean-up of AI text
report.py        performance review + Word financial report
companies.py    covered companies (tech universe), name <-> ticker lookup, peers
app.py           Gradio chat UI            -> http://127.0.0.1:7860
admin.py         Gradio admin panel        -> http://127.0.0.1:7861
db_viewer.py     live, read-only database tab used by admin.py
diag_retrieval.py / diag_metrics.py / diag_sections.py   check retrieval / extracted numbers / stored Items
tests/           automated tests (no API keys or downloads needed)
data/            created at run time: filings_index.db (the database), reports/, company list cache
.env             your keys - NEVER commit; .env.example is the template
```

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate              # Windows     (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt    # includes langgraph, langchain-openai, langchain-core
copy .env.example .env             # macOS/Linux: cp .env.example .env
```

Edit `.env`. Example with three models (adjust to what you have):

```dotenv
OPENROUTER_API_KEY=...
GROQ_API_KEY=...                       # only needed for "groq:" models
EDGAR_IDENTITY=Your Name you@email.com
ADMIN_PASSWORD=...

LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free      # main model
WRITER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
EXTRACT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
DRIVERS_MODEL=nvidia/nemotron-3.5-super-vl:free
PLANNER_MODEL=groq:openai/gpt-oss-120b
AGENT_MODEL=groq:openai/gpt-oss-120b                  # must support tool calling
REVIEWER_MODEL=groq:openai/gpt-oss-120b

# optional
# CHANGES_MODEL=...          model for "what changed" (default: DRIVERS_MODEL)
# REPORT_CHANGES=true        add "what changed" to the Word report (2 extra AI requests per company)
# WRITER=off                 skip the writer step (agents answer directly)
# AGENT_MAX_STEPS=10  AGENT_TIMEOUT=90
# COVERAGE=all               any company on SEC EDGAR (default: the tech list)
```

| Other settings | Meaning |
|---|---|
| `EDGAR_IDENTITY` | `"Your Name you@email.com"`, required by SEC |
| `ADMIN_PASSWORD` | admin panel login (required; `admin.py` refuses to start without it) |
| `LLM_EXTRACT`, `REVIEW`, `REVIEW_LLM` | switch the AI extraction fallback / reviewer / its AI check on or off |
| `CHECK_INTERVAL_HOURS` | how often to ask SEC for new filings (default 12) |
| `VECTOR_BACKEND`, `QDRANT_URL` | vector index (see below) |

> **Never commit `.env` or put real keys in code or chat.** If a key was ever shared or pasted anywhere, revoke it and create a new one.

Before first use, test your models: `python -m agents.check` (JSON answer + tool calling).

## Run

```bash
python admin.py      # 1) load filings: Fetch from SEC -> pick a ticker -> Fetch & store
python app.py        # 2) chat at http://127.0.0.1:7860
```

Asking about a company that is not stored yet also fetches its latest filings automatically.
The terminal prints one `agents:` line per request showing the path, for example
`planner->analyst(NVDA,AMD) > tool:performance_review > tool:calculator > writer`.

### Example questions

```
What are NVIDIA's main risk factors in its latest 10-K?                      (qa agent)
Compare NVDA and AMD revenue growth and margins                              (analyst agent)
What changed in NVIDIA's risk factors compared with its previous 10-K?       (analyst: what_changed)
Why did Intel's operating income change?                                     (analyst: explain_drivers)
Chart NVIDIA revenue growth year over year                                   (chart specialist)
Generate a financial report comparing Intel and AMD                          (report specialist)
```

## Where each kind of question goes

1. **Planner** reads the question (and the chat history) and picks `qa`, `analyst`, `chart`, `report` or `clarify`. If no company can be identified it asks which one.
2. **qa agent**: looks at what is stored, fetches missing filings, searches the right Items by intent, uses the calculator for any arithmetic. The writer turns the evidence into the answer.
3. **analyst agent**: performance review, drivers, what-changed, comparisons.
4. **chart / report**: the existing chart tool and report pipeline.
5. **Retrieval** (inside `search_filing`): qualitative questions search only the matching Items (e.g. Item 1A for risks), spread across the section, boilerplate and tables dropped; numeric questions use table-friendly hybrid search. Hybrid = dense vectors + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.
6. **Writer + reviewer**: the answer uses only the evidence; unsupported figures are removed.

## Financial report: what is in it
1. Executive summary (latest annual and quarter per company, from the tables).
2. Performance commentary: 3 short paragraphs (annual, latest quarter, balance sheet). "Who leads" statements come from code (peer ranks).
3. Per company: business overview and key risks (Items 1 / 1A), scorecard with printed rules, annual and quarterly KPI tables, **what drove the change** (from the MD&A), optional **what changed since the previous 10-K**.
4. Peer comparison tables (latest fiscal year and latest quarter); ties are shown as ties; a warning appears when fiscal periods differ.
5. Four charts: annual revenue, annual margins, quarterly revenue, quarterly margins (FY-labelled).
6. Investment considerations: overall view per company (follows the scorecard), bull / bear case (each company's risks come only from its own 10-K), what would change the view, what to monitor, valuation caveat (no share prices in the data).
7. Sources (accession numbers), an "Automatic review" note when statements were removed, and a disclaimer.

If a model step fails (credit, rate limit), the report still contains every number, table and chart, and a visible **WARNING** names the error. Check the terminal for lines such as `drivers ... model proposed N, M passed the checks` or `investment view skipped`.

## How data stays permanent

- Each ingest is **one SQLite transaction**: filing + Item texts + chunks + embeddings, or nothing.
- `app.py` and `admin.py` share one database file and can run together.
- If SEC is unreachable, answers come from what is already stored.
- Back up by copying `data/filings_index.db`.
- Changed `EMBED_MODEL`? Admin -> Inspect / manage -> **Re-embed all stale filings**.
- By default one 10-K and two 10-Qs are stored per company; `what_changed` fetches the previous 10-K once when needed.

## Vector database
SQLite stays the source of truth. Vectors are mirrored into a pluggable index:

| `.env` | Meaning |
|---|---|
| `VECTOR_BACKEND=sqlite` (default) | vectors read from SQLite, no extra install |
| `VECTOR_BACKEND=qdrant` | needs `pip install qdrant-client`; set `QDRANT_URL` (+ `QDRANT_API_KEY` for cloud) |
| `QDRANT_COLLECTION` | default `filings_chunks` |

If Qdrant is down the app falls back to SQLite vectors automatically. Admin -> Database -> **Vector index** shows counts and has **Repair / sync**.

## Adding a company
Open `companies.py`, add a line to a group in `UNIVERSE`, e.g. `"SMCI": "Super Micro Computer"`, then fetch its filings in the admin panel. Or set `COVERAGE=all`.

## Tests and checks

```bash
python -m pytest tests -q                     # all automated tests
python -m agents.check                        # your configured models: JSON + tool calling
python diag_retrieval.py META 10-K "What are Meta's main risk factors?"   # what retrieval sees
```

Test files: `test_core`, `test_retrieval`, `test_metrics` (engine), `test_drivers`, `test_llm_extract`, `test_reviewer`, `test_graph` (agents, with a scripted fake model), `test_changes`.

## Known limits
- **Free-tier quotas.** OpenRouter's free tier allows 50 requests/day across all `:free` models; one full report uses roughly 20 or more, a chat answer a few. Adding credit raises the limit; Groq has its own limits.
- **Model dependence.** The agents need a model that supports tool calling (check with `python -m agents.check`). AI text varies between runs: the checks make it safe, not identical (for example the drivers pass rate differs per run, Intel's 10-Q most).
- **Extraction.** Numbers are read from filing text by pattern rules (with a verified AI fallback), not from XBRL. Check key figures against the filing, especially for unusual layouts (banks, insurers).
- **Quarterly data.** Only the latest two 10-Qs per company are stored, so some quarter-on-quarter cells show `n/a`; quarterly cash-flow items are not used (10-Q cash flow is year-to-date).
- **What changed.** Matching depends on how the stored text is split into paragraphs; if almost everything shows as changed, check the counts line and the terminal log.
- **Fiscal calendars** differ between companies; each is compared on its own latest period (a warning is shown).
- The investment section is fundamentals only: no prices or valuation multiples. This is an analysis aid, not financial advice.
- Tested mainly on NVIDIA, AMD and Intel; run other companies and compare with the filings before relying on them.

## Security
Never commit `.env` (it is in `.gitignore`). Rotate any key that was ever pasted in a notebook or chat.
The admin panel refuses to start without `ADMIN_PASSWORD`; the Database tab is read-only (SELECT only).
