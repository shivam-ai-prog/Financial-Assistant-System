# 📈 Financial Assistant

A chat assistant that answers questions about **SEC filings** (10-K, 10-Q, 8-K) of tech companies, shows **charts inline**,
measures **financial performance** (scorecards, year-over-year and quarterly trends, peer benchmarks), writes a **financial report**
(chat + downloadable Word file, with investment considerations) and keeps every filing in a **permanent local database**. Built around NVIDIA and Tesla, extended to similar tech peers
(AMD, Intel, Broadcom, Microsoft, Alphabet, Meta, Apple, ...) so companies can be compared.

## What it can do

| Feature | How it works |
|---|---|
| **Answers from filings** | Retrieval-augmented generation (RAG): finds the best passages in the stored filings and asks an LLM to answer from them, with sources |
| **Router** | Decides which company and which filing type (10-K / 10-Q / 8-K / chart) a question needs. A small LLM (Groq) decides sources, chart and question intent in one call; keyword rules are only the fallback if it fails |
| **Question intents** | Risk / business / strategy / legal / etc. questions search only the matching Items of the newest filing and get prose answers (no revenue tables); numeric questions keep the table + calculator path |
| **Vector database** | SQLite is the source of truth; vectors are mirrored into a pluggable index (SQLite default, optional Qdrant) with automatic fallback |
| **Live SEC ingestion** | Checks SEC EDGAR for newer filings (at most every 12 h), skips 10-K/A amendments, splits into Items, embeds and saves |
| **Calculator tool** | Exact maths (growth, margins, ratios) instead of the LLM guessing; no `eval`, a safe expression parser |
| **Chart tool** | Ask "plot NVIDIA and Tesla revenue trend" and the router draws a chart inside the chat: trend, comparison, growth, margins, waterfall |
| **Performance review** | Deterministic scorecard, YoY / quarterly KPI tables and peer ranking, computed in code (no LLM maths) |
| **Financial report** | Full report in chat and as a `.docx`: summary, commentary, business overview, key risks, peer tables, FY-labelled charts, investment considerations, sources |
| **Comparisons** | Name up to 4 companies, or ask for "NVIDIA's competitors" and peers are added automatically |
| **Admin panel** | Password-protected: library, fetch from SEC, inspect sections, delete, re-embed, live read-only database view |

## Architecture

 User ──> app.py (Gradio chat) ──> LLM Router (model.py)  ── one call decides: filings needed, chart?, task, intent
                                     │
        task = qa ───────────────────┼── Live ingestion ──> SEC EDGAR ──> SQLite (filings, sections, chunks, vectors)
                                     ├── Retrieval, by intent (intents.py): section filter -> dense (vectorstore.py)
                                     │     + BM25 -> RRF -> cross-encoder ──> SQLite / Qdrant index
                                     ├── Chart tool (analytics.py) ──> metrics table
                                     └── Answer writer ──> OpenRouter LLM (+ Calculator tool)

        task = performance / report ─> report.py
                                         ├── 1. ensure filings are stored (live ingestion, as above)
                                         ├── 2. analytics.py: read income statement, balance sheet, cash flow
                                         │      from the stored filing text -> `metrics` table (SQLite)
                                         ├── 3. metrics_engine.py (no LLM): ratios, YoY / QoQ, scorecard,
                                         │      peer ranking, "who leads" lines
                                         ├── 4. report.py builds blocks:
                                         │      tables + scorecard + charts      <- code only
                                         │      commentary, overview, risks,
                                         │      investment view                  <- LLM, fed only the tables and
                                         │                                          retrieved 10-K Items 1 / 1A
                                         └── 5. output: Markdown in chat + .docx (data/reports/)
 Admin ─> admin.py ─────────────────> same SQLite database (data/filings_index.db)

## Project layout

```
model.py        engine: SQLite store, SEC ingestion, LLM router, hybrid retrieval, LLM + calculator
intents.py      question intents -> which Items to search and how broadly
vectorstore.py  vector index layer (SQLite default, optional Qdrant, auto-fallback)
diag_retrieval.py  shows what retrieval sees for one question
analytics.py    chart tool: income-statement extraction, chart builders, chart request parsing
companies.py    covered companies (tech universe), name <-> ticker lookup, peers
app.py          Gradio chat UI            -> http://127.0.0.1:7860
admin.py        Gradio admin panel        -> http://127.0.0.1:7861
metrics_engine.py  ratios, YoY/QoQ, scorecard, peer ranking (no LLM)
report.py       performance review + Word financial report (tables by code, prose by LLM)
diag_metrics.py / diag_sections.py   check extracted numbers / stored Items for one filing
db_viewer.py    live, read-only database tab (tables, SQL, vector index panel) used by admin.py
tests/          automated tests (no API keys or downloads needed)
data/           created at run time: filings_index.db (the database), company list cache
.env            your keys - NEVER commit; .env.example is the template
```

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate              # Windows     (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env             # macOS/Linux: cp .env.example .env
```

Edit `.env`:

| Setting | Meaning |
|---|---|
| `OPENROUTER_API_KEY` | key for the answer LLM (required) |
| `GROQ_API_KEY` | key for the backup router (optional) |
| `EDGAR_IDENTITY` | `"Your Name you@email.com"` - SEC requires a real identity (required) |
| `ADMIN_PASSWORD` | admin panel login (required; `admin.py` refuses to start without it) |
| `COVERAGE` | `tech` (default, the covered list) or `all` (any company on SEC EDGAR) |
| `LLM_MODEL`, `LLM_MAX_TOKENS` | answer model and its token budget |

> **Never commit `.env` or put real keys in code.** If a key was ever shared or pasted anywhere, revoke it and create a new one.

## Run

```bash
python admin.py      # 1) load filings: Fetch from SEC -> pick a ticker -> Fetch & store
python app.py        # 2) chat at http://127.0.0.1:7860
```

Asking about a company that is not stored yet also fetches its latest filings automatically.

### Example questions

```
What was NVIDIA's latest quarterly revenue and how did Data Center perform?
What are AMD's main risk factors in its latest 10-K?
Compare AMD and Intel gross margin and operating expenses in the latest 10-K.
Calculate Intel's net income margin for its latest fiscal year.
Plot NVIDIA and Tesla revenue trend over the last 3 years
Show a margin comparison chart for Meta
Compare Microsoft and Alphabet operating margin as a chart
Show a waterfall of how Intel's revenue turns into net income
```

## How data stays permanent

- Each ingest is **one SQLite transaction**: filing + Item texts + chunks + embeddings, or nothing.
- `app.py` and `admin.py` share one database file and can run together.
- If SEC is unreachable, answers come from what is already stored.
- Back up by copying `data/filings_index.db`.
- Changed `EMBED_MODEL`? Admin -> Inspect / manage -> **Re-embed all stale filings**.

## Adding a company

Open `companies.py`, add a line to a group in `UNIVERSE`, for example `"SMCI": "Super Micro Computer"`, then fetch its filings in the admin panel. Or set `COVERAGE=all` in `.env`.

## Tests

```bash
python -m pytest tests -q
```

## Limits to know about

- Chart numbers are read from each company's income statement; companies with unusual layouts (banks, insurers) may need a new rule. Check the "numbers behind the chart" against the real filing.
- Fiscal years differ between companies (NVIDIA's FY2026 ends in January 2026); charts label each company by its own fiscal year.
- This is an analysis aid, not financial advice. Always verify important figures against the filing.

## Financial report: what is in it
1. Executive summary (latest annual and quarter per company, from the tables).
2. Performance commentary: 3 short paragraphs (annual, latest quarter, balance sheet). The "who leads" statements come from
   code (peer ranks), so the model never judges direction itself (for liabilities/equity lower is better).
3. Per company: business overview and key risks (Items 1 / 1A, one retrieval per risk topic), scorecard with printed rules,
   annual KPI table, quarterly KPI table.
4. Peer comparison tables (latest fiscal year and latest quarter), with a warning when fiscal periods differ.
5. Charts (axis shows the calendar year with each company's own fiscal year, e.g. `NVDA FY26`).
6. Investment considerations: overall view per company (follows the scorecard), bull / bear case, what would change the view
   (illustrative thresholds), what to monitor, and a valuation caveat (no share prices in the data).
7. Sources (accession numbers) and a disclaimer.

If the language model fails (credit, rate limit), the report still contains every number, table and chart, and a visible
**WARNING** at the top names the error - nothing is silently dropped. Check the terminal for lines such as
`commentary skipped` / `report prose ... skipped`.

## How a question is answered
1. **LLM router** (one Groq call, `GROQ_API_KEY`): decides which filings (10-K / 10-Q / 8-K), whether a chart is wanted,
   and the question intent. It only picks from fixed lists, never invents sections. It sees your previous question,
   so follow-ups work. If the call fails, keyword rules take over (the `reason` field shows which path ran).
   Company detection is a name/ticker lookup against the SEC list (`companies.py`).
2. **Ingest check**: new filings are fetched from SEC (at most every `CHECK_INTERVAL_HOURS`) and stored atomically.
3. **Retrieval**, by intent (`intents.py`):
   * *Qualitative* (risk, business, strategy, legal, competition, cyber, liquidity...): only the matching Items of the
     newest filing (e.g. Item 1A for risks), chunks spread across the whole section, boilerplate and number-heavy
     tables dropped, document order, prose-only prompt, no calculator, no tables unless asked.
   * *Numeric* (financials): table-friendly hybrid search + calculator for every ratio/difference.
   * Hybrid search = dense vectors + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.
4. **Answer** from the retrieved context only, with a Sources list. If the expected section was missing or tiny in
   storage, a "Data-quality note" is appended.

## Vector database
SQLite stays the source of truth (filings, sections, chunks, embeddings). Vectors are mirrored into a pluggable index:

| `.env` | Meaning |
|---|---|
| `VECTOR_BACKEND=sqlite` (default) | vectors read from SQLite, no extra install |
| `VECTOR_BACKEND=qdrant` | needs `pip install qdrant-client`; set `QDRANT_URL` (+ `QDRANT_API_KEY` for cloud) |
| `QDRANT_COLLECTION` | default `filings_chunks` |

If Qdrant is down the app falls back to SQLite vectors automatically. The index is synced on ingest/delete/re-embed.
Admin -> Database -> **Vector index** shows per-filing counts and has **Repair / sync**.

## Performance review and financial reports
Ask in chat (the LLM router detects the task):
* *"Evaluate NVIDIA's performance with key metrics, year over year and quarterly"* -> scorecard + KPI tables (10-K years, 10-Q quarters) + charts.
* *"Benchmark NVIDIA, AMD and Intel on profitability and growth"* -> side-by-side latest period, rank per metric, peer median.
* *"Generate a financial report for Tesla"* (or "...comparing NVIDIA and AMD") -> the full report shown in chat **and** a Word file
  (download box under the chat; saved in `data/reports/`). It adds business overview, key risks (from Item 1, 1A) and AI commentary.

How it stays trustworthy: `metrics_engine.py` computes every ratio in plain Python from numbers read out of the stored
filings (income statement, balance sheet, cash flow) - the LLM never calculates. Missing inputs show **n/a** and are listed
under "Not available from the stored filings"; nothing is estimated. Scorecard thresholds are printed in every report.
Metrics: growth (YoY, quarterly YoY and vs previous quarter), gross/operating/net/FCF margins, R&D %, ROE, ROA,
current ratio, liabilities/equity, net cash. Limits: quarterly cash-flow data is not used (10-Q cash flow is year-to-date);
peers with different fiscal calendars are compared on each company's own latest period (a warning is shown);
after upgrading, stored filings are re-read once automatically to pick up balance-sheet and cash-flow lines.

## Checking answer quality
```bash
python diag_retrieval.py META 10-K "What are Meta's main risk factors?"   # then read diag_retrieval.txt
```
Lists stored length of every Item (a `<-- SHORT` Item 1A means a truncated filing: delete and re-ingest it in admin),
the detected intent, and the chunks chosen. Tests: `python -m pytest tests -q`.

## Known limits (report and metrics)
- Numbers are read from filing text by pattern rules, not from XBRL. Check key figures against the filing, especially for unusual layouts.
- Only the latest two 10-Qs per company are stored, so some quarter-on-quarter cells show `n/a`.
- Quarterly cash-flow items are not used (10-Q cash flow is year-to-date).
- The investment section is fundamentals only: no prices or valuation multiples.
- Tested mainly on NVIDIA vs AMD; run other pairs and compare with the filings before relying on them.

## Security
Never commit `.env` (it is in `.gitignore`). Rotate any key that was ever pasted in a notebook or chat.
The admin panel refuses to start without `ADMIN_PASSWORD`; the Database tab is read-only (SELECT only).
