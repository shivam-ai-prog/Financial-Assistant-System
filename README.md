# 📈 Financial Assistant

A chat assistant that answers questions about **SEC filings** (10-K, 10-Q, 8-K) of tech companies, shows **charts inline**,
and keeps every filing in a **permanent local database**. Built around NVIDIA and Tesla, extended to similar tech peers
(AMD, Intel, Broadcom, Microsoft, Alphabet, Meta, Apple, ...) so companies can be compared.

## What it can do

| Feature | How it works |
|---|---|
| **Answers from filings** | Retrieval-augmented generation (RAG): finds the best passages in the stored filings and asks an LLM to answer from them, with sources |
| **Router** | Decides which company and which filing type (10-K / 10-Q / 8-K / chart) a question needs. Keyword rules first, a small LLM only when no rule fires |
| **Live SEC ingestion** | Checks SEC EDGAR for newer filings (at most every 12 h), skips 10-K/A amendments, splits into Items, embeds and saves |
| **Calculator tool** | Exact maths (growth, margins, ratios) instead of the LLM guessing; no `eval`, a safe expression parser |
| **Chart tool** | Ask "plot NVIDIA and Tesla revenue trend" and the router draws a chart inside the chat: trend, comparison, growth, margins, waterfall |
| **Comparisons** | Name up to 4 companies, or ask for "NVIDIA's competitors" and peers are added automatically |
| **Admin panel** | Password-protected: library, fetch from SEC, inspect sections, delete, re-embed, live read-only database view |

## Architecture

```
 User ──> app.py (Gradio chat) ──> Router (model.py)
                                     ├── Live ingestion ──> SEC EDGAR ──> SQLite
                                     ├── Retrieval (dense + BM25 -> RRF -> cross-encoder) ──> SQLite
                                     ├── Chart tool (analytics.py) ──> SQLite (extracted income-statement numbers)
                                     └── Answer writer ──> OpenRouter LLM (+ Calculator tool)
 Admin ─> admin.py ─────────────────> same SQLite database (data/filings_index.db)
```

## Project layout

```
model.py        engine: SQLite store, SEC ingestion, router, hybrid retrieval, LLM + calculator
analytics.py    chart tool: income-statement extraction, chart builders, chart request parsing
companies.py    covered companies (tech universe), name <-> ticker lookup, peers
app.py          Gradio chat UI            -> http://127.0.0.1:7860
admin.py        Gradio admin panel        -> http://127.0.0.1:7861
db_viewer.py    live, read-only database tab used by admin.py
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
