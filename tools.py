"""agents/tools.py - the tools every agent uses. Same handlers as before (SEC filings, metrics engine, calculator,
charts, drivers, report) wrapped with guardrails: allow-list, per-tool call limits, result cap, errors returned as text."""
from __future__ import annotations

import json

import model

CALL_LIMITS = {"build_financial_report": 1, "performance_review": 2, "draw_chart": 3, "search_filing": 8,
               "explain_drivers": 4, "ensure_filings": 6, "what_changed": 2}
RESULT_CAP = 7000
INTENTS = ["risk", "business", "strategy", "legal", "competition", "cyber", "liquidity", "financials",
           "performance", "market_risk", "controls"]


class Run:
    def __init__(self, query: str, session_id: str):
        self.query, self.session_id = query, session_id
        self.charts: list = []
        self.sources: list[str] = []
        self.evidence: list[str] = []
        self.report_md: str | None = None
        self.report_file = None
        self.trace: list[str] = []
        self.tickers: list[str] = []
        self.counts: dict[str, int] = {}


def _covered(t: str) -> bool:
    try:
        import companies
        return bool(companies.covered(t))
    except Exception:
        return True


def _clean_tickers(items) -> list[str]:
    out = []
    for t in items or []:
        t = str(t).upper().strip()
        if t and t not in out:
            out.append(t)
    return out


# ── tools ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _t_identify(run: Run, a: dict) -> str:
    found = model.detect_tickers(str(a.get("text", "")))
    if not found:
        return "No covered company could be identified in that text."
    run.tickers = list(dict.fromkeys(run.tickers + list(found)))
    return json.dumps({t: model.company_name(t) for t in found})


def _t_stored(run: Run, a: dict) -> str:
    rows = model.list_filings(str(a.get("ticker", "")).upper())
    return json.dumps([{"form": r["form_type"], "fiscal_year": r.get("fiscal_year"), "period_end": r.get("period_end")}
                       for r in rows]) if rows else "No filings stored for that ticker yet."


def _t_ensure(run: Run, a: dict) -> str:
    t = str(a.get("ticker", "")).upper()
    if not _covered(t):
        return f"{t} is not a covered company, so it cannot be fetched."
    msgs = model.ensure_filings(t, list(a.get("forms") or ["10-K", "10-Q"]))
    return "\n".join(msgs) if msgs else "Already up to date."


def _t_search(run: Run, a: dict) -> str:
    import intents
    t, form, q = str(a.get("ticker", "")).upper(), str(a.get("form", "10-K")), str(a.get("question", ""))
    prof = intents.profile_from_names([a["intent"]]) if a.get("intent") in INTENTS else intents.classify_smart(q)
    hits, info = model.retrieve_ex(q, t, [form], top_n=prof.top_n, profile=prof)
    if not hits:
        return f"No stored {form} text found for {t}."
    ctx, src = model.build_context([hits], max_chars=prof.context_chars, interleave=False)
    run.sources += list(src)
    return ctx


def _t_calc(run: Run, a: dict) -> str:
    return str(model.calculator(str(a.get("expression", ""))))


def _t_performance(run: Run, a: dict) -> str:
    import report
    tickers = _clean_tickers(a.get("tickers"))[:4]
    bad = [t for t in tickers if not _covered(t)]
    if bad or not tickers:
        return f"Not covered or empty: {bad or 'no tickers given'}."
    out = report.performance_review(tickers)
    run.charts += list(out.get("charts") or [])
    run.sources += list(out.get("sources") or [])
    return out["markdown"]


def _t_chart(run: Run, a: dict) -> str:
    import analytics
    tickers = _clean_tickers(a.get("tickers"))[:4]
    decision = {"tickers": tickers, "needs_10k": False, "needs_10q": False, "needs_8k": False}
    out = analytics.run_chart_tool(str(a.get("request", "")), decision, run.session_id)
    run.charts += list(out.get("figures") or [])
    run.sources += list(out.get("sources") or [])
    return out.get("text", "")


def _t_drivers(run: Run, a: dict) -> str:
    import drivers
    items, err = drivers.drivers_for(str(a.get("ticker", "")).upper(), str(a.get("form", "10-K")))
    if err:
        return f"The explanation step failed: {err}"
    if not items:
        return "No verified explanation could be taken from the stored MD&A."
    return "\n".join("- " + i["statement"] for i in items)


def _t_changes(run: Run, a: dict) -> str:
    import changes
    t, form = str(a.get("ticker", "")).upper(), str(a.get("form", "10-K"))
    if not _covered(t):
        return f"{t} is not a covered company."
    res = changes.what_changed(t, form)
    run.sources += list(res.get("sources") or [])
    return changes.to_markdown(res)


def _t_report(run: Run, a: dict) -> str:
    import report
    tickers = _clean_tickers(a.get("tickers"))[:4]
    bad = [t for t in tickers if not _covered(t)]
    if bad or not tickers:
        return f"Not covered or empty: {bad or 'no tickers given'}."
    out = report.financial_report(tickers)
    run.report_md, run.report_file = out["markdown"], out.get("file")
    run.charts += list(out.get("charts") or [])
    run.sources += list(out.get("sources") or [])
    return (f"The full financial report for {', '.join(tickers)} was built ({len(run.report_md)} characters) and "
            "will be shown to the user in full with a Word download. Do not repeat it.")


HANDLERS = {"identify_companies": _t_identify, "stored_filings": _t_stored, "ensure_filings": _t_ensure,
            "search_filing": _t_search, "calculator": _t_calc, "performance_review": _t_performance,
            "draw_chart": _t_chart, "explain_drivers": _t_drivers, "what_changed": _t_changes,
            "build_financial_report": _t_report}


def call_tool(run: Run, name: str, args: dict) -> str:
    """Runs one tool with the guardrails. Errors are returned to the Supervisor as text, never raised."""
    if name not in HANDLERS:
        return f"Unknown tool {name!r}. Available: {', '.join(HANDLERS)}."
    run.counts[name] = run.counts.get(name, 0) + 1
    run.trace.append(f"tool:{name}")
    if run.counts[name] > CALL_LIMITS.get(name, 20):
        return f"Limit reached for {name}; use the results you already have."
    try:
        out = HANDLERS[name](run, args if isinstance(args, dict) else {})
    except Exception as e:
        model.log.warning("agent tool %s failed: %s", name, e)
        out = f"Tool {name} failed: {type(e).__name__}: {str(e)[:200]}"
    out = str(out)[:RESULT_CAP]
    run.evidence.append(out)
    return out


# ── per-request Run registry (so tools running in any thread find their request) ──────────────────────────────
_RUNS: dict[str, Run] = {}


def new_run(run_id: str, query: str, session_id: str) -> Run:
    _RUNS[run_id] = Run(query, session_id)
    return _RUNS[run_id]


def get_run(run_id: str) -> Run:
    return _RUNS[run_id]


def drop_run(run_id: str) -> None:
    _RUNS.pop(run_id, None)


# ── LangChain tool wrappers, grouped by specialist ────────────────────────────────────────────────────────────
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402


def _wrap(name: str, desc: str, schema: dict) -> StructuredTool:
    def _fn(config: RunnableConfig, **kw) -> str:
        run = get_run(config["configurable"]["run_id"])
        return call_tool(run, name, kw)
    return StructuredTool.from_function(func=_fn, name=name, description=desc, args_schema=schema)


_S = lambda props, req: {"type": "object", "properties": props, "required": req}  # noqa: E731
_str = {"type": "string"}
_tick = {"type": "array", "items": {"type": "string"}, "description": "Ticker symbols, e.g. [\"NVDA\", \"AMD\"]"}

ALL_TOOLS = {
    "identify_companies": _wrap("identify_companies", "Find covered companies named or implied in a text.",
                                _S({"text": _str}, ["text"])),
    "stored_filings": _wrap("stored_filings", "List the filings stored for a ticker.", _S({"ticker": _str}, ["ticker"])),
    "ensure_filings": _wrap("ensure_filings", "Fetch the newest filings for a covered ticker if not stored yet.",
                            _S({"ticker": _str, "forms": {"type": "array", "items": _str}}, ["ticker"])),
    "search_filing": _wrap("search_filing", "Search stored filing text. intent is one of: " + ", ".join(INTENTS),
                           _S({"ticker": _str, "question": _str, "form": {"type": "string", "description": "10-K, 10-Q or 8-K"},
                               "intent": _str}, ["ticker", "question"])),
    "calculator": _wrap("calculator", "Exact arithmetic. Use for every difference, percentage or ratio.",
                        _S({"expression": _str}, ["expression"])),
    "performance_review": _wrap("performance_review", "Benchmark 1-4 companies: growth, margins, balance sheet, ranking.",
                                _S({"tickers": _tick}, ["tickers"])),
    "draw_chart": _wrap("draw_chart", "Draw a chart for 1-4 companies (revenue, margins, growth, ...).",
                        _S({"tickers": _tick, "request": _str}, ["tickers", "request"])),
    "explain_drivers": _wrap("explain_drivers", "Verified reasons management gives for why numbers moved (from MD&A).",
                             _S({"ticker": _str, "form": _str}, ["ticker"])),
    "what_changed": _wrap("what_changed", "What is new, removed or reworded in a company's risk factors / business description "
                          "versus its previous 10-K (or 10-Q). Verified against both filings.",
                          _S({"ticker": _str, "form": {"type": "string", "description": "10-K (default) or 10-Q"}}, ["ticker"])),
    "build_financial_report": _wrap("build_financial_report", "Build the full financial report with Word download.",
                                    _S({"tickers": _tick}, ["tickers"])),
}
TEAM_TOOLS = {
    "qa": ["identify_companies", "stored_filings", "ensure_filings", "search_filing", "calculator"],
    "analyst": ["identify_companies", "ensure_filings", "performance_review", "explain_drivers", "what_changed", "search_filing",
                "calculator"],
}


def tools_for(team: str) -> list:
    return [ALL_TOOLS[n] for n in TEAM_TOOLS[team]]