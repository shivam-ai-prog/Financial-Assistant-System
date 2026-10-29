from __future__ import annotations

import contextvars
import re
import zlib
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import model

# ════════════════════════════════════════════════════════════════════
# 1. STORAGE
# ════════════════════════════════════════════════════════════════════
_DDL = """
CREATE TABLE IF NOT EXISTS metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id    INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    ticker       TEXT NOT NULL,
    form_type    TEXT NOT NULL,
    metric       TEXT NOT NULL,
    period_key   TEXT NOT NULL,      -- annual: FY2026   quarterly: 2026-07
    value        REAL NOT NULL,      -- US$ millions
    is_prior     INTEGER NOT NULL,   -- 0 = the filing's own period, 1 = comparison column
    UNIQUE(filing_id, metric, period_key)
);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metrics(ticker, form_type, metric);
CREATE TABLE IF NOT EXISTS metrics_done (
    filing_id   INTEGER PRIMARY KEY REFERENCES filings(id) ON DELETE CASCADE,
    ingested_at TEXT NOT NULL,
    n_metrics   INTEGER NOT NULL
);
"""


EXTRACT_VERSION = 2   # 2 = income statement + balance sheet + cash flow


def _init() -> None:
    with model.db() as c:
        c.executescript(_DDL)
        c.execute("CREATE TABLE IF NOT EXISTS metrics_meta (k TEXT PRIMARY KEY, v TEXT)")
        r = c.execute("SELECT v FROM metrics_meta WHERE k='extract_version'").fetchone()
        if not r or r["v"] != str(EXTRACT_VERSION):  # extractor improved -> re-read every stored filing once
            c.execute("DELETE FROM metrics_done")
            c.execute("INSERT OR REPLACE INTO metrics_meta(k, v) VALUES ('extract_version', ?)", (str(EXTRACT_VERSION),))


_init()

# ════════════════════════════════════════════════════════════════════
# 2. EXTRACTION OF INCOME-STATEMENT LINES
# ════════════════════════════════════════════════════════════════════
# metric key -> label patterns, in priority order (matched at the START of a line)
LINE_PATTERNS: dict[str, list[str]] = {
    "revenue": [r"total\s+(?:net\s+)?revenues?\b", r"net\s+revenues?\b", r"revenues?\b(?!\s+(?:growth|by|from|recogni))",
                r"net\s+sales\b"],
    "cost_of_revenue": [r"total\s+cost\s+of\s+revenues?\b", r"cost\s+of\s+revenues?\b", r"cost\s+of\s+sales\b"],
    "gross_profit": [r"gross\s+profit\b", r"gross\s+margin\b(?!\s*%)"],
    "rnd": [r"research\s+and\s+development\b"],
    "sga": [r"selling,\s+general\s+and\s+administrative\b", r"sales,\s+general\s+and\s+administrative\b",
            r"marketing,\s+general\s+and\s+administrative\b",
            r"general\s+and\s+administrative\b"],
    "opex": [r"total\s+operating\s+expenses\b"],
    "operating_income": [r"income\s+from\s+operations\b", r"operating\s+income\b", r"operating\s+(?:profit|loss)\b"],
    "net_income": [r"net\s+income\s+attributable\s+to\s+common\s+stockholders\b",
                   r"net\s+income\s+\(loss\)\s+attributable\s+to\s+common\s+stockholders\b",
                   r"net\s+income\b(?!\s+(?:per|attributable\s+to\s+non))"],
}
_PATTERNS_C = {k: [re.compile(p, re.I) for p in v] for k, v in LINE_PATTERNS.items()}

_HEADING = re.compile(r"consolidated\s+statements?\s+of\s+(?:operations|income|earnings)", re.I)
_NUM = re.compile(r"\(?\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?")
_ONLY_NUMBERS = re.compile(r"[\s\$\(\)\d,\.\-—–%|]*")


def _numbers(s: str) -> list[float]:
    out = []
    for tok in _NUM.findall(s):
        digits = re.sub(r"[^\d.]", "", tok)
        if not digits or digits == ".":
            continue
        v = float(digits)
        out.append(-v if "(" in tok else v)
    return out


def _values_for_label(lines: list[str], i: int, label: re.Pattern, need: int) -> list[float]:
    """Numbers on the label's line; if the PDF put each cell on its own line, read the lines below."""
    first = lines[i].strip().lstrip("|").strip()
    m = label.match(first)
    if not m:
        return []
    nums = _numbers(first[m.end():])
    j = i + 1
    while len(nums) < need and j < min(len(lines), i + 12):
        s = lines[j].strip().lstrip("|").strip()
        if not s or s in {"$", ")", "|"}:
            j += 1
            continue
        if _ONLY_NUMBERS.fullmatch(s) and any(ch.isdigit() for ch in s):
            nums += _numbers(s)
            j += 1
            continue
        break
    return nums


def _parse_window(window: str, min_cols: int) -> dict[str, list[float]]:
    lines = window.split("\n")
    found: dict[str, list[float]] = {}
    for metric, pats in _PATTERNS_C.items():
        done = False
        for pat in pats:  # priority order
            for i, ln in enumerate(lines):
                s = ln.strip().lstrip("|").strip()
                if not pat.match(s):
                    continue
                vals = _values_for_label(lines, i, pat, 4)
                if len(vals) >= min_cols:
                    found[metric] = vals[:4]
                    done = True
                    break
            if done:
                break
    return found


def _unit_factor(context: str) -> float:
    m = re.search(r"in\s+(millions|thousands|billions)", context, re.I)
    if not m:
        return 1.0
    return {"millions": 1.0, "thousands": 0.001, "billions": 1000.0}[m.group(1).lower()]


def _sane(found: dict[str, list[float]]) -> dict[str, list[float]]:
    """Drop numbers that cannot be right (a wrong row grabbed from a PDF layout)."""
    rev = found.get("revenue")
    if not rev or rev[0] <= 0:
        return {}
    out = dict(found)
    gp = out.get("gross_profit")
    if gp and gp[0] > rev[0] * 1.0001:
        out.pop("gross_profit")
        gp = None
    cost = out.get("cost_of_revenue")
    if cost and gp and abs(rev[0] - cost[0] - gp[0]) > 0.02 * rev[0]:
        out.pop("cost_of_revenue")  # the three should add up; if not, trust revenue + gross profit
    if gp and out.get("operating_income") and out["operating_income"][0] > gp[0] * 1.0001:
        out.pop("operating_income")
    for k in ("opex", "rnd", "sga", "cost_of_revenue"):
        if out.get(k) and abs(out[k][0]) > rev[0] * 1.0001:
            out.pop(k)
    return out


def extract_statement(text: str, form_type: str) -> tuple[dict[str, list[float]], float]:
    """Find the income statement inside `text`; return ({metric: [col0, col1, ...]}, unit_factor)."""
    min_cols = 3 if form_type == "10-K" else 2
    best: tuple[int, dict, float] = (0, {}, 1.0)
    for m in _HEADING.finditer(text):
        window = text[m.start(): m.start() + 7000]
        found = _sane(_parse_window(window, min_cols))
        score = len(found)
        if score > best[0]:
            best = (score, found, _unit_factor(text[max(0, m.start() - 200): m.start() + 1200]))
    return best[1], best[2]



# ---- balance sheet + cash flow (point-in-time / flow statements) ----------------------------------------------
_BS_PATTERNS: dict[str, list[str]] = {
    "cash": [r"cash\s+and\s+cash\s+equivalents\b"],
    "short_investments": [r"(?:short-term\s+investments|marketable\s+securities|short-term\s+marketable\s+securities)\b"],
    "inventory": [r"inventor(?:y|ies)\b"],
    "total_current_assets": [r"total\s+current\s+assets\b"],
    "total_assets": [r"total\s+assets\b"],
    "total_current_liabilities": [r"total\s+current\s+liabilities\b"],
    "total_liabilities": [r"total\s+liabilities\b(?!\s+and)"],
    "long_term_debt": [r"long-term\s+debt\b", r"debt,?\s+net\s+of\s+current", r"notes\s+payable\b"],
    "total_equity": [r"total\s+(?:stockholders|shareholders)['’]?\s+equity\b", r"total\s+equity\b"],
}
_CF_PATTERNS: dict[str, list[str]] = {
    "operating_cash_flow": [r"net\s+cash\b[^\d$|]{0,60}?operating\s+activities"],
    "capex": [r"purchases?\s+(?:of|related\s+to)\s+property(?:,\s+plant)?\s+(?:and|&)\s+equipment\b",
              r"capital\s+expenditures?\b", r"payments?\s+for\s+property"],
    "depreciation": [r"depreciation(?:,?\s+amortization(?:\s+and\s+impairment)?)?\b"],
    "buybacks": [r"(?:repurchases?|purchases?)\s+of\s+(?:common\s+stock|treasury)", r"share\s+repurchases?\b"],
}
_BS_C = {k: [re.compile(p, re.I) for p in v] for k, v in _BS_PATTERNS.items()}
_CF_C = {k: [re.compile(p, re.I) for p in v] for k, v in _CF_PATTERNS.items()}
_BS_HEADING = re.compile(r"consolidated\s+balance\s+sheets?", re.I)
_CF_HEADING = re.compile(r"consolidated\s+statements?\s+of\s+cash\s+flows?", re.I)


def _parse_with(window: str, pats_c: dict, min_cols: int, ncols: int) -> dict[str, list[float]]:
    lines = window.split("\n")
    found: dict[str, list[float]] = {}
    for metric, pats in pats_c.items():
        done = False
        for pat in pats:
            for i, ln in enumerate(lines):
                if not pat.match(ln.strip().lstrip("|").strip()):
                    continue
                vals = _values_for_label(lines, i, pat, ncols)
                if len(vals) >= min_cols:
                    found[metric] = vals[:ncols]
                    done = True
                    break
            if done:
                break
    return found


def _sane_bs(f: dict[str, list[float]]) -> dict[str, list[float]]:
    ta = f.get("total_assets")
    if not ta or ta[0] <= 0:
        return {}
    out = dict(f)
    for k in ("cash", "short_investments", "inventory", "total_current_assets", "total_liabilities",
              "total_current_liabilities", "long_term_debt"):
        if out.get(k) and (out[k][0] < 0 or out[k][0] > ta[0] * 1.0001):
            out.pop(k)
    if out.get("total_equity") and abs(out["total_equity"][0]) > ta[0] * 1.0001:
        out.pop("total_equity")
    if out.get("total_current_assets") and out.get("total_current_liabilities") is None:
        pass
    return out


def _extract_generic(text: str, heading: re.Pattern, pats_c: dict, min_cols: int, ncols: int,
                     sane=None) -> tuple[dict[str, list[float]], float]:
    best: tuple[int, dict, float] = (0, {}, 1.0)
    for m in heading.finditer(text):
        found = _parse_with(text[m.start(): m.start() + 9000], pats_c, min_cols, ncols)
        found = sane(found) if sane else found
        if len(found) > best[0]:
            best = (len(found), found, _unit_factor(text[max(0, m.start() - 200): m.start() + 1500]))
    return best[1], best[2]


def extract_balance_sheet(text: str) -> tuple[dict[str, list[float]], float]:
    return _extract_generic(text, _BS_HEADING, _BS_C, 2, 2, _sane_bs)


def extract_cash_flow(text: str, form_type: str) -> tuple[dict[str, list[float]], float]:
    found, factor = _extract_generic(text, _CF_HEADING, _CF_C, 3 if form_type == "10-K" else 2, 3)
    if not found.get("operating_cash_flow") or found["operating_cash_flow"][0] == 0:
        return {}, 1.0
    for k in ("capex", "buybacks", "depreciation"):  # shown as outflows in brackets; store as positive amounts
        if k in found:
            found[k] = [abs(v) for v in found[k]]
    return found, factor


def _period_keys(form_type: str, fiscal_year: int | None, period_end: str | None, n: int) -> list[tuple[str, int]]:
    """[(period_key, is_prior)] for the first n columns of the statement."""
    if form_type == "10-K" and fiscal_year:
        return [(f"FY{fiscal_year - i}", 0 if i == 0 else 1) for i in range(min(n, 3))]
    pe = (period_end or "")[:7]
    if form_type == "10-Q" and re.fullmatch(r"\d{4}-\d{2}", pe):
        y, mo = int(pe[:4]), pe[5:]
        return [(f"{y}-{mo}", 0), (f"{y - 1}-{mo}", 1)][:n]
    return []


def _filing_text_candidates(filing: dict) -> list[str]:
    """Text to search: the stored Items first, then (for PDFs) the page-text chunks in page order."""
    texts = []
    sections = model.get_sections(filing["accession_no"])
    if sections:
        texts.append("\n".join(sections.values()))
    with model.db() as c:
        rows = c.execute("SELECT content FROM chunks WHERE filing_id=? AND element_type='Text' ORDER BY id",
                         (filing["id"],)).fetchall()
    if rows:
        texts.append("\n".join(r["content"] for r in rows))
    return texts


def compute_filing_metrics(filing: dict) -> list[tuple]:
    rows: list[tuple] = []
    form = filing["form_type"]

    def add(found: dict, factor: float, keys: list[tuple[str, int]]) -> None:
        for metric, vals in found.items():
            for (pkey, prior), v in zip(keys, vals):
                rows.append((filing["id"], filing["ticker"], form, metric, pkey, v * factor, prior))

    texts = _filing_text_candidates(filing)
    for text in texts:  # income statement: first source that yields a statement wins
        found, factor = extract_statement(text, form)
        if found:
            for metric, vals in found.items():
                keys = _period_keys(form, filing["fiscal_year"], filing["period_end"], len(vals))
                for (pkey, prior), v in zip(keys, vals):
                    rows.append((filing["id"], filing["ticker"], form, metric, pkey, v * factor, prior))
            break
    own = set(r[4] for r in rows)
    for text in texts:  # balance sheet: 10-K = [year end, prior year end]; 10-Q = [quarter end, last FY end] -> keep col 0
        found, factor = extract_balance_sheet(text)
        if found:
            keys = _period_keys(form, filing["fiscal_year"], filing["period_end"], 2)
            add(found, factor, keys if form == "10-K" else keys[:1])
            break
    if form == "10-K":  # cash flow: annual only (10-Q cash flow is year-to-date, so it is not comparable)
        for text in texts:
            found, factor = extract_cash_flow(text, form)
            if found:
                add(found, factor, _period_keys(form, filing["fiscal_year"], filing["period_end"], 3))
                break
    return rows


def ensure_metrics() -> int:
    """Extract numbers for every stored 10-K / 10-Q that has none yet (or was re-ingested). Returns count."""
    done = 0
    with model.db() as c:
        status = {r["filing_id"]: r["ingested_at"] for r in c.execute("SELECT * FROM metrics_done").fetchall()}
    for f in model.list_filings():
        if f["form_type"] not in ("10-K", "10-Q") or status.get(f["id"]) == f["ingested_at"]:
            continue
        rows = compute_filing_metrics(f)
        with model.db() as c:
            c.execute("DELETE FROM metrics WHERE filing_id=?", (f["id"],))
            c.executemany("INSERT OR REPLACE INTO metrics(filing_id, ticker, form_type, metric, period_key, value, is_prior) "
                          "VALUES (?,?,?,?,?,?,?)", rows)
            c.execute("INSERT OR REPLACE INTO metrics_done(filing_id, ingested_at, n_metrics) VALUES (?,?,?)",
                      (f["id"], f["ingested_at"], len(rows)))
        done += 1
        model.log.info("metrics: %s %s %s -> %d numbers", f["ticker"], f["form_type"], f["period_end"], len(rows))
    return done


def rebuild_metrics() -> int:
    with model.db() as c:
        c.execute("DELETE FROM metrics_done")
        c.execute("DELETE FROM metrics")
    return ensure_metrics()


# ════════════════════════════════════════════════════════════════════
# 3. DATA ACCESS
# ════════════════════════════════════════════════════════════════════
BASES = {"Annual (10-K)": "10-K", "Quarterly (10-Q)": "10-Q"}

METRIC_CHOICES: dict[str, tuple[str, str]] = {  # label -> (key, kind)  kind: money | pct
    "Revenue": ("revenue", "money"),
    "Gross profit": ("gross_profit", "money"),
    "Operating income": ("operating_income", "money"),
    "Net income": ("net_income", "money"),
    "R&D expense": ("rnd", "money"),
    "SG&A expense": ("sga", "money"),
    "Total operating expenses": ("opex", "money"),
    "Cost of revenue": ("cost_of_revenue", "money"),
    "Gross margin %": ("gross_margin", "pct"),
    "Operating margin %": ("operating_margin", "pct"),
    "Net margin %": ("net_margin", "pct"),
}
_MARGINS = {"gross_margin": "gross_profit", "operating_margin": "operating_income", "net_margin": "net_income"}


def load_wide(form_type: str) -> pd.DataFrame:
    """One row per (ticker, period_key) with a column per metric (+ margins) and the source filing."""
    ensure_metrics()
    with model.db() as c:
        rows = c.execute(
            """SELECT m.ticker, m.metric, m.period_key, m.value, m.is_prior, f.filing_date, f.accession_no
               FROM metrics m JOIN filings f ON f.id = m.filing_id WHERE m.form_type=?""", (form_type,)).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    # a period reported by its own filing beats the comparison column of a later filing; then newest filing
    df = df.sort_values(["is_prior", "filing_date"], ascending=[True, False])
    df = df.drop_duplicates(["ticker", "metric", "period_key"])
    src = df.sort_values(["is_prior", "filing_date"], ascending=[True, False]).drop_duplicates(
        ["ticker", "period_key"])[["ticker", "period_key", "filing_date", "accession_no"]]
    wide = df.pivot(index=["ticker", "period_key"], columns="metric", values="value").reset_index()
    wide = wide.merge(src, on=["ticker", "period_key"], how="left")
    for mk, num in _MARGINS.items():
        if num in wide and "revenue" in wide:
            wide[mk] = wide[num] / wide["revenue"] * 100.0
    return wide.sort_values(["ticker", "period_key"]).reset_index(drop=True)


def _prev_key(key: str) -> str:
    if key.startswith("FY"):
        return f"FY{int(key[2:]) - 1}"
    return f"{int(key[:4]) - 1}{key[4:]}"


def _pretty(key: str) -> str:
    if key.startswith("FY"):
        return key
    try:
        return datetime.strptime(key, "%Y-%m").strftime("%b %Y")
    except ValueError:
        return key


def _name(ticker: str) -> str:
    try:
        import companies
        return companies.short_name(ticker)
    except Exception:
        return model.COMPANIES.get(ticker, {}).get("name", ticker)


def tickers_with_data() -> list[str]:
    with model.db() as c:
        return [r["ticker"] for r in c.execute("SELECT DISTINCT ticker FROM metrics ORDER BY ticker").fetchall()]


# ════════════════════════════════════════════════════════════════════
# 4. CHARTS  (palette validated with the dataviz validator: blue + orange, fixed per company)
# ════════════════════════════════════════════════════════════════════
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
NEUTRAL = "#b9b8b2"
COMPANY_COLORS = {"NVDA": "#2a78d6", "TSLA": "#eb6834"}
_FALLBACK = ["#1baf7a", "#e87ba4", "#4a3aa7", "#eda100", "#008300", "#e34948"]  # validated categorical slots 3-8
_active_colors: contextvars.ContextVar[dict] = contextvars.ContextVar("active_colors", default={})


def assign_colors(tickers: list[str]) -> dict[str, str]:
    """Stable colour per company (same company, same colour, every run) with no two companies in one chart sharing one."""
    out: dict[str, str] = {t: COMPANY_COLORS[t] for t in tickers if t in COMPANY_COLORS}
    used = set(out.values())
    for t in sorted(x for x in tickers if x not in out):
        i = zlib.crc32(t.encode()) % len(_FALLBACK)
        for k in range(len(_FALLBACK)):
            c = _FALLBACK[(i + k) % len(_FALLBACK)]
            if c not in used:
                break
        out[t] = c
        used.add(c)
    return out


def color_for(ticker: str) -> str:
    return _active_colors.get().get(ticker) or assign_colors([ticker])[ticker]


def _base_layout(fig: go.Figure, title: str, subtitle: str, height: int = 470, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{INK2}'>{subtitle}</span>",
                   x=0.01, xanchor="left", font=dict(size=20, color=INK)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK2),
        margin=dict(l=70, r=130, t=95, b=60),
        showlegend=legend, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom",
                                       font=dict(color=INK)),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK, size=13), bordercolor=GRID),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=True, zerolinecolor=NEUTRAL, zerolinewidth=1,
                     linecolor=SURFACE, ticks="")
    return fig


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=16, color=INK2))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, "No data to chart yet", "", height=320, legend=False)


def _fmt(v: float, kind: str) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    if kind == "pct":
        return f"{v:.1f}%"
    return f"${v / 1000:,.1f}B" if abs(v) >= 1000 else f"${v:,.0f}M"


def _scale(series: pd.Series, kind: str) -> pd.Series:
    return series if kind == "pct" else series / 1000.0  # US$ billions


def _axis_title(kind: str) -> str:
    return "% of revenue" if kind == "pct" else "US$ billions"


def _series_table(df: pd.DataFrame, tickers: list[str], mkey: str, kind: str, form: str, yoy: bool = True) -> pd.DataFrame:
    rows = []
    for t in tickers:
        sub = df[df["ticker"] == t].set_index("period_key")
        for k in sub.index:
            v = sub.at[k, mkey] if mkey in sub else None
            if v is None or pd.isna(v):
                continue
            prev = _prev_key(k)
            g = None
            if yoy and kind == "money" and prev in sub.index and mkey in sub and pd.notna(sub.at[prev, mkey]) and sub.at[prev, mkey]:
                g = (v - sub.at[prev, mkey]) / abs(sub.at[prev, mkey]) * 100
            rows.append({"Company": _name(t), "Period": _pretty(k), "Value": _fmt(v, kind),
                         "YoY change": "" if g is None else f"{g:+.1f}%",
                         "Source": f"{form} filed {sub.at[k, 'filing_date']}"})
    return pd.DataFrame(rows, columns=["Company", "Period", "Value", "YoY change", "Source"])


def _note(basis_label: str, tickers: list[str]) -> str:
    base = "Figures are US$ and come from each company's income statement in the stored filings."
    if basis_label.startswith("Annual") and len(tickers) > 1:
        base += (" **Fiscal years can differ:** each company's FY label follows its own year-end "
                 "(e.g. NVIDIA's FY2026 ended in January 2026, Tesla's FY2025 in December 2025).")
    return base

def aligned_year(ticker: str, period_key: str) -> str:
    """NVIDIA FY2026 (ends Jan 2026) -> '2025', comparable with AMD FY2025 (ends Dec 2025)."""
    if not period_key.startswith("FY"):
        return period_key
    with model.db() as c:
        r = c.execute("SELECT fiscal_year, period_end FROM filings WHERE ticker=? AND form_type='10-K' AND "
                      "period_end IS NOT NULL AND fiscal_year IS NOT NULL ORDER BY fiscal_year DESC LIMIT 1",
                      (ticker,)).fetchone()
    if not r or len(r["period_end"] or "") < 7:
        return period_key
    year = int(r["period_end"][:4]) - (r["fiscal_year"] - int(period_key[2:]))
    return str(year - 1 if int(r["period_end"][5:7]) <= 3 else year)


def chart_trend(df, tickers, label, basis_label, bars=False):
    mkey, kind = METRIC_CHOICES[label]
    form = BASES[basis_label]
    fig = go.Figure()
    periods = sorted(df["period_key"].unique(), key=lambda k: (k if not k.startswith("FY") else k[2:]))
    align = form == "10-k" and len(tickers) > 1
    lab = (lambda t, k: aligned_year(t,k)) if align else (lambda t, k: _pretty(k))
    if align:
        cats = sorted({lab(t,k) for t in tickers for k in set(df[df["ticker"] == t]["period_key"])})
    else:
        cats = [_pretty(k) for k in periods]
    plotted = 0
    for t in tickers:
        sub = df[(df["ticker"] == t)].set_index("period_key")
        if mkey not in sub:
            continue
        xs, ys = [], []
        for k in periods:
            if k in sub.index and pd.notna(sub.at[k, mkey]):
                xs.append(lab(t,k))
                ys.append(float(sub.at[k, mkey]))
        if not xs:
            continue
        plotted += 1
        yv = _scale(pd.Series(ys), kind)
        custom = [_fmt(v, kind) for v in ys]
        hover = f"<b>{_name(t)}</b><br>%{{x}}<br>{label}: %{{customdata}}<extra></extra>"
        if bars:
            fig.add_bar(x=xs, y=yv, name=_name(t), marker=dict(color=color_for(t), cornerradius=4,
                        line=dict(color=SURFACE, width=2)), customdata=custom, hovertemplate=hover)
        else:
            fig.add_scatter(x=xs, y=yv, name=_name(t), mode="lines+markers",
                            line=dict(color=color_for(t), width=2.5),
                            marker=dict(size=9, color=color_for(t), line=dict(color=SURFACE, width=2)),
                            customdata=custom, hovertemplate=hover)
            fig.add_annotation(x=xs[-1], y=float(yv.iloc[-1]), text=_name(t), showarrow=False, xanchor="left",
                               xshift=14, font=dict(color=INK, size=13))
    if not plotted:
        return _empty(f"No '{label}' numbers were found in the stored {form} filings."), pd.DataFrame()
    verb = "Company comparison" if bars else "Trend"
    fig = _base_layout(fig, f"{label}: {verb}", f"{basis_label}  -  hover for exact values")
    fig.update_yaxes(title_text=_axis_title(kind), title_font=dict(color=INK2))
    if bars:
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
        fig.update_xaxes(categoryorder="array", categoryarray=cats)
    else:
        fig.update_xaxes(categoryorder="array", categoryarray=cats)
    if kind == "money":
        fig.update_yaxes(rangemode="tozero")
    return fig, _series_table(df, tickers, mkey, kind, form)


def chart_growth(df, tickers, label, basis_label):
    mkey, kind = METRIC_CHOICES[label]
    form = BASES[basis_label]
    if kind != "money":
        return _empty("Growth is shown for money metrics. Pick Revenue, Net income, etc."), pd.DataFrame()
    fig = go.Figure()
    rows = []
    plotted = 0
    for t in tickers:
        sub = df[df["ticker"] == t].set_index("period_key")
        if mkey not in sub:
            continue
        xs, ys = [], []
        for k in sorted(sub.index, key=lambda k: (k if not k.startswith("FY") else k[2:])):
            p = _prev_key(k)
            if p in sub.index and pd.notna(sub.at[k, mkey]) and pd.notna(sub.at[p, mkey]) and sub.at[p, mkey] > 0:
                g = (sub.at[k, mkey] - sub.at[p, mkey]) / sub.at[p, mkey] * 100
                xs.append(_pretty(k))
                ys.append(g)
                rows.append({"Company": _name(t), "Period": _pretty(k), "Value": _fmt(sub.at[k, mkey], kind),
                             "YoY change": f"{g:+.1f}%", "Source": f"{form} filed {sub.at[k, 'filing_date']}"})
        if xs:
            plotted += 1
            fig.add_bar(x=xs, y=ys, name=_name(t), marker=dict(color=color_for(t), cornerradius=4,
                        line=dict(color=SURFACE, width=2)),
                        customdata=[f"{v:+.1f}%" for v in ys],
                        hovertemplate=f"<b>{_name(t)}</b><br>%{{x}}<br>{label} vs a year earlier: %{{customdata}}<extra></extra>")
    if not plotted:
        return _empty("Growth needs two comparable periods; none found yet."), pd.DataFrame()
    fig = _base_layout(fig, f"{label}: change vs a year earlier", f"{basis_label}  -  bars above zero grew, below zero shrank")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    order = sorted(df["period_key"].unique(), key=lambda k: (k if not k.startswith("FY") else k[2:]))
    fig.update_xaxes(categoryorder="array", categoryarray=[_pretty(k) for k in order])
    fig.update_yaxes(title_text="% change", ticksuffix="%")
    return fig, pd.DataFrame(rows, columns=["Company", "Period", "Value", "YoY change", "Source"])


def _tick(key: str) -> str:
    """Axis label that cannot collide: 'Apr<br>2025' (two short lines) instead of 'Apr 2025'."""
    return _pretty(key).replace(" ", "<br>")


def chart_margins(df, tickers, basis_label):
    form = BASES[basis_label]
    names = [("gross_margin", "Gross margin"), ("operating_margin", "Operating margin"), ("net_margin", "Net margin")]
    if not any(m in df for m, _ in names):
        return _empty("Margins need revenue and profit lines; none found yet."), pd.DataFrame()
    fig = make_subplots(rows=1, cols=3, subplot_titles=[n for _, n in names], horizontal_spacing=0.09)
    periods = sorted(df["period_key"].unique(), key=lambda k: (k if not k.startswith("FY") else k[2:]))
    cats = [_tick(k) for k in periods]
    rows, seen = [], set()
    for ci, (mk, mname) in enumerate(names, start=1):
        if mk not in df:
            continue
        for t in tickers:
            sub = df[df["ticker"] == t].set_index("period_key")
            xs = [_tick(k) for k in periods if k in sub.index and pd.notna(sub.at[k, mk])]
            ys = [float(sub.at[k, mk]) for k in periods if k in sub.index and pd.notna(sub.at[k, mk])]
            if not xs:
                continue
            fig.add_scatter(x=xs, y=ys, name=_name(t), mode="lines+markers", row=1, col=ci,
                            showlegend=t not in seen, legendgroup=t,
                            line=dict(color=color_for(t), width=2.5),
                            marker=dict(size=9, color=color_for(t), line=dict(color=SURFACE, width=2)),
                            customdata=[f"{v:.1f}%" for v in ys],
                            hovertemplate=f"<b>{_name(t)}</b><br>%{{x}}<br>{mname}: %{{customdata}}<extra></extra>")
            seen.add(t)
            for k, v in zip([k for k in periods if k in sub.index and pd.notna(sub.at[k, mk])], ys):
                rows.append({"Company": _name(t), "Period": _pretty(k), "Value": f"{v:.1f}%", "YoY change": "",
                             "Source": f"{mname} (computed from {form} filed {sub.at[k, 'filing_date']})"})
    fig = _base_layout(fig, "Profitability: how much of each revenue dollar is kept",
                       f"{basis_label}  -  three margins side by side (higher is better)", height=440)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(categoryorder="array", categoryarray=cats, tickangle=0, tickfont=dict(size=11), automargin=True)
    fig.update_layout(margin=dict(l=60, r=40, t=120, b=90),
                      legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center", yanchor="top", font=dict(color=INK)))
    for a in fig.layout.annotations:
        a.font = dict(size=14, color=INK)
    return fig, pd.DataFrame(rows, columns=["Company", "Period", "Value", "YoY change", "Source"])


def chart_waterfall(df, tickers, basis_label):
    form = BASES[basis_label]
    need = "revenue"
    tick = [t for t in tickers if need in df and not df[df["ticker"] == t].dropna(subset=[need]).empty]
    if not tick:
        return _empty("No revenue found to build the waterfall."), pd.DataFrame()
    fig = make_subplots(rows=1, cols=len(tick), subplot_titles=[_name(t) for t in tick], horizontal_spacing=0.1)
    rows = []
    for ci, t in enumerate(tick, start=1):
        sub = df[df["ticker"] == t].dropna(subset=["revenue"])
        last = sub.sort_values("period_key", key=lambda s: s.str.replace("FY", "")).iloc[-1]
        rev = float(last["revenue"])
        gp = last.get("gross_profit")
        oi = last.get("operating_income")
        steps = [("Revenue", rev, "absolute")]
        if pd.notna(gp):
            steps += [("Cost of revenue", -(rev - float(gp)), "relative"), ("Gross profit", float(gp), "total")]
            if pd.notna(oi):
                steps += [("Operating expenses", -(float(gp) - float(oi)), "relative"),
                          ("Operating income", float(oi), "total")]
        vals = [s[1] / 1000 for s in steps]
        fig.add_trace(go.Waterfall(
            x=[s[0] for s in steps], y=vals, measure=[s[2] for s in steps],
            text=[f"${abs(v):,.1f}B" for v in vals], textposition="outside", textfont=dict(color=INK, size=12),
            totals=dict(marker=dict(color=color_for(t), line=dict(color=SURFACE, width=2))),
            increasing=dict(marker=dict(color=color_for(t))),
            decreasing=dict(marker=dict(color=NEUTRAL, line=dict(color=SURFACE, width=2))),
            connector=dict(line=dict(color=GRID, width=1)), showlegend=False,
            hovertemplate="<b>%{x}</b><br>US$ %{y:,.1f}B<extra></extra>"), row=1, col=ci)
        period = _pretty(str(last["period_key"]))
        fig.layout.annotations[ci - 1].update(text=f"<b>{_name(t)}</b>  -  {period}", font=dict(size=15, color=INK))
        for (n, v, _) in steps:
            rows.append({"Company": _name(t), "Period": period, "Value": f"{n}: {_fmt(abs(v), 'money')}",
                         "YoY change": "", "Source": f"{form} filed {last['filing_date']}"})
    fig = _base_layout(fig, "Where each revenue dollar goes (latest period)",
                       "Colored bars are totals, gray bars are costs taken away  -  hover for values", height=480, legend=False)
    fig.update_yaxes(title_text="US$ billions", rangemode="tozero")
    fig.update_xaxes(tickangle=-20)
    fig.update_layout(margin=dict(l=70, r=40, t=140, b=90))
    return fig, pd.DataFrame(rows, columns=["Company", "Period", "Value", "YoY change", "Source"])


CHART_KINDS = ["Trend over time", "Compare companies", "Growth vs a year earlier",
               "Profit margins", "Where revenue goes (waterfall)"]
_NEEDS_METRIC = {"Trend over time", "Compare companies", "Growth vs a year earlier"}


def _to_tickers(items) -> list[str]:
    """Accepts tickers ('AAPL') or display names ('Apple', 'NVIDIA') and returns tickers, order kept."""
    with model.db() as c:
        stored = {r["ticker"] for r in c.execute("SELECT DISTINCT ticker FROM filings").fetchall()}
    by_name = {}
    for t in stored | set(model.COMPANIES):
        by_name[_name(t).lower()] = t
        by_name[model.company_name(t).lower()] = t
    out = []
    for it in items or []:
        key = str(it).strip()
        t = key.upper() if key.upper() in stored or key.upper() in model.COMPANIES else by_name.get(key.lower())
        if t and t not in out:
            out.append(t)
    return out


def make_chart(kind: str, company_names: list[str], metric_label: str, basis_label: str):
    """Entry point used by the UI. Returns (figure, table DataFrame, note markdown)."""
    tickers = _to_tickers(company_names)
    if not tickers:
        return _empty("Pick at least one company."), pd.DataFrame(), ""
    df = load_wide(BASES[basis_label])
    if df.empty:
        return (_empty("No financial numbers yet. Load filings in admin.py first (10-K PDFs or Fetch from SEC)."),
                pd.DataFrame(), "")
    df = df[df["ticker"].isin(tickers)]
    tickers = [t for t in tickers if t in set(df["ticker"])]
    if df.empty:
        return _empty(f"No {basis_label} numbers stored for the selected company."), pd.DataFrame(), ""
    _active_colors.set(assign_colors(tickers))
    if kind == "Trend over time":
        fig, tab = chart_trend(df, tickers, metric_label, basis_label)
    elif kind == "Compare companies":
        fig, tab = chart_trend(df, tickers, metric_label, basis_label, bars=True)
    elif kind == "Growth vs a year earlier":
        fig, tab = chart_growth(df, tickers, metric_label, basis_label)
    elif kind == "Profit margins":
        fig, tab = chart_margins(df, tickers, basis_label)
    else:
        fig, tab = chart_waterfall(df, tickers, basis_label)
    return fig, tab, _note(basis_label, tickers)


# ════════════════════════════════════════════════════════════════════
# 5. SNAPSHOT (one glance, per company)
# ════════════════════════════════════════════════════════════════════
def snapshot_markdown() -> str:
    lines = []
    for basis_label, form in BASES.items():
        df = load_wide(form)
        if df.empty:
            continue
        lines.append(f"**{basis_label}**\n\n| Company | Period | Revenue | vs year earlier | Gross margin | Operating margin | Net margin |\n"
                     "|---|---|---|---|---|---|---|")
        for t in sorted(df["ticker"].unique()):
            sub = df[df["ticker"] == t].dropna(subset=["revenue"]).set_index("period_key")
            if sub.empty:
                continue
            k = sorted(sub.index, key=lambda k: (k if not k.startswith("FY") else k[2:]))[-1]
            p = _prev_key(k)
            g = ""
            if p in sub.index and sub.at[p, "revenue"]:
                g = f"{(sub.at[k, 'revenue'] - sub.at[p, 'revenue']) / sub.at[p, 'revenue'] * 100:+.1f}%"
            pct = lambda c: _fmt(sub.at[k, c], "pct") if c in sub and pd.notna(sub.at[k, c]) else "n/a"  # noqa: E731
            lines.append(f"| {_name(t)} | {_pretty(k)} | {_fmt(sub.at[k, 'revenue'], 'money')} | {g or 'n/a'} | "
                         f"{pct('gross_margin')} | {pct('operating_margin')} | {pct('net_margin')} |")
        lines.append("")
    return "\n".join(lines) if lines else "_No financial numbers yet - load filings in admin.py first._"


# ════════════════════════════════════════════════════════════════════
# 6. THE CHART TOOL (called by the router, like the calculator)
# ════════════════════════════════════════════════════════════════════
_CHART_WORDS = re.compile(r"\b(?:charts?|graphs?|plot(?:s|ted|ting)?|visuali[sz]\w*|diagrams?|waterfall)\b", re.I)
_ANALYSIS_WORDS = re.compile(
    r"\b(?:explain|why|discuss|analy[sz]\w*|reasons?|drivers?|insights?|implications?|advis\w*|outlook|"
    r"commentary|interpret\w*|caused?|drove|summari[sz]\w*|recommend\w*)\b", re.I)


def wants_chart(query: str) -> bool:
    """True when the user asks to see something drawn: chart, graph, plot, visualise, diagram, waterfall."""
    return bool(_CHART_WORDS.search(query))


def wants_analysis(query: str) -> bool:
    """True when the user also wants written reasoning (explain, why, discuss ...), not only the picture."""
    return bool(_ANALYSIS_WORDS.search(query))


# most specific first; the earliest mention in the question wins
_METRIC_WORDS: list[tuple[str, str]] = [
    ("Gross margin %", r"gross\s+(?:profit\s+)?margins?"),
    ("Operating margin %", r"operating\s+(?:profit\s+|income\s+)?margins?|ebit\s+margins?"),
    ("Net margin %", r"net\s+(?:profit\s+|income\s+)?margins?"),
    ("Total operating expenses", r"operating\s+expenses?|opex"),
    ("R&D expense", r"r\s*&\s*d\b|research\s+and\s+development|\bresearch\b"),
    ("SG&A expense", r"sg\s*&\s*a\b|selling,?\s+general"),
    ("Cost of revenue", r"cost\s+of\s+(?:revenues?|sales|goods)|cogs"),
    ("Gross profit", r"gross\s+profit"),
    ("Operating income", r"operating\s+(?:income|profit)|income\s+from\s+operations|\bebit\b"),
    ("Net income", r"net\s+(?:income|profit|earnings)|\bprofits?\b(?!\s+margin)|\bearnings\b"),
    ("Revenue", r"\brevenues?\b|\bsales\b|top\s+line"),
]
_TO_MARGIN = {"Gross profit": "Gross margin %", "Operating income": "Operating margin %", "Net income": "Net margin %"}


def _find_metric(q: str) -> str | None:
    best: tuple[int, str] | None = None
    for label, pat in _METRIC_WORDS:
        m = re.search(pat, q)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), label)
    if best and best[1] in _TO_MARGIN and re.search(r"\bmargins?\b", q):  # "net income margin"
        return _TO_MARGIN[best[1]]
    return best[1] if best else None


def _find_kind(q: str, metric: str | None) -> str | None:
    if re.search(r"waterfall|where\s+(?:does\s+|do\s+)?(?:the\s+|our\s+)?(?:revenue|money)\s+go|revenue\s+(?:to|into)\s+profit|"
                 r"cost\s+structure|revenue\s+breakdown|breakdown\s+of\s+revenue", q):
        return "Where revenue goes (waterfall)"
    if re.search(r"margins\b|profitability|profit\s+margin", q) and not (metric and metric.endswith("%")):
        return "Profit margins"
    if re.search(r"growth|grew|grow\b|growing|yoy|year[- ]over[- ]year|y/y|% change|percentage change", q):
        return "Growth vs a year earlier"
    if re.search(r"\bbars?\b|bar\s+(?:chart|graph)|side\s+by\s+side|head[- ]to[- ]head|compar|versus|\bvs\.?\b|against", q):
        return "Compare companies"
    if re.search(r"trend|over\s+time|histor|trajectory|\bline\b|evolution|progress|since", q):
        return "Trend over time"
    return None


def parse_chart_request(query: str, tickers: list[str], last: dict | None = None) -> dict:
    """Turn the question into a chart spec: {kind, metric, basis, tickers}. Missing parts come from the previous chart."""
    q = query.lower()
    last = last or {}
    metric = _find_metric(q)
    kind = _find_kind(q, metric)
    if re.search(r"quarter|10-q|\bq[1-4]\b|three\s+months", q):
        basis = "Quarterly (10-Q)"
    elif re.search(r"annual|10-k|full[- ]year|fiscal\s+year|yearly|year[- ]by[- ]year", q):
        basis = "Annual (10-K)"
    else:
        basis = last.get("basis") or "Annual (10-K)"
    picked = list(tickers) or list(last.get("tickers") or [])
    if not picked:  # nobody named: use the stored companies only when that is a short, obvious list
        with model.db() as c:
            have = [r["ticker"] for r in c.execute("SELECT DISTINCT ticker FROM filings ORDER BY ticker").fetchall()]
        picked = have if len(have) <= 3 else []
    if kind is None:
        if metric is None and last.get("kind"):
            kind = last["kind"]
        else:
            kind = "Compare companies" if len(picked) > 1 else "Trend over time"
    if kind == "Compare companies" and len(picked) == 1:
        kind = "Trend over time"
    metric = metric or last.get("metric") or "Revenue"
    if METRIC_CHOICES[metric][1] == "pct" and kind == "Growth vs a year earlier":
        kind = "Trend over time"
    return {"kind": kind, "metric": metric, "basis": basis, "tickers": picked}


_last_chart: dict[str, dict] = {}


def _sorted_keys(keys) -> list[str]:
    return sorted(keys, key=lambda k: (k if not k.startswith("FY") else k[2:]))


def _facts(kind: str, label: str, basis_label: str, tickers: list[str]) -> list[str]:
    """Plain-language findings, computed in Python from the stored numbers (never written by the LLM)."""
    df = load_wide(BASES[basis_label])
    df = df[df["ticker"].isin(tickers)] if not df.empty else df
    if df.empty:
        return []
    mkey, mkind = METRIC_CHOICES[label]
    out: list[str] = []

    def series(t, key):
        sub = df[df["ticker"] == t].set_index("period_key")
        return sub[key].dropna() if key in sub else pd.Series(dtype=float)

    if kind in ("Trend over time", "Compare companies"):
        latest = {}
        for t in tickers:
            s = series(t, mkey)
            if s.empty:
                continue
            keys = _sorted_keys(s.index)
            k, v = keys[-1], float(s[keys[-1]])
            latest[t] = (k, v)
            line = f"- **{_name(t)}:** {label} was **{_fmt(v, mkind)}** in {_pretty(k)}"
            p = _prev_key(k)
            if p in s.index and mkind == "money" and s[p] > 0:
                g = (v - s[p]) / abs(s[p]) * 100
                line += f", {'up' if g >= 0 else 'down'} {abs(g):.1f}% from {_fmt(float(s[p]), mkind)} in {_pretty(p)}"
            elif p in s.index and mkind == "pct":
                line += f" (was {float(s[p]):.1f}% in {_pretty(p)}, a change of {v - float(s[p]):+.1f} percentage points)"
            if len(keys) > 2:
                line += f". Over {len(keys)} periods it moved from {_fmt(float(s[keys[0]]), mkind)} ({_pretty(keys[0])}) to {_fmt(v, mkind)}"
            out.append(line + ".")
        if len(latest) == 2 and mkind == "money":
            (ta, (ka, va)), (tb, (kb, vb)) = latest.items()
            if va > 0 and vb > 0:
                big, small = ((ta, ka, va), (tb, kb, vb)) if va >= vb else ((tb, kb, vb), (ta, ka, va))
                out.append(f"- **Side by side:** {_name(big[0])}'s {label.lower()} ({_pretty(big[1])}) is "
                           f"{big[2] / small[2]:.1f}x {_name(small[0])}'s ({_pretty(small[1])}).")
    elif kind == "Growth vs a year earlier":
        for t in tickers:
            s = series(t, mkey)
            parts = []
            for k in reversed(_sorted_keys(s.index)):
                p = _prev_key(k)
                if p in s.index and s[p] > 0:
                    g = (float(s[k]) - float(s[p])) / float(s[p]) * 100
                    parts.append(f"{'grew' if g >= 0 else 'fell'} **{abs(g):.1f}%** in {_pretty(k)}")
            if parts:
                out.append(f"- **{_name(t)}:** {label} " + ", ".join(parts[:3]) + " (each compared with the year before).")
    elif kind == "Profit margins":
        for t in tickers:
            sub = df[df["ticker"] == t].dropna(subset=["gross_margin", "operating_margin", "net_margin"], how="all") \
                if "gross_margin" in df else pd.DataFrame()
            if sub.empty:
                continue
            row = sub.sort_values("period_key", key=lambda c: c.str.replace("FY", "")).iloc[-1]
            bits = [f"{row[c]:.0f}¢ {txt}" for c, txt in (("gross_margin", "as gross profit"),
                                                           ("operating_margin", "after operating costs"),
                                                           ("net_margin", "as net income")) if c in row and pd.notna(row[c])]
            out.append(f"- **{_name(t)}** ({_pretty(row['period_key'])}): of every $1 of revenue it keeps " + ", ".join(bits) + ".")
    else:  # waterfall
        for t in tickers:
            sub = df[df["ticker"] == t].dropna(subset=["revenue"])
            if sub.empty:
                continue
            r = sub.sort_values("period_key", key=lambda c: c.str.replace("FY", "")).iloc[-1]
            rev = float(r["revenue"])
            line = f"- **{_name(t)}** ({_pretty(r['period_key'])}): of **{_fmt(rev, 'money')}** revenue"
            gp, oi = r.get("gross_profit"), r.get("operating_income")
            if pd.notna(gp):
                line += f", {_fmt(rev - float(gp), 'money')} went to cost of revenue, leaving **{_fmt(float(gp), 'money')}** gross profit"
                if pd.notna(oi):
                    line += (f"; {_fmt(float(gp) - float(oi), 'money')} went to operating expenses, "
                             f"leaving **{_fmt(float(oi), 'money')}** operating income ({float(oi) / rev * 100:.1f}% of revenue)")
            out.append(line + ".")
    return out


def run_chart_tool(query: str, decision: dict, session_id: str = "default") -> dict:
    """The visualisation tool. Called by the orchestrator when the router sets needs_chart.

    Returns {spec, figures, text, sources, ingest}. Numbers come from the stored filings; if a newer filing is
    available on SEC it is fetched first (same live ingestion as the text answers).
    """
    spec = parse_chart_request(query, decision.get("tickers") or [], _last_chart.get(session_id))
    if not spec["tickers"]:
        return {"spec": spec, "figures": [], "ingest": [], "sources": [],
                "text": model._which_company_message(query)}
    decision["tickers"] = spec["tickers"]
    if not any(decision.get(k) for k in ("needs_10k", "needs_10q", "needs_8k")):  # lets a written analysis follow
        decision["needs_10q" if spec["basis"].startswith("Quarterly") else "needs_10k"] = True
    form = BASES[spec["basis"]]
    ingest: list[str] = []
    for t in spec["tickers"]:
        ingest += model.ensure_filings(t, [form])

    fig, table, note = make_chart(spec["kind"], spec["tickers"], spec["metric"], spec["basis"])
    if table.empty:  # nothing worth drawing: explain instead of showing an empty frame
        reason = fig.layout.annotations[0].text if fig.layout.annotations else "no data"
        return {"spec": spec, "figures": [], "ingest": ingest, "sources": [],
                "text": f"I couldn't draw that chart: {reason}"}
    _last_chart[session_id] = spec

    titles = {"Trend over time": f"{spec['metric']} over time", "Compare companies": f"{spec['metric']}: companies compared",
              "Growth vs a year earlier": f"{spec['metric']}: growth vs a year earlier",
              "Profit margins": "Profit margins", "Where revenue goes (waterfall)": "Where revenue goes"}
    facts = _facts(spec["kind"], spec["metric"], spec["basis"], spec["tickers"])
    text = f"### 📊 {titles[spec['kind']]} ({spec['basis']})\n\n" + "\n".join(facts)
    if note:
        text += f"\n\n<sub>{note}</sub>"
    sources = sorted({f"{r['Company']}: {r['Source']}" for _, r in table.iterrows()})
    return {"spec": spec, "figures": [fig], "text": text, "sources": sources, "ingest": ingest}