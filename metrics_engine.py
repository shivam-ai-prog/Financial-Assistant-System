"""
metrics_engine.py - deterministic performance measurement (NO LLM, every number traceable to a stored filing).

  period_table(ticker, basis)     one row per period: raw lines + ratios + YoY (+ sequential QoQ for quarters)
  scorecard(row)                  transparent strong / average / weak rating per area (thresholds are shown)
  peer_table(tickers, basis)      latest period of each company side by side, ranked, gap to peer median
  missing_note(...)               what could not be read from the filings (never guessed)

Basis: "10-K" (annual, periods like FY2025) or "10-Q" (quarterly, periods like 2026-07).
Money is US$ millions as stored by analytics.py. A ratio whose inputs are missing is None ("n/a"), never estimated.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import pandas as pd

import analytics
import model

# metric key -> (label, kind, higher_is_better; None = informational, never ranked)   kind: money | pct | x (multiple)
RATIOS: dict[str, tuple[str, str, bool]] = {
    "revenue": ("Revenue", "money", True),
    "gross_profit": ("Gross profit", "money", True),
    "operating_income": ("Operating income", "money", True),
    "net_income": ("Net income", "money", True),
    "operating_cash_flow": ("Operating cash flow", "money", True),
    "free_cash_flow": ("Free cash flow", "money", True),
    "gross_margin": ("Gross margin", "pct", True),
    "operating_margin": ("Operating margin", "pct", True),
    "net_margin": ("Net margin", "pct", True),
    "fcf_margin": ("Free cash flow margin", "pct", True),
    "rnd_pct": ("R&D as % of revenue", "pct", None),   # informational: higher is not "better"
    "revenue_growth": ("Revenue growth (YoY)", "pct", True),
    "net_income_growth": ("Net income growth (YoY)", "pct", True),
    "roe": ("Return on equity", "pct", True),
    "roa": ("Return on assets", "pct", True),
    "current_ratio": ("Current ratio", "x", True),
    "liabilities_to_equity": ("Liabilities / equity", "x", False),
    "debt_to_equity": ("Long-term debt / equity", "x", False),
    "net_cash": ("Net cash (cash + investments - LT debt)", "money", True),
}

FLOW_GROWTH = ["revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "free_cash_flow"]


def fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "x":
        return f"{v:.2f}x"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v) / 1000:,.2f}B" if abs(v) >= 1000 else f"{sign}${abs(v):,.0f}M"


def _get(d: dict, k: str):
    v = d.get(k)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)


def _div(a, b, pct=False):
    if a is None or b is None or b == 0:
        return None
    return a / b * (100.0 if pct else 1.0)


def _pct_change(new, old):
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old) * 100.0


def _sort_key(key: str):
    return (int(key[2:]), 0) if key.startswith("FY") else (int(key[:4]), int(key[5:7]))


def _prev_quarter_key(key: str) -> str:
    y, m = int(key[:4]), int(key[5:7])
    m -= 3
    if m <= 0:
        y, m = y - 1, m + 12
    return f"{y}-{m:02d}"


def _derive(row: dict) -> dict:
    """Ratios from one period's raw lines (inputs of the SAME period only)."""
    r = dict(row)
    rev, gp, oi, ni = (_get(row, k) for k in ("revenue", "gross_profit", "operating_income", "net_income"))
    ocf, capex, rnd = _get(row, "operating_cash_flow"), _get(row, "capex"), _get(row, "rnd")
    ta, eq, tl = _get(row, "total_assets"), _get(row, "total_equity"), _get(row, "total_liabilities")
    if tl is None and ta is not None and eq is not None:
        tl = ta - eq                      # liabilities = assets - equity (includes any non-controlling interest)
        r["total_liabilities"] = tl
    r["free_cash_flow"] = ocf - capex if ocf is not None and capex is not None else None
    r["gross_margin"] = _div(gp, rev, True)
    r["operating_margin"] = _div(oi, rev, True)
    r["net_margin"] = _div(ni, rev, True)
    r["fcf_margin"] = _div(r["free_cash_flow"], rev, True)
    r["rnd_pct"] = _div(rnd, rev, True)
    r["roe"] = _div(ni, eq, True) if eq and eq > 0 else None
    r["roa"] = _div(ni, ta, True)
    r["current_ratio"] = _div(_get(row, "total_current_assets"), _get(row, "total_current_liabilities"))
    r["liabilities_to_equity"] = _div(tl, eq) if eq and eq > 0 else None
    r["debt_to_equity"] = _div(_get(row, "long_term_debt"), eq) if eq and eq > 0 else None
    cash = _get(row, "cash")
    if cash is not None:
        r["net_cash"] = cash + (_get(row, "short_investments") or 0.0) - (_get(row, "long_term_debt") or 0.0)
    return r


def period_table(ticker: str, basis: str = "10-K") -> pd.DataFrame:
    """Rows oldest -> newest. Extra columns: <metric>_yoy (all flows), <metric>_qoq (quarterly only),
    revenue_growth, net_income_growth, source_filing, source_date."""
    wide = analytics.load_wide(basis)
    if wide.empty:
        return pd.DataFrame()
    sub = wide[wide["ticker"] == ticker]
    if sub.empty:
        return pd.DataFrame()
    raw = {r["period_key"]: r for r in sub.to_dict("records")}
    rows = {k: _derive(v) for k, v in raw.items()}
    out = []
    for key in sorted(rows, key=_sort_key):
        r = rows[key]
        prev_y = rows.get(analytics._prev_key(key))
        prev_q = rows.get(_prev_quarter_key(key)) if basis == "10-Q" else None
        for m in FLOW_GROWTH:
            r[f"{m}_yoy"] = _pct_change(_get(r, m), _get(prev_y, m)) if prev_y else None
            r[f"{m}_qoq"] = _pct_change(_get(r, m), _get(prev_q, m)) if prev_q else None
        r["revenue_growth"] = r["revenue_yoy"]
        r["net_income_growth"] = r["net_income_yoy"]
        r["period"] = key
        out.append(r)
    df = pd.DataFrame(out)
    df["period_label"] = df["period"].map(analytics._pretty)
    df["source_filing"] = df.get("accession_no")
    df["source_date"] = df.get("filing_date")
    return df


# ── scorecard ─────────────────────────────────────────────────────────────────────────────────────────────────
# (area, metric, strong threshold, weak threshold, higher_is_better) - shown in every report so nothing is hidden
RULES = [
    ("Growth", "revenue_growth", 15.0, 3.0, True),
    ("Profitability", "operating_margin", 25.0, 10.0, True),
    ("Cash generation", "fcf_margin", 15.0, 5.0, True),
    ("Returns", "roe", 20.0, 8.0, True),
    ("Liquidity", "current_ratio", 2.0, 1.2, True),
    ("Leverage", "liabilities_to_equity", 0.8, 2.0, False),
]


def scorecard(row: dict | pd.Series) -> dict:
    row = dict(row)
    items, pts, n = [], 0, 0
    for area, key, strong, weak, hib in RULES:
        v = _get(row, key)
        if v is None:
            items.append({"area": area, "metric": RATIOS[key][0], "value": None, "rating": "n/a",
                          "rule": _rule_text(key, strong, weak, hib)})
            continue
        good = (v >= strong) if hib else (v <= strong)
        bad = (v < weak) if hib else (v > weak)
        rating = "Strong" if good else "Weak" if bad else "Average"
        pts += 2 if good else 0 if bad else 1
        n += 1
        items.append({"area": area, "metric": RATIOS[key][0], "value": v, "rating": rating,
                      "rule": _rule_text(key, strong, weak, hib)})
    pct = pts / (2 * n) * 100 if n else None
    overall = None if pct is None else "Strong" if pct >= 80 else "Weak" if pct < 40 else "Average"
    return {"items": items, "overall": overall, "score_pct": pct, "areas_scored": n, "areas_total": len(RULES)}


def _rule_text(key: str, strong: float, weak: float, hib: bool) -> str:
    kind = RATIOS[key][1]
    f = lambda x: fmt(x, kind)  # noqa: E731
    return f"Strong {'>=' if hib else '<='} {f(strong)}; Weak {'<' if hib else '>'} {f(weak)}"


# ── peers ─────────────────────────────────────────────────────────────────────────────────────────────────────
PEER_METRICS = ["revenue", "revenue_growth", "gross_margin", "operating_margin", "net_margin", "fcf_margin",
                "rnd_pct", "roe", "current_ratio", "liabilities_to_equity"]


@dataclass
class PeerResult:
    table: pd.DataFrame          # rows = metrics, columns = tickers (+ Median)
    ranks: pd.DataFrame          # same shape, rank 1 = best
    periods: dict                # ticker -> period label
    period_ends: dict            # ticker -> period end date
    warnings: list


def _period_end(ticker: str, basis: str, period: str) -> str | None:
    for f in model.list_filings():
        if f["ticker"] == ticker and f["form_type"] == basis:
            if (basis == "10-K" and f"FY{f['fiscal_year']}" == period) or \
                    (basis == "10-Q" and (f.get("period_end") or "")[:7] == period):
                return f.get("period_end")
    return None


def latest_row(ticker: str, basis: str) -> dict | None:
    df = period_table(ticker, basis)
    if df.empty:
        return None
    df = df.dropna(subset=["revenue"])
    return None if df.empty else df.iloc[-1].to_dict()


def peer_table(tickers: list[str], basis: str = "10-K") -> PeerResult:
    rows, periods, ends, warns = {}, {}, {}, []
    for t in tickers:
        r = latest_row(t, basis)
        if not r:
            warns.append(f"{analytics._name(t)}: no {basis} numbers stored.")
            continue
        rows[t] = r
        periods[t] = r["period_label"]
        ends[t] = _period_end(t, basis, r["period"])
    cols = list(rows)
    data = {RATIOS[m][0]: {t: _get(rows[t], m) for t in cols} for m in PEER_METRICS}
    tab = pd.DataFrame(data).T if cols else pd.DataFrame()
    ranks = pd.DataFrame(index=tab.index, columns=tab.columns, dtype=float)
    if cols:
        for m in PEER_METRICS:
            label, _, hib = RATIOS[m]
            if hib is None:
                continue  # informational metric: no ranking, no "best"
            ranks.loc[label] = tab.loc[label].rank(ascending=not hib, method="min")
        tab["Peer median"] = tab[cols].median(axis=1, skipna=True)
    if len({e[:7] for e in ends.values() if e}) > 1:
        warns.append("Fiscal periods differ between companies (" + "; ".join(
            f"{analytics._name(t)} {periods[t]} ends {ends[t]}" for t in cols if ends.get(t)) +
            "). Ratios are each company's own latest period, not the same calendar date.")
    return PeerResult(tab, ranks, periods, ends, warns)


def peer_markdown(res: PeerResult, tickers: list[str]) -> str:
    if res.table.empty:
        return "_No peer numbers available._"
    cols = [t for t in tickers if t in res.table.columns]
    kinds = {RATIOS[m][0]: RATIOS[m][1] for m in PEER_METRICS}
    head = "| Metric | " + " | ".join(f"{analytics._name(t)} ({res.periods[t]})" for t in cols) + " | Peer median | Best |\n"
    head += "|---|" + "---|" * (len(cols) + 2) + "\n"
    lines = []
    for label in res.table.index:
        k = kinds[label]
        vals = [res.table.at[label, t] for t in cols]
        best = [t for t in cols if res.ranks.at[label, t] == 1.0]
        lines.append(f"| {label} | " + " | ".join(fmt(v if pd.notna(v) else None, k) for v in vals) +
                     f" | {fmt(res.table.at[label, 'Peer median'], k)} | "
                     f"{', '.join(analytics._name(t) for t in best) if best else 'n/a'} |")
    return head + "\n".join(lines)


def missing_note(df: pd.DataFrame, keys: list[str]) -> list[str]:
    """Metrics that could not be read for the latest period (shown in reports so gaps are never hidden)."""
    if df.empty:
        return list(keys)
    last = df.iloc[-1]
    return [RATIOS[k][0] for k in keys if _get(last.to_dict(), k) is None]