from __future__ import annotations

import re
from datetime import date

import pandas as pd

import analytics
import intents
import metrics_engine as me
import model

# ── content blocks: one structure rendered to Markdown (chat) and to Word ────────────────────────────────────
# ("h1"|"h2"|"h3"|"p"|"note", text) | ("bullets", [str]) | ("table", header, rows) | ("img", path, caption)

ANNUAL_ROWS = ["revenue", "revenue_growth", "gross_profit", "gross_margin", "operating_income", "operating_margin",
               "net_income", "net_margin", "net_income_growth", "rnd_pct", "operating_cash_flow", "free_cash_flow",
               "fcf_margin", "roe", "roa", "current_ratio", "liabilities_to_equity", "net_cash"]
QUARTER_ROWS = [("revenue", "revenue"), ("Revenue growth vs same quarter last year", "revenue_yoy"),
                ("Revenue growth vs previous quarter", "revenue_qoq"), ("gross_margin", "gross_margin"),
                ("operating_margin", "operating_margin"), ("net_margin", "net_margin"), ("net_income", "net_income"),
                ("Net income growth vs same quarter last year", "net_income_yoy")]


def _label(key: str) -> str:
    return me.RATIOS[key][0] if key in me.RATIOS else key


def _kind(key: str) -> str:
    if key in me.RATIOS:
        return me.RATIOS[key][1]
    return "pct" if key.endswith(("_yoy", "_qoq")) else "money"


def _val(row, key):
    v = row.get(key)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _kpi_table(df: pd.DataFrame, rows: list, last_n: int) -> tuple[list, list, list[str]]:
    """Metrics down, periods across. Rows with no value in any period are dropped and reported as missing."""
    d = df.tail(last_n)
    header = ["Metric"] + list(d["period_label"])
    out, missing = [], []
    for item in rows:
        label, key = item if isinstance(item, tuple) else (None, item)
        vals = [_val(r, key) for r in d.to_dict("records")]
        if all(v is None for v in vals):
            missing.append(label if label and " " in label else _label(key))
            continue
        out.append([label if label and " " in label else _label(key)] + [me.fmt(v, _kind(key)) for v in vals])
    return header, out, missing


def _takeaways(name: str, annual: pd.DataFrame, quarterly: pd.DataFrame) -> list[str]:
    out = []
    if not annual.empty:
        r = annual.dropna(subset=["revenue"]).iloc[-1].to_dict() if not annual.dropna(subset=["revenue"]).empty else {}
        if r:
            g = _val(r, "revenue_growth")
            s = f"{name} {r['period_label']}: revenue {me.fmt(r['revenue'], 'money')}"
            s += f" ({g:+.1f}% vs prior year)" if g is not None else ""
            om, nm = _val(r, "operating_margin"), _val(r, "net_margin")
            if om is not None:
                s += f", operating margin {om:.1f}%"
            if nm is not None:
                s += f", net margin {nm:.1f}%"
            out.append(s + ".")
            f = _val(r, "free_cash_flow")
            if f is not None:
                out.append(f"{name} free cash flow was {me.fmt(f, 'money')} ({_val(r, 'fcf_margin'):.1f}% of revenue)."
                           if _val(r, "fcf_margin") is not None else f"{name} free cash flow was {me.fmt(f, 'money')}.")
    if not quarterly.empty and not quarterly.dropna(subset=["revenue"]).empty:
        q = quarterly.dropna(subset=["revenue"]).iloc[-1].to_dict()
        s = f"Latest quarter ({q['period_label']}): revenue {me.fmt(q['revenue'], 'money')}"
        y, sq = _val(q, "revenue_yoy"), _val(q, "revenue_qoq")
        if y is not None:
            s += f", {y:+.1f}% vs the same quarter last year"
        if sq is not None:
            s += f", {sq:+.1f}% vs the previous quarter"
        out.append(s + ".")
    return out


def _company_blocks(t: str, basis_ok: dict) -> tuple[list, dict]:
    name = analytics._name(t)
    annual, quarterly = me.period_table(t, "10-K"), me.period_table(t, "10-Q")
    blocks: list = [("h2", name)]
    info = {"annual": annual, "quarterly": quarterly, "missing": [], "score": None}
    tk = _takeaways(name, annual, quarterly)
    if tk:
        blocks += [("h3", "Key takeaways"), ("bullets", tk)]
    base = None
    if not annual.empty and not annual.dropna(subset=["revenue"]).empty:
        base = annual.dropna(subset=["revenue"]).iloc[-1].to_dict()
        basis_note = f"based on {base['period_label']} (10-K)"
    elif not quarterly.empty and not quarterly.dropna(subset=["revenue"]).empty:
        base = quarterly.dropna(subset=["revenue"]).iloc[-1].to_dict()
        basis_note = f"based on {base['period_label']} (10-Q; no annual data stored)"
    if base:
        sc = me.scorecard(base)
        info["score"] = sc
        blocks.append(("h3", "Performance scorecard"))
        blocks.append(("p", f"Overall: **{sc['overall'] or 'n/a'}**"
                            + (f" ({sc['score_pct']:.0f}% of the maximum score, {sc['areas_scored']} of "
                               f"{sc['areas_total']} areas could be scored)" if sc["score_pct"] is not None else "")
                            + f", {basis_note}."))
        blocks.append(("table", ["Area", "Metric", "Value", "Rating", "Rule"],
                       [[i["area"], i["metric"], me.fmt(i["value"], _kind_by_label(i["metric"])), i["rating"], i["rule"]]
                        for i in sc["items"]]))
    if not annual.empty:
        h, rows, miss = _kpi_table(annual, ANNUAL_ROWS, 5)
        blocks += [("h3", "Annual performance and year-over-year trend (10-K)"), ("table", h, rows),
                   ("note", "Return on equity / assets use year-end balances (not averages), so they differ from "
                            "ratios published with average balances. Net cash = cash + short-term investments - long-term debt.")]
        info["missing"] += miss
    if not quarterly.empty:
        h, rows, miss = _kpi_table(quarterly, QUARTER_ROWS, 6)
        blocks += [("h3", "Quarterly performance (10-Q)"), ("table", h, rows)]
        info["missing"] += [m for m in miss if m not in info["missing"]]
    stored_10k = [f for f in model.list_filings() if f["ticker"] == t and f["form_type"] == "10-K"]
    if annual.empty and stored_10k:
        blocks.append(("note", f"A 10-K is stored for {name} (FY{stored_10k[0]['fiscal_year']}) but its financial statements "
                               "could not be read, so annual figures are missing and the scorecard uses the latest quarter. "
                               "Fix: in admin.py delete this 10-K and fetch it again, then run diag_metrics.py if it persists."))
    if annual.empty and quarterly.empty:
        blocks.append(("note", f"No financial statements could be read for {name} yet (filings missing or not yet processed)."))
    if info["missing"]:
        blocks.append(("note", "Not available from the stored filings (never estimated): " + ", ".join(dict.fromkeys(info["missing"])) + "."))
    return blocks, info


def _kind_by_label(label: str) -> str:
    for k, (lab, kind, _) in me.RATIOS.items():
        if lab == label:
            return kind
    return "money"


def _peer_blocks(tickers: list[str]) -> list:
    blocks: list = []
    for basis, title in (("10-K", "Peer comparison: latest fiscal year (10-K)"),
                         ("10-Q", "Peer comparison: latest quarter (10-Q)")):
        res = me.peer_table(tickers, basis)
        if res.table.empty or len(res.periods) < 2:
            continue
        cols = [t for t in tickers if t in res.periods]
        kinds = {_label(m): me.RATIOS[m][1] for m in me.PEER_METRICS}
        header = ["Metric"] + [f"{analytics._name(t)} ({res.periods[t]})" for t in cols] + ["Peer median", "Best"]
        rows = []
        for lab in res.table.index:
            vals = [res.table.at[lab, t] for t in cols]
            if all(pd.isna(v) for v in vals):  # e.g. quarterly free cash flow (10-Q cash flow is year-to-date)
                continue
            best = [analytics._name(t) for t in cols if res.ranks.at[lab, t] == 1.0]
            rows.append([lab] + [me.fmt(v if pd.notna(v) else None, kinds[lab]) for v in vals]
                        + [me.fmt(res.table.at[lab, "Peer median"], kinds[lab]), ", ".join(best) or "-"])
        blocks += [("h2", title), ("table", header, rows)]
        blocks += [("note", w) for w in res.warnings]
        scores = []
        for t in cols:
            r = me.latest_row(t, basis)
            sc = me.scorecard(r) if r else None
            if sc and sc["score_pct"] is not None:
                scores.append((sc["score_pct"], analytics._name(t), sc["overall"]))
        if scores:
            scores.sort(reverse=True)
            blocks.append(("p", "Scorecard ranking: " + " > ".join(f"{n} ({o}, {p:.0f}%)" for p, n, o in scores) + "."))
    return blocks


# ── charts ────────────────────────────────────────────────────────────────────────────────────────────────────
def _plotly_charts(tickers: list[str]) -> list:
    figs = []
    try:
        analytics._active_colors.set(analytics.assign_colors(tickers))
        for form, basis_label in (("10-K", "Annual (10-K)"), ("10-Q", "Quarterly (10-Q)")):
            df = analytics.load_wide(form)
            df = df[df["ticker"].isin(tickers)] if not df.empty else df
            if df.empty:
                continue
            figs.append(analytics.chart_trend(df, [t for t in tickers if t in set(df["ticker"])], "Revenue", basis_label, bars=True)[0])
            figs.append(analytics.chart_margins(df, [t for t in tickers if t in set(df["ticker"])], basis_label)[0])
    except Exception as e:
        model.log.warning("report charts skipped: %s", e)
    return figs[:4]


def _fy_ticks(ann: dict, periods: list) -> list[str]:
    """x-axis label per aligned year, showing each company's own fiscal year: '2025' / 'NVDA FY26' / 'AMD FY25'."""
    out = []
    for p in periods:
        parts = [f"{t} {d[d['cal'] == p].iloc[0]['period'].replace('FY20', 'FY')}" for t, d in ann.items()
                 if not d[d["cal"] == p].empty]
        out.append(p + "\n" + "\n".join(parts))
    return out


def _png_charts(tickers: list[str], folder) -> list[tuple[str, str]]:
    """Static charts for the Word file (matplotlib; no browser needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = analytics.assign_colors(tickers)
    out = []
    ann = {t: me.period_table(t, "10-K") for t in tickers}
    ann = {t: d.dropna(subset=["revenue"]) for t, d in ann.items() if not d.empty}
    cal = analytics.aligned_year   # same calendar alignment as the chat chart

    for t, d in ann.items():
        d["cal"] = [cal(t, p) for p in d["period"]]
    if ann:
        periods = sorted({p for d in ann.values() for p in d["cal"]})[-5:]
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        w = 0.8 / len(ann)
        for i, (t, d) in enumerate(ann.items()):
            m = dict(zip(d["cal"], d["revenue"] / 1000))
            fy = dict(zip(d["cal"], d["period"]))                      # the company's OWN fiscal-year label
            xs = [j + i * w for j in range(len(periods))]
            ax.bar(xs, [m.get(p, 0) for p in periods], w, label=analytics._name(t), color=colors[t])
            for x, pr in zip(xs, periods):                             # label each bar: "FY2026 $216B"
                if pr in m:
                    ax.annotate(f"{fy[pr].replace('FY20', 'FY')}\n${m[pr]:,.{0 if m[pr] >= 10 else 1}f}B", (x, m[pr]), ha="center",
                                va="bottom", fontsize=6.5, xytext=(0, 2), textcoords="offset points")
        ax.set_xticks([j + w * (len(ann) - 1) / 2 for j in range(len(periods))], _fy_ticks(ann, periods))
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylabel("US$ billions")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)   # headroom so bar labels never touch the title
        ax.set_xlabel("Calendar year of fiscal year-end (Jan-Mar year-ends count to the prior year, e.g. NVIDIA FY2026 -> 2025)", fontsize=8)
        ax.set_title("Annual revenue (10-K); labels show each company's own fiscal year", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        p = folder / "chart_revenue.png"
        fig.tight_layout()
        fig.savefig(p, dpi=170)
        plt.close(fig)
        out.append((str(p), "Annual revenue. Source: stored 10-K filings; fiscal years aligned to the calendar year of year-end."))
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for t, d in ann.items():
            dd = d.dropna(subset=["operating_margin"]).tail(5)
            if not dd.empty:
                ax.plot([periods.index(c) for c in dd["cal"]], dd["operating_margin"], marker="o",
                        label=analytics._name(t), color=colors[t])
        ax.set_xticks(range(len(periods)), _fy_ticks(ann, periods))
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylabel("% of revenue")
        ax.set_xlabel("Fiscal year, aligned to calendar year of year-end", fontsize=8)
        ax.set_title("Operating margin (10-K)", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        p = folder / "chart_margin.png"
        fig.tight_layout()
        fig.savefig(p, dpi=170)
        plt.close(fig)
        out.append((str(p), "Operating margin. Source: stored 10-K filings; fiscal years aligned to the calendar year of year-end."))
    return out


# ── LLM prose (drawn from the filings, optional) ──────────────────────────────────────────────────────────────
_AI_ERRORS: list[str] = []   # language-model failures during the current report (shown to the reader, never silent)

RISK_TOPICS = ["supply chain, manufacturing and dependence on suppliers or foundries",
               "competition, market share and rapid technological change",
               "export controls, regulation, legal proceedings and antitrust",
               "cybersecurity, data breaches and information systems security",
               "financial, tax, liquidity, customer concentration and credit risk",
               "key employees, acquisitions, strategic investments and international operations"]


def _risk_context(t: str) -> tuple[str, list[str], bool]:
    """One retrieval PER risk topic from Item 1A, merged in document order, so no category is crowded out by
    the others (a single search cannot cover a ~100-page section). Returns (context, sources, relaxed)."""
    name = model.company_name(t)
    prof = intents.profile_from_names(["risk"])
    merged: dict = {}
    relaxed = False
    for topic in RISK_TOPICS:
        hits, info = model.retrieve_ex(f"{topic} risk factors {name}", t, ["10-K"], top_n=4, profile=prof)
        relaxed = relaxed or bool(info.get("relaxed"))
        for h, sc in hits:
            merged.setdefault(h["uid"], (h, sc))
    ordered = sorted(merged.values(), key=lambda x: x[0]["id"])
    if not ordered:
        return "", [], True
    context, sources = model.build_context([ordered], max_chars=36000, interleave=False)
    return context, sources, relaxed


def _filing_prose(t: str, intent_name: str, ask: str) -> tuple[str | None, list[str]]:
    name = model.company_name(t)
    try:
        prof = intents.profile_from_names([intent_name])
        if intent_name == "risk":
            context, sources, relaxed = _risk_context(t)
            if not context or relaxed:
                return None, []
            words, extra = 320, (" Cover each of these categories that the Context supports: supply chain, competition and "
                                 "technology, regulation and legal, cybersecurity, financial and tax, people and acquisitions.")
        else:
            hits, info = model.retrieve_ex(f"{ask} {name}", t, ["10-K"], top_n=prof.top_n, profile=prof)
            if not hits or info.get("relaxed"):  # the expected Item is missing/tiny: say nothing rather than guess
                return None, []
            context, sources = model.build_context([hits], max_chars=prof.context_chars, interleave=False)
            words, extra = 220, ""
        prompt = model.QUALITATIVE_PROMPT.format(history="", context=context,
                                                 query=f"{ask} {name}. Maximum {words} words. Use short headed paragraphs.{extra}")
        return model.get_llm().chat(prompt, use_calculator=False), sources
    except Exception as e:
        model.log.warning("report prose (%s/%s) skipped: %s", t, intent_name, e)
        _AI_ERRORS.append(f"{type(e).__name__}: {str(e)[:160]}")
        return None, []


_CAUSE_WORDS = r"(?:demand|pricing power|scale|segment|traction|momentum|mix|operating leverage|efficienc\w*|market share|ai infrastructure)"


def _strip_causes(text: str) -> str:
    """Drop trailing 'reflecting/driven by ...' clauses that explain WHY with causes the tables cannot support."""
    pat = re.compile(rf",?\s+(?:reflecting|demonstrating|indicating|showing|signaling|signalling|driven by|due to|underscoring)\b[^.;\n]*?\b{_CAUSE_WORDS}\b[^.;\n]*", re.I)
    return pat.sub("", text)


def _bulletize(text: str) -> str:
    """'NVIDIA: a. b. c. AMD: d. e.' -> 'NVIDIA:' + one bullet per sentence, 'AMD:' + bullets."""
    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z][\w&' ]{1,40}:\s)", "\n", text)
    out = []
    for ln in text.split("\n"):
        st = ln.strip()
        m = re.match(r"^([A-Z][\w&' ]{1,40}):\s+(.{40,})$", st)
        if (m and not st.startswith("-") and not re.match(rf"^({'|'.join(_HEADS)})$", m.group(1), re.I)
                and len(re.findall(r"[.!?](?:\s|$)", m.group(2))) >= 2):
            out.append(m.group(1) + ":")
            out += ["- " + x for x in re.split(r"(?<=[.!?])\s+", m.group(2).strip()) if x]
        else:
            out.append(ln)
    return "\n".join(out)


def _leaders_text(tickers: list[str]) -> str:
    """Deterministic 'who leads on what' (from the peer ranks), so the AI never has to judge directions itself."""
    lines = []
    for basis, label in (("10-K", "Annual"), ("10-Q", "Latest quarter")):
        res = me.peer_table(tickers, basis)
        if res.table.empty or len(res.periods) < 2:
            continue
        bal = {_label(k) for k in ("current_ratio", "liabilities_to_equity", "net_cash")}
        for group, keep in ((f"{label} performance", False), (f"{label} balance sheet", True)):
            per = []
            for t in res.periods:
                won = [lab for lab in res.ranks.index if res.ranks.at[lab, t] == 1.0 and ((lab in bal) == keep)]
                per.append(f"{analytics._name(t)} leads on: {', '.join(won) if won else 'none'}")
            lines.append(f"LEADERS ({group}; for 'Liabilities / equity' LOWER is better): " + "; ".join(per))
    return "\n".join(lines)


def _useful_note(t: str) -> bool:
    """A coverage note is only worth showing if it says something is MISSING (not 'repeated', 'consolidated'...)."""
    return bool(re.search(r"\b(not|no|missing|lack|without|beyond)\b", t, re.I)) and not re.search(
        r"repeat|fragment|consolidat|overlap|duplicate", t, re.I)


def _clean_note(t: str) -> str:
    t = re.sub(r"\bno\s+outside\s+knowledge\s+was\s+used\.?", "", t.strip(), flags=re.I)
    t = re.sub(r"\bthe\s+context\b", "the retrieved filing sections", t, flags=re.I)
    t = re.sub(r"\b(?:in|from)\s+the\s+(?:provided\s+)?context\b", "in the retrieved filing sections", t, flags=re.I)
    t = re.sub(r",?\s*(?:as|per)\s+(?:the\s+)?(?:instructions?|rules?|prompt)\b\.?", ".", t, flags=re.I)
    t = re.sub(r"\bsections\s+does\b", "sections do", t)
    t = re.sub(r"\.\.+", ".", t).strip()
    return t[:1].upper() + t[1:]


_HEADS = ("Overall view", "Bull case", "Bear case", "What would change the view", "What to monitor", "Valuation caveat")


def _prose_blocks(text: str, skip_title: str = "", keep_notes: bool = False) -> list:
    """LLM markdown-ish text -> clean blocks. Removes '(218 words)' / 'Word count: 218' leaks and stray markdown,
    keeps bullets, bolds sub-headings, and turns 'Coverage note:' into an italic note (or drops it)."""
    text = re.sub(r"\(\s*\d+\s+words?\s*\)", "", text.strip(), flags=re.I)
    text = re.sub(r"\bword\s+count\s*:\s*\d+\.?", "", text, flags=re.I)
    text = re.sub(r"\s*\((?:per\s+(?:the\s+)?(?:risk\s+notes?|tables?|scorecard)[^)]*)\)", "", text, flags=re.I)
    text = re.sub(r",?\s*\bper\s+the\s+(?:risk\s+notes?|tables?)\b", "", text, flags=re.I)
    text = re.sub(r"\s*\((?:FY\s?\d{4}(?:\s*(?:&|and|,)\s*FY\s?\d{4})*\s*(?:10-K|10-Q)?)\)", "", text)   # '(FY2026 10-K)'
    text = re.sub(r"\(\s*\)", "", text)                      # empty brackets left by removed citations
    text = re.sub(r"[ \t]+([.,;])", r"\1", text)
    text = text.replace("**", "").replace("###", "").replace("##", "")
    blocks: list = []
    seen_heads: set = set()
    for para in re.split(r"\n\s*\n", text):
        lines = [ln.rstrip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        if skip_title and lines[0].strip().lower().rstrip(":") == skip_title.lower():
            lines = lines[1:]
            if not lines:
                continue
        # known section labels ("Bull case: ...") -> bold label + remaining text
        m = re.match(rf"^({'|'.join(_HEADS)})\s*:?\s*(.*)$", lines[0].strip(), re.I)
        if m:
            if m.group(1).lower() not in seen_heads:      # "Bull case" once, company names become sub-labels
                blocks.append(("p", f"**{m.group(1)}**"))
                seen_heads.add(m.group(1).lower())
            lines = ([m.group(2)] if m.group(2) else []) + lines[1:]
        elif len(lines[0]) < 70 and not lines[0].rstrip().endswith((".", ";", ",")) and not lines[0].lstrip().startswith(("-", "*", "•")):
            blocks.append(("p", f"**{lines[0].strip().rstrip(':')}**"))
            lines = lines[1:]
        text_buf: list = []
        bul_buf: list = []

        def flush():
            body = " ".join(text_buf).strip()
            text_buf.clear()
            cov = re.search(r"coverage note\s*:\s*(.*)$", body, re.I | re.S)
            if cov:
                body = body[:cov.start()].strip()
                if keep_notes:
                    if body:
                        blocks.append(("p", body))
                    if _useful_note(cov.group(1)):
                        blocks.append(("note", "Coverage note: risk factors are described qualitatively; the filing "
                                               "gives no quantified exposure."))
                    body = ""
            if body:
                blocks.append(("p", body))
            if bul_buf:
                blocks.append(("bullets", list(bul_buf)))
                bul_buf.clear()

        for ln in lines:
            st = ln.strip()
            if st.startswith(("- ", "* ", "• ")):
                if text_buf:
                    flush()
                bul_buf.append(st[1:].strip())
            elif len(st) < 60 and st.endswith(":"):          # sub-label such as "NVIDIA:" -> bold line
                flush()
                blocks.append(("p", f"**{st.rstrip(':')}**"))
            else:
                if bul_buf:
                    flush()
                text_buf.append(st)
        flush()
    return blocks


def _commentary(tables_md: str) -> str | None:
    prompt = ("You are an equity research analyst. Below are TABLES of figures computed from SEC filings. Write the "
              "performance commentary as exactly 3 short paragraphs of 2-3 sentences each, 150 words maximum in total:\n"
              "Each paragraph covers ALL companies together (never one paragraph per company):\n"
              "1) Annual results: growth (say which company grew faster and by how much), profitability, cash generation, side by side.\n"
              "2) Latest quarter: growth (say which company grew faster) and margins, side by side.\n"
              "3) Balance sheet and verdict: liquidity, leverage, and which company leads on which metrics.\n"
              "State only what the numbers show: never explain WHY (no 'pricing power', 'cost efficiency', 'demand', "
              "'operating leverage'). Do not use the word 'across'.\n"
              "Paragraphs 1 and 2 mention ONLY the 'performance' LEADERS lines (never liquidity or leverage); paragraph 3 "
              "mentions ONLY the 'balance sheet' LEADERS lines. Never repeat a leadership statement in another paragraph. "
              "If the companies' fiscal year-ends differ, say so in one short clause in paragraph 1.\n"
              "Rules: use ONLY numbers that appear in the tables, copied exactly, attached to the right company and period. "
              "When you say a company leads or is ahead, use ONLY the metrics listed for it in the LEADERS lines. "
              "Do not calculate anything new. Mention each fact ONCE (no repetition). Name the specific metrics where one "
              "company leads and any metric where the other company is ahead; never use the words 'all', 'every' or "
              "'across' for comparisons. Write for a reader who has not seen the tables: never mention 'the table', "
              "'the Best column', 'peer table' or 'scorecard rules'.\n\n"
              f"{tables_md}\n\nCommentary:")
    try:
        return model.get_llm().chat(prompt, use_calculator=False)
    except Exception as e:
        model.log.warning("commentary skipped: %s", e)
        _AI_ERRORS.append(f"{type(e).__name__}: {str(e)[:160]}")
        return None


def _investment_view(tables_md: str, risks_md: str) -> list | None:
    """Investment considerations: a fundamentals-based view, bull/bear case, what to monitor. Fundamentals only:
    the database has no share prices, so valuation is explicitly out of scope and the text must say so."""
    prompt = (
        "You are a buy-side equity analyst. Using ONLY the tables and risk notes below (figures computed from SEC "
        "filings), write investment considerations. Structure with these bold headings, in this order:\n"
        "Overall view: for EACH company one line: Favourable, Neutral or Cautious on FUNDAMENTALS, consistent with its "
        "scorecard rating in the tables (Strong = Favourable, Average = Neutral, Weak = Cautious) unless you state in one "
        "sentence why you differ, with the 2-3 "
        "strongest reasons (copy numbers exactly from the tables). If several companies, say which looks stronger "
        "on fundamentals and which metrics decide it.\n"
        "Bull case: 3 bullets per company.\n"
        "Bear case: 3 bullets per company, using the risk notes.\n"
        "What would change the view: 3 concrete, measurable triggers (e.g. a margin falling below a level).\n"
        "What to monitor: next-quarter items.\n"
        "Valuation caveat: one sentence stating that share prices and valuation multiples are NOT in this data, so "
        "this view says nothing about whether the stock is cheap or expensive, and that this is not personalised "
        "financial advice.\n"
        "When you say which company is stronger or ahead, use ONLY the metrics listed for it in the LEADERS lines "
        "(for liabilities/equity LOWER is better, so a higher ratio is NOT a strength). "
        "Use '- ' bullets, one point per bullet, under each company label. "
        "Write each heading ONCE (never repeat 'Bull case' or 'Bear case' per company); under it, start each company's "
        "bullets with the company name followed by a colon. Never write '(per risk notes)', 'per the tables' or similar "
        "references: just state the point. "
        "Rules: be balanced and specific; never guarantee returns or give a price target; do not invent numbers; "
        "ONLY use metrics that appear in the tables (revenue, revenue growth, margins, free cash flow margin, ROE, ROA, "
        "current ratio, liabilities/equity, net cash, R&D %). Do NOT mention segments (Data Center, Client, Gaming...), "
        "inventory, gross margin by segment, or any figure not in the tables. Do NOT state WHY a number moved "
        "(pricing power, demand, mix, operating leverage) unless that cause is written in the risk notes; just report the "
        "change. Triggers under 'What would change the view' must be thresholds on table metrics and be called illustrative. "
        "'What to monitor' must also use only table metrics or the named risks. "
        "do not calculate new figures; do not say one company wins 'across all' metrics.\n\n"
        f"TABLES:\n{tables_md}\n\nRISK NOTES:\n{risks_md}\n\nInvestment considerations:")
    try:
        text = model.get_llm().chat(prompt, use_calculator=False)
        return _prose_blocks(_strip_causes(_bulletize(text)))
    except Exception as e:
        model.log.warning("investment view skipped: %s", e)
        _AI_ERRORS.append(f"{type(e).__name__}: {str(e)[:160]}")
        return None


# ── rendering ──────────────────────────────────────────────────────────────────────────────────────────────────
def to_markdown(blocks: list) -> str:
    out = []
    for b in blocks:
        k = b[0]
        if k in ("h1", "h2", "h3"):
            out.append(f"{'#' * int(k[1])} {b[1]}\n")
        elif k == "p":
            out.append(b[1] + "\n")
        elif k == "note":
            out.append(f"> {b[1]}\n")
        elif k == "bullets":
            out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif k == "table":
            h, rows = b[1], b[2]
            out.append("| " + " | ".join(h) + " |\n|" + "---|" * len(h) + "\n"
                       + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n")
    return "\n".join(out)


def to_docx(blocks: list, path, images: list[tuple[str, str]], title: str) -> str:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.8)

    def add_runs(par, text):
        for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
            if part:
                par.add_run(part).bold = bool(i % 2)

    img_after_first_h2 = list(images)
    for b in blocks:
        k = b[0]
        if k == "h1":
            doc.add_heading(b[1], 0)
        elif k in ("h2", "h3"):
            doc.add_heading(b[1], int(k[1]) - 1)
        elif k == "p":
            add_runs(doc.add_paragraph(), b[1])
        elif k == "note":
            p = doc.add_paragraph()
            r = p.add_run(b[1])
            r.italic = True
            r.font.size = Pt(9)
        elif k == "bullets":
            for x in b[1]:
                add_runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "table":
            h, rows = b[1], b[2]
            t = doc.add_table(rows=1, cols=len(h))
            t.style = "Light Grid Accent 1"
            for i, c in enumerate(h):
                t.rows[0].cells[i].text = str(c)
            for r in rows:
                cells = t.add_row().cells
                for i, c in enumerate(r):
                    cells[i].text = str(c)
                    if i:
                        cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            for row in t.rows:
                for c in row.cells:
                    for par in c.paragraphs:
                        for run in par.runs:
                            run.font.size = Pt(8.5)
            doc.add_paragraph()
        elif k == "img" and b[1]:
            doc.add_picture(b[1], width=Inches(6.2))
            cap = doc.add_paragraph(b[2])
            cap.runs[0].italic = True
            cap.runs[0].font.size = Pt(9)
    doc.core_properties.title = title
    doc.save(str(path))
    return str(path)


# ── public API ──────────────────────────────────────────────────────────────────────────────────────────────────
DISCLAIMER = ("This report is generated automatically from SEC filings stored in the local database. Figures are "
              "extracted by software and may contain errors: verify important numbers against the filings. It is an "
              "analysis aid, not personalised financial advice.")


def _prepare(tickers: list[str]) -> None:
    for t in tickers:
        try:
            model.ensure_filings(t, ["10-K", "10-Q"])
        except Exception as e:
            model.log.warning("ensure_filings(%s) failed: %s", t, e)
    analytics.ensure_metrics()


def performance_review(tickers: list[str]) -> dict:
    _prepare(tickers)
    blocks: list = []
    infos = {}
    for t in tickers:
        b, infos[t] = _company_blocks(t, {})
        blocks += b
    if len(tickers) > 1:
        blocks += _peer_blocks(tickers)
    blocks.append(("note", "All figures are computed from stored SEC filings (US$). Say \"generate a financial report for "
                           "<company>\" for the full document."))
    return {"markdown": to_markdown([("h1", "Performance review - " + " vs ".join(analytics._name(t) for t in tickers))] + blocks),
            "charts": _plotly_charts(tickers), "file": None, "sources": _sources(tickers), "infos": infos}


def _sources(tickers: list[str]) -> list[str]:
    out = []
    for f in model.list_filings():
        if f["ticker"] in tickers and f["form_type"] in ("10-K", "10-Q"):
            out.append(f"{model.company_name(f['ticker'])} {f['form_type']} - period ended {f.get('period_end')}, "
                       f"filed {f.get('filing_date')} ({f['accession_no']})")
    return out


def financial_report(tickers: list[str]) -> dict:
    _AI_ERRORS.clear()
    _prepare(tickers)
    names = " vs ".join(analytics._name(t) for t in tickers)
    title = f"Financial Report - {names}"
    body: list = [("p", f"Prepared {date.today():%d %B %Y} from SEC filings (10-K and 10-Q) stored in the database. "
                        "Amounts in US dollars.")]
    infos, all_src = {}, []
    tables_for_llm: list[str] = []
    for t in tickers:
        b, infos[t] = _company_blocks(t, {})
        tables_for_llm.append(to_markdown(b))
        overview, s1 = _filing_prose(t, "business", "Describe the business, main segments and what each does for")
        risks, s2 = _filing_prose(t, "risk", "Summarise the main risk factors, grouped by category, for")
        # the executive summary already carries the takeaways: drop the per-company duplicate
        for i, blk in enumerate(b):
            if blk == ("h3", "Key takeaways"):
                del b[i:i + 2]
                break
        if overview:
            b[1:1] = [("h3", "Business overview (from the latest 10-K)")] + _prose_blocks(overview, "Business Overview")
        elif _AI_ERRORS:
            b.insert(1, ("note", "Business overview and key risks could not be written because the AI model call failed "
                                 "(see the note at the top of this report)."))
        else:
            b.insert(1, ("note", "Business overview and risks were skipped: the 10-K Items 1 / 1A are missing or too short "
                                 "in storage (re-ingest the filing in admin.py) - nothing was guessed."))
        b = list(b)
        if risks:
            b += [("h3", "Key risks (from the latest 10-K, Item 1A)")] + _prose_blocks(risks, keep_notes=True)
        all_src += s1 + s2
        body += b
    if len(tickers) > 1:
        peer = _peer_blocks(tickers)
        body += peer
        tables_for_llm.append(to_markdown(peer))
    leaders = _leaders_text(tickers) if len(tickers) > 1 else ""
    if leaders:
        tables_for_llm.append(leaders)
    comm = _commentary("\n".join(tables_for_llm)[:14000])
    summary: list = []
    for t in tickers:
        summary += _takeaways(analytics._name(t), infos[t]["annual"], infos[t]["quarterly"])
    ai_note: list = []
    if _AI_ERRORS:
        ai_note = [("note", "WARNING: the AI-written parts of this report (commentary, business overview, key risks, "
                            "investment considerations) are missing or incomplete because the language model failed: "
                            f"{_AI_ERRORS[0]}. All numbers, scorecards and tables are unaffected. Check the model "
                            "provider's credit / rate limits and the terminal log, then generate the report again.")]
    head: list = [("h1", title), body[0], *ai_note, ("h2", "Executive summary"), ("bullets", summary or ["No figures available."])]
    if comm:
        head += [("h3", "Performance commentary (AI-written from the tables in this report)"), ("p", comm)]
    risk_txt = "\n".join(x[1] for b_ in body for x in [b_] if x[0] == "p" and len(x[1]) > 200)[:6000]
    inv = _investment_view("\n".join(tables_for_llm)[:12000], risk_txt)
    invest_blocks: list = ([("h2", "Investment considerations (fundamentals only, not personalised advice)")] + inv) if inv else []
    sources = _sources(tickers)
    tail: list = invest_blocks + [("h2", "Sources"), ("bullets", sources or ["(none)"]), ("note", DISCLAIMER)]
    # the coverage note is identical for every company: keep it once, after the last company's risks
    last = max((i for i, b_ in enumerate(body) if b_[0] == "note" and b_[1].startswith("Coverage note:")), default=None)
    body = [b_ for i, b_ in enumerate(body) if not (b_[0] == "note" and b_[1].startswith("Coverage note:") and i != last)]
    blocks = head + body[1:] + tail

    # Word file (static charts inserted after the executive summary)
    reports_dir = model.DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", "_".join(tickers)) + f"_{date.today():%Y%m%d}"
    try:
        imgs = _png_charts(tickers, reports_dir)
    except Exception as e:
        model.log.warning("static charts skipped: %s", e)
        imgs = []
    doc_blocks = head + [("img", p, cap) for p, cap in imgs] + body[1:] + tail
    try:
        path = to_docx(doc_blocks, reports_dir / f"{stem}.docx", imgs, title)
    except Exception as e:
        model.log.warning("docx failed: %s", e)
        path = None
    return {"markdown": to_markdown(blocks), "charts": _plotly_charts(tickers), "file": path,
            "sources": sources + all_src, "infos": infos}