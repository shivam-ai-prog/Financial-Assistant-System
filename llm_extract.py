"""
llm_extract.py - AI fallback for reading a financial statement when the pattern rules in analytics.py find nothing.
 
The AI only READS: it returns the rows of a statement exactly as printed. Code then checks every answer, and anything
that fails a check is thrown away (the report shows 'n/a' rather than a wrong number):
  1. the row label it quotes must occur in the filing text,
  2. every number it returns must occur in the text it was shown (so nothing is invented or calculated),
  3. enough columns, and the same consistency checks the pattern extractor uses (revenue - cost = gross profit,
     assets bound the balance-sheet lines, operating cash flow present, ...),
  4. the unit (millions / thousands / billions) is read by code from the page, never taken from the model.
It runs only when the pattern rules failed, so normal filings never touch the AI. Switch off with LLM_EXTRACT=false.
Optional EXTRACT_MODEL in .env gives it its own OpenRouter model (default: the main model).
"""
from __future__ import annotations
 
import json
import os
import re
 
import model
 
WINDOW, STEP = 9000, 3000
 
KINDS = {
    "income": {
        "title": "the income statement (statement of operations / income / earnings)",
        "metrics": {"revenue": "total net revenue / net sales", "cost_of_revenue": "cost of revenue / cost of sales",
                    "gross_profit": "gross profit / gross margin in dollars", "rnd": "research and development",
                    "sga": "selling, general and administrative (or marketing, general and administrative)",
                    "opex": "total operating expenses",
                    "operating_income": "operating income (loss)",
                    "net_income": "net income (loss) attributable to the parent company's stockholders if shown, "
                                  "otherwise net income (loss)"},
        "keywords": ["revenue", "net sales", "gross profit", "operating income", "operating loss", "net income",
                     "net loss", "cost of", "research and development"],
    },
    "balance": {
        "title": "the balance sheet",
        "metrics": {"cash": "cash and cash equivalents", "short_investments": "short-term investments / marketable securities",
                    "inventory": "inventories", "total_current_assets": "total current assets",
                    "total_assets": "total assets", "total_current_liabilities": "total current liabilities",
                    "total_liabilities": "total liabilities (not 'total liabilities and equity')",
                    "long_term_debt": "long-term debt", "total_equity": "total stockholders' / shareholders' equity"},
        "keywords": ["total assets", "total current assets", "total liabilities", "cash and cash equivalents",
                     "stockholders", "shareholders", "total current liabilities", "inventor"],
    },
    "cashflow": {
        "title": "the statement of cash flows",
        "metrics": {"operating_cash_flow": "net cash provided by (used in) operating activities",
                    "capex": "purchases of / additions to property, plant and equipment (give the amount as printed)",
                    "depreciation": "depreciation (and amortization)",
                    "buybacks": "repurchases of common stock"},
        "keywords": ["operating activities", "investing activities", "financing activities", "depreciation",
                     "property, plant", "repurchase"],
    },
}
 
 
def enabled() -> bool:
    return (os.getenv("LLM_EXTRACT", "true") or "true").lower() != "false"
 
 
def _numbers_in(text: str) -> set[float]:
    return {float(m.replace(",", "").rstrip(".")) for m in re.findall(r"\d[\d,]*\.?\d*", text) if m.strip(",.")}
 
 
def _squash(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())
 
 
def find_window(text: str, kind: str) -> tuple[str, int] | None:
    """The 9,000-character stretch of the filing that looks most like the statement (keyword hits + number rows)."""
    kws = KINDS[kind]["keywords"]
    best, best_i = 0, -1
    for i in range(0, max(1, len(text) - 1000), STEP):
        w = text[i:i + WINDOW]
        low = w.lower()
        hits = sum(1 for k in kws if k in low)
        rows = sum(1 for ln in w.split("\n") if len(re.findall(r"\d[\d,]*", ln)) >= 2)
        score = hits * 6 + min(rows, 40)
        if hits >= 3 and score > best:
            best, best_i = score, i
    return (text[best_i:best_i + WINDOW], best_i) if best_i >= 0 else None
 
 
def _prompt(kind: str, window: str, form: str) -> str:
    spec = KINDS[kind]
    items = "\n".join(f'  - "{k}": {v}' for k, v in spec["metrics"].items())
    return (
        f"Below is text from an SEC {form} filing. Find {spec['title']} and copy the rows listed here, exactly as printed.\n"
        f"{items}\n"
        'Return ONLY JSON: {"metrics": {"<key>": {"line": "<the row label exactly as printed>", '
        '"values": [<every number on that row, left to right, exactly as printed>]}}}.\n'
        "Rules: copy numbers exactly as printed (a number in brackets/parentheses is negative); include every column "
        "of the row in order; leave out any metric that is not on the page; never calculate, add, subtract or "
        "estimate; never use a total from another statement.\n\nTEXT:\n" + window + "\n\nJSON:")
 
 
def _client():
    name = (os.getenv("EXTRACT_MODEL") or "").strip()
    import llm_provider
    cfg = llm_provider.resolve(name)
    if cfg:
        return model.LLMClient(max_tokens=int(os.getenv("EXTRACT_MAX_TOKENS", "2500")), **cfg)
    return model.get_llm()
 
 
def _ask(prompt: str) -> str:
    if (os.getenv("EXTRACT_MODEL") or "").strip():
        try:
            return _client().chat(prompt, use_calculator=False)
        except Exception as e:
            model.log.warning("extract model failed (%s) - using the main model", e)
    return model.get_llm().chat(prompt, use_calculator=False)
 
 
def parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {}
    data = data.get("metrics", data) if isinstance(data, dict) else {}
    return data if isinstance(data, dict) else {}
 
 
def verify(kind: str, data: dict, window: str, form: str, context: str) -> tuple[dict[str, list[float]], float]:
    """Keep only rows whose label and numbers really are in the window; then run the pattern extractor's sanity checks."""
    import analytics
 
    allowed = KINDS[kind]["metrics"]
    seen_nums, squashed = _numbers_in(window), _squash(window)
    ncols = {"income": 4, "balance": 2, "cashflow": 3}[kind]
    min_cols = {"income": 3 if form == "10-K" else 2, "balance": 2, "cashflow": 3 if form == "10-K" else 2}[kind]
    found: dict[str, list[float]] = {}
    for key, row in data.items():
        if key not in allowed or not isinstance(row, dict):
            continue
        label = _squash(str(row.get("line", "")))
        vals = row.get("values")
        if len(label) < 4 or label not in squashed or not isinstance(vals, list):
            continue
        try:
            nums = [float(v) for v in vals]
        except (TypeError, ValueError):
            continue
        if len(nums) < min_cols or any(abs(v) not in seen_nums for v in nums):
            continue
        found[key] = nums[:ncols]
    factor = analytics._unit_factor(context)   # unit from the page, never from the model
    if kind == "income":
        return analytics._sane(found), factor
    if kind == "balance":
        return analytics._sane_bs(found), factor
    if not found.get("operating_cash_flow") or found["operating_cash_flow"][0] == 0:
        return {}, 1.0
    for k in ("capex", "buybacks", "depreciation"):
        if k in found:
            found[k] = [abs(v) for v in found[k]]
    return found, factor
 
 
def extract(kind: str, text: str, form: str) -> tuple[dict[str, list[float]], float]:
    """AI fallback for one statement. Returns ({metric: [columns]}, unit_factor), or ({}, 1.0) when nothing passes."""
    if not enabled():
        return {}, 1.0
    w = find_window(text, kind)
    if not w:
        return {}, 1.0
    window, start = w
    try:
        data = parse(_ask(_prompt(kind, window, form)))
    except Exception as e:
        model.log.warning("AI extraction (%s) skipped: %s", kind, e)
        return {}, 1.0
    found, factor = verify(kind, data, window, form, text[max(0, start - 300): start + 1500])
    model.log.info("AI extraction %s %s: model returned %d rows, %d passed the checks", form, kind, len(data), len(found))
    return found, factor
 
