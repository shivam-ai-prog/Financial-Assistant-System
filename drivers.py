"""
drivers.py - "why did the numbers move?" from the MD&A (10-K Item 7 / 10-Q Item 2), written by the LLM and grounded.

The LLM writes the explanation in its own words, but must back every sentence with EVIDENCE: 1-2 exact sentences copied
from the retrieved MD&A. Code then checks, before anything is shown:
  1. every evidence quote really occurs in the retrieved text (whitespace / case / quote-mark insensitive),
  2. every number in the explanation also occurs in its evidence (no invented or recalculated figures),
  3. most of the explanation's content words occur in its evidence (it may paraphrase, but not add new claims).
Anything that fails is dropped. If nothing survives, the report says so instead of guessing.
"""
from __future__ import annotations

import json
import os
import re

import intents
import model

MAX_ITEMS = 5
MIN_QUOTE = 25        # each evidence sentence
MIN_TOTAL = 40        # all evidence together
MIN_GROUNDED = 0.5   # share of the explanation's content words that must appear in its evidence

_STOP = set("""that this with from were which their they have been has had also into than then over under more most
such these those while during about after before between within across because due driven higher lower increase
increased decrease decreased rose fell grew growth change changed year quarter fiscal period compared prior""".split())

_PROMPT = (
    "You are an equity analyst reading the Management's Discussion and Analysis of an SEC filing. Write up to {n} short "
    "explanations of WHY revenue, margins, operating income, net income or cash flow changed (which segment or product, "
    "price or volume, cost changes, one-off charges); include segment-level reasons when the text gives them. Explain ONLY the most recent period covered by this filing ({period}) "
    "compared with the period before it; ignore comparisons between older years. "
    "Write them in your own words, one plain sentence each, 30 words max.\n"
    "Return ONLY a JSON list, no other text. Each item: "
    '{{"metric": "revenue|gross_margin|operating_income|net_income|cash_flow|other", '
    '"explanation": "your sentence", '
    '"evidence": ["exact sentence copied word for word from the CONTEXT", "optional second exact sentence"]}}.\n'
    "Rules: every explanation must be fully supported by its evidence sentences; use only facts and numbers that appear "
    "in the evidence; never add numbers up or recalculate; never add a cause the CONTEXT does not state; skip anything "
    "that only repeats a figure without a reason; if the CONTEXT gives no reasons return []; never include an item for something the CONTEXT does not discuss. "
    "Copy each evidence sentence completely (a full sentence, not a fragment).\n\nCONTEXT:\n{context}\n\nJSON:"
)


def _norm(t: str) -> str:
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def _squash(t: str) -> str:
    """Letters and digits only, so spacing, table bars, bullets and punctuation never decide whether a quote matches."""
    return re.sub(r"[^a-z0-9]+", "", _norm(t))


def _trigrams(t: str) -> set[tuple]:
    w = re.findall(r"[a-z0-9]+", _norm(t))
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def _quote_in(quote: str, ctx_squashed: str, ctx_trigrams: set) -> bool:
    """True when the quote is in the context: exactly (ignoring spacing and punctuation, piece by piece when the model
    shortened it with '...'), or almost: 85% of its three-word runs occur in the filing text (a word changed or dropped)."""
    parts = [p for p in re.split(r"\.\.\.|\u2026", quote) if len(_squash(p)) >= 20]
    if parts and all(_squash(p) in ctx_squashed for p in parts):
        return True
    tg = _trigrams(quote)
    return len(tg) >= 4 and len(tg & ctx_trigrams) / len(tg) >= 0.85


_NUM_TOKEN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(billion|million|bn\b|%|percent|percentage)?", re.I)


def _tokens(t: str) -> list[tuple[float, int, str | None]]:
    out = []
    for m in _NUM_TOKEN.finditer(t):
        raw = m.group(1).rstrip(",")
        if not raw:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        unit = (m.group(2) or "").lower()
        unit = "billion" if unit in ("billion", "bn") else "million" if unit == "million" else "pct" if unit else None
        out.append((float(raw.replace(",", "")), dec, unit))
    return out


def _unsupported_numbers(text: str, evidence: str) -> list[str]:
    """Numbers in `text` that the evidence does not contain. Rounding and unit changes are fine ($34.6 billion for
    34,639 million; 10.8 for 10,848); a figure that is not in the evidence at all (a recalculated one) is not."""
    cands: list[float] = []
    for v, _, unit in _tokens(evidence):
        cands.append(v)
        if unit == "billion":
            cands.append(v * 1000)
    bad = []
    for v, dec, unit in _tokens(text):
        scale = 1000.0 if unit == "billion" else 1.0
        base, tol = v * scale, 0.5 * 10 ** (-dec) * scale
        if not any(abs(c - base) <= tol + 1e-9 for c in cands):
            bad.append(f"{v:g}")
    return bad


def parse_items(raw: str) -> list[dict]:
    """Pull the JSON list out of the model's reply (tolerates code fences and surrounding text)."""
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return [d for d in data if isinstance(d, dict)]


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", t.lower()) if w not in _STOP}


def verify(items: list[dict], context: str, rejected: list | None = None) -> list[dict]:
    """Keep only explanations backed by real evidence: quotes in the context, numbers and key words from the evidence.
    Reasons for every rejection are appended to `rejected` (for the log)."""
    ctx, ctx_tg = _squash(context), _trigrams(context)
    out, seen = [], set()

    def no(why: str, text: str) -> None:
        if rejected is not None:
            rejected.append(f"{why}: {text[:70]}")

    for it in items:
        text = str(it.get("explanation") or it.get("statement") or "").strip()
        ev = it.get("evidence") or it.get("quote") or []
        ev = [ev] if isinstance(ev, str) else [str(e).strip() for e in ev if str(e).strip()]
        if not text or not ev:
            no("no evidence given", text)
            continue
        ev = [e for e in ev if len(e) >= MIN_QUOTE]
        if not ev or sum(len(e) for e in ev) < MIN_TOTAL:
            no("evidence too short", text)
            continue
        if not all(_quote_in(e, ctx, ctx_tg) for e in ev):
            no("evidence not found in the filing text", text)
            continue
        joined = " ".join(ev)
        extra = _unsupported_numbers(text, joined)
        if extra:
            no(f"numbers {extra} not in the evidence", text)
            continue
        w = _words(text)
        if w and len(w & _words(joined)) / len(w) < MIN_GROUNDED:
            no("wording not supported by the evidence", text)
            continue
        key = _norm(text)
        if key in seen:
            continue
        seen.add(key)
        out.append({"metric": str(it.get("metric", "other")), "statement": text.rstrip(".") + ".", "evidence": ev})
        if len(out) >= MAX_ITEMS:
            break
    return out


def _context(t: str, form: str) -> tuple[str, bool]:
    """MD&A text for one company and form. Returns (context, found)."""
    name = model.company_name(t)
    prof = intents.profile_from_names(["performance"])
    hits, info = model.retrieve_ex(f"management discussion why revenue gross margin operating income changed {name}",
                                   t, [form], top_n=prof.top_n, profile=prof)
    if not hits or info.get("relaxed"):
        return "", False
    context, _ = model.build_context([hits], max_chars=prof.context_chars, interleave=False)
    return context, bool(context)


def _period(t: str, form: str) -> str:
    fl = [f for f in model.list_filings() if f["ticker"] == t and f["form_type"] == form]
    if not fl:
        return "the latest period"
    f = max(fl, key=lambda x: str(x.get("period_end") or ""))
    if form == "10-K":
        return f"fiscal year {f.get('fiscal_year')}, ended {f.get('period_end')}, versus the prior year"
    return f"the quarter ended {f.get('period_end')}, versus the same quarter a year earlier"


def _llm(use_dedicated: bool = True):
    """The model that writes the explanations. Set DRIVERS_MODEL in .env to give this job its own model (any
    OpenRouter model id); leave it empty to use the main answer model. DRIVERS_MAX_TOKENS sets its output budget."""
    name = (os.getenv("DRIVERS_MODEL") or "").strip()
    if not name or not use_dedicated:
        return model.get_llm()
    import llm_provider
    cfg = llm_provider.resolve(name)
    if not cfg:
        return model.get_llm()
    return model.LLMClient(max_tokens=int(os.getenv("DRIVERS_MAX_TOKENS", "3000")), **cfg)


def _ask(prompt: str) -> str:
    """Ask the dedicated drivers model; if it fails (wrong id, outage, rate limit) fall back to the main model once."""
    if (os.getenv("DRIVERS_MODEL") or "").strip():
        try:
            return _llm().chat(prompt, use_calculator=False)
        except Exception as e:
            model.log.warning("drivers model %s failed (%s) - using the main model", os.getenv("DRIVERS_MODEL"), e)
    return _llm(use_dedicated=False).chat(prompt, use_calculator=False)


def drivers_for(t: str, form: str) -> tuple[list[dict], str | None]:
    """Verified reasons for one filing type. Returns (items, error). error is set only when the model call failed."""
    context, found = _context(t, form)
    if not found:
        return [], None
    try:
        raw = _ask(_PROMPT.format(n=MAX_ITEMS, period=_period(t, form), context=context))
    except Exception as e:  # shown to the reader by report.py, never silent
        model.log.warning("drivers (%s/%s) skipped: %s", t, form, e)
        return [], f"{type(e).__name__}: {str(e)[:160]}"
    cand = parse_items(raw)
    rejected: list = []
    kept = verify(cand, context, rejected)
    model.log.info("drivers %s %s: model proposed %d, %d passed the checks", t, form, len(cand), len(kept))
    for r in rejected:
        model.log.info("drivers %s %s rejected -> %s", t, form, r)
    return kept, None


def _label(t: str, form: str) -> str:
    fl = [f for f in model.list_filings() if f["ticker"] == t and f["form_type"] == form]
    if not fl:
        return form
    f = max(fl, key=lambda x: str(x.get("period_end") or ""))
    return f"{form}, period ended {f.get('period_end')}"


def driver_blocks(t: str, errors: list[str]) -> list:
    """Report blocks for one company: reasons from the latest 10-K and the latest 10-Q."""
    blocks: list = []
    for form, title in (("10-K", "annual"), ("10-Q", "latest quarter")):
        items, err = drivers_for(t, form)
        if err:
            errors.append(err)
        if items:
            blocks.append(("p", f"**{title.capitalize()}**"))
            blocks.append(("bullets", [f"{i['statement']} ({_label(t, form)}, MD&A)" for i in items]))
    if blocks:
        blocks = [("h3", "What drove the change (management's explanation, from the MD&A)")] + blocks + [
            ("note", "Explanations are written by the AI from management's discussion in the filing. Each one was checked "
                     "against the filing text before it is shown: its supporting sentences must exist in the filing, and "
                     "its numbers and key terms must come from them.")]
    elif not errors:
        blocks = [("note", "No explanation of the changes could be taken from the stored MD&A sections, so none is given.")]
    return blocks