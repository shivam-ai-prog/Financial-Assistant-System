"""
intents.py - what KIND of question is this?  The answer decides where we look in the filing and how we answer.

Why this exists
  A question about risks, business or strategy is answered from specific parts of a 10-K / 10-Q (Item 1A, Item 1, ...),
  needs BREADTH (the whole section, not the 12 chunks that best match a few words) and must not be pulled toward
  financial tables.  A question about revenue or margins is the opposite.  One retrieval recipe cannot do both, so the
  question is classified first and the recipe follows from the class.

Pure Python, no dependencies, deterministic (no LLM call).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Intent:
    name: str
    label: str                                    # used in the answer title, e.g. "Risk factors"
    qualitative: bool                             # True = explain / describe, False = numbers matter
    patterns: tuple = ()                          # ((regex, weight), ...) - each pattern counts once
    sections_10k: tuple = ()                      # section names (see model._10K_ANCHORS) to search first
    sections_10q: tuple = ()
    strict: bool = False                          # True: search ONLY those sections (when they exist)
    breadth: int = 0                              # >0: cover the section with this many position buckets
    top_n: int = 12                               # chunks per (company, filing) in the context
    context_chars: int = 28000
    expansion: str = ""                           # words real passages use that the question does not


# Order matters: on a tie the earlier intent wins, so qualitative intents come before "financials".
INTENTS: list[Intent] = [
    Intent(
        "risk", "Risk factors", True,
        patterns=((r"\brisk factors?\b", 3), (r"\brisks?\b", 2),
                  (r"\b(?:threats?|headwinds|vulnerabilit\w+|downside|what could go wrong|uncertaint\w+)\b", 2)),
        sections_10k=("Item_1A_Risk_Factors",), sections_10q=("PartII_Item_1A_Risk_Factors",),
        strict=True, breadth=14, top_n=18, context_chars=38000,
        expansion=("could adversely affect our business, financial condition and results of operations; may harm; "
                   "if we fail to; competition; regulation; legal and regulatory; privacy and data; security; "
                   "dependence on; macroeconomic; supply; intellectual property"),
    ),
    Intent(
        "market_risk", "Market risk", False,
        patterns=((r"\bmarket risk\b", 4), (r"\binterest[- ]rate (?:risk|exposure)\b", 4),
                  (r"\b(?:foreign (?:currency|exchange)|currency|fx) (?:risk|exposure|rates?)\b", 4)),
        sections_10k=("Item_7A_Market_Risk",), sections_10q=("Item_3_Market_Risk",),
        strict=True, top_n=10, context_chars=24000,
        expansion="foreign currency exchange rate risk interest rate risk hypothetical change sensitivity",
    ),
    Intent(
        "cyber", "Cybersecurity", True,
        patterns=((r"\bcyber\w*", 4), (r"\bdata breach\w*", 4), (r"\binformation security\b", 4),
                  (r"\bsecurity incidents?\b", 3)),
        sections_10k=("Item_1C_Cybersecurity", "Item_1A_Risk_Factors"), sections_10q=("PartII_Item_1A_Risk_Factors",),
        strict=True, breadth=6, top_n=12, context_chars=28000,
        expansion="cybersecurity risk management governance incidents threats third parties board oversight",
    ),
    Intent(
        "legal", "Legal and regulatory matters", True,
        patterns=((r"\b(?:legal|lawsuits?|litigation|proceedings|antitrust|investigations?|settlements?|"
                   r"regulat\w+|compliance|ftc|doj|gdpr|fines?)\b", 3),),
        sections_10k=("Item_3_Legal", "Item_1A_Risk_Factors", "Item_1_Business"),
        sections_10q=("PartII_Item_1_Legal", "PartII_Item_1A_Risk_Factors"),
        strict=True, breadth=8, top_n=14, context_chars=32000,
        expansion="legal proceedings regulatory investigations lawsuits fines compliance laws and regulations",
    ),
    Intent(
        "competition", "Competition", True,
        patterns=((r"\bcompetit\w+", 3), (r"\brivals?\b", 3), (r"\bmarket share\b", 3), (r"\bmoat\b", 3)),
        sections_10k=("Item_1_Business", "Item_1A_Risk_Factors"), sections_10q=("Item_2_MDA", "PartII_Item_1A_Risk_Factors"),
        strict=True, breadth=8, top_n=14, context_chars=32000,
        expansion="we compete with; competitors; competitive; market share; our ability to compete; alternatives",
    ),
    Intent(
        "strategy", "Strategy and outlook", True,
        patterns=((r"\bstrateg\w+", 3), (r"\b(?:outlook|roadmap|priorit\w+|vision|initiatives?|growth drivers?|"
                                          r"capital allocation|long[- ]term plans?|investments? in)\b", 2),
                  (r"\b(?:ai|artificial intelligence|machine learning|llms?|generative|autonomy|robotaxi)\b", 1)),
        sections_10k=("Item_1_Business", "Item_7_MDA"), sections_10q=("Item_2_MDA",),
        strict=True, breadth=10, top_n=16, context_chars=34000,
        expansion="our strategy; we are investing in; we expect; focus areas; priorities; long-term; opportunities",
    ),
    Intent(
        "business", "Business overview", True,
        patterns=((r"\b(?:describe|overview of|explain|summari[sz]e|tell me about)\b.{0,40}\b(?:business|company|"
                   r"operations)\b", 3),
                  (r"\bbusiness (?:model|overview|description|segments?|lines?|units?)\b", 3),
                  (r"\bwhat (?:does|do)\b.{0,40}\b(?:do|sell|offer|make|provide)\b", 3),
                  (r"\b(?:make money|monetiz\w+|generate revenue|revenue streams?|sources of revenue)\b", 3),
                  (r"\bwhat (?:each|they) (?:does|do|is|are|offers?)\b", 2),
                  (r"\b(?:products?|services|platforms?|customers?|segments?)\b", 1)),
        sections_10k=("Item_1_Business",), sections_10q=("Item_2_MDA",),
        strict=True, breadth=10, top_n=14, context_chars=32000,
        expansion="our products and services; we generate revenue; platform; customers; segments; we offer",
    ),
    Intent(
        "controls", "Internal controls", True,
        patterns=((r"\binternal controls?\b", 4), (r"\bcontrols and procedures\b", 4), (r"\bmaterial weakness\b", 4)),
        sections_10k=("Item_9A_Controls",), sections_10q=("Item_4_Controls",),
        strict=True, top_n=8, context_chars=16000,
    ),
    Intent(
        "liquidity", "Liquidity and capital", False,
        patterns=((r"\b(?:liquidity|cash flows?|capex|capital expenditures?|free cash flow|debt|borrowings?|"
                   r"repurchases?|buybacks?|dividends?|balance sheet)\b", 3),),
        sections_10k=("Item_7_MDA", "Item_8_Financial_Statements", "Item_15_Financial_Statements"),
        sections_10q=("Item_2_MDA", "Item_1_Financial_Statements"),
        top_n=14, context_chars=30000,
        expansion="liquidity and capital resources; cash provided by operating activities; capital expenditures",
    ),
    Intent(
        "performance", "Performance drivers", False,
        patterns=((r"\bwhy (?:did|has|have|was|were|is|are)\b", 3), (r"\bperform\w*", 3),
                  (r"\b(?:what drove|drivers? of|reasons? for|results of operations|md&a|"
                   r"management'?s discussion)\b", 3),
                  (r"\bexplain\w*\b.{0,40}\b(?:change|increase|decrease|decline|growth|results?|margin)\b", 3)),
        sections_10k=("Item_7_MDA",), sections_10q=("Item_2_MDA",),
        top_n=14, context_chars=30000,
        expansion="increase was primarily due to; decrease was primarily driven by; compared to; results of operations",
    ),
    Intent(
        "financials", "Financial analysis", False,
        patterns=((r"\b(?:revenues?|sales|net income|net profit|profit|earnings|eps|ebit|ebitda|margins?|gross profit|"
                   r"operating income|operating expenses?|opex|r&d|sg&a|cost of (?:revenue|sales)|expenses?)\b", 2),
                  (r"\b(?:grew|growth|increase[sd]?|decrease[sd]?|declined?|change[sd]?|compare[sd]?|comparison|"
                   r"how much|how many|total|calculate|ratio|percent(?:age)?|yoy|year[- ]over[- ]year)\b", 1),
                  (r"\bfinancial (?:overview|results|performance|summary|position|statements?)\b", 3)),
        top_n=12, context_chars=28000,
    ),
]

_BY_NAME = {i.name: i for i in INTENTS}


@dataclass
class Profile:
    """The merged retrieval / answer recipe for one question."""
    names: list[str] = field(default_factory=lambda: ["general"])
    label: str = "Financial analysis"
    qualitative: bool = False
    strict: bool = False
    sections_10k: tuple = ()
    sections_10q: tuple = ()
    breadth: int = 0
    top_n: int = 12
    context_chars: int = 28000
    expansion: str = ""
    scores: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "+".join(self.names)

    def sections_for(self, form_type: str) -> tuple:
        if form_type == "10-K":
            return self.sections_10k
        if form_type == "10-Q":
            return self.sections_10q
        return ()


def _score(query: str) -> dict[str, int]:
    q = query.lower()
    out: dict[str, int] = {}
    for it in INTENTS:
        s = sum(w for pat, w in it.patterns if re.search(pat, q))
        if s:
            out[it.name] = s
    return out


def _union(*lists) -> tuple:
    seen: list = []
    for lst in lists:
        for x in lst:
            if x not in seen:
                seen.append(x)
    return tuple(seen)


def classify(query: str) -> Profile:
    """Pick the main intent (and any close second) and merge their recipes."""
    scores = _score(query)
    if not scores:
        return Profile(scores={})
    best = max(scores.values())
    # INTENTS order breaks ties in favour of qualitative intents
    ranked = [it for it in INTENTS if it.name in scores]
    ranked.sort(key=lambda it: -scores[it.name])
    chosen = [it for it in ranked if scores[it.name] >= max(2, 0.6 * best)][:2]
    # "What are the RISKS to Meta's REVENUE?" is a risk question that mentions a metric, not a numbers question:
    # when the main intent is qualitative, a plain "financials" second intent is dropped.
    if len(chosen) == 2 and chosen[0].qualitative and chosen[1].name == "financials":
        chosen = chosen[:1]
    return _profile_from(chosen, scores)


def _profile_from(chosen: list, scores: dict) -> "Profile":
    main = chosen[0]
    # the question mixes "explain" with real figures (e.g. performance + financials): not qualitative
    qualitative = all(i.qualitative for i in chosen)
    return Profile(
        names=[i.name for i in chosen],
        label=main.label if len(chosen) == 1 else " and ".join(i.label.lower() for i in chosen).capitalize(),
        qualitative=qualitative,
        strict=all(i.strict for i in chosen) and any(i.sections_10k or i.sections_10q for i in chosen),
        sections_10k=_union(*(i.sections_10k for i in chosen)),
        sections_10q=_union(*(i.sections_10q for i in chosen)),
        breadth=max(i.breadth for i in chosen) if qualitative else 0,
        top_n=max(i.top_n for i in chosen),
        context_chars=max(i.context_chars for i in chosen),
        expansion=" ".join(i.expansion for i in chosen if i.expansion),
        scores=scores,
    )


# ── hybrid: rules first, LLM only when the rules are unsure ───────────────────────
import json as _json
from functools import lru_cache as _lru

_BY_NAME = {i.name: i for i in INTENTS}


def _confident(scores: dict) -> bool:
    if not scores:
        return False
    vals = sorted(scores.values(), reverse=True)
    return vals[0] >= 3 and (len(vals) == 1 or vals[0] >= 1.5 * vals[1])


@_lru(maxsize=512)
def _llm_names(query: str) -> tuple:
    """Ask the small router LLM to pick 1-2 intents from the FIXED list. Returns () on any problem."""
    import model  # lazy: avoids a circular import
    menu = "\n".join(f"- {i.name}: {i.label}" for i in INTENTS)
    prompt = ("Classify this question about a public company's SEC filings into 1 or 2 intents.\n"
              f"Allowed intents:\n{menu}\n"
              "Use 'financials' only when the user wants numbers/metrics/trends/ratios. Risk, business, strategy, "
              "legal, competition etc. are about explanations, not numbers.\n"
              f'Question: "{query}"\nReply ONLY with JSON: {{"intents": ["name", ...]}}')
    try:
        raw = model._router_chat(prompt) or ""
        m = re.search(r"\{.*\}", raw, re.S)
        names = [n for n in _json.loads(m.group())["intents"] if n in _BY_NAME][:2]
        return tuple(names)
    except Exception:
        return ()


def profile_from_names(names: list[str]) -> Profile:
    """Recipe for intents already chosen by the LLM router (no extra LLM call)."""
    chosen = [_BY_NAME[n] for n in names if n in _BY_NAME][:2]
    if not chosen:
        return Profile(scores={})
    if len(chosen) == 2 and chosen[0].qualitative and chosen[1].name == "financials":
        chosen = chosen[:1]
    return _profile_from(chosen, {n.name: 1 for n in chosen})


def classify_smart(query: str) -> Profile:
    """Rules decide when sure (free, instant); otherwise the LLM picks from the same fixed intent list.
    The retrieval recipe always comes from INTENTS, so the LLM can never invent sections. Any failure -> rules."""
    scores = _score(query)
    if _confident(scores):
        return classify(query)
    names = _llm_names(query)
    if not names:
        return classify(query)
    chosen = [_BY_NAME[n] for n in names]
    if len(chosen) == 2 and chosen[0].qualitative and chosen[1].name == "financials":
        chosen = chosen[:1]
    return _profile_from(chosen, scores or {n: 1 for n in names})


# ── small helpers shared by retrieval ─────────────────────────────────────────────
_BOILERPLATE = re.compile(
    r"(?:risks and uncertainties described below are not the only|forward-looking statements|"
    r"you should carefully consider|this (?:annual|quarterly) report on form|table of contents|"
    r"see (?:part|item)\s+\w+[,.]?\s+(?:item\s+\w+[,.]?\s+)?[\"“]?risk factors)", re.I)


def is_boilerplate(text: str) -> bool:
    """Cover-page / cross-reference style text that mentions 'risk factors' without containing a risk."""
    return len(text) < 350 or bool(_BOILERPLATE.search(text[:600]))


def numeric_density(text: str) -> float:
    """Share of characters that are digits - high for tables, low for prose."""
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


_COMPARE = re.compile(r"\b(?:compare[sd]?|changed?|evolv\w+|versus|vs\.?|over time|since|previous|prior|last year|"
                      r"year[- ]over[- ]year|trend)\b", re.I)


def wants_history(query: str) -> bool:
    """True when the question spans more than the latest filing (so do not restrict to the newest one)."""
    years = set(re.findall(r"\b(?:fy)?(20\d\d)\b", query.lower()))
    return bool(_COMPARE.search(query)) or len(years) >= 2