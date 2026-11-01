"""
reviewer.py - the Reviewer agent: checks the AI-written parts of a report against the tables, then fixes or removes
what does not hold up.

Two layers, in this order:
  1. Number audit (code, no AI, free): every figure written in the commentary or the investment view must appear in
     the report's own tables (rounding and $B / million conversions are fine; a recalculated figure is not).
  2. Semantic review (one AI call): looks for sentences that contradict the tables or the 'who leads' lines (wrong
     company, wrong direction, a cause the tables do not support, self-contradiction). Every sentence it flags must
     really exist in the text, so the reviewer cannot invent a problem.
Fixes: the commentary gets one rewrite by the writer model with the problems listed (re-audited afterwards); anything
that still fails, and any flagged sentence in the investment view, is removed. Nothing is ever replaced by a guess.
Settings: REVIEW=false switches the agent off; REVIEW_LLM=false keeps only the free number audit;
REVIEWER_MODEL gives it its own OpenRouter model.
"""
from __future__ import annotations

import json
import os
import re

import drivers
import model


def _on(name: str, default: str = "true") -> bool:
    return (os.getenv(name, default) or default).lower() != "false"


def _norm_units(t: str) -> str:
    t = re.sub(r"(?<=\d)\s?B\b", " billion", t)
    return re.sub(r"(?<=\d)\s?M\b", " million", t)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _squash(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def _unsupported(sentence: str, source_tokens: list[float]) -> list[str]:
    """Figures in the sentence that the source tables do not contain (years and small counts are ignored)."""
    cands = list(source_tokens)
    bad = []
    for v, dec, unit in drivers._tokens(_norm_units(sentence)):
        if unit is None and dec == 0 and (1990 <= v <= 2100 or v <= 12):
            continue                               # a year, or a small count like 'six areas'
        scale = 1000.0 if unit == "billion" else 1.0
        base, tol = v * scale, 0.5 * 10 ** (-dec) * scale
        if not any(abs(c - base) <= tol + 1e-9 for c in cands):
            bad.append(f"{v:g}")
    return bad


def _source_tokens(tables: str) -> list[float]:
    out: list[float] = []
    for v, _, unit in drivers._tokens(_norm_units(tables)):
        out.append(v)
        if unit == "billion":
            out.append(v * 1000)
    return out


def audit(text: str, tables: str) -> list[str]:
    """Sentences of `text` that contain a figure not found in `tables`."""
    src = _source_tokens(tables)
    return [s for s in _sentences(text) if _unsupported(s, src)]


# ── AI layer ──────────────────────────────────────────────────────────────────────────────────────────────────
def _ask(prompt: str) -> str:
    name = (os.getenv("REVIEWER_MODEL") or "").strip()
    import llm_provider
    cfg = llm_provider.resolve(name)
    if cfg:
        try:
            return model.LLMClient(max_tokens=int(os.getenv("REVIEWER_MAX_TOKENS", "2500")), **cfg
                                   ).chat(prompt, use_calculator=False)
        except Exception as e:
            model.log.warning("reviewer model %s failed (%s) - using the main model", name, e)
    return model.get_llm().chat(prompt, use_calculator=False)


def _flatten_investment(inv: list) -> str:
    out = []
    for b in inv or []:
        if b[0] == "p":
            out.append(b[1].replace("**", ""))
        elif b[0] == "bullets":
            out += [x for x in b[1]]
    return "\n".join(out)


def _parse(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return [d for d in data if isinstance(d, dict)]


def semantic_issues(commentary: str, inv_text: str, tables: str) -> list[dict]:
    """One AI call. Returns [{'sentence', 'problem'}] where the sentence is verified to exist in the texts."""
    prompt = (
        "You are a strict reviewer of an investment report. The TABLES and LEADERS lines are the ground truth. "
        "List ONLY sentences in the TEXT that (a) say a company leads / is ahead / is stronger on a metric that is not "
        "in that company's LEADERS line or contradicts the tables, (b) attach a number to the wrong company or period, "
        "(c) state a cause or reason that the tables do not support, or (d) contradict another sentence. "
        "Do not report style, tone or wording.\n"
        'Return ONLY JSON: [{"sentence": "<the exact sentence copied from the TEXT>", "problem": "<short reason>"}]; '
        "return [] if everything is consistent.\n\n"
        f"TABLES AND LEADERS:\n{tables[:12000]}\n\nTEXT:\n{commentary}\n{inv_text}\n\nJSON:")
    corpus = _squash(commentary + " " + inv_text)
    out = []
    for it in _parse(_ask(prompt)):
        s = str(it.get("sentence", "")).strip()
        if len(_squash(s)) >= 20 and _squash(s) in corpus:
            out.append({"sentence": s, "problem": str(it.get("problem", ""))[:160]})
    return out


def _rewrite(commentary: str, problems: list[str], tables: str) -> str | None:
    prompt = (
        "Rewrite the COMMENTARY below, fixing the listed PROBLEMS. Use only figures that appear in the TABLES, copied "
        "exactly; never calculate anything new; remove any claim you cannot support from the tables. Keep the same "
        "structure (three short paragraphs) and do not add new topics.\n\n"
        "PROBLEMS:\n- " + "\n- ".join(problems) + f"\n\nTABLES AND LEADERS:\n{tables[:12000]}\n\nCOMMENTARY:\n{commentary}\n\nRewritten commentary:")
    try:
        return _ask(prompt).strip() or None
    except Exception as e:
        model.log.warning("reviewer rewrite skipped: %s", e)
        return None


# ── removing what cannot be fixed ───────────────────────────────────────────────────────────────────────────────
def _drop(text: str, bad: list[str]) -> str:
    keys = [_squash(b) for b in bad]
    paras = []
    for para in re.split(r"\n\s*\n", text.strip()):
        kept = [s for s in _sentences(para) if _squash(s) not in keys]
        if kept:
            paras.append(" ".join(kept))
    return "\n\n".join(paras)


def _clean_investment(inv: list, bad: list[str]) -> list:
    keys = [_squash(b) for b in bad]

    def hit(s: str) -> bool:
        q = _squash(s)
        return any(k and (k in q or q in k) for k in keys)

    out = []
    for b in inv or []:
        if b[0] == "bullets":
            items = [x for x in b[1] if not hit(x)]
            if items:
                out.append(("bullets", items))
        elif b[0] == "p" and not b[1].startswith("**"):
            kept = [s for s in _sentences(b[1]) if not hit(s)]
            if kept:
                out.append(("p", " ".join(kept)))
        else:
            out.append(b)
    return out


def review(commentary: str | None, inv: list | None, tables: str) -> tuple[str | None, list | None, list[str]]:
    """Returns (commentary, investment blocks, log). The log lists what was changed (empty = nothing to fix)."""
    log: list[str] = []
    if not _on("REVIEW") or not (commentary or inv):
        return commentary, inv, log
    inv_text = _flatten_investment(inv or [])
    bad_comm = audit(commentary or "", tables)
    bad_inv = audit(inv_text, tables)
    flagged: list[dict] = []
    if _on("REVIEW_LLM"):
        try:
            flagged = semantic_issues(commentary or "", inv_text, tables)
        except Exception as e:
            model.log.warning("reviewer AI check skipped: %s", e)
    flagged_s = [f["sentence"] for f in flagged]
    model.log.info("reviewer: %d sentence(s) with unsupported figures, %d flagged by the AI check",
                   len(bad_comm) + len(bad_inv), len(flagged))

    # commentary: one rewrite, then remove whatever still fails
    comm_issues = list(bad_comm) + [s for s in flagged_s if commentary and _squash(s) in _squash(commentary)]
    if commentary and comm_issues:
        problems = ([f"unsupported figure(s) in: {s}" for s in bad_comm]
                    + [f"{f['problem']}: {f['sentence']}" for f in flagged if _squash(f["sentence"]) in _squash(commentary)])
        new = _rewrite(commentary, problems, tables)
        if new and not audit(new, tables):
            commentary = new
            log.append(f"commentary rewritten ({len(comm_issues)} statement(s) did not match the tables)")
        else:
            commentary = _drop(commentary, comm_issues) or None
            log.append(f"commentary: removed {len(comm_issues)} statement(s) that did not match the tables")
    # investment view: remove the failing statements (never replaced by a guess)
    inv_issues = list(bad_inv) + [s for s in flagged_s if _squash(s) in _squash(inv_text)]
    if inv and inv_issues:
        inv = _clean_investment(inv, inv_issues) or None
        log.append(f"investment view: removed {len(inv_issues)} statement(s) that did not match the tables")
    return commentary, inv, log