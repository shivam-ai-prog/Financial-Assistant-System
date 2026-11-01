"""
changes.py - "what changed since the previous filing?"  (new / removed / reworded risk factors and business language)

Same principle as the rest of the system: the AI reads and explains, code finds and verifies.
  1. CODE splits the same section of the latest and the previous filing into paragraphs and matches them by wording
     (three-word-run overlap). Unchanged paragraphs are ignored; what is new, removed or reworded is found by the program,
     not by the model.
  2. The AI (CHANGES_MODEL, falling back to DRIVERS_MODEL, then the main model) is shown only those passages and writes
     one sentence for an investor on each, plus an exact quote.
  3. CODE verifies every sentence before it is shown: the quote must occur in the passage it is about, every number must
     come from that passage, and most key words must come from it. Anything else is dropped.
Only stored filings are used. If the previous filing is not stored, it is fetched from SEC EDGAR once.
"""
from __future__ import annotations

import os
import re

import drivers as D
import model

SECTIONS = {"10-K": [("Item_1A_Risk_Factors", "risk factors"), ("Item_1_Business", "business description")],
            "10-Q": [("PartII_Item_1A_Risk_Factors", "risk factors")]}
SAME, TOUCHED = 0.80, 0.30           # similarity of a paragraph to its best match in the other filing
MIN_PARA, MAX_PARA, MIN_SECTION = 150, 2500, 1500
CANDIDATES = {"new": 6, "removed": 4, "modified": 6}
MIN_GROUNDED = 0.5
_META = {"new", "removed", "reworded", "wording", "previous", "latest", "filing", "longer", "added", "dropped", "text",
         "passage", "language", "mentions", "mention", "compared", "earlier", "describes", "adds", "introduces", "drops",
         "deleted", "expands", "narrows", "softens", "strengthens", "changed", "changes", "updated", "removes",
         "added", "now", "state", "states", "stated", "version", "prior", "current", "section", "risk", "risks"}


# ── 1. code: find what changed ─────────────────────────────────────────────────────────────────────────────────
def paragraphs(text: str) -> list[str]:
    text = (text or "").replace("\r", "")
    parts = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    if len([p for p in parts if len(p) >= MIN_PARA]) < 8:            # no blank-line structure: cut by sentences
        flat = re.sub(r"\s+", " ", text)
        parts, cur = [], ""
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"])", flat):
            cur = (cur + " " + s).strip()
            if len(cur) >= 450:
                parts.append(cur)
                cur = ""
        if cur:
            parts.append(cur)
    return [p[:MAX_PARA] for p in parts if len(p) >= MIN_PARA]


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def diff(old_text: str, new_text: str) -> dict:
    """{"stats": {...}, "new": [...], "removed": [...], "modified": [...]} - paragraphs, found without any AI."""
    old, new = paragraphs(old_text), paragraphs(new_text)
    og, ng = [D._trigrams(p) for p in old], [D._trigrams(p) for p in new]
    used: set[int] = set()
    same, new_items, mod_items = 0, [], []
    for i, p in enumerate(new):
        best, bj = -1, 0.0
        for j, o in enumerate(og):
            s = _jacc(ng[i], o)
            if s > bj:
                bj, best = s, j
        if bj >= SAME:
            same += 1
            used.add(best)
        elif bj >= TOUCHED:
            mod_items.append({"new": p, "old": old[best], "sim": bj})
            used.add(best)
        else:
            new_items.append({"new": p, "sim": bj})
    removed = [{"old": old[j]} for j in range(len(old))
               if j not in used and max((_jacc(og[j], g) for g in ng), default=0.0) < TOUCHED]
    return {"stats": {"unchanged": same, "reworded": len(mod_items), "new": len(new_items), "removed": len(removed),
                      "old_paragraphs": len(old), "new_paragraphs": len(new)},
            "new": new_items, "removed": removed, "modified": mod_items}


def candidates(d: dict) -> list[dict]:
    """The most substantial changes, with ids the model must refer to (N1, R1, M1...)."""
    out: list[dict] = []
    picks = {"new": sorted(d["new"], key=lambda x: -len(x["new"]))[:CANDIDATES["new"]],
             "removed": sorted(d["removed"], key=lambda x: -len(x["old"]))[:CANDIDATES["removed"]],
             "modified": sorted(d["modified"], key=lambda x: x["sim"])[:CANDIDATES["modified"]]}
    for kind, letter in (("new", "N"), ("removed", "R"), ("modified", "M")):
        for n, c in enumerate(picks[kind], 1):
            out.append({"id": f"{letter}{n}", "kind": kind, "new": c.get("new", ""), "old": c.get("old", "")})
    return out


# ── 2. AI: explain the changes ─────────────────────────────────────────────────────────────────────────────────
_PROMPT = """You compare two filings of the same company: the PREVIOUS {form} ({prev}) and the LATEST {form} ({cur}),
section: {label}. A program already found the passages below that are NEW in the latest filing, REMOVED since the previous
one, or REWORDED. For each passage you can describe, write ONE sentence for an investor saying what is new, what is no
longer there, or how the wording changed.
Rules:
- Use only the passage text. Do not add consequences, opinions or numbers that are not in it.
- Skip passages that are routine boilerplate or only trivially reworded. It is fine to describe fewer passages.
- "evidence": one exact sentence copied from the NEW text (N and M ids) or from the OLD text (R ids).
- For M ids also give "old_evidence": an exact sentence copied from the OLD text that shows what was different.
Reply with ONLY a JSON list, no other text: [{{"id": "N1", "summary": "...", "evidence": "..."}}]

PASSAGES:
{passages}
"""


def _passages(cands: list[dict]) -> str:
    rows = []
    for c in cands:
        if c["kind"] == "new":
            rows.append(f"[{c['id']}] NEW:\n{c['new'][:1200]}")
        elif c["kind"] == "removed":
            rows.append(f"[{c['id']}] REMOVED (was in the previous filing):\n{c['old'][:1200]}")
        else:
            rows.append(f"[{c['id']}] REWORDED\nOLD: {c['old'][:900]}\nNEW: {c['new'][:900]}")
    return "\n\n".join(rows)


def _ask(prompt: str) -> str:
    import llm_provider
    cfg = llm_provider.resolve((os.getenv("CHANGES_MODEL") or os.getenv("DRIVERS_MODEL") or "").strip())
    if cfg:
        try:
            return model.LLMClient(max_tokens=int(os.getenv("CHANGES_MAX_TOKENS", "3000")), **cfg).chat(
                prompt, use_calculator=False)
        except Exception as e:
            model.log.warning("changes model failed (%s) - using the main model", e)
    return model.get_llm().chat(prompt, use_calculator=False)

# ── 3. code: verify ──────────────────────────────────────────────────────────────────────────────────────────
# ── 3. code: verify ──
def _stem(w: str) -> str:
    """Crude stem (first 6 letters) so 'foundry'/'foundries' or 'cost'/'costs' count as the same word."""
    return w[:6]


def verify(items: list[dict], cands: list[dict], rejected: list | None = None) -> list[dict]:
    by_id, out, seen = {c["id"]: c for c in cands}, [], set()

    def no(why: str, text: str) -> None:
        if rejected is not None:
            rejected.append(f"{why}: {text[:70]}")

    for it in items:
        cid = str(it.get("id", "")).strip().upper()
        summ = str(it.get("summary") or "").strip()
        ev, oev = str(it.get("evidence") or "").strip(), str(it.get("old_evidence") or "").strip()
        c = by_id.get(cid)
        if not c or not summ or cid in seen:
            no("unknown, empty or repeated passage id", summ or cid)
            continue
        ev_text = c["old"] if c["kind"] == "removed" else c["new"]
        if len(ev) < D.MIN_QUOTE or not D._quote_in(ev, D._squash(ev_text), D._trigrams(ev_text)):
            no("evidence not found in the passage", summ)
            continue
        if c["kind"] == "modified" and oev and not D._quote_in(oev, D._squash(c["old"]), D._trigrams(c["old"])):
            no("old evidence not found in the previous text", summ)
            continue
        both = f"{c['new']} {c['old']}"
        bad = D._unsupported_numbers(summ, both)
        if bad:
            no(f"numbers {bad} not in the passage", summ)
            continue
        words = {_stem(w) for w in D._words(summ) - _META}
        if words and len(words & {_stem(w) for w in D._words(both)}) / len(words) < MIN_GROUNDED:
            no("wording not supported by the passage", summ)
            continue
        seen.add(cid)
        out.append({"id": cid, "kind": c["kind"], "summary": summ, "evidence": ev})
    return out


# ── the public function ────────────────────────────────────────────────────────────────────────────────────────
def _period(f: dict) -> str:
    return str(f.get("period_end") or f.get("filing_date") or "")


def _label(f: dict, form: str) -> str:
    return f"{form}, period ended {f.get('period_end')}"


def _pair(t: str, form: str):
    def stored():
        return sorted(model.list_filings(t, form), key=_period, reverse=True)
    fl = stored()
    if len(fl) < 2:                       # the previous filing is not stored yet: fetch it once
        try:
            model.ensure_filings(t, [form], limit=2, force_check=True)
        except Exception as e:
            model.log.warning("changes: could not fetch the previous %s for %s: %s", form, t, e)
        fl = stored()
    return (fl[0], fl[1]) if len(fl) >= 2 else None


def what_changed(t: str, form: str = "10-K") -> dict:
    t, form = t.upper(), form if form in SECTIONS else "10-K"
    res: dict = {"ticker": t, "form": form, "sections": [], "error": None, "sources": []}
    pair = _pair(t, form)
    if not pair:
        res["error"] = f"Only one stored {form} for {t}, so there is nothing to compare with."
        return res
    cur, prev = pair
    res["current"], res["previous"] = _label(cur, form), _label(prev, form)
    res["sources"] = [f"{model.company_name(t)} {res['current']}", f"{model.company_name(t)} {res['previous']}"]
    cs, ps = model.get_sections(cur["accession_no"]), model.get_sections(prev["accession_no"])
    for name, label in SECTIONS[form]:
        a, b = ps.get(name, ""), cs.get(name, "")
        sec = {"label": label, "stats": None, "items": [], "note": None}
        res["sections"].append(sec)
        if len(a) < MIN_SECTION or len(b) < MIN_SECTION:
            sec["note"] = f"The {label} section is missing or too short in one of the two stored filings."
            continue
        d = diff(a, b)
        sec["stats"] = d["stats"]
        cands = candidates(d)
        if not cands:
            continue
        try:
            raw = _ask(_PROMPT.format(form=form, prev=res["previous"], cur=res["current"], label=label,
                                      passages=_passages(cands)))
        except Exception as e:
            model.log.warning("changes (%s %s %s) skipped: %s", t, form, label, e)
            sec["note"] = f"The explanation step failed ({type(e).__name__})."
            continue
        parsed, rejected = D.parse_items(raw), []
        sec["items"] = verify(parsed, cands, rejected)
        model.log.info("changes %s %s %s: model proposed %d, %d passed the checks", t, form, label, len(parsed), len(sec["items"]))
        for r in rejected:
            model.log.info("changes %s %s %s rejected -> %s", t, form, label, r)
    return res


# ── presentation ───────────────────────────────────────────────────────────────────────────────────────────────
_KIND = {"new": "New", "removed": "No longer in the filing", "modified": "Reworded"}


def _bullet(i: dict) -> str:
    q = i["evidence"] if len(i["evidence"]) <= 240 else i["evidence"][:237] + "..."
    return f"**{_KIND[i['kind']]}:** {i['summary']} (\"{q}\")"


def _stats_line(label: str, s: dict) -> str:
    return (f"{label.capitalize()}: {s['new']} new, {s['removed']} removed and {s['reworded']} reworded passages; "
            f"{s['unchanged']} of {s['new_paragraphs']} passages unchanged.")


def to_markdown(res: dict) -> str:
    if res.get("error"):
        return res["error"]
    out = [f"Comparison of {res['ticker']}: latest {res['current']} vs previous {res['previous']}."]
    for sec in res["sections"]:
        out.append(f"\n{sec['label'].capitalize()}")
        if sec["note"]:
            out.append(sec["note"])
        if sec["stats"]:
            out.append(_stats_line(sec["label"], sec["stats"]))
        out += ["- " + _bullet(i) for i in sec["items"]]
        if sec["stats"] and not sec["items"] and not sec["note"]:
            out.append("No change substantial enough to describe was found.")
    return "\n".join(out)


def change_blocks(t: str, form: str = "10-K") -> list:
    """Report blocks (same block types as drivers.driver_blocks). Empty when nothing can be compared."""
    try:
        res = what_changed(t, form)
    except Exception as e:
        model.log.warning("changes section skipped for %s: %s", t, e)
        return []
    if res.get("error"):
        return [("note", res["error"])]
    blocks: list = [("h3", f"What changed since the previous {form}")]
    shown = False
    for sec in res["sections"]:
        if sec["stats"]:
            blocks.append(("p", f"**{sec['label'].capitalize()}.** " + _stats_line(sec["label"], sec["stats"])))
        if sec["items"]:
            blocks.append(("bullets", [_bullet(i) for i in sec["items"]]))
            shown = True
    blocks.append(("note", f"Latest {res['current']} compared with {res['previous']}. Changed passages were found by "
                           "program, described by the AI, and each description was checked against the filing text."
                           + ("" if shown else " No change substantial enough to describe was found.")))
    return blocks