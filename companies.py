"""
companies.py - lets the assistant work with ANY company listed on SEC EDGAR, not just NVIDIA and Tesla.

* Downloads SEC's official company list (name <-> ticker <-> CIK) once, caches it in data/company_tickers.json
  and refreshes it weekly. Offline? The cached copy (or the built-in NVIDIA/Tesla list) is used.
* resolve(text)  -> tickers mentioned in a question ("Apple", "Microsoft Corp", "$AMD", "JPM" ...)
* name(ticker)   -> display name used in answers and charts
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from functools import lru_cache

import model

SEC_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_DAYS = 7
MAX_COMPANIES_PER_QUESTION = 4

_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "llc", "lp",
             "holdings", "holding", "group", "the", "sa", "nv", "ag", "de", "cl", "class", "adr", "new", "trust"}
# words that are also (small) companies' names but far more often ordinary words in a finance question
_GENERIC = {"net", "general", "american", "first", "united", "global", "national", "international", "capital",
            "financial", "energy", "technology", "technologies", "systems", "services", "partners", "bank", "health",
            "growth", "income", "value", "revenue", "profit", "cash", "margin", "price", "market", "stock", "sales",
            "cost", "debt", "equity", "dividend", "earnings", "operating", "annual", "quarter", "quarterly", "chart",
            "graph", "trend", "compare", "company", "business", "industry", "total", "report", "growth", "access",
            "target", "common", "select", "premier", "summit", "pioneer", "century", "alpha", "beta", "delta",
            "omega", "sigma", "apex", "core", "focus", "frontier", "legacy", "liberty", "universal", "vision"}
_ALLCAPS_STOP = {"CEO", "CFO", "COO", "CTO", "EPS", "GAAP", "SEC", "USD", "YOY", "QOQ", "ROE", "ROI", "ROA", "TTM",
                 "FCF", "EBIT", "ETF", "IPO", "GDP", "AI", "IT", "US", "USA", "FY", "ALL", "ON", "FOR", "ARE", "A",
                 "I", "AT", "BE", "BY", "DO", "GO", "HAS", "NOW", "ONE", "OUT", "SO", "TO", "UP", "WELL", "NEW",
                 "KEY", "LOW", "BIG", "CAN", "AN", "AND", "THE", "OR", "OF", "IN", "IS", "IF", "IT", "MY", "NO",
                 "R&D", "SGA", "COGS", "PE", "EV", "Q1", "Q2", "Q3", "Q4", "QTR", "VS", "TOP", "BEST", "FAST"}
# popular names people use that are not the legal name
_ALIASES = {"google": "GOOGL", "youtube": "GOOGL", "alphabet": "GOOGL", "facebook": "META", "instagram": "META",
            "meta": "META", "whatsapp": "META", "amazon": "AMZN", "aws": "AMZN", "apple": "AAPL", "microsoft": "MSFT",
            "berkshire": "BRK-B", "jpmorgan": "JPM", "jp morgan": "JPM", "chase": "JPM", "coca cola": "KO",
            "coke": "KO", "pepsi": "PEP", "disney": "DIS", "walmart": "WMT", "mcdonalds": "MCD", "amd": "AMD",
            "intel": "INTC", "netflix": "NFLX", "paypal": "PYPL", "uber": "UBER", "airbnb": "ABNB"}


# ── the companies this assistant covers (tech peers of NVIDIA and Tesla) ──────────────────
# Edit this list freely: add a ticker + name to a group and it is covered. COVERAGE=all in .env removes the limit.
UNIVERSE: dict[str, dict[str, str]] = {
    "AI chips & semiconductors (NVIDIA peers)": {
        "NVDA": "NVIDIA", "AMD": "Advanced Micro Devices", "INTC": "Intel", "AVGO": "Broadcom",
        "QCOM": "Qualcomm", "MU": "Micron Technology", "MRVL": "Marvell Technology", "TXN": "Texas Instruments"},
    "EV & autos (Tesla peers)": {
        "TSLA": "Tesla", "GM": "General Motors", "F": "Ford Motor", "RIVN": "Rivian Automotive",
        "LCID": "Lucid Group"},
    "Big tech, cloud & AI software": {
        "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms", "AAPL": "Apple",
        "ORCL": "Oracle", "IBM": "IBM", "CRM": "Salesforce", "ADBE": "Adobe", "NFLX": "Netflix",
        "CSCO": "Cisco Systems", "PLTR": "Palantir Technologies"},
}
COVERAGE = (model._env("COVERAGE", "tech") or "tech").strip().lower()  # "tech" (default) or "all"
_UNIVERSE_NAMES = {t: n for g in UNIVERSE.values() for t, n in g.items()}


def _tidy(s: str) -> str:
    s = re.sub(r"[^a-z0-9& ]+", " ", s.lower().replace("&", " and "))
    return re.sub(r"\s+", " ", s).strip()


_UNIVERSE_KEYS = {k for n in _UNIVERSE_NAMES.values() for k in (_tidy(n), _tidy(n).split()[0])}


def _norm_name(title: str) -> str:
    words = [w for w in _tidy(title).split() if w not in _SUFFIXES]
    return " ".join(words)


def _pretty(title: str) -> str:
    if title.isupper() or title.islower():
        title = " ".join(w.capitalize() if len(w) > 3 or w.lower() in {"the", "inc", "co"} else w.upper()
                         for w in title.split())
    return title.strip()


# ── the directory ────────────────────────────────────────────────────
def _cache_path():
    return model.DATA_DIR / "company_tickers.json"


def _download() -> dict | None:
    identity = model._env("EDGAR_IDENTITY") or "financial-assistant research contact@example.com"
    try:
        req = urllib.request.Request(SEC_URL, headers={"User-Agent": identity, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # offline / blocked -> fall back to the cache
        model.log.warning("Could not download SEC company list (%s) - using cached copy", e)
        return None


def _load_raw() -> dict:
    path = _cache_path()
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_DAYS * 86400
    if not fresh:
        data = _download()
        if data:
            model.DATA_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@lru_cache(maxsize=1)
def _index() -> dict:
    """{tickers: {T: (cik, title)}, names: {normalised name: T}, first: {first word: T}}"""
    tickers: dict[str, tuple[int, str]] = {}
    names: dict[str, str] = {}
    seen_cik: set[int] = set()
    first_count: dict[str, int] = {}
    first_map: dict[str, str] = {}
    two_count: dict[str, int] = {}
    two_map: dict[str, str] = {}
    for row in _load_raw().values():  # SEC lists the biggest companies first
        t, title, cik = str(row.get("ticker", "")).upper(), str(row.get("title", "")), int(row.get("cik_str", 0))
        if not t or not title:
            continue
        tickers[t] = (cik, title)
        n = _norm_name(title)
        if cik in seen_cik or not n:  # extra share classes of the same company
            continue
        seen_cik.add(cik)
        names.setdefault(n, t)
        fw = n.split()[0]
        first_count[fw] = first_count.get(fw, 0) + 1
        first_map.setdefault(fw, t)
        if len(n.split()) >= 3:  # "apple hospitality" for "Apple Hospitality REIT"
            tw = " ".join(n.split()[:2])
            two_count[tw] = two_count.get(tw, 0) + 1
            two_map.setdefault(tw, t)
    for t, disp in _UNIVERSE_NAMES.items():  # built-in names so the covered companies resolve even offline
        tickers.setdefault(t, (0, disp))
        names.setdefault(_norm_name(disp), t)
    first = {w: first_map[w] for w, c in first_count.items() if c == 1 and len(w) >= 5 and w not in _GENERIC}
    fw_seen: dict[str, int] = {}
    for disp in _UNIVERSE_NAMES.values():
        fw_seen[_norm_name(disp).split()[0]] = fw_seen.get(_norm_name(disp).split()[0], 0) + 1
    for t, disp in _UNIVERSE_NAMES.items():  # "Ford" for "Ford Motor", "Intel", "Oracle" ...
        w = _norm_name(disp).split()[0]
        if fw_seen[w] == 1 and w not in _GENERIC:
            first[w] = t
    for tw, c in two_count.items():
        if c == 1 and tw not in names:
            names[tw] = two_map[tw]
    return {"tickers": tickers, "names": names, "first": first}


def reload() -> None:
    _index.cache_clear()


def directory_size() -> int:
    return len(_index()["tickers"])


# ── public helpers ───────────────────────────────────────────────────
def name(ticker: str) -> str:
    """Display name: built-in list, then SEC list, then what is stored with the filings, then the ticker."""
    t = (ticker or "").upper()
    if t in model.COMPANIES:
        return model.COMPANIES[t]["name"]
    if t in _UNIVERSE_NAMES:
        return _UNIVERSE_NAMES[t]
    hit = _index()["tickers"].get(t)
    if hit:
        return _pretty(hit[1])
    try:
        with model.db() as c:
            r = c.execute("SELECT company FROM filings WHERE ticker=? LIMIT 1", (t,)).fetchone()
        if r and r["company"]:
            return r["company"]
    except Exception:
        pass
    return t


def short_name(ticker: str) -> str:
    """'Apple Inc.' -> 'Apple' (for chart legends)."""
    n = name(ticker)
    words = [w for w in n.replace(",", " ").split() if w.lower().strip(".") not in _SUFFIXES]
    return " ".join(words) or n


def exists(ticker: str) -> bool:
    t = (ticker or "").upper()
    return t in model.COMPANIES or t in _index()["tickers"]


def stored() -> list[tuple[str, str]]:
    """[(ticker, name)] of every company that already has filings in the database."""
    with model.db() as c:
        rows = c.execute("SELECT DISTINCT ticker, company FROM filings ORDER BY ticker").fetchall()
    return [(r["ticker"], r["company"] or r["ticker"]) for r in rows]


def choices() -> list[str]:
    """Names for dropdowns: stored companies first, then the built-in ones."""
    seen, out = set(), []
    for t, n in stored():
        if t not in seen:
            seen.add(t)
            out.append(short_name(t))
    for t in list(model.COMPANIES) + list(_UNIVERSE_NAMES):
        if t not in seen:
            seen.add(t)
            out.append(short_name(t))
    return out


def resolve(text: str, limit: int = MAX_COMPANIES_PER_QUESTION) -> list[str]:
    """Tickers of the companies mentioned in `text` (a question or a dropdown entry), in order of appearance."""
    if not text or not text.strip():
        return []
    idx, found = _index(), []  # (position, ticker)

    def add(pos: int, t: str) -> None:
        if t and t not in [x[1] for x in found]:
            found.append((pos, t))

    # 1. tickers written in capitals / with a $: "AAPL", "$AMD", "BRK-B"
    for m in re.finditer(r"(?<![A-Za-z0-9])(\$?)([A-Z]{1,5}(?:[.-][A-Z])?)(?![A-Za-z0-9])", text):
        dollar, sym = m.group(1), m.group(2).replace(".", "-")
        if not (sym in idx["tickers"] or sym in model.COMPANIES):
            continue
        if sym in _ALLCAPS_STOP and not dollar:
            continue
        if len(sym) <= 2 and not dollar:
            continue
        add(m.start(), sym)

    # 2. names: longest word sequence first (so "Advanced Micro Devices" beats "Advanced")
    low = _tidy(text)
    words = low.split()
    taken = [False] * len(words)
    for n in range(5, 0, -1):
        for i in range(len(words) - n + 1):
            if any(taken[i:i + n]):
                continue
            gram = " ".join(words[i:i + n])
            t = _ALIASES.get(gram) or idx["names"].get(gram) or (idx["first"].get(gram) if n == 1 else None)
            if not t:
                continue
            if n == 1 and (gram in _GENERIC or len(gram) < 4) and gram not in _ALIASES \
                    and gram not in _UNIVERSE_KEYS:
                continue
            for k in range(i, i + n):
                taken[k] = True
            add(low.find(gram), t)

    # 3. a dropdown entry that IS a ticker typed in lower case ("aapl")
    if not found and re.fullmatch(r"[A-Za-z]{1,5}([.-][A-Za-z])?", text.strip()):
        sym = text.strip().upper().replace(".", "-")
        if sym in idx["tickers"] or sym in model.COMPANIES:
            add(0, sym)

    found.sort()
    return [t for _, t in found][:limit]


# ── coverage (which companies the assistant answers about) ───────────
def covered(ticker: str) -> bool:
    """True if questions about this company are allowed (always true when COVERAGE=all)."""
    t = (ticker or "").upper()
    if COVERAGE == "all":
        return True
    if t in _UNIVERSE_NAMES or t in model.COMPANIES:
        return True
    return t in {x for x, _ in stored()}  # anything an admin already loaded stays available


def groups() -> dict[str, list[str]]:
    return {g: list(m) for g, m in UNIVERSE.items()}


def peers_of(ticker: str, n: int = 3) -> list[str]:
    """Same-group companies, ones already stored in the database first."""
    t = (ticker or "").upper()
    for members in UNIVERSE.values():
        if t in members:
            have = {x for x, _ in stored()}
            others = [m for m in members if m != t]
            others.sort(key=lambda m: (m not in have,))
            return others[:n]
    return []


def coverage_message(query: str = "") -> str:
    """What to tell the user when they ask about a company outside the coverage list."""
    outside = [name(t) for t in resolve(query) if not covered(t)]
    lines = ["I currently cover tech companies similar to NVIDIA and Tesla:"]
    lines += [f"- **{g}:** " + ", ".join(m.values()) for g, m in UNIVERSE.items()]
    lead = (f"**{', '.join(outside)}** is not in my coverage list. " if outside else "")
    return lead + "\n\n".join(["", "\n".join(lines),
                                  "An admin can add a company in `companies.py` (UNIVERSE) or set `COVERAGE=all` in `.env`."]).lstrip("\n")