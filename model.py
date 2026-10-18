"""
model.py - core engine of the Financial Assistant (RAG over SEC filings).

Everything that must survive a restart lives in ONE SQLite file (data/filings_index.db):
    filings    one row per filing (10-K / 10-Q / 8-K) + metadata
    sections   full text of every Item of every filing
    chunks     searchable text chunks + their embedding vectors
    fetch_log  when we last asked SEC EDGAR for each (ticker, form)

app.py (chat) and admin.py (management) both import this module and share that file.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import operator
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("financial_assistant")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import intents  # noqa: E402  - question type -> where to look and how to answer
import vectorstore  # noqa: E402  - vector index (SQLite or Qdrant) on top of the chunks table


# ════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ════════════════════════════════════════════════════════════════════
def _env(name: str, default=None):
    """Read an env var; treat empty / 'your_...' placeholders as 'not set'."""
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    if not v or v.lower().startswith("your_") or v.lower() in {"changeme", "none"}:
        return default
    return v


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR / "data"))).resolve()
DB_PATH = DATA_DIR / "filings_index.db"
PDF_DIR = Path(_env("PDF_DIR", str(DATA_DIR / "pdfs"))).resolve()

EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANK_MODEL = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = (_env("USE_RERANKER", "true") or "true").lower() == "true"

LLM_MODEL = _env("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
LLM_BASE_URL = _env("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "4500"))
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.3"))

ROUTER_MODEL = _env("ROUTER_MODEL", "openai/gpt-oss-20b")
ROUTER_BASE_URL = _env("ROUTER_BASE_URL", "https://api.groq.com/openai/v1")

CHECK_INTERVAL_HOURS = float(_env("CHECK_INTERVAL_HOURS", "12"))
MAX_CONTEXT_CHARS = int(_env("MAX_CONTEXT_CHARS", "28000"))

COMPANIES = {
    "NVDA": {"name": "NVIDIA", "aliases": ["nvidia", "nvda"]},
    "TSLA": {"name": "Tesla", "aliases": ["tesla", "tsla"]},
}

def company_name(ticker: str) -> str:
    """Display name for ANY ticker (built-in list, SEC company list, or what is stored with its filings)."""
    try:
        import companies
        return companies.name(ticker)
    except Exception:
        return COMPANIES.get(ticker, {}).get("name", ticker)


# 10-Ks you already had as PDFs (file must be inside PDF_DIR). Used by backfill_known_pdfs().
KNOWN_10K_PDFS = [
    {"file": "Nvidia.pdf", "ticker": "NVDA", "fiscal_year": 2026, "period_end": "2026-01-25"},
    {"file": "Tesla.pdf", "ticker": "TSLA", "fiscal_year": 2025, "period_end": "2025-12-31"},
]

MIN_FILING_CHARS = int(_env("MIN_FILING_CHARS", "30000") or 30000)  # a real 10-K/10-Q is 100k+ characters
DEFAULT_LIMITS = {"10-K": 1, "10-Q": 2, "8-K": 3}


# ════════════════════════════════════════════════════════════════════
# 2. SQLITE - the permanent store
# ════════════════════════════════════════════════════════════════════
SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    company      TEXT NOT NULL,
    form_type    TEXT NOT NULL,
    fiscal_year  INTEGER,
    period       TEXT,
    period_end   TEXT,
    filing_date  TEXT NOT NULL,
    accession_no TEXT UNIQUE NOT NULL,
    source       TEXT,
    source_file  TEXT,
    embed_model  TEXT,
    ingested_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_filings_tf  ON filings(ticker, form_type);
CREATE INDEX IF NOT EXISTS idx_filings_tfy ON filings(ticker, form_type, fiscal_year);

CREATE TABLE IF NOT EXISTS sections (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    position  INTEGER NOT NULL,
    name      TEXT NOT NULL,
    content   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sections_filing ON sections(filing_id);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id    INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    uid          TEXT UNIQUE NOT NULL,
    section      TEXT,
    element_type TEXT,
    page         INTEGER,
    chunk_idx    INTEGER,
    content      TEXT NOT NULL,
    embedding    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_filing ON chunks(filing_id);

CREATE TABLE IF NOT EXISTS fetch_log (
    ticker     TEXT NOT NULL,
    form_type  TEXT NOT NULL,
    checked_at REAL NOT NULL,
    PRIMARY KEY (ticker, form_type)
);
"""


@contextmanager
def db():
    """Open a connection, commit on success, roll back on error, always close."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


init_db()


def filing_exists(accession_no: str) -> bool:
    with db() as c:
        return c.execute("SELECT 1 FROM filings WHERE accession_no=?", (accession_no,)).fetchone() is not None


def filing_exists_for_year(ticker: str, form_type: str, fiscal_year: int | None) -> bool:
    if fiscal_year is None:
        return False
    with db() as c:
        return c.execute(
            "SELECT 1 FROM filings WHERE ticker=? AND form_type=? AND fiscal_year=?",
            (ticker.upper(), form_type, fiscal_year),
        ).fetchone() is not None


def count_chunks(accession_no: str) -> int:
    with db() as c:
        r = c.execute(
            "SELECT COUNT(*) n FROM chunks ch JOIN filings f ON f.id=ch.filing_id WHERE f.accession_no=?",
            (accession_no,),
        ).fetchone()
        return int(r["n"])


def list_filings(ticker: str | None = None, form_type: str | None = None) -> list[dict]:
    sql = """SELECT f.*, (SELECT COUNT(*) FROM chunks c WHERE c.filing_id=f.id) AS n_chunks,
                    (SELECT COUNT(*) FROM sections s WHERE s.filing_id=f.id) AS n_sections
             FROM filings f WHERE 1=1"""
    params: list = []
    if ticker:
        sql += " AND f.ticker=?"
        params.append(ticker.upper())
    if form_type:
        sql += " AND f.form_type=?"
        params.append(form_type)
    sql += " ORDER BY f.ticker, f.form_type, f.filing_date DESC"
    with db() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def get_sections(accession_no: str) -> dict[str, str]:
    with db() as c:
        rows = c.execute(
            """SELECT s.name, s.content FROM sections s JOIN filings f ON f.id=s.filing_id
               WHERE f.accession_no=? ORDER BY s.position""",
            (accession_no,),
        ).fetchall()
    return {r["name"]: r["content"] for r in rows}


def get_10k_sections(ticker: str, fiscal_year: int | None = None) -> dict | None:
    """Read a stored 10-K back (latest fiscal year if none given)."""
    with db() as c:
        if fiscal_year:
            row = c.execute("SELECT * FROM filings WHERE ticker=? AND form_type='10-K' AND fiscal_year=?",
                            (ticker.upper(), fiscal_year)).fetchone()
        else:
            row = c.execute("SELECT * FROM filings WHERE ticker=? AND form_type='10-K' "
                            "ORDER BY fiscal_year DESC LIMIT 1", (ticker.upper(),)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["sections"] = get_sections(d["accession_no"])
    return d


def delete_filing(accession_no: str) -> bool:
    with db() as c:
        cur = c.execute("DELETE FROM filings WHERE accession_no=?", (accession_no,))  # cascades
        deleted = cur.rowcount > 0
    if deleted:
        vectorstore.delete_filing(accession_no)
    return deleted


def db_stats() -> dict:
    with db() as c:
        return {
            "filings": c.execute("SELECT COUNT(*) FROM filings").fetchone()[0],
            "sections": c.execute("SELECT COUNT(*) FROM sections").fetchone()[0],
            "chunks": c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "db_file": str(DB_PATH),
            "db_size_mb": round(DB_PATH.stat().st_size / 1e6, 2) if DB_PATH.exists() else 0,
        }


def _check_due(ticker: str, form: str) -> bool:
    """True if SEC has not been asked recently, or we hold nothing for this (ticker, form)."""
    with db() as c:
        have = c.execute("SELECT 1 FROM filings WHERE ticker=? AND form_type=? LIMIT 1",
                         (ticker, form)).fetchone()
        row = c.execute("SELECT checked_at FROM fetch_log WHERE ticker=? AND form_type=?",
                        (ticker, form)).fetchone()
    if not have or not row:
        return True
    return (time.time() - row["checked_at"]) > CHECK_INTERVAL_HOURS * 3600


def _mark_checked(ticker: str, form: str) -> None:
    with db() as c:
        c.execute("INSERT INTO fetch_log(ticker, form_type, checked_at) VALUES (?,?,?) "
                  "ON CONFLICT(ticker, form_type) DO UPDATE SET checked_at=excluded.checked_at",
                  (ticker, form, time.time()))


# ════════════════════════════════════════════════════════════════════
# 3. EMBEDDINGS + CROSS-ENCODER (lazy: loaded on first use)
# ════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def _embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    device = _env("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading embedding model %s on %s", EMBED_MODEL, device)
    return SentenceTransformer(EMBED_MODEL, device=device)


def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Normalised float32 embeddings, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = _embedder().encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    prefix = "Represent this sentence for searching relevant passages: " if "bge" in EMBED_MODEL.lower() else ""
    return embed([prefix + query])[0]


@lru_cache(maxsize=1)
def get_cross_encoder():
    if not USE_RERANKER:
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(RERANK_MODEL)
    except Exception as e:  # reranking is an improvement, never a requirement
        log.warning("Cross-encoder unavailable (%s) - continuing without reranking", e)
        return None


# ════════════════════════════════════════════════════════════════════
# 4. TEXT CLEANING, SPLITTING, CHUNKING
# ════════════════════════════════════════════════════════════════════
def _tidy(text: str) -> str:
    text = text.replace("\f", "\n")
    text = re.sub(r"[─━═]{3,}|-{5,}", "", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)  # padded table columns -> two spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""])
    return [c for c in splitter.split_text(text) if len(c.strip()) >= 50]


_D = r"[\s\.\:\-–—\*\|]*"  # any mix of space / dot / colon / dash / bold markers after "Item 1"


def _split_by_anchors(text: str, anchors: list[tuple[str, str]], min_len: int = 300,
                      max_chars: int = 400_000) -> dict[str, str]:
    """Cut a filing into Items. A heading must START a line (ignores in-text cross references).
    Table-of-contents hits are tiny, so for every Item the LONGEST block wins."""
    clean = text.replace("\f", "\n")
    hits: list[tuple[int, str]] = []
    for name, pat in anchors:
        # a heading may follow "PART I" on the same line (AMD, Intel ...), e.g. "PART I ITEM 1. BUSINESS"
        rx = re.compile(r"^[ \t#*|>]*(?:part\s+[iv]+[\s,.:\-–—|*]*)?" + pat, re.I | re.M)
        hits.extend((m.start(), name) for m in rx.finditer(clean))
    hits.sort()
    items: dict[str, str] = {}
    for i, (start, name) in enumerate(hits):
        if name == "_END":
            continue
        end = hits[i + 1][0] if i + 1 < len(hits) else len(clean)
        block = clean[start:end].strip()
        if len(block) > len(items.get(name, "")):
            items[name] = block[:max_chars]
    return {k: v for k, v in items.items() if len(v) >= min_len}


_10K_ANCHORS = [
    ("Item_1_Business", rf"item\s*1\s*{_D}\s*business"),
    ("Item_1A_Risk_Factors", rf"item\s*1a\s*{_D}\s*risk\s+factors"),
    ("Item_1B_Unresolved_Staff", rf"item\s*1b\s*{_D}\s*unresolved"),
    ("Item_1C_Cybersecurity", rf"item\s*1c\s*{_D}\s*cybersecurity"),
    ("Item_2_Properties", rf"item\s*2\s*{_D}\s*properties"),
    ("Item_3_Legal", rf"item\s*3\s*{_D}\s*legal\s+proceedings"),
    ("Item_5_Market", rf"item\s*5\s*{_D}\s*market\s+for"),
    ("Item_7_MDA", rf"item\s*7\s*{_D}\s*management"),
    ("Item_7A_Market_Risk", rf"item\s*7a\s*{_D}\s*quantitative"),
    ("Item_8_Financial_Statements", rf"item\s*8\s*{_D}\s*(?:consolidated\s+)?financial\s+statements"),
    ("Item_9A_Controls", rf"item\s*9a\s*{_D}\s*controls"),
    # NVIDIA (and others) put the financial statements after Item 15/16, not under Item 8
    ("Item_15_Financial_Statements", rf"item\s*15\s*{_D}\s*(?:exhibit|financial)"),
    ("_END", rf"item\s*(?:4|6|9|9b|10|11|12|13|14)\s*{_D}\s*[a-z]"),
]

_10Q_ANCHORS = [
    ("Item_1_Financial_Statements",
     rf"item\s*1\s*{_D}\s*(?:condensed\s+)?(?:consolidated\s+)?financial\s+statements"),
    ("Item_2_MDA", rf"item\s*2\s*{_D}\s*management"),
    ("Item_3_Market_Risk", rf"item\s*3\s*{_D}\s*quantitative"),
    ("Item_4_Controls", rf"item\s*4\s*{_D}\s*controls"),
    ("PartII_Item_1_Legal", rf"item\s*1\s*{_D}\s*legal\s+proceedings"),
    ("PartII_Item_1A_Risk_Factors", rf"item\s*1a\s*{_D}\s*risk\s+factors"),
    ("_END", rf"item\s*(?:2\s*{_D}\s*unregistered|3\s*{_D}\s*defaults|4\s*{_D}\s*mine|"
            rf"5\s*{_D}\s*other\s+information|6\s*{_D}\s*exhibits)"),
]


# Some companies (Intel, ...) do not label sections "Item 1.", "Item 7." - they use plain titles such as
# "Risk Factors" or "Management's Discussion and Analysis". These anchors are used when no Item headings are found.
_APOS = r"[\'\u2019\u2018]?"
_E = r"[ \t]*(?:\(.*\))?[ \t]*$"  # the title must be the whole line
_10K_PLAIN_ANCHORS = [
    ("Item_1_Business", rf"(?:business|our\s+business|business\s+overview){_E}"),
    ("Item_1A_Risk_Factors", rf"risk\s+factors{_E}"),
    ("Item_7_MDA", rf"management{_APOS}s\s+discussion\s+and\s+analysis[^\n]{{0,110}}$"),
    ("Item_7A_Market_Risk", rf"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk{_E}"),
    ("Item_8_Financial_Statements",
     rf"(?:consolidated\s+financial\s+statements|financial\s+statements\s+and\s+supplementary\s+data|"
     rf"notes\s+to\s+(?:the\s+)?consolidated\s+financial\s+statements){_E}"),
    ("Item_9A_Controls", rf"controls\s+and\s+procedures{_E}"),
]
_10Q_PLAIN_ANCHORS = [
    ("Item_1_Financial_Statements",
     rf"(?:condensed\s+)?consolidated\s+(?:financial\s+statements|statements\s+of\s+(?:income|operations)){_E}"),
    ("Item_2_MDA", rf"management{_APOS}s\s+discussion\s+and\s+analysis[^\n]{{0,110}}$"),
    ("Item_3_Market_Risk", rf"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk{_E}"),
    ("Item_4_Controls", rf"controls\s+and\s+procedures{_E}"),
    ("PartII_Item_1A_Risk_Factors", rf"risk\s+factors{_E}"),
]


def extract_items_from_10k(text: str) -> dict[str, str]:
    items = _split_by_anchors(text, _10K_ANCHORS)
    if len(items) < 3:  # no "Item N." headings -> try the plain section titles
        plain = _split_by_anchors(text, _10K_PLAIN_ANCHORS)
        log.info("10-K without Item headings: plain-title split found %d sections", len(plain))
        if len(plain) >= 2:
            return plain
        log.warning("10-K: sections not recognised - storing the whole text as one section")
        items = {"Full_10K_Body": text.replace("\f", "\n")[:1_500_000]}
    return items


def extract_items_from_10q(text: str) -> dict[str, str]:
    items = _split_by_anchors(text, _10Q_ANCHORS, min_len=500)
    if len(items) < 2:
        plain = _split_by_anchors(text, _10Q_PLAIN_ANCHORS, min_len=500)
        log.info("10-Q without Item headings: plain-title split found %d sections", len(plain))
        if len(plain) >= 2:
            return plain
        clean = text.replace("\f", "\n")  # last resort: keep the whole text (never just a tail)
        items = {"Full_Filing_Body": clean[:600_000]}
    return items


def extract_items_from_8k(text: str) -> dict[str, str]:
    clean = text.replace("\f", "\n")
    rx = re.compile(r"^[ \t#*|>]*item\s+(\d\.\d{2})", re.I | re.M)
    marks = [(m.start(), m.group(1)) for m in rx.finditer(clean)]
    items: dict[str, str] = {}
    for i, (start, num) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(clean)
        block = clean[start:end].strip()
        if len(block) > 300 and len(block) > len(items.get(f"Item_{num}", "")):
            items[f"Item_{num}"] = block[:100_000]
    return items or {"Full_8K": clean[:100_000]}


def extract_key_blocks(filing_text: str, form_type: str) -> dict[str, str]:
    ft = form_type.upper()
    if ft == "10-K":
        return extract_items_from_10k(filing_text)
    if ft == "8-K":
        return extract_items_from_8k(filing_text)
    return extract_items_from_10q(filing_text)


def blocks_to_chunks(blocks: dict[str, str], accession_no: str) -> list[dict]:
    out = []
    for section, content in blocks.items():
        for i, piece in enumerate(chunk_text(_tidy(content))):
            out.append({"uid": f"{accession_no}:{section}:{i}", "section": section,
                        "element_type": "Text", "page": None, "chunk_idx": i, "content": piece})
    return out


# ════════════════════════════════════════════════════════════════════
# 5. SAVING A FILING (atomic: filing + sections + chunks, or nothing)
# ════════════════════════════════════════════════════════════════════
def save_filing(meta: dict, sections: dict[str, str], chunks: list[dict]) -> int:
    """Embed first (slow, outside the DB), then write everything in ONE transaction."""
    vecs = embed([c["content"] for c in chunks]) if chunks else np.zeros((0, 0), dtype=np.float32)
    now = datetime.now(timezone.utc).isoformat()
    acc = meta["accession_no"]

    replaced: list[str] = []
    with db() as c:
        if meta["form_type"] == "10-K" and meta.get("fiscal_year"):
            # a PDF-sourced row gets replaced by the real EDGAR one (and vice versa)
            replaced = [r["accession_no"] for r in c.execute(
                "SELECT accession_no FROM filings WHERE ticker=? AND form_type='10-K' AND fiscal_year=? "
                "AND accession_no<>?", (meta["ticker"], meta["fiscal_year"], acc)).fetchall()]
            c.execute("DELETE FROM filings WHERE ticker=? AND form_type='10-K' AND fiscal_year=? "
                      "AND accession_no<>?", (meta["ticker"], meta["fiscal_year"], acc))
        row = c.execute("SELECT id FROM filings WHERE accession_no=?", (acc,)).fetchone()
        fields = (meta["ticker"], meta["company"], meta["form_type"], meta.get("fiscal_year"),
                  meta.get("period"), meta.get("period_end"), str(meta["filing_date"]),
                  meta.get("source"), meta.get("source_file"), EMBED_MODEL, now)
        if row:
            fid = row["id"]
            c.execute("""UPDATE filings SET ticker=?, company=?, form_type=?, fiscal_year=?, period=?,
                         period_end=?, filing_date=?, source=?, source_file=?, embed_model=?, ingested_at=?
                         WHERE id=?""", fields + (fid,))
            c.execute("DELETE FROM sections WHERE filing_id=?", (fid,))
            c.execute("DELETE FROM chunks WHERE filing_id=?", (fid,))
        else:
            cur = c.execute("""INSERT INTO filings(ticker, company, form_type, fiscal_year, period,
                               period_end, filing_date, source, source_file, embed_model, ingested_at,
                               accession_no) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", fields + (acc,))
            fid = cur.lastrowid
        c.executemany("INSERT INTO sections(filing_id, position, name, content) VALUES (?,?,?,?)",
                      [(fid, i, n, t) for i, (n, t) in enumerate(sections.items())])
        c.executemany(
            """INSERT INTO chunks(filing_id, uid, section, element_type, page, chunk_idx, content, embedding)
               VALUES (?,?,?,?,?,?,?,?)""",
            [(fid, ch["uid"], ch.get("section"), ch.get("element_type"), ch.get("page"),
              ch.get("chunk_idx"), ch["content"], vecs[i].astype(np.float32).tobytes())
             for i, ch in enumerate(chunks)],
        )
    # SQLite is committed (the filing is safe). Now update the vector index; a failure here never loses data -
    # vectorstore falls back to SQLite vectors and sync() repairs the index later.
    for old in replaced:
        vectorstore.delete_filing(old)
    vectorstore.index_filing(acc)
    return fid


def reembed_filing(accession_no: str) -> int:
    """Re-compute vectors from the stored chunk text (use after changing EMBED_MODEL)."""
    with db() as c:
        rows = c.execute("""SELECT ch.id, ch.content FROM chunks ch JOIN filings f ON f.id=ch.filing_id
                            WHERE f.accession_no=?""", (accession_no,)).fetchall()
    if not rows:
        return 0
    vecs = embed([r["content"] for r in rows])
    with db() as c:
        c.executemany("UPDATE chunks SET embedding=? WHERE id=?",
                      [(vecs[i].astype(np.float32).tobytes(), r["id"]) for i, r in enumerate(rows)])
        c.execute("UPDATE filings SET embed_model=? WHERE accession_no=?", (EMBED_MODEL, accession_no))
    vectorstore.index_filing(accession_no)
    return len(rows)


def reembed_stale() -> list[str]:
    """Re-embed every filing whose vectors were made with a different model than EMBED_MODEL."""
    with db() as c:
        accs = [r["accession_no"] for r in c.execute(
            "SELECT accession_no FROM filings WHERE embed_model IS NOT ?", (EMBED_MODEL,)).fetchall()]
    return [f"{a}: {reembed_filing(a)} chunks re-embedded" for a in accs]


# ════════════════════════════════════════════════════════════════════
# 6. INGESTION - SEC EDGAR + PDFs
# ════════════════════════════════════════════════════════════════════
def _edgar_filings(ticker: str, form: str, limit: int) -> list:
    """Isolated so tests can stub the network."""
    identity = _env("EDGAR_IDENTITY")
    if not identity:
        raise RuntimeError("EDGAR_IDENTITY is not set in .env (SEC requires 'Your Name your@email.com')")
    from edgar import Company, set_identity

    set_identity(identity)
    # SEC's "10-K" list also contains amendments (10-K/A, 10-Q/A ...). Keep only the exact form,
    # fetching a few extra so skipped amendments never leave us short.
    found = Company(ticker).get_filings(form=form).head(limit + 10)
    exact = [f for f in found if str(getattr(f, "form", form)).strip().upper() == form.upper()]
    return exact[:limit]


def _period_fields(period_end: str) -> tuple[int | None, str | None]:
    period_end = (period_end or "")[:10]
    return (int(period_end[:4]) if period_end[:4].isdigit() else None), (period_end[:7] or None)


def _looks_like_markup(text: str) -> bool:
    head = (text or "")[:3000].lower()
    return head.lstrip().startswith("<") or "<?xml" in head or "<html" in head or "<xbrl" in head


def html_to_text(markup: str) -> str:
    """Raw SEC (inline-XBRL) HTML -> clean text. Each table row becomes ONE line ("Revenue  $ 130,497  $ 60,922"),
    every heading/paragraph sits on its own line, and the hidden XBRL header is dropped."""
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(markup, "lxml")
    except Exception:
        soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name and t.name.lower() in ("ix:header", "ix:hidden")):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    for tr in soup.find_all("tr"):
        cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", ")", "%"}]
        tr.replace_with(soup.new_string("  ".join(cells) + "\n"))
    for tag in soup.find_all(["br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"]):
        tag.append(soup.new_string("\n"))
    text = soup.get_text().replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _is_amendment_text(text: str, form: str) -> bool:
    """True if the cover page marks the document as an amendment (FORM 10-K/A, 'Amendment No. 1')."""
    head = (text or "")[:6000]
    f = re.escape(form.lower())
    return bool(re.search(rf"form\s+{f}\s*/\s*a\b|\(\s*amendment\s+no\.?\s*\d+\s*\)|^\s*amendment\s+no\.?\s*\d+\s*$",
                          head, re.I | re.M))


def get_filing_text(filing) -> str:
    """Clean text of an EDGAR filing. edgartools sometimes hands back the raw XBRL/HTML markup (seen with AMD);
    in that case it is converted here so Items and income statements can be found."""
    text = filing.text()
    if not text:
        raise RuntimeError("SEC returned no text for this filing")  # counts as a failure -> retried next time
    if _looks_like_markup(text):
        text = html_to_text(text)
    if len(text) < MIN_FILING_CHARS:  # suspiciously short for a 10-K/10-Q -> try the other renderings, keep the longest
        for name in ("html", "markdown"):
            fn = getattr(filing, name, None)
            if not callable(fn):
                continue
            try:
                alt = fn()
                alt = html_to_text(alt) if alt and _looks_like_markup(alt) else (alt or "")
                if len(alt) > len(text):
                    log.info("filing.text() gave %d chars; filing.%s() gave %d - using that", len(text), name, len(alt))
                    text = alt
            except Exception as e:
                log.warning("filing.%s() failed: %s", name, e)
    return text


def ingest_edgar_filing(ticker: str, form: str, filing, force: bool = False) -> str:
    ticker = ticker.upper()
    acc = filing.accession_no
    actual_form = str(getattr(filing, "form", form)).strip().upper()
    if actual_form != form.upper():  # e.g. a 10-K/A must never be stored as the 10-K
        return f"{actual_form} {filing.filing_date}: amendment skipped (only original {form} is stored)"
    period_end = str(getattr(filing, "period_of_report", "") or "")[:10]
    fy, period = _period_fields(period_end)

    if not force:
        if filing_exists(acc) and count_chunks(acc) > 0:
            return f"{form} {filing.filing_date}: already stored"
        if form == "10-K" and filing_exists_for_year(ticker, "10-K", fy):
            return f"{form} FY{fy}: already stored (from another source)"

    full_text = get_filing_text(filing)
    if _is_amendment_text(full_text, form):  # second line of defence: the cover page itself says "10-K/A"
        return f"{form} {filing.filing_date}: amendment skipped (cover page says it is an amendment)"
    if form in ("10-K", "10-Q") and len(full_text) < MIN_FILING_CHARS:
        return (f"{form} {filing.filing_date}: SEC text is only {len(full_text)} characters - NOT saved "
                f"(run diag_sections.py to inspect). It will be retried next time.")
    blocks = extract_key_blocks(full_text, form)
    chunks = blocks_to_chunks(blocks, acc)
    if not chunks:
        return f"{form} {filing.filing_date}: nothing extractable - NOT saved"

    meta = {"ticker": ticker, "company": company_name(ticker), "form_type": form,
            "fiscal_year": fy, "period": period, "period_end": period_end,
            "filing_date": str(filing.filing_date), "accession_no": acc, "source": "edgar"}
    save_filing(meta, blocks, chunks)
    return f"🆕 {form} {filing.filing_date} (FY{fy}) saved: {len(blocks)} sections, {len(chunks)} chunks"


def ensure_filings(ticker: str, forms: list[str], limit: int | None = None,
                   force_check: bool = False, force_reingest: bool = False) -> list[str]:
    """Make sure the newest filings of each form are in the DB. Throttled; works offline."""
    ticker = ticker.upper()
    msgs: list[str] = []
    for form in forms:
        if not force_check and not _check_due(ticker, form):
            continue
        try:
            filings = _edgar_filings(ticker, form, limit or DEFAULT_LIMITS.get(form, 2))
        except Exception as e:
            msgs.append(f"{ticker} {form}: SEC check skipped ({e}) - using stored data")
            continue
        failed = False
        for f in filings:
            try:
                msgs.append(f"{ticker} " + ingest_edgar_filing(ticker, form, f, force=force_reingest))
            except Exception as e:
                failed = True
                msgs.append(f"{ticker} {form} {getattr(f, 'filing_date', '?')}: FAILED ({e})")
                log.exception("ingest failed")
        if not failed:
            _mark_checked(ticker, form)
    return msgs


def _lookup_edgar_accession(ticker: str, period_end: str) -> tuple[str | None, str | None]:
    try:
        for f in _edgar_filings(ticker, "10-K", 8):
            if str(getattr(f, "period_of_report", ""))[:10] == period_end:
                return f.accession_no, str(f.filing_date)
    except Exception as e:
        log.info("EDGAR lookup skipped: %s", e)
    return None, None


def ingest_pdf_10k(pdf_path: str | Path, ticker: str, fiscal_year: int, period_end: str,
                   lookup_edgar: bool = True, force: bool = False) -> str:
    import pdfplumber
    import pymupdf

    ticker, pdf_path = ticker.upper(), Path(pdf_path)
    if not pdf_path.exists():
        return f"{pdf_path.name}: file not found"
    if not force and filing_exists_for_year(ticker, "10-K", fiscal_year):
        return f"{ticker} 10-K FY{fiscal_year}: already stored"

    acc, fdate = _lookup_edgar_accession(ticker, period_end) if lookup_edgar else (None, None)
    acc = acc or f"PDF-{ticker}-10K-FY{fiscal_year}"
    fdate = fdate or period_end

    page_texts: list[str] = []
    chunks: list[dict] = []
    with pymupdf.open(pdf_path) as doc, pdfplumber.open(pdf_path) as plumb:
        for pno in range(len(doc)):
            text = doc[pno].get_text("text").strip()
            page_texts.append(text)
            if len(text) < 30:
                continue
            for i, piece in enumerate(chunk_text(_tidy(text))):
                chunks.append({"uid": f"{acc}:p{pno + 1}:x{i}", "section": None, "element_type": "Text",
                               "page": pno + 1, "chunk_idx": i, "content": piece})
            try:
                for ti, table in enumerate(plumb.pages[pno].extract_tables()):
                    if table and len(table) > 1:
                        ttxt = "\n".join(" | ".join("" if c is None else str(c) for c in row) for row in table)
                        for j, piece in enumerate(chunk_text(ttxt, size=2500, overlap=100) or [ttxt]):
                            chunks.append({"uid": f"{acc}:p{pno + 1}:T{ti}:{j}", "section": None,
                                           "element_type": "Table", "page": pno + 1, "chunk_idx": j,
                                           "content": piece})
            except Exception:
                continue

    sections = extract_items_from_10k("\n".join(page_texts))
    if not chunks:
        return f"{pdf_path.name}: no text found (scanned PDF?) - NOT saved"
    meta = {"ticker": ticker, "company": company_name(ticker), "form_type": "10-K",
            "fiscal_year": fiscal_year, "period": f"FY{fiscal_year}", "period_end": period_end,
            "filing_date": fdate, "accession_no": acc, "source": "pdf", "source_file": pdf_path.name}
    save_filing(meta, sections, chunks)
    n_tab = sum(1 for c in chunks if c["element_type"] == "Table")
    return (f"🆕 {ticker} 10-K FY{fiscal_year} saved from {pdf_path.name}: {len(sections)} sections, "
            f"{len(chunks)} chunks ({n_tab} table chunks), accession {acc}")


def backfill_known_pdfs() -> list[str]:
    msgs = []
    for cfg in KNOWN_10K_PDFS:
        path = PDF_DIR / cfg["file"]
        if not path.exists():
            msgs.append(f"{cfg['file']}: not found in {PDF_DIR} - copy it there first")
            continue
        try:
            msgs.append(ingest_pdf_10k(path, cfg["ticker"], cfg["fiscal_year"], cfg["period_end"]))
        except Exception as e:
            msgs.append(f"{cfg['file']}: FAILED ({e})")
            log.exception("backfill failed")
    return msgs


# ════════════════════════════════════════════════════════════════════
# 7. CALCULATOR (safe - no eval)
# ════════════════════════════════════════════════════════════════════
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Arithmetic only: + - * / ** % and parentheses. Used by the LLM for every change / ratio."""
    try:
        expr = expression.replace(",", "").replace("$", "").strip()
        expr = re.sub(r"(\d)\s*%", r"\1/100", expr)
        if len(expr) > 300:
            return "Calculation Error: expression too long"
        result = _safe_eval(ast.parse(expr, mode="eval"))
        return f"{result:.4f}" if isinstance(result, float) else str(result)
    except ZeroDivisionError:
        return "Calculation Error: division by zero"
    except Exception as e:
        return f"Calculation Error: {e}"


CALC_TOOL = [{"type": "function", "function": {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression. Use it for every difference, % change, margin, ratio or growth rate.",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string",
                                                 "description": "e.g. ((96221 - 46743) / 46743) * 100"}},
                   "required": ["expression"]}}}]


# ════════════════════════════════════════════════════════════════════
# 8. LLM CLIENTS (OpenRouter for answers, Groq for routing)
# ════════════════════════════════════════════════════════════════════
def _clean_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.I)
    return text.replace("</think>", "").strip()


class LLMClient:
    def __init__(self, api_key: str, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL,
                 max_tokens: int = LLM_MAX_TOKENS, temperature: float = LLM_TEMPERATURE):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.disable_reasoning = (_env("LLM_DISABLE_REASONING", "true") or "true").lower() == "true"

    def _create(self, messages, tools=None, max_tokens=None, think=False, tool_choice=None):
        kw = dict(model=self.model, messages=messages, temperature=self.temperature,
                  max_tokens=max_tokens or self.max_tokens)
        if tools:
            kw["tools"] = tools
            if tool_choice:
                kw["tool_choice"] = tool_choice
        if self.disable_reasoning and not think:
            kw["extra_body"] = {"reasoning": {"enabled": False}}  # reasoning models: no hidden thinking
        for attempt in range(3):  # OpenRouter sometimes answers HTTP 200 with an error body and no choices
            resp = self.client.chat.completions.create(**kw)
            if getattr(resp, "choices", None):
                return resp
            err = getattr(resp, "error", None) or getattr(resp, "model_extra", None) or "no details"
            log.warning("LLM returned no choices (%s) - retry %d/3", err, attempt + 1)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"The LLM provider returned an empty response (often a rate limit or provider "
                           f"outage: {err}). Please try again in a moment.")

    def chat(self, prompt: str, use_calculator: bool = False, max_steps: int = 14) -> str:
        messages = [{"role": "user", "content": prompt}]
        tools = CALC_TOOL if use_calculator else None
        calc_cache: dict[str, str] = {}
        for step in range(max_steps):
            try:  # last step: forbid tools so the model must write its answer
                resp = self._create(messages, tools, tool_choice="none" if step == max_steps - 1 else None)
            except Exception as e:
                if tools:  # model/route without tool support -> carry on without the calculator
                    log.warning("Tool calling failed (%s) - retrying without calculator", e)
                    tools = None
                    continue
                raise
            msg = resp.choices[0].message
            if getattr(msg, "tool_calls", None):
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls]})
                for tc in msg.tool_calls:
                    try:
                        expr = json.loads(tc.function.arguments).get("expression", "")
                    except Exception:
                        expr = ""
                    result = calc_cache.get(expr) or calculator(expr)
                    calc_cache[expr] = result
                    log.info("calculator(%s) = %s", expr, result)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue
            content = _clean_text(msg.content or "")
            if not content:  # reasoning model ate the budget -> retry once with more room
                log.warning("Empty answer (finish_reason=%s) - retrying with a bigger budget",
                            resp.choices[0].finish_reason)
                resp = self._create(messages, tools, max_tokens=self.max_tokens * 2, think=True)
                content = _clean_text(resp.choices[0].message.content or "")
            return content or "⚠️ The model returned no text. Please try again."
        return "⚠️ Calculator loop limit reached. Please try again."


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    key = _env("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    return LLMClient(api_key=key)


def _router_chat(prompt: str) -> str | None:
    key = _env("GROQ_API_KEY")
    if not key:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=ROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=ROUTER_MODEL, temperature=0.1, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}], extra_body={"reasoning_effort": "low"})
    return resp.choices[0].message.content or ""


# ════════════════════════════════════════════════════════════════════
# 9. ROUTER
# ════════════════════════════════════════════════════════════════════
_Q_PATTERNS = [r"latest quarter", r"quarterly", r"10-q", r"\bq[1-4]\b", r"most recent quarter",
               r"latest reported quarter", r"three months ended", r"current quarter", r"this quarter",
               r"last quarter"]
_EV_PATTERNS = [r"8-k", r"announcement", r"press release", r"recent event", r"just announced"]
_K_PATTERNS = [r"10-k", r"annual", r"full[- ]year", r"fiscal year", r"long[- ]term", r"strategy",
               r"risk factors?", r"competit", r"supply chain", r"export", r"capital allocation",
               r"business model", r"architecture"]


def detect_tickers(query: str, company: str | None = None) -> list[str]:
    """Companies in the question: built-in NVIDIA/Tesla aliases, then the SEC list - limited to the covered
    tech universe (companies.py) unless COVERAGE=all. "peers/competitors" adds same-group companies."""
    def match(text: str) -> list[str]:
        t = text.lower()
        found = [tk for tk, info in COMPANIES.items()
                 if any(re.search(rf"\b{re.escape(a)}\b", t) for a in info["aliases"])]
        try:
            import companies
            found += [tk for tk in companies.resolve(text) if tk not in found]
            found = [tk for tk in found if companies.covered(tk)]
        except Exception as e:  # never let the company lookup break a question
            log.warning("company lookup failed: %s", e)
        return found[:4]

    found = match(company) if company else []
    if not found:
        found = match(query)
    if found and re.search(r"\b(peers?|competitors?|rivals?|similar\s+compan\w+|alternatives?)\b", query, re.I):
        try:
            import companies
            for p in companies.peers_of(found[0], 3):
                if p not in found and len(found) < 4:
                    found.append(p)
        except Exception as e:
            log.warning("peer lookup failed: %s", e)
    return found


def rule_route(query: str) -> dict:
    q = query.lower()
    hit = lambda pats: any(re.search(p, q) for p in pats)  # noqa: E731
    return {"needs_10q": hit(_Q_PATTERNS), "needs_8k": hit(_EV_PATTERNS), "needs_10k": hit(_K_PATTERNS)}


_ROUTE_CACHE: dict = {}


def _llm_route(query: str, prev: str | None) -> dict | None:
    """ONE LLM call decides everything: which filings, whether a chart is wanted, and the question intent."""
    key = (query, prev)
    if key in _ROUTE_CACHE:
        return _ROUTE_CACHE[key]
    menu = "\n".join(f"- {i.name}: {i.label}" for i in intents.INTENTS)
    prompt = (
        "You are the routing engine of an SEC-filings assistant. Read the question and decide:\n"
        "1. Sources: 10-K (annual: business description, risk factors, strategy, full-year results), "
        "10-Q (latest quarter results and updates), 8-K (recent events, earnings releases, announcements). "
        "Several may be needed. 'latest'/'recent' results usually mean 10-Q; business/risks/strategy mean 10-K.\n"
        "2. needs_chart: true ONLY if the user asks to plot/chart/graph/visualise/show a trend visually.\n"
        f"3. intents: 1-2 from this list:\n{menu}\n"
        "   Use 'financials' only when the user wants numbers/metrics/ratios/trends. Explanations of risks, "
        "business, strategy, competition, legal etc. are NOT financials even if they mention revenue.\n"
        + (f'Previous question in this chat (for follow-ups like "and gross profit?"): "{prev}"\n' if prev else "")
        + f'Question: "{query}"\n'
        'Reply ONLY with JSON: {"needs_10k": bool, "needs_10q": bool, "needs_8k": bool, "needs_chart": bool, '
        '"intents": ["name"], "reason": "short"}')
    try:
        m = re.search(r"\{.*\}", _router_chat(prompt) or "", re.DOTALL)
        data = json.loads(m.group())
        out = {k: bool(data.get(k)) for k in ("needs_10k", "needs_10q", "needs_8k", "needs_chart")}
        out["intents"] = [n for n in data.get("intents", []) if n in {i.name for i in intents.INTENTS}][:2]
        out["reason"] = "LLM router: " + str(data.get("reason", ""))[:120]
        if len(_ROUTE_CACHE) > 500:
            _ROUTE_CACHE.clear()
        _ROUTE_CACHE[key] = out
        return out
    except Exception as e:
        log.warning("LLM router failed (%s) - falling back to keyword rules", e)
        return None


def route_query(query: str, company: str | None = None, prev_query: str | None = None) -> dict:
    """The LLM decides sources, chart and intent. Keyword rules are only a safety net when the LLM is
    unavailable (no GROQ_API_KEY, rate limit, bad JSON)."""
    r = _llm_route(query, prev_query)
    if r:
        flags = {k: r[k] for k in ("needs_10k", "needs_10q", "needs_8k")}
        needs_chart, reason, names = r["needs_chart"], r["reason"], r["intents"]
        if not any(flags.values()):
            flags["needs_10k"] = True
    else:
        try:
            import analytics
            needs_chart = analytics.wants_chart(query)
        except Exception as e:
            log.warning("Chart tool unavailable (%s)", e)
            needs_chart = False
        flags, reason, names = rule_route(query), "keyword rules (LLM router unavailable)", []
        if not any(flags.values()) and not needs_chart:
            flags["needs_10k"], reason = True, "default (10-K)"
    return {"tickers": detect_tickers(query, company), **flags, "needs_chart": needs_chart,
            "reason": reason, "intent_names": names}


# ════════════════════════════════════════════════════════════════════
# 10. RETRIEVAL: dense (cosine on stored vectors) + BM25 -> RRF -> cross-encoder -> boosts
# ════════════════════════════════════════════════════════════════════
_corpus_cache: dict = {}
_cache_lock = threading.Lock()
_TOKEN = re.compile(r"[a-z]+|\d[\d,\.]*\d|\d")
_BOOST_WORDS = ["year ended", "three months ended", "consolidated statements of income", "revenue by",
                "in millions", "gross profit", "operating income", "operating expense", "net income"]


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _signature() -> tuple:
    with db() as c:
        r = c.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()
        return (r[0], r[1])


MIN_SECTION_CHUNKS = 5   # fewer chunks than this in the wanted sections = the filing was not split into sections


def _load_corpus(ticker: str | None, form_types: list[str] | None, fiscal_year: int | None = None) -> dict:
    """Chunk TEXT + metadata (no vectors) and a BM25 index. Cached; refreshes when ANY process changes the DB.
    The vectors are searched by `vectorstore` (SQLite or Qdrant); this corpus supplies text, order and keyword search."""
    from rank_bm25 import BM25Okapi

    sig = _signature()
    key = (ticker, tuple(sorted(form_types)) if form_types else None, fiscal_year)
    with _cache_lock:
        cached = _corpus_cache.get(key)
        if cached and cached["sig"] == sig:
            return cached

    sql = """SELECT c.id, c.uid, c.section, c.element_type, c.page, c.chunk_idx, c.content,
                    f.ticker, f.form_type, f.fiscal_year, f.filing_date, f.period_end, f.accession_no
             FROM chunks c JOIN filings f ON f.id = c.filing_id WHERE 1=1"""
    params: list = []
    if ticker:
        sql += " AND f.ticker=?"
        params.append(ticker)
    if form_types:
        sql += f" AND f.form_type IN ({','.join('?' * len(form_types))})"
        params += list(form_types)
    if fiscal_year:
        sql += " AND f.fiscal_year=?"
        params.append(fiscal_year)
    sql += " ORDER BY c.id"
    with db() as c:
        rows = c.execute(sql, params).fetchall()

    records = [dict(r) for r in rows]
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault((r["accession_no"], r["section"]), []).append(i)
    for idxs in groups.values():  # position of a chunk inside its section: 0.0 = start, 1.0 = end
        for k, i in enumerate(idxs):
            records[i]["pos"] = k / max(1, len(idxs) - 1)
    bm25 = BM25Okapi([_tokenize(r["content"]) for r in records]) if records else None
    latest: dict = {}
    latest_acc: dict = {}
    for r in records:
        if r["filing_date"] >= latest.get(r["form_type"], ""):
            latest[r["form_type"]], latest_acc[r["form_type"]] = r["filing_date"], r["accession_no"]
    entry = {"sig": sig, "records": records, "bm25": bm25, "latest": latest, "latest_acc": latest_acc,
             "idx": {r["uid"]: i for i, r in enumerate(records)}}
    with _cache_lock:
        _corpus_cache[key] = entry
    return entry


def _near_duplicate(a: set, b: set) -> bool:
    return bool(a and b) and len(a & b) / len(a | b) >= 0.8


def _coverage_pick(scored: list[tuple[dict, float]], breadth: int, top_n: int) -> list[tuple[dict, float]]:
    """Cover the whole section: the best chunk from each position zone, then the best of the rest.

    A question like "main risk factors" cannot be answered from the few chunks that best match its words - the real
    risks are spread over the entire section. Zone winners must still beat the median relevance of the shortlist, so a
    narrow question ("risks from AI") only keeps the zones that are actually about it."""
    if not scored:
        return []
    floor = float(np.median([s for _, s in scored]))
    zone_best: dict[tuple, tuple[dict, float]] = {}
    for rec, s in scored:
        z = (rec["accession_no"], rec["section"], min(breadth - 1, int(rec["pos"] * breadth)))
        if z not in zone_best or s > zone_best[z][1]:
            zone_best[z] = (rec, s)
    ranked = sorted(zone_best.values(), key=lambda x: -x[1])
    chosen = [x for x in ranked if x[1] >= floor][:top_n]
    taken = {id(x[0]) for x in chosen}
    sigs = [set(_tokenize(x[0]["content"])) for x in chosen]
    for rec, s in sorted(scored, key=lambda x: -x[1]):  # fill the remaining slots with the best of the rest
        if len(chosen) >= top_n:
            break
        if id(rec) in taken:
            continue
        sig = set(_tokenize(rec["content"]))
        if any(_near_duplicate(sig, o) for o in sigs):
            continue
        chosen.append((rec, s))
        sigs.append(sig)
    return chosen


def retrieve_ex(query: str, ticker: str | None, form_types: list[str] | None, fiscal_year: int | None = None,
                top_n: int = 10, pool: int = 40, prefer_latest: bool = True,
                profile: "intents.Profile | None" = None) -> tuple[list[tuple[dict, float]], dict]:
    """Hybrid retrieval that depends on WHAT is asked.

    Quantitative questions: dense + BM25 -> RRF -> cross-encoder, boosting tables and statement keywords.
    Qualitative questions (risks, business, strategy ...): search only the relevant filing sections of the newest
    filing, expand the query with the words real passages use, never boost tables, and cover the whole section."""
    corpus = _load_corpus(ticker, form_types, fiscal_year)
    recs = corpus["records"]
    info = {"sections": None, "relaxed": False, "filing": None, "strategy": "standard", "candidates": 0}
    if not recs:
        return [], info
    prof = profile
    qualitative = bool(prof and prof.qualitative)
    form = form_types[0] if form_types and len(form_types) == 1 else None

    # 1. which filing: the newest of that form, unless the question spans several
    acc_filter = None
    if qualitative and prefer_latest and not fiscal_year and form and not intents.wants_history(query):
        newest = corpus["latest_acc"].get(form)
        acc_filter = {newest} if newest else None
        info["filing"] = newest

    # 2. which sections: only if the filing really was split into sections (else search everything)
    wanted = None
    if prof and form and prof.sections_for(form):
        want = set(prof.sections_for(form))
        n_in = sum(1 for r in recs if r["section"] in want and (acc_filter is None or r["accession_no"] in acc_filter))
        if n_in >= MIN_SECTION_CHUNKS:
            wanted = want
        else:
            info["relaxed"] = True
    info["sections"] = sorted(wanted) if wanted else None
    hard = wanted if (wanted and prof.strict) else None
    soft = wanted if (wanted and not prof.strict) else None

    allowed = [i for i, r in enumerate(recs)
               if (hard is None or r["section"] in hard) and (acc_filter is None or r["accession_no"] in acc_filter)]
    if not allowed:
        allowed = list(range(len(recs)))
        info["relaxed"] = True
    allowed_set = set(allowed)
    pool_n = max(pool, 80) if qualitative else pool

    # 3. candidates: dense (vector index, with metadata filters) + BM25 (keywords), fused with RRF
    dq = f"{query}. {prof.expansion}" if (prof and prof.expansion) else query
    dense = vectorstore.search(embed_query(dq), ticker=ticker, form_types=form_types, fiscal_year=fiscal_year,
                               sections=sorted(hard) if hard else None,
                               accession_nos=sorted(acc_filter) if acc_filter else None, limit=pool_n * 2)
    dense_rank = [corpus["idx"][u] for u, _ in dense if u in corpus["idx"] and corpus["idx"][u] in allowed_set]
    bm = corpus["bm25"].get_scores(_tokenize(dq))
    bm_rank = [int(i) for i in np.argsort(-bm) if int(i) in allowed_set and bm[i] > 0][:pool_n * 2]

    rrf: dict[int, float] = {}
    for rank, i in enumerate(dense_rank):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank)
    for rank, i in enumerate(bm_rank):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank)
    cand = sorted(rrf, key=rrf.get, reverse=True)[:pool_n] or allowed[:pool_n]
    info["candidates"] = len(cand)

    # 4. rerank with the cross-encoder (uses the question as asked, not the expanded one)
    ce = get_cross_encoder()
    if ce is not None:
        raw = ce.predict([[query, recs[i]["content"][:2000]] for i in cand])
        base = {i: 1.0 / (1.0 + math.exp(-float(s))) for i, s in zip(cand, raw)}
    else:
        top = max(rrf.get(i, 0.0) for i in cand) or 1.0
        base = {i: rrf.get(i, 0.0) / top for i in cand}

    # 5. adjust: numbers only help a numbers question
    scored = []
    for i in cand:
        r, s = recs[i], base[i]
        if qualitative:
            if intents.is_boilerplate(r["content"]):
                s -= 0.25
            if r["element_type"] == "Table" or intents.numeric_density(r["content"]) > 0.18:
                s -= 0.25
        else:
            if r["element_type"] == "Table":
                s += 0.15
            if any(w in r["content"].lower() for w in _BOOST_WORDS):
                s += 0.10
        if soft and r["section"] in soft:
            s += 0.12
        if prefer_latest and not fiscal_year and r["filing_date"] == corpus["latest"].get(r["form_type"]):
            s += 0.10
        scored.append((r, s))

    # 6. choose
    if qualitative and prof.breadth:
        chosen = _coverage_pick(scored, prof.breadth, top_n)
        chosen.sort(key=lambda x: (x[0]["accession_no"], x[0]["id"]))  # read it in filing order
        info["strategy"] = f"coverage ({prof.breadth} zones per section)"
        return chosen, info
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n], info


def retrieve(query: str, ticker: str | None, form_types: list[str] | None, fiscal_year: int | None = None,
             top_n: int = 10, pool: int = 40, prefer_latest: bool = True,
             profile: "intents.Profile | None" = None) -> list[tuple[dict, float]]:
    return retrieve_ex(query, ticker, form_types, fiscal_year, top_n, pool, prefer_latest, profile)[0]


def _source_label(r: dict) -> str:
    loc = r.get("section") or (f"page {r['page']}" if r.get("page") else "")
    return f"{r['ticker']} {r['form_type']} (FY{r['fiscal_year']}, filed {r['filing_date']})" + (f" - {loc}" if loc else "")


def build_context(groups: list[list[tuple[dict, float]]], max_chars: int = MAX_CONTEXT_CHARS,
                  interleave: bool = True) -> tuple[str, list[str]]:
    """Join the chosen chunks. interleave=True shares the space between groups (numbers questions);
    interleave=False keeps each (company, filing) together and in reading order (explanations)."""
    parts, sources, used = [], [], 0

    def add(rec) -> None:
        nonlocal used
        block = f"[{_source_label(rec)}]\n{rec['content']}"
        if used + len(block) > max_chars:
            return
        parts.append(block)
        used += len(block)
        if _source_label(rec) not in sources:
            sources.append(_source_label(rec))

    if interleave:
        for rank in range(max((len(g) for g in groups), default=0)):
            for g in groups:
                if rank < len(g):
                    add(g[rank][0])
    else:
        share = max_chars // max(1, len(groups))  # each group gets an equal share, so none is crowded out
        for g in groups:
            start = used
            for rec, _ in g:
                if used - start + len(rec["content"]) > share:
                    break
                add(rec)
    return "\n\n---\n\n".join(parts), sources


# ════════════════════════════════════════════════════════════════════
# 11. ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════
_memory: dict[str, list[tuple[str, str]]] = {}
_last_route: dict[str, dict] = {}

ANALYST_PROMPT = """You are a Senior Institutional Financial Analyst.

ABSOLUTE RULES - NEVER BREAK THEM
1. Use ONLY numbers that appear VERBATIM in the Context below. Never invent, estimate, round, or use numbers from memory.
2. LABEL LOCKING: every number stays attached to the exact label it has in the Context (Revenue only for Revenue,
   Operating expenses only for Operating expenses, Net income only for Net income, and so on). Never swap labels.
3. Always state the period and filing for a number (e.g. "three months ended July 26, 2026, 10-Q filed 2026-08-26").
4. For EVERY difference, percentage change, margin, ratio or growth rate, call the calculator tool with the numbers
   from the Context and use its returned value. Never calculate mentally. Show the formula you passed to it.
5. DIRECTION: compare the two numbers of the SAME metric before writing increased / decreased. Later > earlier = increased.
6. If a metric (or one of its two periods) is missing from the Context, write: "not disclosed in the retrieved sections."
7. Different filings have different labels such as [10-K ...] and [10-Q ...]: never mix annual and quarterly figures.

FORMAT
A. Financial-figure questions: start with a Markdown table (Metric | Earlier | Later | Change | % Change), then a
   structured analysis that uses only numbers from the table.
B. Strategy / technology / risk / competition questions: no table; write a clear, structured analysis from the Context
   and quote short passages when useful.
Write in a clear professional financial tone. If the user asks for investor implications, give balanced, clearly
reasoned considerations (not a guarantee of returns) and note this is not personalised financial advice.

{history}Context:
{context}

Question:
{query}

Financial Analysis:"""


QUALITATIVE_PROMPT = """You are a Senior Equity Research Analyst writing from SEC filings.

This question is QUALITATIVE (business, risks, strategy, competition, legal, etc.). Rules:
1. Answer ONLY from the Context. Do not use outside knowledge. If something is not in the Context, say so.
2. Do NOT include financial tables or revenue / income figures unless the question explicitly asks for them.
   At most mention Financial Figures when they are essential to explain a point.
3. Be thorough and specific: cover EVERY distinct topic the Context supports, grouped under clear headings
   (for risks: one heading per risk category, with 2-4 sentences each on what the risk is and why it matters).
   Prefer the filing's own concrete wording; quote short phrases when useful.
4. Name the filing (form, period) the points come from. Ignore cross-reference boilerplate and table-of-contents text.
5. If the Context looks incomplete for the question, add a final line: "Coverage note: ..." explaining what seems missing.
6. If the user asks for investor implications, be balanced and note this is not personalised financial advice.

{history}Context:
{context}

Question:
{query}

Answer:"""


def _history_text(session_id: str, max_turns: int = 3) -> str:
    turns = _memory.get(session_id, [])[-max_turns:]
    if not turns:
        return ""
    lines = ["Previous conversation (for follow-up questions only):"]
    for q, a in turns:
        lines.append(f"User: {q}\nAssistant: {a[:600]}{'…' if len(a) > 600 else ''}")
    return "\n".join(lines) + "\n\n"


def _which_company_message(query: str) -> str:
    try:
        import companies
        if any(not companies.covered(t) for t in companies.resolve(query)):
            return companies.coverage_message(query)
        return ("Which company do you mean? Mention its name or ticker (e.g. *Apple*, *AMD*, *Ford*) "
                "or pick it in the Company box.\n\n" + companies.coverage_message(""))
    except Exception:
        return "Which company do you mean? Mention its name or ticker in your question."


def financial_assistant(query: str, company: str | None = None, session_id: str = "default") -> dict:
    """Route -> make sure the data is stored (fetch from SEC if new) -> retrieve -> answer."""
    last = _last_route.get(session_id)
    decision = route_query(query, company, prev_query=last.get("query") if last else None)
    if last:  # follow-ups ("and gross profit?") inherit company / filing type from the previous turn
        if not decision["tickers"]:
            decision["tickers"] = list(last["tickers"])
        if decision["reason"].startswith(("default", "keyword")):
            decision.update({k: last[k] for k in ("needs_10k", "needs_10q", "needs_8k")},
                             reason="follow-up: same sources as previous question")
    chart_out = None
    if decision.get("needs_chart"):
        import analytics
        chart_out = analytics.run_chart_tool(query, decision, session_id)
        decision["chart_spec"] = chart_out["spec"]
        if not analytics.wants_analysis(query) or not chart_out["figures"]:
            _memory.setdefault(session_id, []).append((query, chart_out["text"]))
            _memory[session_id] = _memory[session_id][-6:]
            return {"answer": chart_out["text"], "sources": chart_out["sources"], "decision": decision,
                    "ingest": chart_out["ingest"], "charts": chart_out["figures"]}
    tickers = decision["tickers"]
    _last_route[session_id] = {k: decision[k] for k in ("tickers", "needs_10k", "needs_10q", "needs_8k")}
    _last_route[session_id]["query"] = query
    if not tickers:
        return {"answer": _which_company_message(query),
                "sources": [], "decision": decision, "ingest": []}

    forms = [f for f, flag in (("10-K", decision["needs_10k"]), ("10-Q", decision["needs_10q"]),
                               ("8-K", decision["needs_8k"])) if flag]
    ingest_msgs: list[str] = []
    for t in tickers:
        ingest_msgs += ensure_filings(t, forms)  # no-op when recently checked or SEC is unreachable

    n_groups = len(tickers) * len(forms)
    prof = (intents.profile_from_names(decision.get("intent_names")) if decision.get("intent_names")
            else intents.classify_smart(query))
    decision["intent"] = prof.name
    top_n = 12 if n_groups == 1 else 8 if n_groups == 2 else 6
    if prof.qualitative:
        top_n = max(top_n, prof.top_n // max(1, n_groups) if n_groups > 1 else prof.top_n)
    groups, notes = [], []
    for t in tickers:
        for f in forms:
            hits, info = retrieve_ex(query, t, [f], top_n=top_n, profile=prof)
            if hits:
                groups.append(hits)
            if prof.qualitative and info.get("relaxed"):
                notes.append(f"{company_name(t)} {f}: the expected section(s) were missing or tiny in storage, "
                             "so broader retrieval was used - answer may be incomplete (re-ingest the filing).")
    if not groups:
        names = ", ".join(company_name(t) for t in tickers)
        return {"answer": f"I have no stored {'/'.join(forms)} data for {names} yet. "
                          "Ask an admin to ingest it (admin.py), or check the SEC connection settings.",
                "sources": [], "decision": decision, "ingest": ingest_msgs,
                "charts": chart_out["figures"] if chart_out else []}

    context, sources = build_context(groups, max_chars=prof.context_chars if prof.qualitative else MAX_CONTEXT_CHARS,
                                     interleave=not prof.qualitative)
    prompt = (QUALITATIVE_PROMPT if prof.qualitative else ANALYST_PROMPT).format(
        history=_history_text(session_id), context=context, query=query)
    try:
        answer = get_llm().chat(prompt, use_calculator=not prof.qualitative)
    except Exception as e:
        if not chart_out:
            raise
        log.warning("LLM failed after chart was drawn: %s", e)
        return {"answer": f"{chart_out['text']}\n\n⚠️ The written analysis failed ({e}). The chart above is still valid.",
                "sources": chart_out["sources"], "decision": decision, "ingest": ingest_msgs,
                "charts": chart_out["figures"]}

    _memory.setdefault(session_id, []).append((query, answer))
    _memory[session_id] = _memory[session_id][-6:]

    title = " & ".join(company_name(t) for t in tickers)
    pre = f"{chart_out['text']}\n\n---\n\n" if chart_out else ""
    head = prof.label if prof.qualitative else "Financial Analysis"
    if notes:
        answer += "\n\n> ⚠️ Data-quality note: " + " ".join(notes)
    md = (f"{pre}# {head} - {title}\n\n{answer}\n\n## Sources\n" + "\n".join(f"- {s}" for s in sources))
    return {"answer": md, "sources": sources, "decision": decision, "ingest": ingest_msgs, "context": context,
            "charts": chart_out["figures"] if chart_out else []}