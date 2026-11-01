"""
vectorstore.py - where the chunk VECTORS live.

SQLite (data/filings_index.db) stays the source of truth for filings, sections, chunk text AND a copy of every
vector.  A vector database is an INDEX on top of it that is fast to search with metadata filters (company, form,
section, fiscal year).  Because SQLite keeps the vectors, the index can always be rebuilt without re-embedding.

Backends (choose with VECTOR_BACKEND in .env)
  sqlite   default. Vectors are read from SQLite and searched in memory with numpy. No extra service. Fine up to a few
           hundred thousand chunks.
  qdrant   a real vector database.
             QDRANT_URL + QDRANT_API_KEY   Qdrant Cloud (free tier) or a Qdrant server (docker run -p 6333:6333 qdrant/qdrant)
             QDRANT_PATH=./data/qdrant     embedded, single process only (app.py and admin.py cannot both open it)
             QDRANT_PATH=:memory:          tests

Resilience: if the vector database is unreachable, searches fall back to the SQLite backend automatically, and
`sync()` repairs the index afterwards.  A failed index write never loses data - the filing is already in SQLite.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("financial_assistant")

COLLECTION = os.getenv("QDRANT_COLLECTION", "filings_chunks")
BATCH = 128


@dataclass
class VecRow:
    uid: str
    vector: np.ndarray
    payload: dict


# ───────────────────────── reading from SQLite (lazy import: model imports this module) ─────────────────────────
def _db():
    import model

    return model.db()


def _sqlite_counts() -> dict[str, int]:
    """accession_no -> number of chunks stored in SQLite."""
    with _db() as c:
        rows = c.execute("""SELECT f.accession_no a, COUNT(ch.id) n FROM filings f
                            LEFT JOIN chunks ch ON ch.filing_id = f.id GROUP BY f.id""").fetchall()
    return {r["a"]: int(r["n"]) for r in rows}


def _rows_for_filing(accession_no: str) -> list[VecRow]:
    with _db() as c:
        rows = c.execute("""SELECT ch.uid, ch.section, ch.element_type, ch.page, ch.chunk_idx, ch.embedding,
                                   f.ticker, f.form_type, f.fiscal_year, f.filing_date, f.accession_no
                            FROM chunks ch JOIN filings f ON f.id = ch.filing_id
                            WHERE f.accession_no = ?""", (accession_no,)).fetchall()
    out = []
    for r in rows:
        out.append(VecRow(
            uid=r["uid"], vector=np.frombuffer(r["embedding"], dtype=np.float32),
            payload={"uid": r["uid"], "ticker": r["ticker"], "form_type": r["form_type"],
                     "fiscal_year": r["fiscal_year"], "section": r["section"], "element_type": r["element_type"],
                     "page": r["page"], "chunk_idx": r["chunk_idx"], "filing_date": r["filing_date"],
                     "accession_no": r["accession_no"]}))
    return out


# ═════════════════════════════════════════ backend 1: SQLite + numpy ═════════════════════════════════════════
class SqliteBackend:
    name = "sqlite"

    def __init__(self):
        self._cache: dict = {}
        self._lock = threading.Lock()

    # the vectors already live in the chunks table, so there is nothing to write
    def upsert(self, rows: list[VecRow]) -> None:
        return None

    def delete_filing(self, accession_no: str) -> None:
        return None

    def count(self, accession_no: str | None = None) -> int:
        counts = _sqlite_counts()
        return counts.get(accession_no, 0) if accession_no else sum(counts.values())

    def accessions(self) -> dict[str, int]:
        return {a: n for a, n in _sqlite_counts().items() if n}

    @staticmethod
    def _signature() -> tuple:
        with _db() as c:
            r = c.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()
        return (r[0], r[1])

    def _load(self, ticker, form_types, fiscal_year) -> dict:
        key = (ticker, tuple(sorted(form_types)) if form_types else None, fiscal_year)
        sig = self._signature()
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit["sig"] == sig:
                return hit
        sql = """SELECT ch.uid, ch.section, f.accession_no, ch.embedding
                 FROM chunks ch JOIN filings f ON f.id = ch.filing_id WHERE 1=1"""
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
        with _db() as c:
            rows = c.execute(sql, params).fetchall()
        entry = {"sig": sig, "uids": [r["uid"] for r in rows], "sections": [r["section"] for r in rows],
                 "acc": [r["accession_no"] for r in rows],
                 "matrix": (np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                            if rows else None)}
        with self._lock:
            self._cache[key] = entry
        return entry

    def search(self, qvec, *, ticker=None, form_types=None, fiscal_year=None, sections=None,
               accession_nos=None, limit=40) -> list[tuple[str, float]]:
        e = self._load(ticker, form_types, fiscal_year)
        if e["matrix"] is None:
            return []
        sims = e["matrix"] @ np.asarray(qvec, dtype=np.float32)
        mask = np.ones(len(sims), dtype=bool)
        if sections:
            allowed = set(sections)
            mask &= np.array([s in allowed for s in e["sections"]])
        if accession_nos:
            allowed_acc = set(accession_nos)
            mask &= np.array([a in allowed_acc for a in e["acc"]])
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        order = idx[np.argsort(-sims[idx])[:limit]]
        return [(e["uids"][i], float(sims[i])) for i in order]

    def status(self) -> dict:
        return {"backend": "sqlite", "ok": True, "detail": "vectors read from SQLite and searched in memory"}


# ═════════════════════════════════════════ backend 2: Qdrant ═════════════════════════════════════════
class QdrantBackend:
    name = "qdrant"

    def __init__(self, url: str | None = None, api_key: str | None = None, path: str | None = None):
        from qdrant_client import QdrantClient

        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
            self.where = url
        elif path == ":memory:":
            self.client = QdrantClient(":memory:")
            self.where = "memory"
        elif path:
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
            self.where = path
        else:
            raise ValueError("Qdrant needs QDRANT_URL (server / cloud) or QDRANT_PATH (embedded)")
        self.client.get_collections()  # fail fast if the server is unreachable
        self._dim: int | None = None
        self._lock = threading.Lock()

    # ---- helpers ----
    @staticmethod
    def _pid(uid: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, uid))

    def _exists(self) -> bool:
        return any(c.name == COLLECTION for c in self.client.get_collections().collections)

    def _ensure(self, dim: int) -> None:
        from qdrant_client import models

        with self._lock:
            if self._dim == dim:
                return
            if self._exists():
                size = self.client.get_collection(COLLECTION).config.params.vectors.size
                if size != dim:  # EMBED_MODEL changed: the index is derived data, so rebuild it
                    log.warning("Qdrant collection has dimension %s but vectors are %s - recreating it", size, dim)
                    self.client.delete_collection(COLLECTION)
            if not self._exists():
                self.client.create_collection(
                    COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
                for field, schema in (("ticker", "keyword"), ("form_type", "keyword"), ("section", "keyword"),
                                      ("accession_no", "keyword"), ("fiscal_year", "integer")):
                    try:  # not supported by the embedded mode (harmless)
                        self.client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
                    except Exception:
                        pass
            self._dim = dim

    @staticmethod
    def _filter(ticker=None, form_types=None, fiscal_year=None, sections=None, accession_nos=None, accession_no=None):
        from qdrant_client import models as m

        must = []
        if ticker:
            must.append(m.FieldCondition(key="ticker", match=m.MatchValue(value=ticker)))
        if form_types:
            must.append(m.FieldCondition(key="form_type", match=m.MatchAny(any=list(form_types))))
        if fiscal_year:
            must.append(m.FieldCondition(key="fiscal_year", match=m.MatchValue(value=int(fiscal_year))))
        if sections:
            must.append(m.FieldCondition(key="section", match=m.MatchAny(any=list(sections))))
        if accession_nos:
            must.append(m.FieldCondition(key="accession_no", match=m.MatchAny(any=list(accession_nos))))
        if accession_no:
            must.append(m.FieldCondition(key="accession_no", match=m.MatchValue(value=accession_no)))
        return m.Filter(must=must) if must else None

    # ---- API ----
    def upsert(self, rows: list[VecRow]) -> None:
        from qdrant_client import models

        if not rows:
            return
        self._ensure(len(rows[0].vector))
        for i in range(0, len(rows), BATCH):
            self.client.upsert(COLLECTION, wait=True, points=[
                models.PointStruct(id=self._pid(r.uid), vector=np.asarray(r.vector, dtype=np.float32).tolist(),
                                   payload=r.payload) for r in rows[i:i + BATCH]])

    def delete_filing(self, accession_no: str) -> None:
        from qdrant_client import models

        if not self._exists():
            return
        self.client.delete(COLLECTION, wait=True,
                           points_selector=models.FilterSelector(filter=self._filter(accession_no=accession_no)))

    def count(self, accession_no: str | None = None) -> int:
        if not self._exists():
            return 0
        return int(self.client.count(COLLECTION, exact=True,
                                     count_filter=self._filter(accession_no=accession_no)).count)

    def accessions(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if not self._exists():
            return out
        offset = None
        while True:
            pts, offset = self.client.scroll(COLLECTION, limit=1000, offset=offset, with_vectors=False,
                                             with_payload=["accession_no"])
            for p in pts:
                a = (p.payload or {}).get("accession_no")
                if a:
                    out[a] = out.get(a, 0) + 1
            if offset is None:
                return out

    def search(self, qvec, *, ticker=None, form_types=None, fiscal_year=None, sections=None,
               accession_nos=None, limit=40) -> list[tuple[str, float]]:
        if not self._exists():
            return []
        res = self.client.query_points(
            COLLECTION, query=np.asarray(qvec, dtype=np.float32).tolist(), limit=limit, with_payload=["uid"],
            query_filter=self._filter(ticker, form_types, fiscal_year, sections, accession_nos))
        return [((p.payload or {}).get("uid"), float(p.score)) for p in res.points if (p.payload or {}).get("uid")]

    def status(self) -> dict:
        try:
            return {"backend": "qdrant", "ok": True, "detail": f"{self.where} - collection '{COLLECTION}', "
                                                              f"{self.count()} points"}
        except Exception as e:
            return {"backend": "qdrant", "ok": False, "detail": f"{self.where}: {e}"}


# ═════════════════════════════════════════ selection, fallback, sync ═════════════════════════════════════════
_lock = threading.Lock()
_backend = None
_fallback = SqliteBackend()
_down_until = 0.0          # after a failure, use the SQLite fallback until this time (seconds since epoch)
_last_error = ""
_synced_sig = None


def _build_backend():
    kind = (os.getenv("VECTOR_BACKEND") or "sqlite").strip().lower()
    if kind in ("", "sqlite", "numpy", "none"):
        return _fallback
    if kind == "qdrant":
        url = (os.getenv("QDRANT_URL") or "").strip() or None
        key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
        path = (os.getenv("QDRANT_PATH") or "").strip() or None
        return QdrantBackend(url=url, api_key=key, path=path)
    raise ValueError(f"Unknown VECTOR_BACKEND '{kind}' (use sqlite or qdrant)")


def get_backend():
    global _backend, _last_error
    with _lock:
        if _backend is None:
            try:
                _backend = _build_backend()
                log.info("Vector backend: %s", _backend.name)
            except Exception as e:  # never stop the app because the vector DB is down
                _last_error = f"{type(e).__name__}: {e}"
                log.warning("Vector backend unavailable (%s) - using SQLite vectors", _last_error)
                _backend = _fallback
        return _backend


def reset_backend() -> None:
    """Forget the chosen backend (used by tests and after changing settings)."""
    global _backend, _down_until, _synced_sig, _last_error
    with _lock:
        _backend, _down_until, _synced_sig, _last_error = None, 0.0, None, ""
        _fallback._cache.clear()


def _mark_down(err: Exception) -> None:
    global _down_until, _last_error
    _last_error = f"{type(err).__name__}: {err}"
    _down_until = time.time() + 60
    log.warning("Vector database error (%s) - using SQLite vectors for 60 s", _last_error)


def search(qvec, **filters) -> list[tuple[str, float]]:
    """Dense search with metadata filters. Falls back to SQLite vectors if the vector DB fails."""
    be = get_backend()
    if be is not _fallback and time.time() >= _down_until:
        try:
            ensure_synced()
            return be.search(qvec, **filters)
        except Exception as e:
            _mark_down(e)
    return _fallback.search(qvec, **filters)


def index_filing(accession_no: str) -> int:
    """(Re)write one filing's vectors into the index. Safe to call any time; best effort."""
    be = get_backend()
    if be is _fallback:
        return 0
    try:
        rows = _rows_for_filing(accession_no)
        be.delete_filing(accession_no)  # drop points of an earlier version of this filing
        be.upsert(rows)
        return len(rows)
    except Exception as e:
        _mark_down(e)
        return 0


def delete_filing(accession_no: str) -> None:
    be = get_backend()
    if be is _fallback:
        return
    try:
        be.delete_filing(accession_no)
    except Exception as e:
        _mark_down(e)


def sync(force: bool = False) -> dict:
    """Make the index match SQLite: add missing filings, fix wrong counts, remove deleted filings."""
    global _synced_sig
    be = get_backend()
    report = {"backend": be.name, "indexed": [], "removed": [], "ok": True}
    if be is _fallback:
        return report
    try:
        want = _sqlite_counts()
        have = be.accessions()
        for acc, n in want.items():
            if n and (force or have.get(acc, 0) != n):
                report["indexed"].append((acc, index_filing(acc)))
        for acc in have:
            if acc not in want:
                be.delete_filing(acc)
                report["removed"].append(acc)
        _synced_sig = _SIG()
    except Exception as e:
        _mark_down(e)
        report["ok"] = False
        report["error"] = str(e)
    return report


def _SIG():
    return SqliteBackend._signature()


def ensure_synced() -> None:
    """Cheap check done before searching: if SQLite changed since the last check, compare totals and repair."""
    global _synced_sig
    sig = _SIG()
    if sig == _synced_sig:
        return
    be = get_backend()
    if be is _fallback:
        return
    if be.count() != sig[0]:
        sync()
    _synced_sig = sig


def status() -> dict:
    be = get_backend()
    st = be.status()
    st["sqlite_chunks"] = SqliteBackend().count()
    if be is _fallback:
        st["wanted"] = (os.getenv("VECTOR_BACKEND") or "sqlite").lower()
    st["fallback_active"] = time.time() < _down_until
    if _last_error:
        st["last_error"] = _last_error
    return st


def per_filing_report() -> list[dict]:
    """SQLite chunk count vs index count for every filing (admin view)."""
    want = _sqlite_counts()
    try:
        have = get_backend().accessions()
    except Exception as e:
        _mark_down(e)
        have = {}
    be_is_sqlite = get_backend() is _fallback
    return [{"accession_no": a, "sqlite_chunks": n,
             "indexed_chunks": n if be_is_sqlite else have.get(a, 0),
             "in_sync": be_is_sqlite or have.get(a, 0) == n} for a, n in sorted(want.items())]
