"""
model.py
--------
Core RAG pipeline for the Financial Research Assistant:
  - PDF ingestion (text + tables) from SEC filings
  - Semantic chunking
  - Embedding + Chroma vector store (built once, then persisted/reloaded)
  - Hybrid retrieval (semantic + BM25) with cross-encoder reranking
  - Corrective RAG (widen search if retrieval quality is low)
  - Local LLM (Llama-3.1-8B-Instruct, 4-bit) for answer generation
  - BERTScore grounding/faithfulness metric

app.py imports `financial_rag_with_bertscore` (and a couple of helpers) from
this module and wraps them in a Gradio UI.
"""

import os
import re
import gc
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import fitz  # PyMuPDF
import pdfplumber

from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# All of these can be overridden with environment variables so the app is
# portable outside of the original Kaggle notebook environment.

PDF_DIR = os.environ.get("PDF_DIR", "./data/SEC Filings")
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./financial_db")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# NOTE: Never hardcode your HuggingFace token in source code.
# Set it as an environment variable before running:
#   export HF_TOKEN="hf_xxx"           (Linux/Mac)
#   setx HF_TOKEN "hf_xxx"             (Windows)
# or put it in a local .env file that is excluded via .gitignore.
HF_TOKEN = os.environ.get("HF_TOKEN")


def hf_login():
    """Log in to HuggingFace Hub using a token from the environment."""
    if not HF_TOKEN:
        print("⚠️  HF_TOKEN not set. Skipping HuggingFace login "
              "(gated models like Llama-3.1 will fail to download).")
        return
    from huggingface_hub import login
    login(HF_TOKEN)
    print("✅ Logged into HuggingFace")


# ---------------------------------------------------------------------------
# Globals populated by initialize()
# ---------------------------------------------------------------------------
pdf_files = []
all_documents = []
chunks = []
embedding_model = None
vectorstore = None
llm = None
chat_llm = None
cross_encoder = None
_memory_store = {}
_bert_log = []


# ---------------------------------------------------------------------------
# 1. PDF ingestion (text + tables)
# ---------------------------------------------------------------------------
def load_pdfs(pdf_dir: str = PDF_DIR):
    """Extract text and table blocks from every PDF in pdf_dir."""
    global pdf_files, all_documents

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")

    all_documents = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if len(text) < 30:
                continue

            # Text block
            all_documents.append(Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "file_name": pdf_path.name,
                    "element_type": "Text",
                    "page_number": page_num + 1,
                }
            ))

            # Table extraction
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    plumber_page = pdf.pages[page_num]
                    tables = plumber_page.extract_tables()
                    for idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            table_text = "\n".join(
                                [" | ".join(str(cell) if cell is not None else "" for cell in row)
                                 for row in table]
                            )
                            all_documents.append(Document(
                                page_content=table_text,
                                metadata={
                                    "source": str(pdf_path),
                                    "file_name": pdf_path.name,
                                    "element_type": "Table",
                                    "page_number": page_num + 1,
                                    "table_index": idx,
                                }
                            ))
            except Exception:
                continue

        doc.close()

    print(f"\n✅ Extraction complete!")
    print(f"Total Documents : {len(all_documents)}")
    print(f"Text Blocks     : {sum(1 for d in all_documents if d.metadata['element_type'] == 'Text')}")
    print(f"Tables          : {sum(1 for d in all_documents if d.metadata['element_type'] == 'Table')}")

    all_documents = [_to_okf_concept(d) for d in all_documents]
    print(f"✅ Wrapped {len(all_documents)} documents as OKF concept blocks (frontmatter + content)")
    return all_documents


def _to_okf_concept(doc: Document) -> Document:
    """Attach a small YAML frontmatter header describing each chunk."""
    company_guess = doc.metadata.get("file_name", "Unknown").split(".")[0].replace("_", " ")
    frontmatter = {
        "type": "FinancialTable" if doc.metadata.get("element_type") == "Table" else "FinancialText",
        "company": company_guess,
        "source_file": doc.metadata.get("file_name", "Unknown"),
        "page": doc.metadata.get("page_number", "?"),
    }
    fm_str = yaml.dump(frontmatter, sort_keys=False)
    doc.page_content = f"---\n{fm_str}---\n\n{doc.page_content.strip()}"
    return doc


# ---------------------------------------------------------------------------
# 2. Embeddings + Semantic Chunking
# ---------------------------------------------------------------------------
def load_embedding_model():
    global embedding_model
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded!")
    return embedding_model


def build_chunks(documents):
    """GPU-optimized semantic chunking of text docs; tables are kept whole."""
    global chunks
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings

    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    gc.collect()

    chunk_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 4,
        }
    )

    table_docs = [doc for doc in documents if doc.metadata.get("element_type") == "Table"]
    text_docs = [doc for doc in documents if doc.metadata.get("element_type") == "Text"]

    semantic_splitter = SemanticChunker(
        embeddings=chunk_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    print("🔄 Performing semantic chunking...")
    text_chunks = semantic_splitter.split_documents(text_docs)

    chunks = table_docs + text_chunks

    del chunk_embeddings, semantic_splitter
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"✅ Done! Tables: {len(table_docs)} | Text Chunks: {len(text_chunks)} | Total: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# 3. Vector store
# ---------------------------------------------------------------------------
def build_vectorstore(doc_chunks, persist_directory: str = PERSIST_DIR):
    global vectorstore
    from langchain_chroma import Chroma

    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    print("✅ Vector Store Created and Persisted")
    return vectorstore


def load_vectorstore(persist_directory: str = PERSIST_DIR):
    """Reload an already-persisted Chroma store instead of rebuilding it."""
    global vectorstore
    from langchain_chroma import Chroma

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
    print("✅ Vector Store Loaded from disk")
    return vectorstore


# ---------------------------------------------------------------------------
# 4. LLM (Llama-3.1-8B-Instruct, 4-bit quantized)
# ---------------------------------------------------------------------------
def load_llm(model_id: str = LLM_MODEL_ID):
    global llm, chat_llm
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1700,
        temperature=0.4,
        top_p=0.92,
        do_sample=True,
        repetition_penalty=1.15,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    chat_llm = ChatHuggingFace(llm=llm)

    print(f"{model_id} loaded (4-bit).")
    return llm, chat_llm


def load_cross_encoder():
    global cross_encoder
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("✅ Cross-encoder loaded")
    return cross_encoder


# ---------------------------------------------------------------------------
# 5. Memory / text utilities
# ---------------------------------------------------------------------------
def get_memory(company=None, session_id="default"):
    key = (company.lower().strip() if company else None, session_id)
    if key not in _memory_store:
        _memory_store[key] = InMemoryChatMessageHistory()
    return _memory_store[key]


def _strip_sec_header(text: str) -> str:
    if "[SEC FILING DATA]" not in text:
        return text.strip()
    parts = text.split("---\n", maxsplit=2)
    return parts[2].strip() if len(parts) >= 3 else text.strip()


def _clean_text(text):
    markers = ["<think>", "</think>", "**Final Answer**", "Final Answer:", "Changes made:"]
    for m in markers:
        if m in text:
            text = text.split(m)[0]
    return re.sub(r'\n+(I have|Note that|Please note).*', '', text,
                  flags=re.IGNORECASE | re.DOTALL).strip()


# ---------------------------------------------------------------------------
# 6. Hybrid retrieval (semantic + BM25) + reranking + corrective RAG
# ---------------------------------------------------------------------------
def hybrid_retrieval(query, vs, company=None, k=40):
    from rank_bm25 import BM25Okapi

    # 1. Semantic search
    results = vs.similarity_search_with_score(query, k=k * 3 if company else k)

    semantic_list = []
    for doc, score in results:
        if company and company.lower() not in doc.metadata.get("source", "").lower():
            continue
        semantic_list.append((doc, 1.0 / (1.0 + score)))

    # 2. BM25 keyword search
    all_data = vs._collection.get(include=["documents", "metadatas"])
    filtered_texts, filtered_metas = [], []

    for text, meta in zip(all_data["documents"], all_data["metadatas"]):
        if company and company.lower() not in str(meta.get("source", "")).lower():
            continue
        filtered_texts.append(_strip_sec_header(text))
        filtered_metas.append(meta)

    bm25_list = []
    if filtered_texts:
        bm25 = BM25Okapi([t.lower().split() for t in filtered_texts])
        scores = bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:k]
        for i in top_idx:
            if scores[i] > 0:
                score_norm = scores[i] / max(scores.max(), 1)
                doc = SimpleNamespace(page_content=filtered_texts[i], metadata=filtered_metas[i])
                bm25_list.append((doc, score_norm))

    # Merge semantic + BM25
    merged = {id(d[0]): d for d in semantic_list}
    for doc, score in bm25_list:
        merged[id(doc)] = (doc, merged.get(id(doc), (None, 0))[1] + score * 0.7)

    return sorted(merged.values(), key=lambda x: x[1], reverse=True)[:k]


def rerank_with_cross_encoder(query, candidates, top_n=15):
    if not candidates:
        return []
    docs = [pair[0] for pair in candidates]
    texts = [_strip_sec_header(d.page_content) for d in docs]
    scores = cross_encoder.predict([[query, t] for t in texts])
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]


def multimodal_boost(reranked_pairs):
    boosted = []
    for doc, score in reranked_pairs:
        new_score = float(score)
        if doc.metadata.get("element_type") == "Table":
            new_score += 0.25
        boosted.append((doc, new_score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)


def evaluate_retrieval_quality(query, docs):
    if not docs or len(docs) < 3:
        return False
    combined_text = " ".join(
        _strip_sec_header(doc.page_content)[:1000] for doc, _ in docs[:6]
    ).lower()
    query_words = [w for w in query.lower().split() if len(w) > 3]
    if not query_words:
        return True
    overlap = sum(1 for w in query_words if w in combined_text)
    required_overlap = max(2, len(query_words) // 3)
    print(f"   Retrieval Quality: {overlap}/{required_overlap} words matched")
    return overlap >= required_overlap


def _build_history_text(memory, max_turns=3):
    msgs = memory.messages
    pairs = []
    i = 0
    while i < len(msgs) - 1:
        if msgs[i].type == "human" and msgs[i + 1].type == "ai":
            pairs.append((msgs[i].content, msgs[i + 1].content))
            i += 2
        else:
            i += 1
    recent = pairs[-max_turns:]
    if not recent:
        return ""
    lines = ["Previous conversation:"]
    for turn_idx, (q, a) in enumerate(reversed(recent), 1):
        short_a = a[:500] + "…" if len(a) > 500 else a
        lines.append(f"\n[Turn {turn_idx}] User: {q}")
        lines.append(f"AI: {short_a}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Main RAG entrypoint
# ---------------------------------------------------------------------------
def financial_rag(query: str, company: str = None, session_id: str = "default"):
    global vectorstore, embedding_model, llm

    if not all([vectorstore, embedding_model, llm]):
        return "❌ Error: vectorstore, embedding_model or llm not initialized."

    memory = get_memory(company, session_id)

    query_preview = (query.strip()[:60] + "...") if len(query.strip()) > 60 else query.strip()
    print(f"🔍 Company: {company or 'All'} | Query: {query_preview}")

    # ── Retrieval ──────────────────────────────────────────────────────────
    candidates = hybrid_retrieval(query, vectorstore, company=company, k=50)
    reranked = rerank_with_cross_encoder(query, candidates, top_n=12)
    reranked = multimodal_boost(reranked)

    # Corrective RAG
    if not evaluate_retrieval_quality(query, reranked):
        print(" Corrective RAG triggered — widening search...")
        candidates = hybrid_retrieval(query, vectorstore, company=company, k=80)
        reranked = rerank_with_cross_encoder(query, candidates, top_n=15)
        reranked = multimodal_boost(reranked)

    # Filter & cap
    filtered_docs = [
        (doc, score) for doc, score in reranked
        if not company or company.lower() in str(doc.metadata.get("source", "")).lower()
    ][:16]

    if len(filtered_docs) < 3:
        return f"❌ Not enough relevant information found for '{company}'."

    context = "\n\n---\n\n".join(_strip_sec_header(doc.page_content) for doc, _ in filtered_docs)

    # ── Pass 1: Generate Response ───────────────────────────────────────────
    print("📝 Pass 1: Generating Response")
    pass1_prompt = f"""You are a Senior Institutional Financial Analyst.

**Instructions:**
- Use ONLY the context provided below.
- Never hallucinate numbers, facts, or events.
- Every financial figure must be directly from the context.
- Write in clear, professional, institutional tone.

**Context:**
{context}

**Question:**
{query}

**Output Format:**
Provide a structured, concise, and accurate financial analysis.

Financial Analysis:"""

    raw_pass1 = llm.invoke(pass1_prompt)
    final_response = raw_pass1.content if hasattr(raw_pass1, "content") else str(raw_pass1)
    final_response = _clean_text(final_response)

    # Memory
    memory.add_user_message(query)
    memory.add_ai_message(final_response)

    # Sources
    sources = [
        f"{doc.metadata.get('file_name', 'Unknown')} | Page {doc.metadata.get('page_number', '?')}"
        for doc, _ in filtered_docs
    ]
    unique_sources = list(dict.fromkeys(sources))

    final_output = (
        f"# Financial Analysis — {company or 'All Companies'}\n\n"
        + final_response
        + "\n\n## Sources\n"
        + "\n".join(f"- {s}" for s in unique_sources)
    )

    # Debug metadata
    financial_rag._last_context = context
    financial_rag._last_sources = unique_sources
    financial_rag._debug = {
        "initial_docs": len(candidates),
        "final_docs": len(filtered_docs),
        "corrective_triggered": not evaluate_retrieval_quality(query, reranked),
    }

    return final_output


# ---------------------------------------------------------------------------
# 8. BERTScore grounding metric
# ---------------------------------------------------------------------------
def compute_bertscore(reference_text: str, candidate_text: str):
    """
    BERTScore of candidate_text against reference_text.
    reference = retrieved source context, candidate = generated answer.

    High F1  -> answer's meaning is well supported by the retrieved context
    Low F1   -> answer may be drifting from / hallucinating beyond the context
    """
    from bert_score import score as bert_score

    if not reference_text.strip() or not candidate_text.strip():
        return 0.0, 0.0, 0.0

    ref = reference_text[:4000]
    cand = candidate_text[:4000]

    P, R, F1 = bert_score(
        [cand], [ref],
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    return P.item(), R.item(), F1.item()


def financial_rag_with_bertscore(query: str, company: str = None, session_id: str = "default"):
    """Wrapper around financial_rag() that also computes/logs a BERTScore."""
    response = financial_rag(query, company=company, session_id=session_id)

    context = getattr(financial_rag, "_last_context", "")
    precision, recall, f1 = compute_bertscore(context, response) if context else (0.0, 0.0, 0.0)

    print(f"📊 BERTScore — Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    _bert_log.append({
        "query": query.strip()[:80],
        "company": company or "All",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    })

    return response


# Backward-compatible alias
financial_rag_with_bleu = financial_rag_with_bertscore


def show_bertscore_log():
    """Pretty-print the BERTScore for every query run so far."""
    if not _bert_log:
        print("No queries logged yet.")
        return
    print(f"{'#':<3} {'Company':<10} {'Precision':<10} {'Recall':<10} {'F1':<8} Query")
    print("-" * 100)
    for i, entry in enumerate(_bert_log, 1):
        print(f"{i:<3} {entry['company']:<10} {entry['precision']:<10} {entry['recall']:<10} {entry['f1']:<8} {entry['query']}")
    avg_p = sum(e['precision'] for e in _bert_log) / len(_bert_log)
    avg_r = sum(e['recall'] for e in _bert_log) / len(_bert_log)
    avg_f1 = sum(e['f1'] for e in _bert_log) / len(_bert_log)
    print("-" * 100)
    print(f"Average -> Precision: {avg_p:.4f} | Recall: {avg_r:.4f} | F1: {avg_f1:.4f}")


# ---------------------------------------------------------------------------
# 9. One-shot initializer used by app.py
# ---------------------------------------------------------------------------
def initialize(pdf_dir: str = PDF_DIR,
               persist_dir: str = PERSIST_DIR,
               force_rebuild: bool = False):
    """
    Set up the full pipeline:
      - HF login
      - embedding model
      - vector store (rebuilt from PDFs, or reloaded from persist_dir)
      - LLM + cross-encoder

    Call this once at app startup, e.g. in app.py before demo.launch().
    """
    hf_login()
    load_embedding_model()

    if not force_rebuild and Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        load_vectorstore(persist_dir)
    else:
        docs = load_pdfs(pdf_dir)
        doc_chunks = build_chunks(docs)
        build_vectorstore(doc_chunks, persist_dir)

    load_llm()
    load_cross_encoder()

    print("✅ model.py initialization complete — RAG pipeline is ready")
