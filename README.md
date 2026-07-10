# 📊 Financial Research Assistant — RAG over SEC Filings

A Retrieval-Augmented Generation (RAG) chatbot that answers deep financial-analysis
questions (revenue trends, risk factors, competitive positioning, supply chain, etc.)
grounded strictly in real SEC filings (10-Ks) — currently loaded with NVIDIA, Tesla, Meta, Oracle and Apple.

Built as a hands-on project to go beyond "basic RAG" and implement production-grade
retrieval techniques: hybrid search, reranking, corrective RAG, and automated
faithfulness scoring.

---

## 🚀 What it does

- Ingests SEC filing PDFs (text **and** tables) and chunks them semantically
- Retrieves context using **hybrid search**: dense embeddings + BM25 keyword search
- **Reranks** candidates with a cross-encoder, with a boost for table content
- Runs **Corrective RAG**: automatically widens the search if initial retrieval quality is low
- Generates grounded answers with **Llama-3.1-8B-Instruct** (4-bit quantized)
- Scores every answer's faithfulness to its source context with **BERTScore**
- Maintains **per-company, per-session conversation memory**
- Serves everything through a clean **Gradio** chat UI

---

## 🧠 What I learned building this

- How to design a **hybrid retrieval pipeline** (semantic + BM25) instead of relying on
  vector similarity alone, and why keyword search still matters for numbers/tickers/names
  that embeddings can blur together
- The value of a **reranking stage** (cross-encoder) as a second, more precise filter
  after a cheap high-recall retrieval pass
- How to implement **Corrective RAG** — detecting weak retrieval and automatically
  re-querying with a wider net instead of silently generating a low-quality answer
- Why **table extraction** needs a separate code path from plain text (tables lose all
  meaning if flattened the wrong way)
- How to evaluate RAG output quality with **BERTScore** as a grounding/faithfulness
  proxy, rather than trusting the LLM's output at face value
- Practical trade-offs of running a **4-bit quantized 8B LLM** locally (memory vs. speed
  vs. quality) instead of calling a hosted API
- How to structure a notebook-first ML project into clean, reusable, deployable
  **Python modules** (`model.py` for the pipeline, `app.py` for the UI) instead of one
  giant notebook

---

## 🏗️ Architecture

```
PDF Filings (10-Ks)
      │
      ▼
Text + Table Extraction (PyMuPDF + pdfplumber)
      │
      ▼
Semantic Chunking (LangChain SemanticChunker)
      │
      ▼
Embeddings (BAAI/bge-base-en-v1.5) → Chroma Vector Store
      │
      ▼
Hybrid Retrieval (Semantic + BM25)
      │
      ▼
Cross-Encoder Reranking + Table Boost
      │
      ▼
Corrective RAG (quality check → widen search if needed)
      │
      ▼
Llama-3.1-8B-Instruct (4-bit) → Grounded Answer
      │
      ▼
BERTScore Faithfulness Check
      │
      ▼
Gradio Chat UI
```

---

## 🛠️ Tech Stack

- **LangChain** / **LangChain-Chroma** / **LangChain-HuggingFace** — orchestration
- **Chroma** — vector database
- **BAAI/bge-base-en-v1.5** — embedding model
- **rank-bm25** — keyword retrieval
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — reranking
- **Llama-3.1-8B-Instruct** (4-bit, via `bitsandbytes`) — generation
- **bert-score** — faithfulness evaluation
- **Gradio** — chat UI
- **PyMuPDF** / **pdfplumber** — PDF text & table extraction

---

## 📂 Project Structure

```
├── app.py                                    # Gradio UI
├── model.py                                  # RAG pipeline (ingestion, retrieval, generation)
├── requirements.txt
├── financial-assistant-using-rag.ipynb       # Original research/development notebook
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/financial-research-assistant-rag.git
cd financial-research-assistant-rag
pip install -r requirements.txt
```



> **Note:** Requires a CUDA-capable GPU with enough VRAM to run an 8B model in 4-bit
> (roughly 6–8GB+). On CPU-only machines, swap `LLM_MODEL_ID` for a smaller model.

---



## 📄 License

MIT — feel free to fork and build on this.
