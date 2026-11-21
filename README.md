# 📄 AI-Powered PDF Assistant Using RAG

An intelligent, production-ready document Q&A assistant built with **Retrieval-Augmented Generation (RAG)**, local dense vector embeddings, persistent storage with **ChromaDB**, ultra-fast inference via the **Groq Cloud API (`openai/gpt-oss-120b`)**, and a modern **Streamlit** user interface.

---

## 🌟 Key Highlights

- **Strict Grounding & Zero Hallucinations**: Answers questions strictly based on uploaded PDF facts. Automatically abstains when context is absent or unrelated.
- **Scope & Category Awareness**: Distinguishes general concepts from specific subcategories (e.g. will not confuse the disadvantages of Model-Based Learning with the disadvantages of Machine Learning as a whole).
- **Exact Page-Level Citations**: Every generated fact is accompanied by verified citations showing the source filename, page number, and semantic distance score.
- **High-Speed Inference**: Powered by the **Groq API** with the powerful open-weights `openai/gpt-oss-120b` model.
- **Local & Private Embeddings**: Semantic embeddings (384-d) are generated locally using HuggingFace's `all-MiniLM-L6-v2` via `sentence-transformers` — no external API calls for embedding generation.
- **Persistent Vector Storage**: Chunks and vectors are stored locally in **ChromaDB** with automated duplicate prevention and institutional header noise stripping.
- **Empirical Distance Filtering**: Queries scoring distance $> 1.25$ against indexed chunks are automatically recognized as unrelated, preventing false citations.
- **Modern Responsive Web UI**: Drag-and-drop ingestion, glass-card metric summaries, expandable citation cards, and real-time engine health indicators.

---

## 🏛️ System Architecture

```text
       ┌────────────────────────┐
       │   Uploaded PDF File    │
       └───────────┬────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  1. Text Extraction (PyMuPDF)        │  Page-by-page text parsing
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  2. Text Chunking (LangChain)        │  RecursiveCharacterTextSplitter
│                                      │  (chunk_size=500, overlap=50)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  3. Header Cleaning & Embeddings     │  Institutional boilerplate stripping
│     (all-MiniLM-L6-v2, 384-d)        │  Dense vector representation
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  4. Local Vector Store (ChromaDB)    │  Persistent storage at ./chroma_db
│                                      │  Deduplication by source filename
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  5. Semantic Retrieval (Retriever)   │  Cosine/L2 similarity search
│                                      │  top_k=5, distance_threshold <= 1.25
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  6. Prompt Builder                   │  Scope-aware, citation-grounded RAG
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  7. Answer Generation (Groq API)     │  openai/gpt-oss-120b Chat Completions
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  8. Interactive Frontend (Streamlit) │  Modern chat UI with citation cards
└──────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
AI Powered PDF Assistant Using RAG/
├── app.py                  # Streamlit web application & chat frontend
├── answer_generator.py     # Groq Chat Completions generation module
├── embedding_model.py      # SentenceTransformer (all-MiniLM-L6-v2) singleton
├── llm_client.py           # Groq API client initialization & key loader
├── pdf_processor.py        # PyMuPDF page-by-page extraction
├── prompt_builder.py       # Grounded, scope-aware prompt engineering
├── retriever.py            # ChromaDB vector query & distance filtering
├── text_chunker.py         # Recursive text chunking (500/50)
├── vector_store.py         # ChromaDB management & boilerplate cleaner
├── main.py                 # Comprehensive offline test suite (All PASS)
├── requirements.txt        # Pinned project dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules (protects .env & chroma_db)
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- A free [Groq API Key](https://console.groq.com/keys)

### 2. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai-powered-pdf-assistant-rag.git
   cd ai-powered-pdf-assistant-rag
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure your API Key:**
   Create a `.env` file in the project root and set your Groq key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   ```

---

## 🖥️ Running the Application

Launch the Streamlit web app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. Drag and drop any PDF into the **"Ingest & Index PDF Document"** box.
2. Click **"⚡ Index Document"** — the document is split, embedded, and indexed into ChromaDB.
3. Start asking questions in the chat! Click **"📚 View Grounded Sources & Citations"** under any response to verify the exact page numbers and quotes used.

---

## 🧪 Running the Test Suite

Run the complete regression test suite covering all components:

```powershell
.\.venv\Scripts\python.exe main.py
```

Verified test coverage:
- ✅ **Sprint 1**: PyMuPDF extraction & page metadata
- ✅ **Sprint 2**: Text chunking (500 chars, 50 overlap)
- ✅ **Sprint 3**: ChromaDB connection & persistent storage
- ✅ **Sprint 4**: 384-dimensional dense embeddings
- ✅ **Sprint 5**: Semantic vector retrieval & top-k ranking
- ✅ **Sprint 6**: Groq Chat Completions integration (Mocked)
- ✅ **Fix 1**: Source-based duplicate record prevention
- ✅ **Fix 2**: Grounding & synthesis prompt validation
- ✅ **Sprint 7**: Scope awareness, category distinction, distance threshold filtering, and limitation disclosure

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Interactive web UI with glass-card metrics and citation cards |
| **LLM Inference** | [Groq Cloud API](https://groq.com/) | Ultra-low latency inference using `openai/gpt-oss-120b` |
| **Embeddings** | [Sentence-Transformers](https://sbert.net/) | Local `all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | Persistent local vector store with cosine/L2 distance metric |
| **PDF Extraction**| [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | High-accuracy page-by-page document parser |
| **Chunking** | [LangChain Splitters](https://python.langchain.com/) | `RecursiveCharacterTextSplitter` preserving semantic boundaries |
| **Secrets** | [python-dotenv](https://pypi.org/project/python-dotenv/) | Zero-leak credential management via `.env` |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
