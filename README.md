# FastAPI RAG Microservice (local)
# Repository structure (single-file presentation for convenience)
# README.md
FastAPI RAG microservice (ingestion + query) - local-ready


Features:
- Upload PDF/CSV/XLSX via /ingest endpoint (admin) -> extracts text, chunks, creates embeddings, stores metadata
- Stores chunk metadata in SQLite (SQLAlchemy)
- Stores vectors in FAISS index on disk
- Query endpoint /query: embed query, nearest neighbor search, threshold check
- Optional Ollama LLM call if OLLAMA_URL configured; otherwise returns concatenated retrieved context and a conservative response that refuses if below threshold


Run locally:
1. python -m venv .venv
2. source .venv/bin/activate (or .venv\Scripts\activate on Windows)
3. pip install -r requirements.txt
4. uvicorn app.main:app --reload --port 8001


Env variables:
- EMBED_MODEL (optional) default: sentence-transformers/all-MiniLM-L6-v2
- VECTOR_DIM (optional) default: 384
- OLLAMA_URL (optional) e.g. http://localhost:11434/api/generate
- THRESHOLD (optional) similarity threshold between 0 and 1 (default: 0.60)


API:
- POST /ingest (multipart form): file -> returns file_id
- POST /query (json): {"question": "...", "top_k": 5} -> returns answer + sources
- GET /status -> basic status


Notes:
- This is a minimal but complete example for local dev. For production, swap FAISS for pgvector or Milvus, secure endpoints (auth), use object storage for files, run embedding service separate, etc.