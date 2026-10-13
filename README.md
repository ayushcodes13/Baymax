# Baymax

A Flask + LangChain RAG web app that lets you chat with your own documents
(PDFs). It retrieves relevant passages from a Pinecone vector store and
generates responses through Groq's Llama 3.

> **Scope:** This is a wellbeing conversation prototype, **not** a medical
> product. It does not diagnose, treat, or replace consultation with a
> qualified health professional. The LLM may produce inaccurate or
> outdated information — always verify with authoritative sources.

---

## Architecture

```
User → Flask web UI (chat.html)
              ↓
        /get endpoint
              ↓
   RAG chain (LangChain)
      ├─ Retriever: Pinecone vector store (similarity search, k=3)
      └─ Generator: ChatGroq (llama3-70b-8192)
              ↓
        Retrieved context + reply
```

## Quick start

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/ayushcodes13/Baymax.git
cd Baymax
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Only the runtime requirements are needed for the web app:

```bash
pip install -r requirements.txt
```

For experimental audio/vision/model modules (STT, TTS, BioGPT):

```bash
pip install -r requirements-research.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your PINECONE_API_KEY and GROQ_API_KEY
```

### 4. Build the vector index

Place your PDF documents in `data/` (or specify another directory):

```bash
mkdir -p data
# copy your PDFs into data/
python3 store_index.py --data-dir data --index-name baymax-index
```

This will:
- Load all PDFs from the directory
- Split them into 500-token chunks
- Embed them with `sentence-transformers/all-MiniLM-L6-v2`
- Create or update the Pinecone index `baymax-index`

### 5. Run the app

```bash
python3 app.py
```

Open `http://localhost:8080` in your browser.

---

## Project structure

| File | Description |
|------|-------------|
| `app.py` | Flask server and RAG chain entry point |
| `store_index.py` | Builds the Pinecone vector index from PDFs |
| `src/helper.py` | PDF loading, text splitting, embedding setup |
| `src/prompt.py` | System prompt with wellbeing boundaries |
| `templates/chat.html` | Web chat UI |
| `static/style.css` | Chat UI styling |
| `convert_pdf_to_txt.py` | Utility to batch-convert PDFs to text |
| `requirements.txt` | Runtime dependencies (Flask, LangChain, Pinecone, Groq) |
| `requirements-research.txt` | Heavy optional deps (torch, whisper, vosk, pygame) |

## Data handling

- API keys are loaded from environment variables / `.env` and are never
  committed to version control.
- PDF documents in `data/` are not shared; they are processed locally and
  embedded into your own Pinecone index.
- Chat messages are sent to Groq for reply generation. Do not enter
  sensitive personal or health information.
- No chat history is persisted to a database.

## Safety & limitations

- This prototype is **not** therapy, diagnosis, medical advice, or a
  clinical service.
- The LLM generates responses from a general-purpose model. Content may be
  inaccurate, outdated, or hallucinated.
- Retrieved documents inform the answer but are not independently verified.
- There is no authentication, audit logging, or production privacy review.

## License

Apache 2.0. See [LICENSE](./LICENSE).
