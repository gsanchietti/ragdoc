# ragdoc

Supervisor-driven RAG on Python 3.12. This repo includes fetching from HTTP and Git with caching, text extraction, embedding and indexing into Postgres+pgvector, a hybrid retriever with a test CLI, and a LangGraph agent with clarify → retrieve → answer/escalate.

## Features
- HTTP fetch with conditional requests (If-None-Match / If-Modified-Since)
- Git clone/update using system `git` (shallow by default)
- JSON-based local state cache under `.ragdoc/state.json`
- Deterministic output directory structure under `data/`
- Simple YAML config to declare sources
- Text extraction and normalization for HTML, Markdown, plain text, and reStructuredText
- Embedding via OpenAI (default `text-embedding-3-small`) and indexing in Postgres with `pgvector`
- Idempotent upsert with provenance+content hash de-duplication
- Hybrid Retriever (vector + simple lexical merge)
- Test CLI `ragdoc-test query` for similarity search
- LangGraph agent: clarify → retrieve → answer or escalate, with confidence threshold

## Quickstart (with uv)

1) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Create venv and install package

```bash
uv venv
source .venv/bin/activate
uv pip install -U pip
uv pip install -e .
```

3) Configure sources

Edit `configs/sources.yaml`:

```yaml
http:
  - url: https://example.com/
    out_dir: data/raw/http
  - url: https://raw.githubusercontent.com/psf/requests/main/README.md
    out_dir: data/raw/http

git:
  - repo: https://github.com/python/cpython.git
    ref: main           # optional; default is remote HEAD
    shallow: true       # optional; default true
    out_dir: data/repos # optional; defaults to data/repos
```

4) Run fetch + index job (requires Postgres + OpenAI key)

```bash
export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/ragdoc"
export OPENAI_API_KEY="sk-..."
ragdoc-fetch --config configs/sources.yaml \
  --root data/repos \
  --glob "**/*.md"
```

Outputs will be placed under `data/` and state under `.ragdoc/state.json`.

5) Test retrieval via CLI

```bash
ragdoc-test query \
  --dsn "$DATABASE_URL" \
  --k 8 \
  --alpha 0.7 \
  --query "How do I configure pgvector?"
```

6) Run the chat (LangGraph prebuilt UI)

```bash
export RAGDOC_CONFIDENCE_THRESHOLD=0.65   # optional
export RAGDOC_RETRIEVAL_TOP_K=8           # optional
langgraph dev apps/chat/graph.py
```

## Environment
- Python: 3.12
- Dependencies: `httpx`, `PyYAML`, `beautifulsoup4`, `openai`, `psycopg[binary]`, `pgvector`, `docutils`, `langgraph`, `langchain-core`, `langchain-openai`
- A Postgres instance with the `vector` extension is required for indexing. Fetch-only still works without DB (files + local state).

### Configuration (env)
- `DATABASE_URL` e.g. `postgresql+psycopg://user:pass@host:5432/ragdoc`
- `OPENAI_API_KEY`
- `RAGDOC_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `RAGDOC_EMBEDDING_DIM` (default: `1536`)
- `RAGDOC_RETRIEVAL_TOP_K` (default: `8`)
- `RAGDOC_CONFIDENCE_THRESHOLD` (default: `0.65`)
- `RAGDOC_ANSWER_MODEL` (default: `gpt-4o-mini`)

## Run Postgres + pgvector in a container

Use a ready image that ships the pgvector extension.

Docker:

```bash
docker run --name ragdoc-pg \
  -e POSTGRES_USER=ragdoc \
  -e POSTGRES_PASSWORD=ragdoc \
  -e POSTGRES_DB=ragdoc \
  -p 5432:5432 \
  -v ragdoc-pgdata:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16

export DATABASE_URL="postgresql+psycopg://ragdoc:ragdoc@127.0.0.1:5432/ragdoc"
```

Podman:

```bash
podman run --rm --name ragdoc-pg \
  -e POSTGRES_USER=ragdoc \
  -e POSTGRES_PASSWORD=ragdoc \
  -e POSTGRES_DB=ragdoc \
  -p 5432:5432 \
  -v ragdoc-pgdata:/var/lib/postgresql/data \
  -d docker.io/pgvector/pgvector:pg17

export DATABASE_URL="postgresql+psycopg://ragdoc:ragdoc@127.0.0.1:5432/ragdoc"
```

Notes:
- The CLI also attempts `CREATE EXTENSION IF NOT EXISTS vector` on startup; using the pgvector image ensures it succeeds.
- Adjust ports/credentials as needed; the example uses a named volume `ragdoc-pgdata` for persistence.

## Notes
- The job avoids re-downloading unchanged HTTP resources using ETag/Last-Modified when available.
- Git operations use the local `git` executable; ensure it’s installed and available on PATH.
- Network credentials or private repo access are not handled in this minimal version.
 - Vectorization uses OpenAI embeddings. See env vars to customize model/dim.
 - Postgres must have the `vector` extension available. The table `ragdoc_embeddings` will be created if missing, with a unique constraint on (source_type, source_url, path, content_sha256).
 - The agent asks one clarifying question when needed and loops back to retrieval. It routes to answer vs. escalate based on `RAGDOC_CONFIDENCE_THRESHOLD`.

## Roadmap (next steps)
- Add PDF parsing
- Upgrade lexical retrieval to Postgres FTS (tsvector/ts_rank) and add recency/domain filters
- Confidence calibration and evaluation harness
- Containerization for the agent service and healthcheck endpoint
