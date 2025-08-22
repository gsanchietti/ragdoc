# ragdoc

Minimal scaffold focused on the Fetch tool job from AGENTS.md. This repository sets up a Python 3.12 package managed with uv and provides a runnable CLI to fetch sources from HTTP and Git with caching (ETag/Last-Modified for HTTP, last commit SHA for Git).

Note: Only the fetch tool is implemented here. Parsing, ingestion, retrieval, and agent graph are out of scope for this step.

## Features
- HTTP fetch with conditional requests (If-None-Match / If-Modified-Since)
- Git clone/update using system `git` (shallow by default)
- JSON-based local state cache under `.ragdoc/state.json`
- Deterministic output directory structure under `data/`
- Simple YAML config to declare sources

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
  --md-root data/repos \
  --md-glob "**/*.md"
```

Outputs will be placed under `data/` and state under `.ragdoc/state.json`.

## Environment
- Python: 3.12
- Dependencies: `httpx`, `PyYAML`, `beautifulsoup4`, `openai`, `psycopg[binary]`, `pgvector`
- No DB required for this step; state is stored locally.

## Notes
- The job avoids re-downloading unchanged HTTP resources using ETag/Last-Modified when available.
- Git operations use the local `git` executable; ensure it’s installed and available on PATH.
- Network credentials or private repo access are not handled in this minimal version.
 - Vectorization uses OpenAI embeddings (`text-embedding-3-small` by default). You can override via env:
   - `RAGDOC_EMBEDDING_MODEL` (default: text-embedding-3-small)
   - `RAGDOC_EMBEDDING_DIM` (default: 1536)
 - Postgres must have the `vector` extension available. The table `ragdoc_embeddings` will be created if missing, with a unique constraint on (source_type, source_url, path, content_sha256).

## Roadmap (next steps)
- Add parsers (HTML/MD/PDF) and content normalization
- Add embedding/indexing to Postgres + pgvector
- Integrate with LangGraph supervisor and chat UI
