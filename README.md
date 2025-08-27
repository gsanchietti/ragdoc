# ragdoc

Supervisor-driven RAG on Python 3.12. This repo includes fetching from HTTP and Git with caching, text extraction, embedding and indexing into ### HTTP Recursive Crawling

All HTTP sources are now crawled recursively using LangChain's RecursiveUrlLoader. This automatically:

- Follows links within the same domain
- Extracts clean text content from HTML pages
- Respects the configured `max_depth` and `exclude_dirs`
- Filters out binary files and external domains automatically

Configure crawling behavior in your `configs/sources.yaml`:

```yaml
http:
  - url: https://docs.example.com/
    out_dir: data/raw/http
    max_depth: 3                    # crawl up to 3 levels deep
    exclude_dirs: ["api", "legacy"] # skip these URL paths
    timeout: 15                     # request timeout in seconds
    link_regex: ".*\\.html$"        # optional: only follow HTML links
```
A hybrid retriever with a test CLI, and a LangGraph agent with clarify → retrieve → answer/escalate.

## Features

- HTTP recursive crawling using LangChain's RecursiveUrlLoader
- Git clone/update using system `git` (shallow by default)
- JSON-based local state cache under `.ragdoc/state.json`
- Deterministic output directory structure under `data/`
- Simple YAML config to declare sources
- Text extraction and normalization for HTML, Markdown, plain text, and reStructuredText
- Embedding via OpenAI (default `text-embedding-3-small`) and indexing in Postgres with `pgvector`
- Idempotent upsert with provenance+content hash de-duplication
- Hybrid Retriever with advanced text search:
  - Vector similarity search using OpenAI embeddings
  - PostgreSQL full-text search (tsvector/ts_rank) with fallback to pattern matching
  - Configurable weighting between vector and lexical scores
  - Title boost for enhanced relevance of document titles
  - Text-only result discovery (finds high-relevance text matches beyond top vector results)
- Test CLI `ragdoc-test` with comprehensive hybrid search and debug logging
- LangGraph agent: router → (intro|clarify|retrieve) → answer or escalate; confidence gating, automatic translation to English for database search, and history-aware retrieval

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
    max_depth: 2          # optional; default 2
    timeout: 10           # optional; default 10 seconds
  - url: https://docs.python.org/3.12/
    out_dir: data/raw/http
    max_depth: 3
    exclude_dirs: ["_downloads", "_static"]  # optional; skip these paths

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
# Basic hybrid search test with BM25 sparse vectors
ragdoc-test query "How do I configure pgvector?" \
  --k 8 \
  --database-url "$DATABASE_URL"

# Advanced hybrid search with custom parameters and debug logging
ragdoc-test query "database configuration" \
  --k 10 \
  --alpha 0.6 \
  --title-boost 2.0 \
  --sparse-weight 0.4 \
  --bm25-k1 1.5 \
  --debug \
  --database-url "$DATABASE_URL"

# Disable BM25 sparse vector search (use only dense vectors + lexical)
ragdoc-test query "machine learning models" \
  --no-bm25 \
  --alpha 0.8 \
  --database-url "$DATABASE_URL"
```

The test CLI always executes hybrid search combining:
- Dense vector similarity search using OpenAI embeddings
- PostgreSQL full-text search with tsvector/ts_rank  
- BM25 sparse vector search for keyword-based matching
- Text-only result discovery for comprehensive coverage
- Configurable three-way scoring balance via `--alpha` and `--sparse-weight` parameters

6) Run the chat (LangGraph Studio prebuilt UI)

```bash
export RAGDOC_CONFIDENCE_THRESHOLD=0.65   # optional
export RAGDOC_RETRIEVAL_TOP_K=8           # optional
export RAGDOC_INFO_CONF_THRESHOLD=0.35    # optional, router confidence to proceed to retrieve
export RAGDOC_LOG_LEVEL=DEBUG             # optional, verbose agent logs

# BM25 Sparse Vector Search Configuration (optional)
export RAGDOC_RETRIEVAL_USE_BM25=true     # enable BM25 sparse vectors (default: true)
export RAGDOC_RETRIEVAL_BM25_K1=1.2       # term frequency saturation (default: 1.2)
export RAGDOC_RETRIEVAL_BM25_B=0.75       # length normalization (default: 0.75)
export RAGDOC_RETRIEVAL_SPARSE_WEIGHT=0.3 # sparse vector weight in hybrid search (default: 0.3)

# Use langgraph.json: this will start a dev server and open a browser page to https://smith.langchain.com/
langgraph dev
```

## Using with Agent Chat UI

LangChain provides a ready to use [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui).
To use it with the above UI:

- Start ragdoc in server mode:
  ```
  langgraph dev --config langgraph.json --port 8000
  ```

- Clone the [Agent Chat UI repository](https://github.com/langchain-ai/agent-chat-ui) and follow the setup instructions. If you need to install `pnpm` see https://pnpm.io/it/installation

- Enter `agent-chat-app` directory and run the development server:
  ```
  cd agent-chat-app
  pnpm install
  NEXT_PUBLIC_API_URL=http://localhost:8000 NEXT_PUBLIC_ASSISTANT_ID=chat pnpm dev
  ```

- Access the UI at port 3000:[http://localhost:3000](http://localhost:3000)

## Enhanced Retriever

The hybrid retriever combines vector similarity with advanced text search and BM25 sparse vector search:

### Search Methods
1. **Vector Search**: Uses OpenAI embeddings with cosine similarity
2. **Full-Text Search**: PostgreSQL `tsvector`/`ts_rank` for natural language queries
3. **BM25 Sparse Vector Search**: Keyword-based search using BM25 algorithm for sparse embeddings
4. **Pattern Matching**: Fallback ILIKE search for exact phrases and keywords
5. **Text-Only Discovery**: Finds high-relevance documents that might not appear in top vector results

### Configuration Options
- `alpha`: Weight between vector (default: 0.7) and lexical (0.3) scores
- `use_fts`: Enable PostgreSQL full-text search (default: True)
- `fts_language`: Language for FTS stemming (default: "english")
- `title_boost`: Multiplier for title matches (default: 1.5)
- `use_bm25`: Enable BM25 sparse vector search (default: True)
- `bm25_k1`: BM25 term frequency saturation parameter (default: 1.2)
- `bm25_b`: BM25 length normalization parameter (default: 0.75)
- `sparse_weight`: Weight for sparse vector component in hybrid search (default: 0.3)

### BM25 Sparse Vector Search
BM25 (Best Matching 25) implements keyword-based sparse vector search using TF-IDF principles:

- **Sparse Embeddings**: Generates vectors where most values are zero except for relevant terms
- **TF-IDF Foundation**: Built on Term Frequency-Inverse Document Frequency with BM25 improvements
- **Saturation Function**: Uses k1 parameter to control term frequency saturation
- **Length Normalization**: Uses b parameter to normalize document length bias
- **Hybrid Integration**: Combines with dense vectors and lexical search for optimal results

### How It Works
1. **Query Translation**: Automatically translates non-English queries to English for optimal database search
2. Gets top vector similarity results using dense embeddings
3. Runs parallel text search across title and content fields
4. Computes BM25 sparse vector scores for keyword matching

## Agent Configuration

The ragdoc agent supports extensive configuration through both YAML configuration files and environment variables.

### Configuration Methods

#### 1. YAML Configuration File
The primary configuration is stored in `configs/prompts.yaml`:

```yaml
system:
  prompt_it: |
    Sei ragdoc, un assistente di supporto. Rispondi in modo conciso...
  prompt_en: |
    You are ragdoc, a support assistant. Answer concisely...

summarization:
  system_prompt_it: |
    Analizza la seguente conversazione e riassumi i punti chiave...
  user_prompt_it: |
    Conversazione da riassumere:
    {conversation}
    
    Fornisci un riassunto conciso che evidenzi:
    - Prodotti/servizi coinvolti
    - Problemi tecnici specifici
    - Passi già tentati
```

#### 2. Environment Variable Overrides
Any YAML configuration can be overridden using environment variables with the pattern:
`RAGDOC_<SECTION>_<KEY>_<LANGUAGE>`

Examples:
```bash
# Override Italian system prompt
RAGDOC_SYSTEM_PROMPT_IT="Your custom Italian prompt here"

# Override English summarization prompt  
RAGDOC_SUMMARIZATION_SYSTEM_PROMPT_EN="Your custom summarization prompt"

# Configure model settings
RAGDOC_SUMMARIZATION_MODEL="gpt-4o-mini"
RAGDOC_ANSWER_MODEL="gpt-4o-mini"
```

### New Features

#### Conversation Summarization
Before performing database searches, the agent now automatically summarizes multi-turn conversations to provide better context for retrieval:

- **Automatic Context Enhancement**: Summarizes conversation history to improve search relevance
- **Configurable Prompts**: Customize summarization behavior via YAML or environment variables
- **Language-Aware**: Supports both Italian and English summarization
- **Model Configuration**: Use `RAGDOC_SUMMARIZATION_MODEL` to specify the LLM model

The summarization improves search results by:
- Identifying mentioned products and services
- Capturing technical issues and error details
- Recording troubleshooting steps already attempted
- Providing contextual information for better database matching

#### System Prompts
Configure the main agent personality and behavior:

```bash
# Italian system prompt
RAGDOC_SYSTEM_PROMPT_IT="Sei ragdoc, un assistente di supporto..."

# English system prompt
RAGDOC_SYSTEM_PROMPT_EN="You are ragdoc, a support assistant..."
```

#### Answer Generation
Configure how the agent formats responses:

```bash
# Answer instruction templates (use {refs_text} placeholder)
RAGDOC_ANSWER_INSTRUCTION_IT="Usa i seguenti estratti per rispondere..."
RAGDOC_NO_SOURCES_IT="- (nessuna fonte trovata)"
```

#### Escalation Messages
Configure messages when escalating to human support:

```bash
# Base escalation messages
RAGDOC_ESCALATE_BASE_IT="Sto passando il caso a un collega umano..."
RAGDOC_ESCALATE_BASE_EN="I'm escalating this case to a human colleague..."

# Additional escalation details (use {attempts}, {confidence}, {keywords} placeholders)
RAGDOC_ESCALATE_ATTEMPTS_IT="\n\nTentativi di raffinamento effettuati: {attempts}"
RAGDOC_ESCALATE_CONFIDENCE_IT="\nMigliore confidenza raggiunta: {confidence:.2f}"
```

#### Placeholder Messages
Configure messages when no good search results are found:

```bash
# Placeholder context titles and previews
RAGDOC_PLACEHOLDER_TITLE_IT="Dettagli richiesti"
RAGDOC_PLACEHOLDER_PREVIEW_IT="Fornisci maggiori dettagli per migliorare la ricerca."
```

#### Fallback Questions
Set default clarification questions:

```bash
# Fallback when specific questions don't apply
RAGDOC_FALLBACK_IT="Ho bisogno di maggiori dettagli per trovare le informazioni giuste..."
RAGDOC_FALLBACK_EN="I need more details to find the right information for you..."
```

See `.env.example` for a complete list of configurable prompts and their default values.

### Enhanced Iterative Query Refinement

The agent now features an intelligent iterative refinement system that:

- **Analyzes retrieval quality** and suggests specific improvements
- **Generates targeted questions** based on missing information types:
  - Product specificity (NethServer, NethSecurity, etc.)
  - Version information
  - Error details and log messages
  - Configuration context
- **Tracks improvement trajectory** across refinement attempts
- **Escalates strategically** only after 5 iterations or when confidence is very low
- **Provides detailed refinement history** for debugging and optimization

This system transforms the agent from a simple Q&A tool into an intelligent assistant that actively helps users formulate better queries.
5. Combines all result sets with configurable three-way weighting (dense + lexical + sparse)
6. Applies title boost for documents with query terms in titles
7. Returns deduplicated results ranked by combined hybrid score

## Automatic Query Translation

The agent automatically translates non-English queries to English before searching the database, ensuring optimal retrieval performance regardless of the query language.

### Translation Logic
- **Smart Detection**: Uses linguistic heuristics to identify non-English text patterns
- **Multi-language Support**: Recognizes Italian, Spanish, French, German, Portuguese, and other languages
- **Conservative Approach**: Only translates when confident the text is non-English
- **Graceful Fallback**: If translation fails, uses the original query

### Supported Languages
The translation detection supports common patterns from:
- Italian: "Come configurare...", "Dove si trova..."
- Spanish: "Cómo configurar...", "Dónde está..."  
- French: "Comment configurer...", "Où se trouve..."
- German: "Wie konfigurieren...", "Wo befindet sich..."
- Portuguese: "Como configurar...", "Onde fica..."
- And more through OpenAI's translation capabilities

### Configuration
Use `RAGDOC_TRANSLATION_MODEL` to specify the model for translation (default: `gpt-4o-mini`)

## Environment
- Python: 3.12
- Dependencies: `httpx`, `PyYAML`, `beautifulsoup4`, `lxml`, `openai`, `psycopg[binary]`, `pgvector`, `docutils`, `langgraph`, `langchain-core`, `langchain-openai`, `langchain-community`
- A Postgres instance with the `vector` extension is required for indexing. Fetch-only still works without DB (files + local state).

### Configuration (env)
- `DATABASE_URL` e.g. `postgresql+psycopg://user:pass@host:5432/ragdoc`
- `OPENAI_API_KEY`
- `RAGDOC_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `RAGDOC_EMBEDDING_DIM` (default: `1536`)
- `RAGDOC_EMBEDDING_MAX_TOKENS` (default: `8000`)
- `RAGDOC_RETRIEVAL_TOP_K` (default: `8`)
- `RAGDOC_RETRIEVAL_ALPHA` (default: `0.7`)
- `RAGDOC_RETRIEVAL_TITLE_BOOST` (default: `1.5`)
- `RAGDOC_RETRIEVAL_USE_FTS` (default: `true`)
- `RAGDOC_RETRIEVAL_FTS_LANGUAGE` (default: `english`)
- `RAGDOC_CONFIDENCE_THRESHOLD` (default: `0.65`)
- `RAGDOC_ANSWER_MODEL` (default: `gpt-4o-mini`)
- `RAGDOC_TRANSLATION_MODEL` (default: `gpt-4o-mini`) - Model used for translating queries to English
- `RAGDOC_INFO_CONF_THRESHOLD` (default: `0.35`)
- `RAGDOC_RETRIEVAL_HISTORY_CHARS` (default: `800`)
- `RAGDOC_RETRIEVAL_HISTORY_PARTS` (default: `6`)
- `RAGDOC_LOG_LEVEL` (default: `INFO`)

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
- HTTP crawling recursively follows links up to the specified max_depth (default: 2).
- Git operations use the local `git` executable; ensure it’s installed and available on PATH.
- Network credentials or private repo access are not handled in this minimal version.
- Vectorization uses OpenAI embeddings. See env vars to customize model/dim.
- Postgres must have the `vector` extension available. The table `ragdoc_embeddings` will be created if missing, with a unique constraint on (source_type, source_url, path, content_sha256).
- The agent introduces itself (router), asks clarifying questions with human-in-the-loop, and uses conversation history to form retrieval queries; it then answers or escalates based on `RAGDOC_CONFIDENCE_THRESHOLD`.

### Crawl whole sites via sitemap

You can enable sitemap-based recursive fetching for a whole site using LangChain’s SitemapLoader.

Add to your `configs/sources.yaml`:

```yaml
http:
  - url: https://example.com/
    out_dir: data/raw/http
    sitemap: true                # enable sitemap crawl
    # sitemap_url: https://example.com/sitemap.xml  # optional override
```

Notes:
- Requires `langchain-community` (included). The fetcher reads the sitemap and saves each page under `out_dir` as `.html`.
- Non-HTML assets like images/PDFs are skipped by default.
- ETag/Last-Modified caching applies to single-URL fetches; sitemap mode stores pages directly.

## Roadmap (next steps)
- Add PDF parsing
- ~~Upgrade lexical retrieval to Postgres FTS (tsvector/ts_rank)~~ ✅ **Completed**
- Add recency/domain filters and metadata-based filtering
- Confidence calibration and evaluation harness
- Containerization for the agent service and healthcheck endpoint
