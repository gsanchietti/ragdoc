from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Iterable, Sequence

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector


DEFAULT_MODEL = os.getenv("RAGDOC_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_DIM = int(os.getenv("RAGDOC_EMBEDDING_DIM", "1536"))
DEFAULT_MAX_TOKENS = int(os.getenv("RAGDOC_EMBEDDING_MAX_TOKENS", "8000"))

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingChunk:
    text: str
    source_type: str  # 'http' | 'md'
    source_url: str = ""
    path: str = ""
    title: str | None = None
    metadata: dict | None = None

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class EmbeddingClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.client = OpenAI()
        self.model = model
        # Conservative token limit for embeddings API
        self.max_tokens_per_request = max_tokens

    def _estimate_tokens(self, text: str) -> int:
        """Conservative estimation: ~3 characters per token to be safer."""
        return len(text) // 3 + 1

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # For small batches, send directly
        if len(texts) == 1:
            resp = self.client.embeddings.create(model=self.model, input=list(texts))
            return [item.embedding for item in resp.data]
        
        # For larger batches, check token count and split if needed
        total_tokens = sum(self._estimate_tokens(text) for text in texts)
        
        # Use 60% of max tokens as safety margin to account for estimation errors
        safe_max_tokens = int(self.max_tokens_per_request * 0.6)
        
        if total_tokens <= safe_max_tokens:
            # Safe to send as one batch
            logger.debug(f"Embedding batch: {len(texts)} texts, ~{total_tokens} tokens")
            resp = self.client.embeddings.create(model=self.model, input=list(texts))
            return [item.embedding for item in resp.data]
        else:
            # Split into smaller batches based on token count
            logger.info(f"Large batch detected: {len(texts)} texts, ~{total_tokens} tokens - splitting")
            results = []
            current_batch = []
            current_tokens = 0
            batch_count = 0
            
            for text in texts:
                text_tokens = self._estimate_tokens(text)
                
                # If single text is too large, truncate it
                if text_tokens > safe_max_tokens:
                    logger.warning(f"Text too large ({text_tokens} tokens), truncating")
                    # Truncate to safe character count
                    max_chars = int(safe_max_tokens * 3 * 0.8)
                    text = text[:max_chars]
                    text_tokens = self._estimate_tokens(text)
                
                # If adding this text would exceed limit, process current batch
                if current_tokens + text_tokens > safe_max_tokens and current_batch:
                    batch_count += 1
                    logger.debug(f"Processing sub-batch {batch_count}: {len(current_batch)} texts, ~{current_tokens} tokens")
                    resp = self.client.embeddings.create(model=self.model, input=current_batch)
                    results.extend([item.embedding for item in resp.data])
                    current_batch = []
                    current_tokens = 0
                
                current_batch.append(text)
                current_tokens += text_tokens
            
            # Process final batch
            if current_batch:
                batch_count += 1
                logger.debug(f"Processing final sub-batch {batch_count}: {len(current_batch)} texts, ~{current_tokens} tokens (safe limit: {safe_max_tokens})")
                resp = self.client.embeddings.create(model=self.model, input=current_batch)
                results.extend([item.embedding for item in resp.data])
            
            logger.info(f"Completed embedding in {batch_count} sub-batches")
            return results


class PgVectorIndexer:
    def __init__(self, dsn: str, dim: int = DEFAULT_DIM) -> None:
        self.dsn = self._normalize_dsn(dsn)
        self.dim = dim
        self._ensure_schema()

    @staticmethod
    def _normalize_dsn(dsn: str) -> str:
        # Accept SQLAlchemy-style DSNs like postgresql+psycopg:// and normalize to postgresql://
        return dsn.replace("postgresql+psycopg://", "postgresql://").replace("postgres+psycopg://", "postgres://")

    def _ensure_schema(self) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS ragdoc_embeddings (
                        id UUID PRIMARY KEY,
                        source_type TEXT NOT NULL,
                        source_url TEXT NOT NULL DEFAULT '',
                        path TEXT NOT NULL DEFAULT '',
                        title TEXT,
                        content TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        embedding vector({self.dim}) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_ragdoc_embeddings_source_content
                    ON ragdoc_embeddings (source_type, source_url, path, content_sha256)
                    """
                )
            register_vector(conn)


    def upsert(self, chunks: Iterable[EmbeddingChunk], embeddings: Iterable[Sequence[float]]) -> int:
        count = 0
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO ragdoc_embeddings
                        (id, source_type, source_url, path, title, content, content_sha256, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source_type, source_url, path, content_sha256) DO NOTHING
                        """,
                        (
                            uuid.uuid4(),
                            chunk.source_type,
                            chunk.source_url,
                            chunk.path,
                            chunk.title,
                            chunk.text,
                            chunk.content_sha256,
                            emb,
                            chunk.metadata,
                        ),
                    )
                    count += cur.rowcount if cur.rowcount is not None else 0
        return count
