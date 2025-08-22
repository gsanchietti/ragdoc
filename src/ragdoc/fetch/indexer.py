from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Iterable, Sequence

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector


DEFAULT_MODEL = os.getenv("RAGDOC_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_DIM = int(os.getenv("RAGDOC_EMBEDDING_DIM", "1536"))


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
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.client = OpenAI()
        self.model = model

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in resp.data]


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
            register_vector(conn)
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
