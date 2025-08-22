from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector, Vector

from .models import RetrievalResult


@dataclass
class RetrieverConfig:
    dsn: str
    k: int = int(os.getenv("RAGDOC_RETRIEVAL_TOP_K", "8"))
    alpha: float = 0.7  # weight for vector score vs lexical
    embedding_model: str = os.getenv("RAGDOC_EMBEDDING_MODEL", "text-embedding-3-small")


class Retriever:
    def __init__(self, cfg: RetrieverConfig) -> None:
        self.cfg = cfg
        self.client = OpenAI()

    def _embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.cfg.embedding_model, input=[text])
        return resp.data[0].embedding

    def _vector_search(self, conn: psycopg.Connection, vec: list[float], k: int) -> list[RetrievalResult]:
        sql = (
            "SELECT id::text, source_type, source_url, path, title, 1 - (embedding <=> %s) AS vscore, content, metadata "
            "FROM ragdoc_embeddings ORDER BY embedding <=> %s ASC LIMIT %s"
        )
        results: list[RetrievalResult] = []
        with conn.cursor() as cur:
            v = Vector(vec)
            cur.execute(sql, (v, v, k))
            for r in cur.fetchall():
                content: str = r[6] or ""
                results.append(
                    RetrievalResult(
                        id=r[0],
                        source_type=r[1],
                        source_url=r[2],
                        path=r[3],
                        title=r[4],
                        vscore=float(r[5]),
                        lscore=0.0,
                        score=float(r[5]),
                        content_preview=content[:400],
                        metadata=r[7],
                    )
                )
        return results

    def _lexical_search(self, conn: psycopg.Connection, query: str, k: int) -> dict[str, float]:
        # Simple ILIKE frequency count; swap to tsvector/ts_rank for production
        sql = (
            "SELECT id::text, ((length(content) - length(replace(lower(content), lower(%s), ''))) / NULLIF(length(%s),0)) AS freq "
            "FROM ragdoc_embeddings ORDER BY freq DESC NULLS LAST LIMIT %s"
        )
        scores: dict[str, float] = {}
        with conn.cursor() as cur:
            cur.execute(sql, (query, query, k))
            for r in cur.fetchall():
                scores[r[0]] = float(r[1] or 0.0)
        return scores

    def search(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        dsn = self.cfg.dsn.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn) as conn:
            register_vector(conn)
            vec = self._embed(query)
            k = k or self.cfg.k
            vec_results = self._vector_search(conn, vec, k)
            lex_scores = self._lexical_search(conn, query, max(10, k))

            # Merge by id: alpha * vscore + (1-alpha) * normalized lexical
            max_lex = max(lex_scores.values()) if lex_scores else 1.0
            combined: list[RetrievalResult] = []
            for r in vec_results:
                lscore = (lex_scores.get(r.id, 0.0) / max_lex) if max_lex > 0 else 0.0
                score = self.cfg.alpha * r.vscore + (1 - self.cfg.alpha) * lscore
                combined.append(
                    RetrievalResult(
                        id=r.id,
                        source_type=r.source_type,
                        source_url=r.source_url,
                        path=r.path,
                        title=r.title,
                        vscore=r.vscore,
                        lscore=lscore,
                        score=score,
                        content_preview=r.content_preview,
                        metadata=r.metadata,
                    )
                )

            combined.sort(key=lambda x: x.score, reverse=True)
            return combined[:k]
