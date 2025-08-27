from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Any, Dict

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector, Vector

from .models import RetrievalResult

# Logger setup
logger = logging.getLogger("ragdoc.retrieval")

def _parse_bool(value: str, default: bool = True) -> bool:
    """Parse boolean environment variable values."""
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False
    else:
        return default

try:
    # Optional: used only for building history-aware query
    from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
except Exception:  # pragma: no cover - soft dependency
    AnyMessage = object  # type: ignore
    HumanMessage = object  # type: ignore
    AIMessage = object  # type: ignore


@dataclass
class RetrieverConfig:
    dsn: str
    k: int = int(os.getenv("RAGDOC_RETRIEVAL_TOP_K", "8"))
    alpha: float = float(os.getenv("RAGDOC_RETRIEVAL_ALPHA", "0.7"))  # weight for vector score vs lexical
    embedding_model: str = os.getenv("RAGDOC_EMBEDDING_MODEL", "text-embedding-3-small")
    use_fts: bool = _parse_bool(os.getenv("RAGDOC_RETRIEVAL_USE_FTS", "true"))  # Use PostgreSQL full-text search
    fts_language: str = os.getenv("RAGDOC_RETRIEVAL_FTS_LANGUAGE", "english")  # Language for full-text search
    title_boost: float = float(os.getenv("RAGDOC_RETRIEVAL_TITLE_BOOST", "1.5"))  # Boost factor for title matches
    # BM25 parameters for sparse vector search
    use_bm25: bool = _parse_bool(os.getenv("RAGDOC_RETRIEVAL_USE_BM25", "true"))  # Enable BM25 sparse vector search
    bm25_k1: float = float(os.getenv("RAGDOC_RETRIEVAL_BM25_K1", "1.2"))  # Term frequency saturation parameter
    bm25_b: float = float(os.getenv("RAGDOC_RETRIEVAL_BM25_B", "0.75"))  # Length normalization parameter
    sparse_weight: float = float(os.getenv("RAGDOC_RETRIEVAL_SPARSE_WEIGHT", "0.3"))  # Weight for sparse vs dense search


class Retriever:
    def __init__(self, cfg: RetrieverConfig) -> None:
        self.cfg = cfg
        self.client = OpenAI()

    def _embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.cfg.embedding_model, input=[text])
        return resp.data[0].embedding

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25 - split on whitespace and punctuation, lowercase."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [token.strip() for token in text.split() if token.strip()]
        return tokens

    def _compute_bm25_scores(self, query_tokens: List[str], documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute BM25 scores for documents given query tokens."""
        if not query_tokens or not documents:
            return {}
        
        logger.debug("BM25: Computing scores for %d documents with query tokens: %s", 
                    len(documents), query_tokens[:10])  # Log first 10 tokens
        
        # Tokenize all documents and compute statistics
        doc_tokens = {}
        doc_lengths = {}
        term_doc_freq = defaultdict(int)  # How many documents contain each term
        total_docs = len(documents)
        avg_doc_length = 0
        
        # First pass: tokenize and compute document statistics
        for doc in documents:
            doc_id = doc['id']
            # Combine title and content for BM25 scoring
            full_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            tokens = self._tokenize(full_text)
            doc_tokens[doc_id] = tokens
            doc_lengths[doc_id] = len(tokens)
            avg_doc_length += len(tokens)
            
            # Count unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                term_doc_freq[term] += 1
        
        avg_doc_length = avg_doc_length / total_docs if total_docs > 0 else 0
        
        logger.debug("BM25: Processed %d documents, avg_length=%.1f, unique_terms=%d", 
                    total_docs, avg_doc_length, len(term_doc_freq))
        
        # Compute BM25 scores
        scores = {}
        k1, b = self.cfg.bm25_k1, self.cfg.bm25_b
        
        for doc in documents:
            doc_id = doc['id']
            tokens = doc_tokens.get(doc_id, [])
            doc_length = doc_lengths.get(doc_id, 0)
            
            if doc_length == 0:
                scores[doc_id] = 0.0
                continue
            
            # Term frequency in document
            term_freq = Counter(tokens)
            score = 0.0
            
            for query_term in query_tokens:
                if query_term in term_freq:
                    tf = term_freq[query_term]
                    df = term_doc_freq.get(query_term, 0)
                    
                    if df > 0:
                        # IDF component: log((N - df + 0.5) / (df + 0.5))
                        # Use max() to avoid negative IDF for very common terms
                        idf = max(0.0, math.log((total_docs - df + 0.5) / (df + 0.5)))
                        
                        # BM25 TF component: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                        if avg_doc_length > 0:
                            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                        else:
                            tf_component = (tf * (k1 + 1)) / (tf + k1)
                        
                        score += idf * tf_component
            
            scores[doc_id] = score
        
        # Normalize scores to 0-1 range - ensure we only normalize positive scores
        if scores:
            max_score = max(scores.values())
            min_score = min(scores.values())
            if max_score > min_score and max_score > 0:
                # Normalize to 0-1 range with proper handling of negative scores
                score_range = max_score - min_score
                scores = {doc_id: max(0.0, (score - min_score) / score_range) for doc_id, score in scores.items()}
            elif max_score > 0:
                # All scores are the same positive value
                scores = {doc_id: score / max_score for doc_id, score in scores.items()}
            else:
                # All scores are zero or negative
                scores = {doc_id: 0.0 for doc_id in scores}
        
        logger.debug("BM25: Computed scores for %d documents, max_score=%.4f", 
                    len(scores), max(scores.values()) if scores else 0.0)
        
        return scores

    def _sparse_vector_search(self, conn: psycopg.Connection, query: str, k: int) -> Dict[str, float]:
        """Perform BM25-based sparse vector search."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.debug("Sparse vector search: No query tokens, returning empty scores")
            return {}
        
        logger.debug("Sparse vector search: Processing query with %d tokens", len(query_tokens))
        
        # Get all documents for BM25 computation
        # Note: In a production system, you might want to pre-compute BM25 scores
        # or use a more efficient approach for large corpora
        sql = """
            SELECT id::text, title, content 
            FROM ragdoc_embeddings 
            ORDER BY id
        """
        
        documents = []
        with conn.cursor() as cur:
            cur.execute(sql)
            results = cur.fetchall()
            
            for r in results:
                documents.append({
                    'id': r[0],
                    'title': r[1] or '',
                    'content': r[2] or ''
                })
        
        logger.debug("Sparse vector search: Retrieved %d documents for BM25 computation", len(documents))
        
        if not documents:
            return {}
        
        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(query_tokens, documents)
        
        # Return top-k scoring documents
        sorted_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        top_scores = dict(sorted_scores[:k * 2])  # Get more for diversity
        
        logger.debug("Sparse vector search: Returning %d top scores", len(top_scores))
        return top_scores

    def _content_to_text(self, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(str(p.get("text") or p.get("content") or p))
                else:
                    parts.append(str(p))
            return " ".join(s for s in parts if s).strip()
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or content).strip()
        return str(content).strip()

    def _build_history_query(self, messages: Optional[Sequence[Any]], query: Optional[str]) -> str:
        # Combine explicit query with compacted recent messages to form a retrieval query.
        max_chars = int(os.getenv("RAGDOC_RETRIEVAL_HISTORY_CHARS", "800"))
        max_parts = int(os.getenv("RAGDOC_RETRIEVAL_HISTORY_PARTS", "6"))
        parts: list[str] = []
        if query:
            q = query.strip()
            if q:
                parts.append(f"User: {q}")
        if messages:
            count = 0
            for m in reversed(messages):
                # Prefer recent human messages; include at most one assistant clarify
                if isinstance(m, HumanMessage):
                    txt = self._content_to_text(getattr(m, "content", ""))
                    if txt:
                        parts.append(f"User: {txt}")
                        count += 1
                elif isinstance(m, AIMessage):
                    if count < max_parts:
                        txt = self._content_to_text(getattr(m, "content", ""))
                        # heuristically include only short clarifications
                        if txt and len(txt) <= 280:
                            parts.append(f"Assistant: {txt}")
                if count >= max_parts:
                    break
        if not parts:
            return ""
        combined = " | ".join(reversed(parts))
        if len(combined) > max_chars:
            combined = combined[-max_chars:]
        return combined

    def _vector_search(self, conn: psycopg.Connection, vec: list[float], k: int) -> list[RetrievalResult]:
        logger.debug("Vector search: executing cosine similarity search with k=%d", k)
        
        sql = (
            "SELECT id::text, source_type, source_url, path, title, 1 - (embedding <=> %s) AS vscore, content, metadata "
            "FROM ragdoc_embeddings ORDER BY embedding <=> %s ASC LIMIT %s"
        )
        results: list[RetrievalResult] = []
        with conn.cursor() as cur:
            v = Vector(vec)
            cur.execute(sql, (v, v, k))
            db_results = cur.fetchall()
            
            logger.debug("Vector search: found %d results from database", len(db_results))
            
            for r in db_results:
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
        
        if results:
            logger.debug("Vector search: top result has vscore=%.4f", results[0].vscore)
            
        return results

    def _lexical_search(self, conn: psycopg.Connection, query: str, k: int) -> dict[str, float]:
        # Use PostgreSQL full-text search with tsvector and ts_rank for better relevance
        # Also include simple pattern matching as fallback for exact phrases
        
        # Prepare query for full-text search - handle phrase queries and clean input
        clean_query = query.strip()
        if not clean_query:
            logger.debug("Lexical search: empty query, returning empty scores")
            return {}
        
        logger.debug("Lexical search: processing query=%r with FTS=%s", clean_query, self.cfg.use_fts)
        
        scores: dict[str, float] = {}
        
        if self.cfg.use_fts:
            # Try full-text search first (best for natural language queries)
            fts_sql = f"""
                SELECT id::text, 
                       ts_rank(
                           to_tsvector('{self.cfg.fts_language}', COALESCE(title, '') || ' ' || content), 
                           plainto_tsquery('{self.cfg.fts_language}', %s)
                       ) +
                       CASE 
                           WHEN COALESCE(title, '') ILIKE %s THEN {self.cfg.title_boost}
                           ELSE 0 
                       END AS fts_score
                FROM ragdoc_embeddings 
                WHERE to_tsvector('{self.cfg.fts_language}', COALESCE(title, '') || ' ' || content) @@ plainto_tsquery('{self.cfg.fts_language}', %s)
                   OR COALESCE(title, '') ILIKE %s
                ORDER BY fts_score DESC 
                LIMIT %s
            """
            
            logger.debug("Lexical search: executing FTS query with language=%s", self.cfg.fts_language)
            
            try:
                # Try full-text search
                with conn.cursor() as cur:
                    pattern = f"%{clean_query}%"
                    cur.execute(fts_sql, (clean_query, pattern, clean_query, pattern, k))
                    fts_results = cur.fetchall()
                    
                    logger.debug("Lexical search: FTS found %d results", len(fts_results))
                    
                    if fts_results:
                        # Normalize FTS scores to 0-1 range
                        max_fts = max(r[1] for r in fts_results) if fts_results else 1.0
                        for r in fts_results:
                            if max_fts > 0:
                                scores[r[0]] = float(r[1]) / max_fts
                            else:
                                scores[r[0]] = 0.0
                        logger.debug("Lexical search: FTS scores normalized (max_score=%.4f)", max_fts)
                    else:
                        # Fallback to ILIKE pattern matching
                        logger.debug("Lexical search: FTS found no results, falling back to pattern matching")
                        self._fallback_pattern_search(conn, clean_query, k, scores)
                        
            except Exception as e:
                # If FTS fails, fall back to simple ILIKE search
                logger.warning("Lexical search: FTS failed (%s), falling back to pattern matching", e)
                self._fallback_pattern_search(conn, clean_query, k, scores)
        else:
            # Direct pattern matching if FTS is disabled
            logger.debug("Lexical search: FTS disabled, using pattern matching")
            self._fallback_pattern_search(conn, clean_query, k, scores)
        
        logger.debug("Lexical search: completed with %d scored documents", len(scores))
        return scores
    
    def _fallback_pattern_search(self, conn: psycopg.Connection, query: str, k: int, scores: dict[str, float]) -> None:
        """Fallback pattern-based search using ILIKE."""
        logger.debug("Pattern search: executing ILIKE-based search for query=%r", query)
        
        fallback_sql = f"""
            SELECT id::text, 
                   CASE 
                       WHEN COALESCE(title, '') ILIKE %s THEN {self.cfg.title_boost}
                       WHEN content ILIKE %s THEN 0.8
                       ELSE ((length(content) - length(replace(lower(content), lower(%s), ''))) / 
                             GREATEST(length(%s), 1)::float) * 0.6
                   END AS pattern_score
            FROM ragdoc_embeddings 
            WHERE COALESCE(title, '') ILIKE %s OR content ILIKE %s
            ORDER BY pattern_score DESC 
            LIMIT %s
        """
        
        with conn.cursor() as cur:
            pattern = f"%{query}%"
            cur.execute(fallback_sql, (pattern, pattern, query, query, pattern, pattern, k))
            results = cur.fetchall()
            for r in results:
                scores[r[0]] = float(r[1] or 0.0)
        
        logger.debug("Pattern search: found %d results", len(results))

    def search(self, query: str, k: int | None = None, messages: Optional[Sequence[Any]] = None) -> list[RetrievalResult]:
        dsn = self.cfg.dsn.replace("postgresql+psycopg://", "postgresql://")
        # Build a history-aware query if messages are provided
        eff_query = self._build_history_query(messages, query) if messages is not None else (query or "").strip()
        
        # Debug: Log the executed queries
        logger.debug("Search initiated with original query: %r", query)
        logger.debug("Effective query for search: %r", eff_query)
        logger.debug("Hybrid search configuration: alpha=%.2f, use_fts=%s, title_boost=%.1f, use_bm25=%s, sparse_weight=%.2f", 
                    self.cfg.alpha, self.cfg.use_fts, self.cfg.title_boost, self.cfg.use_bm25, self.cfg.sparse_weight)
        
        if not eff_query:
            logger.debug("Empty effective query, returning no results")
            return []
            
        with psycopg.connect(dsn) as conn:
            register_vector(conn)
            vec = self._embed(eff_query)
            k = k or self.cfg.k
            
            logger.debug("Starting hybrid search with k=%d", k)
            
            # Always get vector similarity results (hybrid search component 1)
            vec_results = self._vector_search(conn, vec, k * 2)  # Get more for better diversity
            logger.debug("Vector search completed: found %d results", len(vec_results))
            
            # Always get text search results (hybrid search component 2) - search independently to catch text-only matches
            lex_scores = self._lexical_search(conn, eff_query, k * 2)
            logger.debug("Lexical search completed: found %d scored documents", len(lex_scores))
            
            # BM25 sparse vector search (hybrid search component 3)
            sparse_scores = {}
            if self.cfg.use_bm25:
                sparse_scores = self._sparse_vector_search(conn, eff_query, k * 2)
                logger.debug("Sparse vector search (BM25) completed: found %d scored documents", len(sparse_scores))
            
            # Always get purely text-based results that might not be in vector top results (hybrid search component 4)
            text_only_results = self._get_text_only_results(conn, eff_query, lex_scores, vec_results, k)
            logger.debug("Text-only search completed: found %d additional results", len(text_only_results))

            # Hybrid scoring with three components: dense vectors + lexical + sparse vectors
            max_lex = max(lex_scores.values()) if lex_scores else 1.0
            max_sparse = max(sparse_scores.values()) if sparse_scores else 1.0
            combined: list[RetrievalResult] = []
            
            logger.debug("Combining results with hybrid scoring (alpha=%.2f, sparse_weight=%.2f)", 
                        self.cfg.alpha, self.cfg.sparse_weight)
            
            # Process vector results with hybrid scoring
            for r in vec_results:
                lscore = (lex_scores.get(r.id, 0.0) / max_lex) if max_lex > 0 else 0.0
                sscore = (sparse_scores.get(r.id, 0.0) / max_sparse) if max_sparse > 0 else 0.0
                
                # Three-way hybrid scoring: dense vector + lexical + sparse vector
                if self.cfg.use_bm25:
                    # Weighted combination: dense vector, lexical, and sparse vector
                    dense_weight = self.cfg.alpha * (1 - self.cfg.sparse_weight)
                    lexical_weight = (1 - self.cfg.alpha) * (1 - self.cfg.sparse_weight)
                    sparse_weight = self.cfg.sparse_weight
                    score = dense_weight * r.vscore + lexical_weight * lscore + sparse_weight * sscore
                else:
                    # Original two-way scoring
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
                        sscore=sscore,
                        score=score,
                        content_preview=r.content_preview,
                        metadata=r.metadata,
                    )
                )
            
            # Add sparse-only results (high BM25 relevance but not in dense vector top results)
            if self.cfg.use_bm25:
                vec_ids = {r.id for r in vec_results}
                sparse_only_results = self._get_sparse_only_results(conn, sparse_scores, vec_ids, max_sparse, k)
                combined.extend(sparse_only_results)
                logger.debug("Added %d sparse-only results", len(sparse_only_results))
            
            # Always add text-only results (high text relevance, low vector similarity)
            combined.extend(text_only_results)

            # Remove duplicates and sort by final hybrid score
            seen_ids = set()
            unique_results = []
            for result in combined:
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: x.score, reverse=True)
            final_results = unique_results[:k]
            
            search_components = f"vector: {len(vec_results)}, text-only: {len(text_only_results)}"
            if self.cfg.use_bm25:
                sparse_only_count = len([r for r in combined if hasattr(r, '_sparse_only') and r._sparse_only])
                search_components += f", sparse-only: {sparse_only_count}"
            
            logger.debug("Hybrid search completed: returning %d results (%s, total candidates: %d)", 
                        len(final_results), search_components, len(combined))
            
            return final_results
    
    def _get_text_only_results(self, conn: psycopg.Connection, query: str, lex_scores: dict[str, float], vec_results: list[RetrievalResult], k: int) -> list[RetrievalResult]:
        """Get high-relevance text matches that weren't in top vector results."""
        vec_ids = {r.id for r in vec_results}
        
        logger.debug("Text-only search: looking for high-relevance text matches not in top %d vector results", len(vec_ids))
        
        # Find high-scoring text matches not in vector results
        high_text_matches = []
        max_lex = max(lex_scores.values()) if lex_scores else 1.0
        
        for doc_id, lex_score in lex_scores.items():
            if doc_id not in vec_ids and lex_score > 0.3:  # Only consider high text relevance
                high_text_matches.append((doc_id, lex_score))
        
        logger.debug("Text-only search: found %d high-scoring text matches (threshold > 0.3)", len(high_text_matches))
        
        if not high_text_matches:
            return []
        
        # Get document details for high text matches
        high_text_matches.sort(key=lambda x: x[1], reverse=True)
        top_text_ids = [x[0] for x in high_text_matches[:k//2]]  # Take up to half of k
        
        logger.debug("Text-only search: selecting top %d text-only results", len(top_text_ids))
        
        if not top_text_ids:
            return []
        
        # Fetch full document details
        placeholders = ",".join(["%s"] * len(top_text_ids))
        sql = f"""
            SELECT id::text, source_type, source_url, path, title, content, metadata 
            FROM ragdoc_embeddings 
            WHERE id::text IN ({placeholders})
        """
        
        text_results = []
        with conn.cursor() as cur:
            cur.execute(sql, top_text_ids)
            db_results = cur.fetchall()
            
            logger.debug("Text-only search: retrieved %d document details from database", len(db_results))
            
            for r in db_results:
                doc_id = r[0]
                content = r[5] or ""
                lscore = (lex_scores.get(doc_id, 0.0) / max_lex) if max_lex > 0 else 0.0
                
                # For text-only matches, give heavy weight to lexical score
                score = 0.2 * 0.5 + 0.8 * lscore  # Low vector score, high lexical weight
                
                text_results.append(
                    RetrievalResult(
                        id=doc_id,
                        source_type=r[1],
                        source_url=r[2],
                        path=r[3],
                        title=r[4],
                        vscore=0.5,  # Assume medium vector relevance for text-only matches
                        lscore=lscore,
                        score=score,
                        content_preview=content[:400],
                        metadata=r[6],
                    )
                )
        
        logger.debug("Text-only search: created %d text-only result objects", len(text_results))
        return text_results

    def _get_sparse_only_results(self, conn: psycopg.Connection, sparse_scores: Dict[str, float], vec_ids: set, max_sparse: float, k: int) -> list[RetrievalResult]:
        """Get high-relevance BM25 matches that weren't in top vector results."""
        logger.debug("Sparse-only search: looking for high-relevance BM25 matches not in top %d vector results", len(vec_ids))
        
        # Find high-scoring sparse matches not in vector results
        high_sparse_matches = []
        
        for doc_id, sparse_score in sparse_scores.items():
            if doc_id not in vec_ids and sparse_score > 0.3:  # Only consider high sparse relevance
                high_sparse_matches.append((doc_id, sparse_score))
        
        logger.debug("Sparse-only search: found %d high-scoring sparse matches (threshold > 0.3)", len(high_sparse_matches))
        
        if not high_sparse_matches:
            return []
        
        # Get document details for high sparse matches
        high_sparse_matches.sort(key=lambda x: x[1], reverse=True)
        top_sparse_ids = [x[0] for x in high_sparse_matches[:k//3]]  # Take up to third of k
        
        logger.debug("Sparse-only search: selecting top %d sparse-only results", len(top_sparse_ids))
        
        if not top_sparse_ids:
            return []
        
        # Fetch full document details
        placeholders = ",".join(["%s"] * len(top_sparse_ids))
        sql = f"""
            SELECT id::text, source_type, source_url, path, title, content, metadata 
            FROM ragdoc_embeddings 
            WHERE id::text IN ({placeholders})
        """
        
        sparse_results = []
        with conn.cursor() as cur:
            cur.execute(sql, top_sparse_ids)
            db_results = cur.fetchall()
            
            logger.debug("Sparse-only search: retrieved %d document details from database", len(db_results))
            
            for r in db_results:
                doc_id = r[0]
                content = r[5] or ""
                sscore = (sparse_scores.get(doc_id, 0.0) / max_sparse) if max_sparse > 0 else 0.0
                
                # For sparse-only matches, give heavy weight to sparse score
                score = 0.2 * 0.5 + 0.8 * sscore  # Low vector score, high sparse weight
                
                result = RetrievalResult(
                    id=doc_id,
                    source_type=r[1],
                    source_url=r[2],
                    path=r[3],
                    title=r[4],
                    vscore=0.5,  # Assume medium vector relevance for sparse-only matches
                    lscore=0.0,  # No lexical score computed for these
                    sscore=sscore,
                    score=score,
                    content_preview=content[:400],
                    metadata=r[6],
                )
                # Mark as sparse-only for debugging
                result._sparse_only = True
                sparse_results.append(result)
        
        logger.debug("Sparse-only search: created %d sparse-only result objects", len(sparse_results))
        return sparse_results
