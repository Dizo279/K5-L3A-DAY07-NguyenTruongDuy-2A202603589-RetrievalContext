"""Retrieval models and implementations for Day 7 Agentic RAG:
1. Dense Retriever (PostgreSQL pgvector + OpenRouter text-embedding-3-small)
2. Lexical Retriever (BM25 / TF-IDF)
3. Hybrid Retriever (Dense + Lexical via Reciprocal Rank Fusion)
"""

import json
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional
import urllib.request
import psycopg
from dotenv import load_dotenv

load_dotenv()

def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())

class Embedder:
    """Embedder using OpenRouter OpenAI-compatible API."""
    def __init__(self, model_name: str = "openai/text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

    def embed_query(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "input": [text]
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"][0]["embedding"]

class DenseRetriever:
    """Dense retrieval querying PostgreSQL pgvector table day6_chunks."""
    def __init__(self, database_url: Optional[str] = None, strategy: str = "A"):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.strategy = strategy
        self.embedder = Embedder()

    def search(self, query: str, top_k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        filter_dict = filter_dict or {}

        conn = psycopg.connect(self.database_url)
        cur = conn.cursor()

        sql = """
            SELECT chunk_id, document_id, source, page, text, metadata, (embedding <=> %s::vector) as distance
            FROM day6_chunks
            WHERE strategy = %s
        """
        params = [query_vector, self.strategy]

        if "section" in filter_dict and filter_dict["section"]:
            sql += " AND metadata->>'section' = %s"
            params.append(filter_dict["section"])

        sql += " ORDER BY distance ASC LIMIT %s"
        params.append(top_k)

        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        results = []
        for rank, r in enumerate(rows, start=1):
            score = 1.0 - float(r[6]) # cosine similarity approx
            results.append({
                "id": r[0],
                "chunk_id": r[0],
                "source_id": r[1],
                "source": r[2],
                "page": r[3],
                "text": r[4],
                "metadata": r[5] if isinstance(r[5], dict) else {},
                "score": score,
                "rank": rank
            })
        return results

class BM25Retriever:
    """BM25 Lexical Retriever implemented in pure Python."""
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(dt) for dt in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1.0
        self.N = len(chunks)

        self.df = Counter()
        for dt in self.doc_tokens:
            for term in set(dt):
                self.df[term] += 1

        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        scores = []
        filter_dict = filter_dict or {}

        for i, doc in enumerate(self.chunks):
            # check filter
            if "section" in filter_dict and filter_dict["section"]:
                if doc.get("metadata", {}).get("section") != filter_dict["section"]:
                    continue

            tf = Counter(self.doc_tokens[i])
            score = 0.0
            doc_len = self.doc_lens[i]
            for term in q_tokens:
                if term in tf:
                    t_idf = self.idf.get(term, 0.0)
                    t_tf = tf[term]
                    numerator = t_tf * (self.k1 + 1.0)
                    denominator = t_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += t_idf * (numerator / denominator)
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, doc) in enumerate(scores[:top_k], start=1):
            results.append({
                "id": doc.get("id", doc.get("chunk_id")),
                "chunk_id": doc.get("id", doc.get("chunk_id")),
                "source_id": doc.get("source_id", doc.get("document_id")),
                "source": doc.get("source"),
                "page": doc.get("page"),
                "text": doc.get("text"),
                "metadata": doc.get("metadata", {}),
                "score": score,
                "rank": rank
            })
        return results

class HybridRetriever:
    """Hybrid Retriever combining Dense + Lexical using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3, filter_dict: Optional[Dict[str, Any]] = None, fetch_k: int = 15) -> List[Dict[str, Any]]:
        dense_results = self.dense.search(query, top_k=fetch_k, filter_dict=filter_dict)
        bm25_results = self.bm25.search(query, top_k=fetch_k, filter_dict=filter_dict)

        rrf_scores = {}
        doc_map = {}

        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            doc_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            if cid not in doc_map:
                doc_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            item = dict(doc_map[cid])
            item["score"] = rrf_scores[cid]
            item["rank"] = rank
            results.append(item)
        return results
