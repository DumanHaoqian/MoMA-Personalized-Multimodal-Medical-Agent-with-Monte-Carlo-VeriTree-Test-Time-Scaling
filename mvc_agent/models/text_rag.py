"""
Text retrieval augmented generation (RAG) module.

This module defines the ``TextRAG`` class, which implements semantic
retrieval over a QA dataset using sentence embeddings and FAISS.  It
provides methods for building an index, retrieving relevant chunks and
formatting prompts for downstream models.
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch


class TextRAG:
    def __init__(
        self,
        dataset_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        similarity_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = None  # type: Optional[Any]
        self.index = None  # type: Optional[Any]
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []

    def _load_dataset(self) -> Dict[str, Any]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_chunks(self, data: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        chunks: List[str] = []
        metadata_list: List[Dict[str, Any]] = []
        for pmid, entry in data.items():
            question = entry.get("QUESTION", "")
            contexts = entry.get("CONTEXTS", [])
            labels = entry.get("LABELS", [])
            meshes = entry.get("MESHES", [])
            year = entry.get("YEAR", "")
            final_decision = entry.get("final_decision", "")
            long_answer = entry.get("LONG_ANSWER", "")
            for idx, context in enumerate(contexts):
                chunk_text = f"Question: {question}\n\nContext: {context}"
                chunks.append(chunk_text)
                metadata = {
                    "pmid": pmid,
                    "question": question,
                    "context": context,
                    "context_idx": idx,
                    "label": labels[idx] if idx < len(labels) else "",
                    "meshes": meshes,
                    "year": year,
                    "final_decision": final_decision,
                    "long_answer": long_answer,
                }
                metadata_list.append(metadata)
        return chunks, metadata_list

    def _initialize_embedder(self):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.embedder = SentenceTransformer(self.embedding_model_name, device=self.device)

    def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        if self.embedder is None:
            self._initialize_embedder()
        embeddings = self.embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        return embeddings

    def _build_vector_store(self, embeddings: np.ndarray):
        import faiss  # type: ignore
        embedding_dim = embeddings.shape[1]
        embeddings = np.array(embeddings, dtype=np.float32, order='C')
        embeddings_norm = embeddings.copy()
        norms = np.linalg.norm(embeddings_norm, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings_norm / norms
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings_norm)
        self.index = index

    def build_index(self):
        data = self._load_dataset()
        self.chunks, self.chunk_metadata = self._create_chunks(data)
        embeddings = self._generate_embeddings(self.chunks)
        self._build_vector_store(embeddings)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        import faiss  # type: ignore
        if self.index is None:
            raise ValueError("Vector index not built. Call build_index() first.")
        if self.embedder is None:
            self._initialize_embedder()
        if top_k is None:
            top_k = self.top_k
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype=np.float32, order='C')
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        query_embedding = query_embedding / norm
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if self.similarity_threshold is not None and similarity < self.similarity_threshold:
                continue
            meta = self.chunk_metadata[idx].copy()
            meta['similarity'] = float(similarity)
            results.append((meta, float(similarity)))
        return results

    def format_prompt(self, query: str, retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
        prompt = (
            "You are a medical question‑answering assistant. Use the following retrieved contexts to answer the user's question.\n\n"
        )
        prompt += "=" * 80 + "\n\n"
        for i, (metadata, similarity) in enumerate(retrieved, 1):
            prompt += f"Context {i} (PMID: {metadata['pmid']}, Similarity: {similarity:.3f}):\n"
            prompt += f"Question: {metadata['question']}\n\n"
            prompt += f"{metadata['context']}\n\n"
            prompt += "-" * 80 + "\n\n"
        prompt += "=" * 80 + "\n\n"
        prompt += f"User Question: {query}\n\n"
        prompt += (
            "Based on the retrieved contexts above, provide a comprehensive and accurate answer to the user's question.\n"
            "If the contexts do not contain sufficient information, please indicate that clearly.\n"
        )
        return prompt


__all__ = ['TextRAG']