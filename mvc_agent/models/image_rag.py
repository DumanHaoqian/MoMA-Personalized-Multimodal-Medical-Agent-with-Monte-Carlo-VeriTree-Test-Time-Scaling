"""
Image retrieval augmented generation module.

This module defines the ``ImageRAG`` class, which builds a CLIP embedding
index over a dataset of images and captions.  It supports retrieval by
image or by text and returns ranked results with similarity scores.
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageRAG:
    def __init__(
        self,
        dataset_jsonl_path: str,
        image_embedding_model: str = "openai/clip-vit-base-patch32",
        top_k: int = 3,
        device: Optional[str] = None,
    ):
        self.dataset_jsonl_path = dataset_jsonl_path
        self.image_embedding_model_name = image_embedding_model
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None
        self.index: Optional[Any] = None
        self.image_paths: List[str] = []
        self.captions: List[str] = []
        self._image_embeddings: Optional[np.ndarray] = None

    def _load_jsonl(self) -> None:
        if not os.path.exists(self.dataset_jsonl_path):
            raise FileNotFoundError(f"JSONL dataset not found: {self.dataset_jsonl_path}")
        image_paths: List[str] = []
        captions: List[str] = []
        with open(self.dataset_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                image_path = obj.get("image_path") or obj.get("image") or obj.get("path")
                caption = obj.get("caption") or obj.get("text") or ""
                if image_path:
                    image_paths.append(image_path)
                    captions.append(caption)
        if not image_paths:
            raise RuntimeError("No valid image_path entries found in JSONL dataset.")
        self.image_paths = image_paths
        self.captions = captions

    def _initialize_clip(self):
        if self.clip_model is None or self.clip_processor is None:
            self.clip_model = CLIPModel.from_pretrained(self.image_embedding_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.image_embedding_model_name)
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()

    def _load_images(self, paths: List[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(220, 220, 220))
            images.append(img)
        return images

    def _generate_image_embeddings(self, batch_size: int = 64) -> np.ndarray:
        if not self.image_paths:
            self._load_jsonl()
        self._initialize_clip()
        images = self._load_images(self.image_paths)
        all_embeds: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i + batch_size]
                inputs = self.clip_processor(images=batch_imgs, return_tensors="pt").to(self.device)
                image_features = self.clip_model.get_image_features(**inputs)
                all_embeds.append(image_features.detach().cpu().numpy())
        embeddings_np = np.concatenate(all_embeds, axis=0)
        self._image_embeddings = embeddings_np
        return embeddings_np

    def _build_vector_store(self, embeddings: Optional[np.ndarray] = None):
        import faiss  # type: ignore
        if embeddings is None:
            embeddings = self._image_embeddings
            if embeddings is None:
                embeddings = self._generate_image_embeddings()
        embedding_dim = embeddings.shape[1]
        embeddings = np.array(embeddings, dtype=np.float32, order="C")
        normalized = embeddings.copy()
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = normalized / norms
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(normalized)
        self.index = index

    def build_index(self):
        self._load_jsonl()
        embeddings = self._generate_image_embeddings()
        self._build_vector_store(embeddings)

    def _ensure_index(self):
        if self.index is None:
            if self._image_embeddings is None:
                self._generate_image_embeddings()
            self._build_vector_store(self._image_embeddings)

    def retrieve_by_image(self, query_image_path: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_index()
        self._initialize_clip()
        if top_k is None:
            top_k = self.top_k
        query_img = self._load_images([query_image_path])[0]
        with torch.no_grad():
            inputs = self.clip_processor(images=[query_img], return_tensors="pt").to(self.device)
            query_features = self.clip_model.get_image_features(**inputs)
            query_embedding = query_features.detach().cpu().numpy().astype(np.float32, copy=False)
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        query_embedding = query_embedding / norm
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        scores, indices = scores[0], indices[0]
        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            results.append({
                "rank": rank,
                "index": int(idx),
                "image_path": self.image_paths[idx],
                "caption": self.captions[idx],
                "similarity": float(score),
            })
        return results

    def retrieve_by_text(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_index()
        self._initialize_clip()
        if top_k is None:
            top_k = self.top_k
        with torch.no_grad():
            inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
            text_features = self.clip_model.get_text_features(**inputs)
            query_embedding = text_features.detach().cpu().numpy().astype(np.float32, copy=False)
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        query_embedding = query_embedding / norm
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        scores, indices = scores[0], indices[0]
        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            results.append({
                "rank": rank,
                "index": int(idx),
                "image_path": self.image_paths[idx],
                "caption": self.captions[idx],
                "similarity": float(score),
            })
        return results

    def format_prompt(self, user_query_text: str, retrieved: List[Dict[str, Any]], max_caps: int = 3) -> str:
        prompt = "You are a helpful assistant. Use the following retrieved captions to answer the user.\n\n"
        prompt += "=" * 80 + "\n\n"
        for i, item in enumerate(retrieved[:max_caps], 1):
            prompt += f"Retrieved Caption {i} (Similarity: {item['similarity']:.3f}):\n"
            prompt += item["caption"].strip() + "\n\n"
        prompt += "=" * 80 + "\n\n"
        prompt += f"User Input: {user_query_text}\n"
        prompt += "Provide a concise and accurate answer grounded in the retrieved captions. If insufficient, say so.\n"
        return prompt


__all__ = ['ImageRAG']