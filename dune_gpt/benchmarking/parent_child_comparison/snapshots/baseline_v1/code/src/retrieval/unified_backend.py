"""Unified DocDB + Indico retrieval backend for AskDUNE development."""

from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import CrossEncoder

from src.embedder.embedding_wrappers import OriginalEmbedder
from src.retrieval.parent_child_adapter import format_parent_child_results
from src.retrieval.unified_retriever import search_unified_parent_contexts


class UnifiedRetrievalBackend:
    """Search versioned DocDB and Indico parent-child collections."""

    def __init__(self, index_path: str):
        path = Path(index_path).resolve()

        if not path.is_dir():
            raise FileNotFoundError(f"Unified index not found: {path}")

        self.client = chromadb.PersistentClient(path=str(path))

        self.docdb = self.client.get_collection("DUNE_DOCDB_CHILDREN")
        self.manifest = self.client.get_collection("DUNE_DOCDB_VERSION_MANIFEST")
        self.indico = self.client.get_collection("DUNE_INDICO_CHILDREN")

        self.embedder = OriginalEmbedder(
            model_name="intfloat/e5-small-v2",
            device="cpu",
        )

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
        )

    def search(
        self,
        question: str,
        *,
        top_k: int = 5,
        k_embeddings: int = 30,
        **filters: Any,
    ) -> tuple[list[str], list[str]]:
        """Return contexts and reference URLs for the existing LLM pipeline."""

        result = search_unified_parent_contexts(
            self.docdb,
            self.manifest,
            self.indico,
            self.embedder,
            question,
            self.reranker,
            candidate_count=k_embeddings,
            top_k=top_k,
            **filters,
        )

        return format_parent_child_results(result)
