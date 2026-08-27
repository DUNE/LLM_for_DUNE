import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config import CHUNK_SIZE, EMBEDDING_MODEL
from src.extractors.base import BaseExtractor
from src.indexing.chroma_manager import ChromaManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LocalTextExtractor(BaseExtractor):
    def extract_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        return []


class LocalAttachmentProcessor:
    """Indexes previously cached raw attachments into Chroma."""

    def __init__(
        self,
        chroma_path: str,
        cache_path: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_strategy: str = "word",
        chunk_overlap: int = 0,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.root = Path(__file__).resolve().parents[2]
        self.cache_path = self._resolve_path(cache_path)
        self.chunk_size = int(chunk_size)
        self.chunk_strategy = chunk_strategy
        self.chunk_overlap = int(chunk_overlap or 0)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.embedding_model = embedding_model
        self.extractor = LocalTextExtractor()
        self.chroma_manager = ChromaManager(
            chroma_path,
            embedding_model=self.embedding_model,
            load_reranker=False,
        )

    def _resolve_path(self, value: str | Path) -> Path:
        normalized = str(value).replace("\\", os.sep).replace("/", os.sep)
        path = Path(normalized)
        return path if path.is_absolute() else self.root / path

    def _iter_manifest_records(self, source: str) -> Iterable[Dict[str, Any]]:
        manifest = self.cache_path / source / "metadata.jsonl"
        if not manifest.exists():
            logger.warning(f"No manifest found for {source}: {manifest}")
            return

        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("error") or not record.get("path"):
                    continue
                yield record

    def _content_type(self, record: Dict[str, Any], path: Path) -> str:
        content_type = record.get("content_type") or ""
        if content_type:
            return content_type
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or ""

    def _base_metadata(self, record: Dict[str, Any], source: str) -> Dict[str, Any]:
        if source == "docdb":
            return {
                "source": "docdb",
                "docid": str(record.get("docid", "")),
                "docdb_version": int(record.get("docdb_version") or 1),
                "title": record.get("title", ""),
                "event_url": record.get("page_url", ""),
                "attachment_url": record.get("attachment_url", ""),
                "filename": record.get("filename", ""),
            }

        return {
            "source": "indico",
            "event_id": str(record.get("event_id", "")),
            "category_id": str(record.get("category_id", "")),
            "meeting_name": record.get("event_title", ""),
            "event_url": record.get("event_url", ""),
            "attachment_url": record.get("attachment_url", ""),
            "filename": record.get("filename", ""),
            "attachment_title": record.get("attachment_title", ""),
        }

    def _document_root_id(self, record: Dict[str, Any], source: str, index: int) -> str:
        if source == "docdb":
            return f"local_docdb_{record.get('docid', 'unknown')}_att{index:03d}"
        return f"local_indico_{record.get('event_id', 'unknown')}_att{index:03d}"

    def _chunk_text(self, raw_text: str) -> List[Dict[str, Any]]:
        if self.chunk_strategy == "word":
            words = raw_text.split()
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            for start in range(0, len(words), step):
                end = min(start + self.chunk_size, len(words))
                chunks.append(
                    {
                        "text": " ".join(words[start:end]),
                        "chunk_start_word": start,
                        "chunk_end_word": end,
                        "chunk_start_char": -1,
                        "chunk_end_char": -1,
                    }
                )
                if end >= len(words):
                    break
            return chunks
        if self.chunk_strategy == "char":
            chunks = []
            step = self.chunk_size - self.chunk_overlap
            for start in range(0, len(raw_text), step):
                end = min(start + self.chunk_size, len(raw_text))
                chunk = raw_text[start:end]
                chunks.append(
                    {
                        "text": chunk,
                        "chunk_start_word": len(raw_text[:start].split()),
                        "chunk_end_word": len(raw_text[:end].split()),
                        "chunk_start_char": start,
                        "chunk_end_char": end,
                    }
                )
                if end >= len(raw_text):
                    break
            return chunks
        raise ValueError(f"Unsupported chunk_strategy={self.chunk_strategy}. Use 'word' or 'char'.")

    def _record_to_chunks(self, record: Dict[str, Any], source: str, index: int) -> List[Dict[str, Any]]:
        file_path = self._resolve_path(record["path"])
        if not file_path.exists():
            logger.warning(f"Cached attachment missing: {file_path}")
            return []

        content = file_path.read_bytes()
        content_type = self._content_type(record, file_path)
        raw_text, document_type = self.extractor.get_raw_text(content_type, content)
        if not raw_text:
            logger.warning(f"No text extracted from {file_path}")
            return []

        chunks = self._chunk_text(raw_text)
        root_id = self._document_root_id(record, source, index)
        metadata = self._base_metadata(record, source)
        metadata["content_type"] = content_type
        metadata["document_type"] = document_type or "document"
        metadata["local_path"] = str(file_path.relative_to(self.root))
        metadata["benchmark_chunk_size"] = self.chunk_size
        metadata["benchmark_chunk_strategy"] = self.chunk_strategy
        metadata["benchmark_chunk_overlap"] = self.chunk_overlap
        metadata["benchmark_embedding_model"] = self.embedding_model

        docs = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            docs.append(
                {
                    **metadata,
                    "document_id": f"{root_id}_chunk{chunk_index:03d}",
                    "attachment_index": index,
                    "chunk_index": chunk_index,
                    "chunk_start_word": chunk["chunk_start_word"],
                    "chunk_end_word": chunk["chunk_end_word"],
                    "chunk_start_char": chunk["chunk_start_char"],
                    "chunk_end_char": chunk["chunk_end_char"],
                    "cleaned_text": chunk["text"],
                }
            )
        return docs

    def process(self, source: str = "both", limit: int | None = None) -> Dict[str, int]:
        sources = ["docdb", "indico"] if source == "both" else [source]
        results = {
            "attachments_seen": 0,
            "attachments_with_text": 0,
            "chunks_created": 0,
            "embeddings_added": 0,
        }

        for source_name in sources:
            batch: List[Dict[str, Any]] = []
            for index, record in enumerate(self._iter_manifest_records(source_name), start=1):
                if limit is not None and results["attachments_seen"] >= limit:
                    break

                results["attachments_seen"] += 1
                docs = self._record_to_chunks(record, source_name, index)
                if docs:
                    results["attachments_with_text"] += 1
                    results["chunks_created"] += len(docs)
                    batch.extend(docs)

            if batch:
                results["embeddings_added"] += self.chroma_manager.add_documents(batch, results["attachments_with_text"])

        return results

    def get_index_stats(self) -> Dict[str, int]:
        return self.chroma_manager.get_stats()

    def cleanup(self) -> None:
        self.extractor = None
        self.chroma_manager.cleanup()
