from copy import deepcopy
from collections import defaultdict
import os
import pickle
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from src.embedder.embedding_wrappers import OriginalEmbedder
from sentence_transformers import CrossEncoder
from config import (
    EMBEDDING_MODEL,
    MAX_VARIABLE_NUMBER,
    create_directories,
    DEFAULT_TOP_K,
)
import re
from src.utils.logger import get_logger
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import torch.nn as nn
import chromadb
from langchain_core.runnables import chain

logger = get_logger(__name__)
CHROMA_DB_NAME='DUNE_VECTOR_DB'
class ChromaManager:
    """Manager for ChromaDB index operations"""

    def __init__(self, db_path, indexing_only=False):
        # Prevent thread‐related segfaults
        self.bm25_cache=None
        self._configure_threading()

        self.db_path = db_path

        self.indexing_only = indexing_only
        self.reranker = None if indexing_only else CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512
        )

        self.chroma_client = chromadb.PersistentClient(path=self.db_path, settings=Settings())
        print("Collections available:", self.chroma_client.list_collections())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device: {self.device}")

        self.model = OriginalEmbedder(EMBEDDING_MODEL)
        logger.debug("Creating collection")
        try:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_DB_NAME,
                embedding_function=self.model
            )
        except Exception as e:
            logger.error(f"Error initiating chroma {e}")

        logger.debug(f"Using Chroma collection '{CHROMA_DB_NAME}' (count deferred)")
        
        # Setup device & model


        logger.debug(f"Loaded sentence transformer {EMBEDDING_MODEL}")

        self.indico_ids = defaultdict()
        self.docdb_versions = defaultdict()
        self.docdb_content_modified = defaultdict()
        self.docdb_metadata_modified = defaultdict()
        self.metadata= defaultdict()
        self.embedding_contents= defaultdict()
        self.entry_ids=set()
        self.attachment_ids=set()
        self.embedding_ids = []

        # Ensure directories exist
        create_directories(self.db_path)

        if indexing_only:
            self.fetch_ddb_ind_data_indexing()
        else:
            self.fetch_ddb_ind_data()

    
    def fetch_docdb_versions(self, batch_size: int = 1000) -> Dict[str, int]:
        """Build a lightweight DocDB index from IDs and metadata only."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        versions = {}
        docdb_index = {}
        offset = 0

        while True:
            results = self.chroma_collection.get(
                where={"source": "docdb"},
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )

            ids = results["ids"]
            metadatas = results["metadatas"]

            for chunk_id, md in zip(ids, metadatas):
                try:
                    docid = str(chunk_id).split("_")[0].split("/")[0]
                    version = int(md["docdb_version"])
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    logger.warning(
                        f"Invalid DocDB metadata for {chunk_id}: {exc}"
                    )
                    continue

                entry = docdb_index.setdefault(
                    docid,
                    {
                        "version": version,
                        "chunk_ids": [],
                        "attachments": set(),
                    },
                )

                entry["version"] = max(entry["version"], version)
                entry["chunk_ids"].append(chunk_id)

                filename = md.get("filename")
                if filename:
                    entry["attachments"].add(str(filename))

                versions[docid] = entry["version"]

            if len(ids) < batch_size:
                break

            offset += len(ids)

        self.docdb_index = docdb_index
        return versions

    def fetch_ddb_ind_data_indexing(self, batch_size=1000):
        """Load only existing IDs and metadata needed for indexing."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.docdb_index = {}
        offset = 0

        while True:
            results = self.chroma_collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )

            ids = results["ids"]
            metadatas = results["metadatas"]

            for chunk_id, md in zip(ids, metadatas):
                md = md or {}
                self.embedding_ids.append(chunk_id)

                entry_id = str(chunk_id).split("_")[0]
                docid = entry_id.split("/")[0]
                self.entry_ids.add(entry_id)

                filename = md.get("filename")
                if filename:
                    self.attachment_ids.add(f"{entry_id}_{filename}")

                if md.get("source") == "indico":
                    self.indico_ids[docid] = True

                elif md.get("source") == "docdb":
                    entry = self.docdb_index.setdefault(
                        docid,
                        {
                            "version": 0,
                            "chunk_ids": [],
                            "attachments": set(),
                        },
                    )
                    entry["chunk_ids"].append(chunk_id)

                    if filename:
                        entry["attachments"].add(str(filename))

                    try:
                        version = int(md["docdb_version"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid DocDB version for existing chunk "
                            f"{chunk_id}: {exc}"
                        ) from exc

                    entry["version"] = max(entry["version"], version)
                    self.docdb_versions[docid] = entry["version"]

            if len(ids) < batch_size:
                break

            offset += len(ids)

        logger.info(
            "Lightweight indexing initialization: %d chunks, "
            "%d DocDB documents, %d Indico events",
            len(self.embedding_ids),
            len(self.docdb_index),
            len(self.indico_ids),
        )

    def fetch_ddb_ind_data(self):
        results = self.chroma_collection.get(include=["documents", "metadatas", "uris"])
        for id, md, text in zip(results['ids'], results['metadatas'], results['documents'] ):
            if md.get('source') == 'indico':
                self.indico_ids[id.split('_')[0].split("/")[0]]=True
            elif md.get('source') == 'docdb':
                try:
                    docid = id.split("_")[0].split("/")[0]
                    version = int(md["docdb_version"])
                    self.docdb_versions[docid] = max(
                        int(self.docdb_versions.get(docid, 0)),
                        version,
                    )
                    self.docdb_content_modified[id]  = md['content_last_modified_date']
                    self.docdb_metadata_modified[id] = md['metadata_last_modified_date']
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(
                        f"DocDB metadata incomplete for {id}: {e}"
                    )
            self.metadata[id] = md
            self.embedding_contents[id]=text

            entry_id = id.split("_")[0]
            self.entry_ids.add(entry_id)

            filename = md.get("filename")

            if filename:
                unique_attachment_id = f"{entry_id}_{filename}"
                self.attachment_ids.add(unique_attachment_id)
            else:
                logger.warning(f"Missing filename for chunk {id}. Skipping attachment tracking.")

        self.embedding_ids = list(self.metadata.keys())
    
    
    def _configure_threading(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def add_to_chroma(self, chunks, ids, ids_to_idx_map, mode):
        '''
        Updates self.embedding_ids and adds/updates chroma with new chunks
        '''

        metadatas= []
        embedding_texts=[]
        up_ids=[]
        for i in ids:
            if chunks[ids_to_idx_map[i]].get('cleaned_text', None):
                up_ids.append(i)
            else:
                logger.warning(f"Cannot add {i} because didn't extract text from it")

        for id_ in up_ids:
            md = {}
            chunk_idx = ids_to_idx_map[id_]
            metadata_of_chunk_to_update = chunks[chunk_idx]
            embedding_texts.append(metadata_of_chunk_to_update['cleaned_text'])

            for k,v in metadata_of_chunk_to_update.items():
                if k not in ['cleaned_text', 'raw_text', 'document_id']:
                    md[k] = v
            metadatas.append(md)


        length = 0
        for text in embedding_texts:
            length += len(text)
        logger.debug(f'Storing chunks of total length {length}')

        if length == 0: return 0
        

        for i in range(0, len(up_ids), MAX_VARIABLE_NUMBER):
            if mode == 'update':
                self.chroma_collection.update(
                    ids = up_ids[i:i+MAX_VARIABLE_NUMBER],
                    documents = embedding_texts[i:i+MAX_VARIABLE_NUMBER],
                    metadatas = metadatas[i:i+MAX_VARIABLE_NUMBER],
                )

            elif mode == 'add':
                self.chroma_collection.add(
                    ids = up_ids[i:i+MAX_VARIABLE_NUMBER],
                    documents = embedding_texts[i:i+MAX_VARIABLE_NUMBER],
                    metadatas = metadatas[i:i+MAX_VARIABLE_NUMBER],
                )
            else:
                logger.error(f"Invalid argument mode={mode}. Must be 'add' or 'update")
                raise ValueError

        logger.debug(f"Added {len(up_ids)} to Chroma")
        return len(up_ids)


    def update_indico_docdb_record(self, chunks):
        """
        Updates global variables recording the status of the database with respect to specific metadata of each chunk
        """

        try:
            for chunk in chunks:
                source = chunk.get("source", '')
                did = chunk['document_id']
                if source == 'docdb':
                    self.docdb_versions[did.split('_')[0].split("/")[0]] = chunk['docdb_version']
                    self.docdb_content_modified[did] = chunk.get('content_last_modified_date', 'Unknown Content Date')
                    self.docdb_metadata_modified[did] = chunk.get('metadata_last_modified_date', 'Unknown Metadata Date')

                elif source == 'indico':
                    self.indico_ids[did.split("_")[0].split("/")[0]]=True

                entry_id = did.split("_")[0]
                self.entry_ids.add(entry_id)

                filename = chunk.get("filename")

                if filename:
                    unique_attachment_id = f"{entry_id}_{filename}"
                    self.attachment_ids.add(unique_attachment_id)
                else:
                    logger.warning(f"Missing filename for chunk {id}. Skipping attachment tracking.")

        except Exception as e:
            logger.error(f"Error in updating list of embedding IDs with Indico/DocDB: {e}")

    def add_entries(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Adds new and updates existing chunks to ChromaDB
        """
        map_ids_to_idx = {chunk['document_id']:idx for idx,chunk in enumerate(chunks)}

        self.update_indico_docdb_record(chunks)

        ids = set(map_ids_to_idx.keys())
        existing_ids = set(self.embedding_ids)

        ids_to_update = list(existing_ids.intersection(ids))
        added = self.add_to_chroma(chunks=chunks, ids=ids_to_update, ids_to_idx_map=map_ids_to_idx, mode='update')

        ids_to_add = ids - existing_ids

        if ids_to_add:
            added += self.add_to_chroma(chunks= chunks, ids= list(ids_to_add), ids_to_idx_map=map_ids_to_idx, mode='add')

        self.embedding_ids = list(existing_ids.union(ids))
        return added
                
    def get_indico_ids(self) -> Dict[str, bool]:
        """
        Return a map { entry_id: True } for all Indico events.
        """
        return self.indico_ids


    def get_docdb_document_backup(self, docid: int) -> Dict[str, Any]:
        """Back up one DocDB document directly from ChromaDB."""
        prefix = f"{int(docid)}_"
        records = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }

        # Use the lightweight startup index to locate this document's chunks.
        docdb_index = getattr(self, "docdb_index", None)
        if docdb_index is None:
            raise RuntimeError(
                "DocDB index is not initialized; call fetch_docdb_versions first"
            )

        entry = docdb_index.get(str(int(docid)))
        target_ids = list(entry["chunk_ids"]) if entry else []

        if len(target_ids) != len(set(target_ids)):
            raise RuntimeError(
                f"DocDB {docid}: duplicate chunk IDs in lightweight index"
            )

        # Retrieve full records only for the document being replaced.
        for start in range(0, len(target_ids), MAX_VARIABLE_NUMBER):
            batch = target_ids[start:start + MAX_VARIABLE_NUMBER]
            page = self.chroma_collection.get(
                ids=batch,
                include=["documents", "metadatas", "embeddings"],
            )

            if set(page["ids"]) != set(batch):
                raise RuntimeError(
                    f"DocDB {docid}: backup retrieval incomplete"
                )

            for i, record_id in enumerate(page["ids"]):
                records["ids"].append(record_id)
                records["documents"].append(page["documents"][i])
                records["metadatas"].append(page["metadatas"][i])
                records["embeddings"].append(page["embeddings"][i])

        return records

    def validate_docdb_replacement(
        self,
        docid: int,
        version: int,
        chunks: List[Dict[str, Any]],
        backup: Dict[str, Any],
    ) -> bool:
        """Validate a complete, strictly newer DocDB replacement."""
        docid = int(docid)
        version = int(version)

        if not chunks:
            raise ValueError(f"DocDB {docid}: no replacement chunks")

        ids = []
        for chunk in chunks:
            chunk_id = str(chunk.get("document_id", ""))
            if not chunk_id.startswith(f"{docid}_"):
                raise ValueError(f"DocDB {docid}: invalid chunk ID {chunk_id}")
            if chunk.get("source") != "docdb":
                raise ValueError(f"DocDB {docid}: incorrect chunk source")
            if int(chunk.get("docdb_version", -1)) != version:
                raise ValueError(f"DocDB {docid}: inconsistent chunk version")
            if not str(chunk.get("cleaned_text", "")).strip():
                raise ValueError(f"DocDB {docid}: empty chunk")
            ids.append(chunk_id)

        if len(ids) != len(set(ids)):
            raise ValueError(f"DocDB {docid}: duplicate chunk IDs")

        old_versions = [
            int(md["docdb_version"])
            for md in backup["metadatas"]
        ]

        if old_versions and version <= max(old_versions):
            logger.info(
                f"DocDB {docid}: version {version} is not newer "
                f"than indexed version {max(old_versions)}. Skipping."
            )
            return False

        return True

    def check_docdb_id_collisions(
        self,
        chunks: List[Dict[str, Any]],
        backup: Dict[str, Any],
    ) -> None:
        """Reject new chunk IDs already owned by unrelated records."""
        new_ids = [str(chunk["document_id"]) for chunk in chunks]
        old_ids = set(backup["ids"])

        for start in range(0, len(new_ids), MAX_VARIABLE_NUMBER):
            batch = new_ids[start:start + MAX_VARIABLE_NUMBER]
            existing = self.chroma_collection.get(
                ids=batch,
                include=["metadatas"],
            )

            for record_id, md in zip(
                existing["ids"], existing["metadatas"]
            ):
                if record_id not in old_ids:
                    raise ValueError(
                        f"Chunk ID collision with unrelated record: "
                        f"{record_id}, source={md.get('source')}"
                    )

    def validate_docdb_backup(
        self,
        docid: int,
        backup: Dict[str, Any],
    ) -> None:
        """Check backup integrity before any destructive operation."""
        ids = backup["ids"]
        n = len(ids)

        for key in ("documents", "metadatas", "embeddings"):
            if len(backup[key]) != n:
                raise ValueError(
                    f"DocDB {docid}: incomplete backup field {key}"
                )

        if len(set(ids)) != n:
            raise ValueError(f"DocDB {docid}: duplicate backup IDs")

        prefix = f"{int(docid)}_"

        for i, record_id in enumerate(ids):
            md = backup["metadatas"][i]

            if not record_id.startswith(prefix):
                raise ValueError(
                    f"DocDB {docid}: unrelated record in backup"
                )

            if md is None or md.get("source") != "docdb":
                raise ValueError(
                    f"DocDB {docid}: invalid backup source"
                )

            if backup["documents"][i] is None:
                raise ValueError(
                    f"DocDB {docid}: missing backed-up document"
                )

            if backup["embeddings"][i] is None:
                raise ValueError(
                    f"DocDB {docid}: missing backed-up embedding"
                )

    def restore_docdb_backup(
        self,
        backup: Dict[str, Any],
    ) -> None:
        """Restore previously backed-up chunks and their original embeddings."""
        ids = backup["ids"]
        if not ids:
            return

        for start in range(0, len(ids), MAX_VARIABLE_NUMBER):
            end = start + MAX_VARIABLE_NUMBER
            self.chroma_collection.add(
                ids=ids[start:end],
                documents=backup["documents"][start:end],
                metadatas=backup["metadatas"][start:end],
                embeddings=[
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                    for embedding in backup["embeddings"][start:end]
                ],
            )

        expected = {
            record_id: i for i, record_id in enumerate(ids)
        }
        restored = set()

        for start in range(0, len(ids), MAX_VARIABLE_NUMBER):
            page = self.chroma_collection.get(
                ids=ids[start:start + MAX_VARIABLE_NUMBER],
                include=["documents", "metadatas", "embeddings"],
            )

            for j, record_id in enumerate(page["ids"]):
                i = expected[record_id]
                if page["documents"][j] != backup["documents"][i]:
                    raise RuntimeError(
                        f"DocDB rollback: text mismatch for {record_id}"
                    )
                if page["metadatas"][j] != backup["metadatas"][i]:
                    raise RuntimeError(
                        f"DocDB rollback: metadata mismatch for {record_id}"
                    )
                if not np.allclose(
                    page["embeddings"][j],
                    backup["embeddings"][i],
                    rtol=1e-5,
                    atol=1e-7,
                ):
                    raise RuntimeError(
                        f"DocDB rollback: embedding mismatch for {record_id}"
                    )
                restored.add(record_id)

        if restored != set(ids):
            raise RuntimeError(
                "DocDB rollback failed: not all original chunks were restored"
            )

    def verify_docdb_backup_completeness(
        self,
        docid: int,
        backup: Dict[str, Any],
    ) -> None:
        """Verify the backup against the lightweight DocDB index."""
        docid = str(int(docid))

        if not hasattr(self, "docdb_index"):
            raise RuntimeError("Lightweight DocDB index is not initialized")

        entry = self.docdb_index.get(docid)
        expected_list = list(entry["chunk_ids"]) if entry else []
        expected_ids = set(expected_list)
        backed_up_ids = set(backup["ids"])

        if (
            len(expected_list) != len(expected_ids)
            or expected_ids != backed_up_ids
            or len(backup["ids"]) != len(backed_up_ids)
        ):
            raise RuntimeError(
                f"DocDB {docid}: backup incomplete; "
                f"expected {len(expected_ids)} chunks, "
                f"backed up {len(backed_up_ids)}"
            )

    def insert_verified_docdb_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Insert complete DocDB chunks and verify stored records."""
        ids = [str(chunk["document_id"]) for chunk in chunks]

        for start in range(0, len(chunks), MAX_VARIABLE_NUMBER):
            batch = chunks[start:start + MAX_VARIABLE_NUMBER]
            batch_ids = [str(c["document_id"]) for c in batch]
            documents = [c["cleaned_text"] for c in batch]
            metadatas = [
                {
                    k: v for k, v in c.items()
                    if k not in ("document_id", "cleaned_text", "raw_text")
                }
                for c in batch
            ]

            self.chroma_collection.add(
                ids=batch_ids,
                documents=documents,
                metadatas=metadatas,
            )

        expected = {
            str(chunk["document_id"]): chunk for chunk in chunks
        }
        verified = set()

        for start in range(0, len(ids), MAX_VARIABLE_NUMBER):
            page = self.chroma_collection.get(
                ids=ids[start:start + MAX_VARIABLE_NUMBER],
                include=["documents", "metadatas"],
            )

            for i, record_id in enumerate(page["ids"]):
                chunk = expected[record_id]
                expected_md = {
                    k: v for k, v in chunk.items()
                    if k not in ("document_id", "cleaned_text", "raw_text")
                }

                if page["documents"][i] != chunk["cleaned_text"]:
                    raise RuntimeError(
                        f"DocDB insertion: text mismatch for {record_id}"
                    )
                if page["metadatas"][i] != expected_md:
                    raise RuntimeError(
                        f"DocDB insertion: metadata mismatch for {record_id}"
                    )
                verified.add(record_id)

        if verified != set(ids):
            raise RuntimeError(
                "DocDB insertion failed: not all chunks were stored"
            )

    def update_docdb_replacement_cache(
        self,
        docid: int,
        version: int,
        chunks: List[Dict[str, Any]],
        backup: Dict[str, Any],
    ) -> None:
        """Synchronize in-memory caches after successful DocDB replacement."""
        docid = str(int(docid))
        old_ids = set(backup["ids"])

        for record_id in old_ids:
            self.metadata.pop(record_id, None)
            self.embedding_contents.pop(record_id, None)
            self.docdb_content_modified.pop(record_id, None)
            self.docdb_metadata_modified.pop(record_id, None)

        for chunk in chunks:
            record_id = str(chunk["document_id"])
            if not self.indexing_only:
                self.metadata[record_id] = {
                    k: v for k, v in chunk.items()
                    if k not in ("document_id", "cleaned_text", "raw_text")
                }
                self.embedding_contents[record_id] = chunk["cleaned_text"]
            if not self.indexing_only:
                self.docdb_content_modified[record_id] = chunk.get(
                    "content_last_modified_date", "Unknown Content Date"
                )
                self.docdb_metadata_modified[record_id] = chunk.get(
                    "metadata_last_modified_date", "Unknown Metadata Date"
                )

        self.docdb_versions[docid] = int(version)
        if self.indexing_only:
            old_id_set = set(old_ids)
            self.embedding_ids = [
                record_id for record_id in self.embedding_ids
                if record_id not in old_id_set
            ]
            self.embedding_ids.extend(
                str(chunk["document_id"]) for chunk in chunks
            )
        else:
            self.embedding_ids = list(self.metadata.keys())
        self.entry_ids.add(docid)

        self.attachment_ids = {
            attachment_id
            for attachment_id in self.attachment_ids
            if not attachment_id.startswith(f"{docid}_")
        }

        for chunk in chunks:
            filename = chunk.get("filename")
            if filename:
                self.attachment_ids.add(f"{docid}_{filename}")

        # Synchronize the lightweight DocDB index after replacement.
        self.docdb_index[docid] = {
            "version": int(version),
            "chunk_ids": [str(c["document_id"]) for c in chunks],
            "attachments": {
                str(c["filename"])
                for c in chunks
                if c.get("filename")
            },
        }

        self.bm25_cache = None

    def replace_docdb_document(
        self,
        docid: int,
        version: int,
        chunks: List[Dict[str, Any]],
    ) -> bool:
        """Replace one fully extracted DocDB document, with rollback."""
        docid = int(docid)
        version = int(version)

        # Skip unchanged or older versions before retrieving a backup.
        indexed_version = int(
            self.docdb_versions.get(str(docid), 0)
        )
        if version <= indexed_version:
            return False

        # No database mutation until every preflight check passes.
        backup = self.get_docdb_document_backup(docid)
        self.validate_docdb_backup(docid, backup)
        self.verify_docdb_backup_completeness(docid, backup)

        if not self.validate_docdb_replacement(
            docid, version, chunks, backup
        ):
            return False

        self.check_docdb_id_collisions(chunks, backup)

        old_ids = list(backup["ids"])
        new_ids = [str(c["document_id"]) for c in chunks]

        # Snapshot only this document's cache state before mutation.
        prefix = f"{docid}_"
        cache_names = (
            "metadata",
            "embedding_contents",
            "docdb_content_modified",
            "docdb_metadata_modified",
        )
        cache_snapshot = {}
        for name in cache_names:
            cache = getattr(self, name)
            cache_snapshot[name] = {
                key: value
                for key, value in cache.items()
                if key.startswith(prefix)
            }

        # Preserve the lightweight index entry for rollback.
        previous_docdb_index_entry = deepcopy(
            self.docdb_index.get(str(docid))
        )
        previous_version = self.docdb_versions.get(str(docid))
        previous_embedding_ids = self.embedding_ids
        previous_entry_ids = set(self.entry_ids)
        previous_attachment_ids = set(self.attachment_ids)
        previous_bm25_cache = self.bm25_cache

        try:
            # Delete the old version before inserting the new version.
            for start in range(0, len(old_ids), MAX_VARIABLE_NUMBER):
                self.chroma_collection.delete(
                    ids=old_ids[start:start + MAX_VARIABLE_NUMBER]
                )

            self.insert_verified_docdb_chunks(chunks)
            self.update_docdb_replacement_cache(
                docid, version, chunks, backup
            )

        except Exception as original_error:
            try:
                # Remove both partial new records and surviving old records.
                # This allows restoration even when IDs overlap.
                cleanup_ids = list(set(old_ids + new_ids))
                for start in range(
                    0, len(cleanup_ids), MAX_VARIABLE_NUMBER
                ):
                    self.chroma_collection.delete(
                        ids=cleanup_ids[start:start + MAX_VARIABLE_NUMBER]
                    )

                if old_ids:
                    self.restore_docdb_backup(backup)

                # Restore this document's in-memory cache entries.
                for name in cache_names:
                    cache = getattr(self, name)
                    for key in list(cache):
                        if key.startswith(prefix):
                            cache.pop(key, None)
                    cache.update(cache_snapshot[name])

                if previous_version is None:
                    self.docdb_versions.pop(str(docid), None)
                else:
                    self.docdb_versions[str(docid)] = previous_version

                if previous_docdb_index_entry is None:
                    self.docdb_index.pop(str(docid), None)
                else:
                    self.docdb_index[str(docid)] = previous_docdb_index_entry

                self.embedding_ids = previous_embedding_ids
                self.entry_ids = previous_entry_ids
                self.attachment_ids = previous_attachment_ids
                self.bm25_cache = previous_bm25_cache

            except Exception as rollback_error:
                raise RuntimeError(
                    f"CRITICAL: DocDB {docid} replacement failed and "
                    f"rollback failed. Original error: {original_error}; "
                    f"rollback error: {rollback_error}"
                ) from rollback_error

            raise RuntimeError(
                f"DocDB {docid} replacement failed; "
                f"original version restored: {original_error}"
            ) from original_error

        return True

    def get_docdb_versions(self) -> Dict[str, int]:
        """
        Return a map { entry_id: max_version_indexed } for all DocDB entries.
        """
        return self.docdb_versions
    
    def get_content_modification_dates(self) -> Dict[str, str]:
        return self.docdb_content_modified

    def get_metadata_modification_dates(self) -> Dict[str, str]:
        return self.docdb_metadata_modified


    def without_reranker_search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
            Finds the embedding_id associated with the retrieved text, then extracts header info from metadata
        """
        results = self.chroma_collection.query(query_texts=[query],  n_results=top_k)
        snippets, refs = [], []
        for md in results['metadatas'][0]:
            link  = md.get("event_url", "")
            title = md.get("meeting_name", "")
            if link:
                refs.append(link)
        for text in results['documents'][0]:
            snippets.append(text)

        return snippets, refs

    def search_old(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
            Finds the embedding_id associated with the retrieved text, then extracts header info from metadata
        """
        results = self.chroma_collection.query(query_texts=[query],  n_results=top_k*6)

        scores = []
        scores = self.reranker.predict(
            [(query, text) for text in results["documents"][0]],
            batch_size=6)

        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        snippets, refs = [], []
        for i in top_k_indices:
            md = results['metadatas'][0][i]
            link  = md.get("event_url", "")
            title = md.get("meeting_name", "")
            if link:
                refs.append(link)
            snippets.append(results['documents'][0][i])
        return snippets, refs

    def tokenize(self, text):
        return re.findall(r"\w+", text.lower())
    
    def build_bm25_index(self, embedding_texts):
        tokenized_texts = [self.tokenize(text) for text in embedding_texts]
        self.bm25_cache = BM25Okapi(tokenized_texts)
        
    def keyword_search(self, query, k_embeddings):
        """
        Performs keyword search returning k_embeddings chunks
        """
        all_entries = self.chroma_collection.get()
        if not self.bm25_cache:
            self.build_bm25_index(all_entries['documents'])
        
        bm25_scores = self.bm25_cache.get_scores(query.split())
        top_indices = np.argsort(bm25_scores)[::-1][:k_embeddings]
        results = [(all_entries['ids'][idx], bm25_scores[idx]) for idx in top_indices]
        return results

    def semantic_search(self, query, doc_type, k_embeddings):

        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=k_embeddings,
            where={'document_type': doc_type}
        )
        if not results['ids'][0]:
            return []
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        # Assuming cosine distance, similarity = 1 - distance
        similarities = [1 - dist for dist in results['distances'][0]]
        
        return list(zip(results['ids'][0], similarities))
    
    def merge(self, keyword_embedding_ids, semantic_embedding_ids):
        """
            Merges embedding_ids extracted from semantic search and keyword search
        """

        def normalize_scores(results):
            if not results:
                return {}
            scores = [score for _, score in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score if max_score > min_score else 1
            return {
                embedding_id: (score - min_score) / score_range 
                for embedding_id, score in results
            }
        keyword_scores = normalize_scores(keyword_embedding_ids)
        semantic_scores = normalize_scores(semantic_embedding_ids)
        all_selected_ids = keyword_scores.keys() | semantic_scores.keys()
        combined_score=defaultdict()
        for id_ in all_selected_ids:
            keyword_score = keyword_scores.get(id_,0)
            semantic_score = semantic_scores.get(id_,0)
            combined_score[id_] = 0.5*keyword_score + 0.5*semantic_score
        
        merged_ids = sorted(
            combined_score.keys(),
            key=lambda x: combined_score[x],
            reverse=True
        )
        return merged_ids

        

    def reranker_search(self, query, merged_embedding_ids, top_k):
        embedding_texts=[]
        try:
            embedding_texts = [self.embedding_contents[embedding_id] for embedding_id in merged_embedding_ids]
        except:
            print("No embeddings to rerank")
            return []
                
        # Create query-document pairs
        pairs = [[query, text] for text in embedding_texts]
        
        # Get cross-encoder scores
        ce_scores = self.reranker.predict(pairs)
        
        # Return sorted by score
        results = list(zip(merged_embedding_ids, ce_scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


    def get_links(self, reranked_embedding_ids):
        return [self.metadata[id_[0]]['event_url'] for id_ in reranked_embedding_ids]

    def get_content(self, reranked_embedding_ids):
        return [self.embedding_contents[id_[0]] for id_ in reranked_embedding_ids]

    def search(self, query: str,top_k: int = 3, k_embeddings: int = 2,  keyword=True) -> Tuple[List[str], List[str]]:
        """
            Performs keyword and semantic search (k_embeddings chunks each), reranking each output and selecting the best top_k
        """
      
        if keyword:
            keyword_embedding_ids = self.keyword_search(query, 3*k_embeddings)
        else:
            keyword_embedding_ids = []
        
        semantic_embedding_ids = self.semantic_search(query, 'document', k_embeddings)
        semantic_embedding_ids.extend(self.semantic_search(query, 'slides', k_embeddings))
        merged_embedding_ids = self.merge(keyword_embedding_ids, semantic_embedding_ids)

        reranked_embedding_ids = self.reranker_search(query, merged_embedding_ids, top_k)

        links = self.get_links(reranked_embedding_ids)
        content = self.get_content(reranked_embedding_ids)
        return content, links

    def save_all(self):
        """Persist metadata, embedding_ids, and ChromaDB index."""
        pass

    def get_docdb_index_stats(self) -> Dict[str, int]:
        """Return cumulative DocDB counts without loading document contents."""
        if not self.indexing_only:
            raise RuntimeError(
                "DocDB index statistics require indexing-only mode"
            )

        return {
            "total_docdb_pages": len(self.docdb_index),
            "total_docdb_attachments": sum(
                len(entry["attachments"])
                for entry in self.docdb_index.values()
            ),
            "total_docdb_chunks": sum(
                len(entry["chunk_ids"])
                for entry in self.docdb_index.values()
            ),
            "total_chromadb_chunks": self.chroma_collection.count(),
        }

    def get_stats(self) -> Dict[str, int]:
        disk_size = 0
        for path, dirs, files in os.walk(self.db_path):
            for f in files:
                fp = os.path.join(path, f)
                disk_size += os.path.getsize(fp)

        return {
            "total_entries": len(self.entry_ids),
            "total_attachments": len(self.attachment_ids),
            "total_embeddings": self.chroma_collection.count(),
            "disk_size": disk_size
        }

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.chroma_collection = None
        self.entry_ids.clear()
        self.attachment_ids.clear()
        self.indico_ids = None
        self.docdb_versions = None
        self.docdb_content_modified = None
        self.docdb_metadata_modified = None
        self.embedding_ids = None

    def as_retriever(self, **kwargs) -> dict:
        @chain
        def fetch_documents(question: str):
            return self.search(question, **kwargs)
        
        return fetch_documents
