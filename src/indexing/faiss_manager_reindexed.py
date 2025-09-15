from collections import defaultdict
import os
import pickle
import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    DOC_IDS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    create_directories,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSManager:
    """Manager for FAISS index operations"""

    def __init__(self):
        # Prevent thread‐related segfaults
        self._configure_threading()
        self.count=0
        # Setup device & model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device, token=False)
        logger.info(f"Loaded sentence transformer {EMBEDDING_MODEL}")
        
        self.metadata_store=defaultdict( dict)


        # Ensure directories exist
        create_directories()

        # Load or initialize on‐disk data
        self.metadata_store: Dict[str, Dict[str, Any]] = self._load_metadata()
        
      
        self.doc_ids: List[str] = self._load_doc_ids()
        self.faiss_index: faiss.Index = self._load_faiss_index()

        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} entries")
        logger.info(f"Loaded metadata store with {len(self.metadata_store)} entries")
        logger.info(f"Loaded doc_ids with {len(self.doc_ids)} entries")

    def _configure_threading(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def _load_metadata(self) -> Dict[str, Any]:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_metadata(self):
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata_store, f)

    def _load_doc_ids(self) -> List[str]:
        if DOC_IDS_PATH.exists():
            with open(DOC_IDS_PATH, "rb") as f:
                return pickle.load(f)
        return []

    def _save_doc_ids(self):
        with open(DOC_IDS_PATH, "wb") as f:
            pickle.dump(self.doc_ids, f)

    def _load_faiss_index(self) -> faiss.Index:
        if FAISS_INDEX_PATH.exists():
            return faiss.read_index(str(FAISS_INDEX_PATH))
        return faiss.IndexFlatL2(EMBEDDING_DIM)

    def _save_faiss_index(self):
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add new documents to the index.
        Expects each `doc` to have these top-level fields:
          - document_id (str)
          - cleaned_text (str)
          - raw_text (str)
          - title, link, author, created_date, last_modified_date
          - source, docdb_version, filename, content_type
        """
        # Filter out any IDs we've already stored
        new_docs = [d for d in documents if d["document_id"]  not in self.metadata_store or self.metadata_store[d["document_id"]]['source'] != d['source'] ]
        if not new_docs:
            logger.info("All documents already indexed")
            return 0

        logger.info(f"Adding {len(new_docs)} {new_docs[-1].keys()}new documents to index")

        # 1) Embed the cleaned text
        texts = [d.get("cleaned_text", "Placeholder") for d in new_docs]
        embeddings = self.generate_embeddings(texts)
        logger.info(f"Embedded {len(new_docs)} {len(embeddings)} new documents to index")


        # 2) Add to FAISS
        self.count+=1
        self.faiss_index.add(np.array(embeddings))
        print(f"Added {len(np.array(embeddings))} embeddings")

        # 3) Update doc_ids
        self.doc_ids.extend(d["document_id"] for d in new_docs)

        # 4) Write out rich metadata for each
        u=set()
        try:
            for d in new_docs:

                did = d["document_id"]
                u.add(did)
                if d.get("source", "") == 'docdb':
                    self.metadata_store[did] = {
                        "title": d.get("title", ""),
                        "url": d.get("url", ""),
                        "author": d.get("author", ""),
                        "submitted_by": d.get("submitted_by", ''),
                        "updated_by": d.get("updated_by", ''),
                        "created_date": d.get("created_date", ""),
                        "content_last_modified_date": d.get("content_last_modified_date", ""),
                        "metadata_last_modified_date": d.get("metadata_last_modified_date", ""),
                        "source": d.get("source", ""),
                        "docdb_version": d.get("docdb_version", ""),
                        "filename": d.get("filename", ""),
                        'topics': d.get("topics", ""), 
                        "keywords":d.get("keywords", ""),
                        "abstract": d.get("abstract", ""),
                        "content_type": d.get("content_type", ""),
                        "cleaned_text": d.get("cleaned_text", ""),
                    
                    }
                else:
                    #logger.info(f"Storing {d} into store")
                    self.metadata_store[did]={}
                    for key in d.keys():
                        self.metadata_store[did][key] = d[key]
            
        except Exception as e:
            print(f"Error in adding odcumentb {e}, {key}, {d[key]}")

                


        # 5) Persist everything
        self.save_all()
        logger.info(f"Successfully added {len(new_docs)} documents to index")
        return len(new_docs)
    
    def get_indico_ids(self) -> Dict[int, int]:
        """
        Return a map { doc_id: max_version_indexed } for all DocDB docs.
        """
        ids: Dict[int, bool] = {}
        for did_str, meta in self.metadata_store.items():
            if meta.get("source", '') == "indico":
                ids[did_str] = True
        return ids
    

    def get_docdb_versions(self) -> Dict[int, int]:
        """
        Return a map { doc_id: max_version_indexed } for all DocDB docs.
        """
        versions: Dict[int, int] = {}
        for did_str, meta in self.metadata_store.items():
            if meta.get("source") == "docdb":
                try:
                    did = int(did_str)
                    v = int(meta.get("docdb_version", "") or "0")
                except ValueError:
                    continue
                versions[did] = max(versions.get(did, 0), v)
        return versions
    
    #def contentdatamodof
    def get_content_modification_dates(self) -> Dict[int, str]:
        dates = {}
        for did_str, meta in self.metadata_store.items():
            if meta.get('source') == 'docdb':
                try:
                    did=int(did_str)
                    cmd = meta.get('content_last_modified_date', None)
                except ValueError:
                    continue
                dates[did] = cmd if cmd != 'Unknown Content Date' else None
        return dates 
      
    #def metamodf
    def get_metadata_modification_dates(self) -> Dict[int, str]:
        dates = {}
        for did_str, meta in self.metadata_store.items():
            if meta.get('source') == 'docdb':
                try:
                    did=int(did_str)
                    cmd = meta.get('metadata_last_modified_date', None)
                except ValueError:
                    continue
                dates[did] = cmd if cmd != 'Unknown Metadata Date' else None
        return dates   
    

    def rebuild_index(self):
        """
        Rebuild the FAISS index from scratch using the current
        metadata_store & doc_ids (after pruning).
        """
        logger.info("Rebuilding FAISS index from pruned metadata_store...")
        # 1) fresh index
        self.faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)

        # 2) re-embed every remaining doc
        texts = [self.metadata_store[did].get("cleaned_text", "")
                 for did in self.doc_ids
                 if did in self.metadata_store]

        if texts:
            embeddings = self.generate_embeddings(texts)
            self.faiss_index.add(np.array(embeddings))

        # 3) persist
        self._save_faiss_index()
        logger.info(f"Rebuilt FAISS index; now contains {self.faiss_index.ntotal} vectors")

    def search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
            Finds the docID_number associated with the retrieved text, then goes back to the docID to find header info
        """
        query_emb = self.generate_embeddings([query])
        distances, indices = self.faiss_index.search(query_emb, top_k)

        snippets, refs = [], []
        for idx in indices[0]:
            if idx < len(self.doc_ids):
                did_of_txt = self.doc_ids[idx]
                did_root = did_of_txt.split("_")[0]
                md_root = self.metadata_store.get(did_root, {})
                md_text = self.metadata_store.get(did_of_txt, {})
                title = md_root.get("meeting_name", "")
                print(md_text)
                raw   = md_text.get("raw_text", "")
                link  = md_root.get("event_url", "")
                snippets.append(f"Title: {title}\n{raw}")
                if link:
                    refs.append(link)
        return snippets, refs

    def save_all(self):
        """Persist metadata, doc_ids, and FAISS index."""
        self._save_metadata()
        self._save_doc_ids()
        self._save_faiss_index()

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_documents": len(self.doc_ids),
            "total_vectors": self.faiss_index.ntotal,
            "metadata_entries": len(self.metadata_store),
        }

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.faiss_index = None
        self.model = None
        self.metadata_store = None
        self.doc_ids = None