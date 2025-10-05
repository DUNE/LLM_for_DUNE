from collections import defaultdict
import os
import pickle

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from pathlib import Path
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    create_directories,
)
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embedder.embedding_wrappers import OriginalEmbedder
from src.utils.logger import get_logger
logger = get_logger(__name__)

class FAISSManager:
    """Manager for FAISS index operations"""

    def __init__(self, data_path):
        # Prevent thread‐related segfaults
        self._configure_threading()
       
        # Setup device & model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = OriginalEmbedder()
        logger.info(f"Loaded sentence transformer {EMBEDDING_MODEL}")
        
        self.metadata_store=defaultdict( dict)
        self.num_events = 0

        # Ensure directories exist
        create_directories(data_path)

        # Load or initialize on‐disk data
     
        self.metadata_path = os.path.join(data_path, 'metadata_store.pkl')

        self.metadata_store: Dict[str, Dict[str, Any]] = self._load_metadata()
        
        self.doc_ids_path = os.path.join(data_path, 'doc_ids.pkl')
        self.doc_ids: List[str] = self._load_doc_ids()

        self.faiss_index_path = os.path.join(data_path, 'faiss_index.index')
        self.faiss_index = self._load_faiss_index()
        print(type(self.faiss_index))
        

        logger.info(f"Loaded FAISS index with {len(self.faiss_index.index_to_docstore_id)} entries")
        logger.info(f"Loaded metadata store with {len(self.metadata_store)} entries")
        logger.info(f"Loaded doc_ids with {len(self.doc_ids)} entries")

    def _configure_threading(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def _load_metadata(self) -> Dict[str, Any]:
        
        if Path(self.metadata_path).exists():
            with open(self.metadata_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_metadata(self):
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata_store, f)

    def _load_doc_ids(self) -> List[str]:
        if Path(self.doc_ids_path).exists():
            with open(self.doc_ids_path , "rb") as f:
                return pickle.load(f)
        return []

    def _save_doc_ids(self):
        with open(self.doc_ids_path , "wb") as f:
            pickle.dump(self.doc_ids, f)

    def _load_faiss_index(self) -> faiss.Index:
        if Path(self.faiss_index_path).exists():
            return FAISS.load_local(
                    self.faiss_index_path, self.model, allow_dangerous_deserialization=True
                )
        index= faiss.IndexFlatL2(384)
        vector_store = FAISS(
            embedding_function=self.model,
            index=index,
            docstore= InMemoryDocstore(),
            index_to_docstore_id={}
        )

        return vector_store

    def _save_faiss_index(self):
        self.faiss_index.save_local(self.faiss_index_path)
        #faiss.write_index(self.faiss_index, str(self.faiss_index_path))

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def add_documents(self, documents: List[Dict[str, Any]], num_events: int) -> int:
        self.num_events += num_events
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
        all_docs = [d for d in documents if d["document_id"] not in self.metadata_store or self.metadata_store[d["document_id"]]['source'] != d['source'] ]
        new_docs = [d for d in all_docs if 'cleaned_text' in d ]
        if not new_docs:
            logger.info("All documents already indexed")
            return 0

        logger.info(f"Adding {len(new_docs)} new documents to FAISS")

        # 1) Embed the cleaned text
        #texts = [d["cleaned_text"] for d in new_docs]
       
        #embeddings = self.generate_embeddings(texts)
        #logger.info(f"Embedded {len(new_docs)} new docs to FAISS")


        # 2) Add to FAISS

        
        #print(f"Added {len(np.array(embeddings))} embeddings to FAISS")
        # Check current FAISS index dimension
        print("FAISS index dimension:", self.faiss_index.index.d)

        # Check dimension of new vectors
        test_vector = self.model.embed_query("test")
        print("Embedding model output dimension:", len(test_vector))

        # 3) Update doc_ids
        self.doc_ids.extend(d["document_id"] for d in new_docs)
       
        # 4) Write out rich metadata for each
        u=set()
        try:
            documents=[]
            for d in new_docs:
                did = d["document_id"]
                u.add(did)
                if d.get("source", "") == 'docdb':
                    self.metadata_store[did] = {
                        "title": d.get("title", ""),
                        "url": d.get("event_url", ""),
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
                        "cleaned_text": d.get('cleaned_text', ''),
                    }
                    documents.append(Document(page_content = d.get('cleaned_text', ''), metadata=self.metadata_store[did] ))
                
                else:
                    self.metadata_store[did]={}
                    for key in d.keys():
                        self.metadata_store[did][key] = d[key]
            
                    documents.append(Document(page_content = d.get('cleaned_text', ''), metadata=self.metadata_store[did] ))
            
            self.faiss_index.add_documents(documents=documents)
         
        except Exception as e:
            import traceback
            traceback.print_exc()
        
        self.save_all()
        print("metadata store ", len(self.metadata_store))
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
        

        # 2) re-embed every remaining doc
        texts = [Document(page_content=self.metadata_store[did].get("cleaned_text", ""), metadata=self.metadata_store[did])
                 for did in self.doc_ids
                 if did in self.metadata_store]
        
        dids_to_remove=[did for did in self.doc_ids
                 if did in self.metadata_store]
        
        

        if texts and dids_to_remove:
            #embeddings = self.generate_embeddings(texts)
            #embeddings = np.array(embeddings) 
            self.faiss_index.delete(ids=dids_to_remove)
            self.faiss_index.add_documents(texts)

        # 3) persist
        self._save_faiss_index()
        logger.info(f"Rebuilt FAISS index; now contains {len(self.faiss_index.index_to_docstore_id)} vectors")

    def search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
            Finds the docID_number associated with the retrieved text, then goes back to the docID to find header info
        """
        #query_emb = self.generate_embeddings([query])
        snippets, refs=[],[]
        #results = self.faiss_index.max_marginal_relevance_search(query, top_k)
        retriever = self.faiss_index.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        results = retriever.invoke(query)
        for doc in results:
            snippets.append(doc.page_content)
            try:
                refs.append(doc.metadata['url'])
            except:
                refs.append(doc.metadata['event_url'])
        #logger.info(distances)
        logger.info(f"REFS { refs}")
        #logger.warning(f"Links : {refs}")
        return snippets, refs

    def save_all(self):
        """Persist metadata, doc_ids, and FAISS index."""
        self._save_faiss_index()
        self._save_metadata()

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_documents": self.num_events,
            "total_embeddings":  len(self.faiss_index.docstore._dict),
            "total_number_attachments_in_metadata": len(self.metadata_store),
        }

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.faiss_index = None
        self.model = None
        self.metadata_store = None
        self.doc_ids = None

