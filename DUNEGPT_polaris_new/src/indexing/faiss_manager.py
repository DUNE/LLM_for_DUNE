import os
import pickle
import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from config import (
    FAISS_INDEX_PATH, METADATA_PATH, DOC_IDS_PATH, 
    EMBEDDING_MODEL, EMBEDDING_DIM, create_directories
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FAISSManager:
    """Manager for FAISS index operations"""
    
    def __init__(self):
        # Set thread limits to prevent segfaults
        self._configure_threading()
        
        # Setup device and model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
        
        # Create directories
        create_directories()
        
        # Initialize storage
        self.metadata_store = self._load_metadata()
        self.doc_ids = self._load_doc_ids()
        self.faiss_index = self._load_faiss_index()
        
        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} entries")
        logger.info(f"Loaded metadata store with {len(self.metadata_store)} entries")
        logger.info(f"Loaded doc_ids with {len(self.doc_ids)} entries")
    
    def _configure_threading(self):
        """Configure threading to prevent issues with FAISS"""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata store from disk"""
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "rb") as f:
                return pickle.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata store to disk"""
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata_store, f)
    
    def _load_doc_ids(self) -> List[str]:
        """Load document IDs from disk"""
        if DOC_IDS_PATH.exists():
            with open(DOC_IDS_PATH, "rb") as f:
                return pickle.load(f)
        return []
    
    def _save_doc_ids(self):
        """Save document IDs to disk"""
        with open(DOC_IDS_PATH, "wb") as f:
            pickle.dump(self.doc_ids, f)
    
    def _load_faiss_index(self) -> faiss.Index:
        """Load FAISS index from disk"""
        if FAISS_INDEX_PATH.exists():
            return faiss.read_index(str(FAISS_INDEX_PATH))
        return faiss.IndexFlatL2(EMBEDDING_DIM)
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add new documents to the index"""
        new_docs = [doc for doc in documents if doc["document_id"] not in self.metadata_store]
        
        if not new_docs:
            logger.info("All documents already indexed")
            return 0
        
        logger.info(f"Adding {len(new_docs)} new documents to index")
        
        # Generate embeddings
        texts = [doc["cleaned_text"] for doc in new_docs]
        embeddings = self.generate_embeddings(texts)
        
        # Add to FAISS index
        self.faiss_index.add(np.array(embeddings))
        
        # Update document IDs
        self.doc_ids.extend([doc["document_id"] for doc in new_docs])
        
        # Update metadata store
        for doc in new_docs:
            doc_id = doc["document_id"]
            self.metadata_store[doc_id] = {
                "title": doc.get("title", "Untitled"),
                "link": doc.get("link", ""),
                "author": doc.get("author", "Unknown"),
                "date": doc.get("date", ""),
                "source": doc.get("source", ""),
                "raw_text": doc.get("raw_text", "")
            }
        
        # Save all changes
        self.save_all()
        
        logger.info(f"Successfully added {len(new_docs)} documents to index")
        return len(new_docs)
    
    def search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """Search the FAISS index for similar documents"""
        logger.debug("Starting FAISS search")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        context_snippets = []
        references = []
        
        for idx in indices[0]:
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                metadata = self.metadata_store.get(doc_id, {})
                
                # Get content from metadata
                content = metadata.get("raw_text", "") or metadata.get("content", "")
                title = metadata.get("title", "")
                link = metadata.get("link", "")
                
                context_snippets.append(f"Title: {title}\nContent: {content}")
                if link:
                    references.append(link)
        
        logger.debug(f"Returning {len(context_snippets)} context snippets and {len(references)} references")
        return context_snippets, references
    
    def save_all(self):
        """Save all index data to disk"""
        self._save_metadata()
        self._save_doc_ids()
        self._save_faiss_index()
    
    def get_stats(self) -> Dict[str, int]:
        """Get index statistics"""
        return {
            "total_documents": len(self.doc_ids),
            "total_vectors": self.faiss_index.ntotal,
            "metadata_entries": len(self.metadata_store)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.faiss_index = None
        self.model = None
        self.metadata_store = None
        self.doc_ids = None 