from collections import defaultdict
import os
import pickle
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    create_directories,
)
CHROMA_DB_NAME='DUNE_VECTOR_DB'
from src.utils.logger import get_logger

logger = get_logger(__name__)

import chromadb
class ChromaManager:
    """Manager for FAISS index operations"""

    def __init__(self):
        # Prevent thread‐related segfaults
        self._configure_threading()
        
        chroma_client = chromadb.Client()
        self.chroma_collection = chroma_client.create_collection(name=CHROMA_DB_NAME)
        self.chroma_ntotal=0
        # Setup device & model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device, token=False)
        logger.info(f"Loaded sentence transformer {EMBEDDING_MODEL}")
        
        self.indico_ids = defaultdict()
        self.docdb_versions = defaultdict()
        self.docdb_content_modified = defaultdict()
        self.docdb_metadata_modified = defaultdict()


        self.doc_ids=[]
        # Ensure directories exist
        create_directories()

        # Load or initialize on‐disk data
        

    def _configure_threading(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

   
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def add_to_chroma(self, documents, ids, ids_to_idx_map, mode):
        metadatas= []
        doc_texts=[]
        for id_ in ids:
            md = {}
            doc_idx = ids_to_idx_map[id_]
            metadata_of_doc_to_update = documents[doc_idx]
            doc_texts.append(metadata_of_doc_to_update['raw_text'])

            for k,v in metadata_of_doc_to_update.items():
                if k not in ['raw_text', 'document_id']:
                    md[k] = v
            metadatas.append(md)
           
        if mode == 'update':
            self.chroma_collection.update(
                ids = ids,
                documents = doc_texts,
                metadatas = metadatas,
            )
            self.chroma_ntotal += 1
        elif mode == 'add':
            self.chroma_collection.add(
                ids = ids,
                documents = doc_texts,
                metadatas = metadatas,
            )
            self.chroma_ntotal += 1
        else:
            logger.error(f"Invalid argument mode={mode}. Must be 'add' or 'update")
            raise ValueError
         
        logger.info(f"Added len(ids) to Chrome")
        return 
        

    def  update_indico_docdb_record(self, documents):
        try: 
            for doc in documents:
                source = doc.get("source", '')
                did = doc['document_id']
                if source == 'docdb':
                    self.docdb_versions[did] = doc['docdb_version']
                    self.docdb_content_modified[did] = doc.get('content_last_modified_date', 'Unknown Content Date')
                    self.docdb_metadata_modified[did] = doc.get('metadata_last_modified_date', 'Unknown Metadata Date')

                elif source == 'indico':
                    self.indico_ids[did]=True
                self.doc_ids.append(did)
        except Exception as e:
            logger.error(f"Error in updating list of doc_ids with Indico/DocDB: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
    
        #which index each doc id is at in documents
        map_ids_to_idx = {d['document_id']:idx for idx,d in enumerate(documents)}

        self.update_indico_docdb_record(documents)
        
        #all ids to add/update
        ids = set(map_ids_to_idx.keys())

        #update existing ids in chroma
        ids_to_update = list(self.existing_ids.intersection(ids))
        self.add_to_chroma(documents = documents, ids = ids_to_update, ids_to_idx_map =  map_ids_to_idx, mode = 'update')
        

        #add new ids to chroma
        ids_to_add = ids - self.existing_ids 
        self.add_to_chroma(documents= documents, ids= ids_to_add, ids_to_idx_map=map_ids_to_idx, mode='add')
           

    
    def get_indico_ids(self) -> Dict[int, int]:
        """
        Return a map { doc_id: max_version_indexed } for all DocDB docs.
        """
        return self.indico_ids
    

    

    def get_docdb_versions(self) -> Dict[int, int]:
        """
        Return a map { doc_id: max_version_indexed } for all DocDB docs.
        """
        return self.docdb_versions
    
    #def contentdatamodof
    def get_content_modification_dates(self) -> Dict[int, str]:
        return self.docdb_content_modified
      
    #def metamodf
    def get_metadata_modification_dates(self) -> Dict[int, str]:
        return self.docdb_metadata_modified
    

    
    def search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
            Finds the docID_number associated with the retrieved text, then goes back to the docID to find header info
        """
        query_emb = self.generate_embeddings([query])
        results = self.faiss_index.query(query_emb, top_k)

        snippets, refs = [], []
        for md in results['metadatas']:
            link  = md.get("event_url", "")
            title = md.get("meeting_name", "")
            if link:
                refs.append(link)
        for docs in results['documents']:
            snippets.append(docs)

        #might be able to zip these for oops together. 
        #check if results are returned as connected idices 
        #ie if metadata idx1 is assoc w dcyments idx 1

        
        return snippets, refs

    def save_all(self):
        """Persist metadata, doc_ids, and FAISS index."""
        pass

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_documents": len(self.doc_ids),
            "total_vectors": self.chroma_ntotal,
            "metadata_entries": self.chroma_collection.count(),
        }

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.chroma_collection = None
        self.chroma_ntotal=0
        self.indico_ids = None
        self.docdb_versions = None
        self.docdb_content_modified = None
        self.docdb_metadata_modified = None

        self.doc_ids = None