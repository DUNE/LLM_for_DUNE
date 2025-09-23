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
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
logger = get_logger(__name__)

import chromadb
class ChromaManager:
    """Manager for FAISS index operations"""

    def __init__(self, data):
        # Prevent thread‐related segfaults
        self._configure_threading()
        

        self.chroma_client = chromadb.PersistentClient(path=data, settings=Settings())
        print("Collections available:", self.chroma_client.list_collections())

        self.chroma_collection = self.chroma_client.get_or_create_collection(name=CHROMA_DB_NAME)
        self.chroma_ntotal = self.chroma_collection.count()


        if self.chroma_ntotal == 0:
            logger.info(f"Initiating new DB named {CHROMA_DB_NAME}")
        else:
            logger.info(f"Retrieving existing DB named {CHROMA_DB_NAME}")
        
        # Setup device & model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
        logger.info(f"Loaded sentence transformer {EMBEDDING_MODEL}")
        
        self.indico_ids = defaultdict()
        self.docdb_versions = defaultdict()
        self.docdb_content_modified = defaultdict()
        self.docdb_metadata_modified = defaultdict()
        self.doc_ids=set()
        self.fetch_ddb_ind_data()



        # Ensure directories exist
        create_directories(data)

        # Load or initialize on‐disk data
        
    def fetch_ddb_ind_data(self):
        results = self.chroma_collection.get(include=["metadatas", "uris"])
        for id, md in zip(results['ids'], results['metadatas']):
            if md.get('source') == 'indico':
                self.indico_ids[id]=True
            elif md.get('source') == 'docdb':
                self.docdb_versions[id] = md['docdb_version']
                self.docdb_content_modified[id]  = md['content_last_modified_date']
                self.docdb_metadata_modified[id] = md['metadata_last_modified_date']
            self.doc_ids.add(id)
        print(len(self.doc_ids))
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
        up_ids=[]
        for i in ids:
            if documents[ids_to_idx_map[i]].get('cleaned_text', None):
                up_ids.append(i)
        for id_ in up_ids:
            md = {}
            doc_idx = ids_to_idx_map[id_]
            metadata_of_doc_to_update = documents[doc_idx]
            doc_texts.append(metadata_of_doc_to_update['cleaned_text'])

            for k,v in metadata_of_doc_to_update.items():
                if k not in ['cleaned_text', 'raw_text', 'document_id']:
                    md[k] = v
            metadatas.append(md)
        if not up_ids: return    
        if mode == 'update':
            print("Update 1")
            self.chroma_collection.update(
                ids = up_ids,
                documents = doc_texts,
                metadatas = metadatas,
            )
            
        elif mode == 'add':
            print(f"Adding {up_ids}")
            self.chroma_collection.add(
                ids = up_ids,
                documents = doc_texts,
                metadatas = metadatas,
            )
            self.chroma_ntotal += len(up_ids)
        else:
            logger.error(f"Invalid argument mode={mode}. Must be 'add' or 'update")
            raise ValueError
         
        logger.info(f"Added len(ids) to Chrome")
        self.chroma_client.persist()
        return len(ids)
        

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
                #self.doc_ids.add(did)
        except Exception as e:
            logger.error(f"Error in updating list of doc_ids with Indico/DocDB: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
    
        #which index each doc id is at in documents
        map_ids_to_idx = {d['document_id']:idx for idx,d in enumerate(documents)}

        self.update_indico_docdb_record(documents)
        
        #all ids to add/update
        ids = set(map_ids_to_idx.keys())

        #update existing ids in chroma
        ids_to_update = list(self.doc_ids.intersection(ids))
        self.add_to_chroma(documents = documents, ids = ids_to_update, ids_to_idx_map =  map_ids_to_idx, mode = 'update')
        

        #add new ids to chroma
        ids_to_add = ids - self.doc_ids 
        if ids_to_add:
            self.add_to_chroma(documents= documents, ids= ids_to_add, ids_to_idx_map=map_ids_to_idx, mode='add')
        self.doc_ids.update(ids_to_add)

        return len(ids_to_add)
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
        results = self.chroma_collection.query(query_texts=[query],  n_results=top_k)
        snippets, refs = [], []
        print(results['metadatas'][0], type(results['metadatas'][0]))
        for md in results['metadatas'][0]:
            link  = md.get("event_url", "")
            title = md.get("meeting_name", "")
            if link:
                refs.append(link)
        for docs in results['documents'][0]:
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
