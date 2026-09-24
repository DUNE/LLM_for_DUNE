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

    def __init__(self, db_path):
        # Prevent thread‐related segfaults
        self.bm25_cache=None
        self._configure_threading()

        self.db_path = db_path

        self.reranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

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

        self.fetch_ddb_ind_data()

    
    def fetch_ddb_ind_data(self):
        results = self.chroma_collection.get(include=["documents", "metadatas", "uris"])
        for id, md, text in zip(results['ids'], results['metadatas'], results['documents'] ):
            if md.get('source') == 'indico':
                self.indico_ids[id.split('_')[0].split("/")[0]]=True
            elif md.get('source') == 'docdb':
                try:
                    self.docdb_versions[id.split('_')[0].split("/")[0]] = md['docdb_version']
                    self.docdb_content_modified[id]  = md['content_last_modified_date']
                    self.docdb_metadata_modified[id] = md['metadata_last_modified_date']
                except:
                    logger.error(f"error with {id}")
                    continue
            self.metadata[id] = md
            self.embedding_contents[id]=text

            entry_id = id.split("_")[0]
            self.entry_ids.add(entry_id)

            id_parts = id.split('_')
            if len(id_parts) >= 2:
                unique_attachment_id = f"{id_parts[0]}_{id_parts[1]}"
            else:
                unique_attachment_id = id
            self.attachment_ids.add(unique_attachment_id)

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

                id_parts = did.split('_')
                if len(id_parts) >= 2:
                    unique_attachment_id = f"{id_parts[0]}_{id_parts[1]}"
                else:
                    unique_attachment_id = did
                self.attachment_ids.add(unique_attachment_id)

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
