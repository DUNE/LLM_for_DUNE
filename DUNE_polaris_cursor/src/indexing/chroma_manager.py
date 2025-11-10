from collections import defaultdict
import os
import pickle
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from src.embedder.embedding_wrappers import ChatlasEmbedder, OriginalEmbedder
from sentence_transformers import CrossEncoder
from config import (
    EMBEDDING_MODEL,
    MAX_VARIABLE_NUMBER,
    create_directories,
)
import re
CHROMA_DB_NAME='DUNE_VECTOR_DB'
from src.utils.logger import get_logger
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
logger = get_logger(__name__)
import torch.nn as nn
import chromadb
class ChromaManager:
    """Manager for FAISS index operations"""

    def __init__(self, data):
        # Prevent thread‐related segfaults
        self.bm25_cache=None
        self._configure_threading()
        self.reranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        self.sigmoid = nn.Sigmoid()
        #self.reranker=CrossEncoder('valhalla/longformer-base-4096-finetuned-squadv1',max_length=4096)
        #self.reranker = AutoModel.from_pretrained("allenai/longformer-base-4096", torch_dtype="auto")

        #self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        self.chroma_client = chromadb.PersistentClient(path=data, settings=Settings())
        print("Collections available:", self.chroma_client.list_collections())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model = OriginalEmbedder(EMBEDDING_MODEL)
        logger.info("Creating collection")
        try:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_DB_NAME,
                embedding_function=self.model
            )
        except Exception as e:
            logger.error(f"Error intiiateing chroma {e}")

        self.chroma_ntotal = self.chroma_collection.count()


        if self.chroma_ntotal == 0:
            logger.info(f"Initiating new DB named {CHROMA_DB_NAME}")
        else:
            logger.info(f"Number of documents = {self.chroma_ntotal}")
            logger.info(f"Retrieving existing DB named {CHROMA_DB_NAME}")

        # Setup device & model


        logger.info(f"Loaded sentence transformer {EMBEDDING_MODEL}")

        self.indico_ids = defaultdict()
        self.docdb_versions = defaultdict()
        self.docdb_content_modified = defaultdict()
        self.docdb_metadata_modified = defaultdict()
        self.metadata= defaultdict()
        self.documents= defaultdict()
        self.doc_ids=set()
        self.events_ids=set()
        self.fetch_ddb_ind_data()
        self.num_events=len(self.events_ids)


        # Ensure directories exist
        create_directories(data)

    def fetch_ddb_ind_data(self):
        results = self.chroma_collection.get(include=["documents", "metadatas", "uris"])
        for id, md, doc in zip(results['ids'], results['metadatas'],results['documents'] ):
            if md.get('source') == 'indico':
                self.indico_ids[id]=True
            elif md.get('source') == 'docdb':
                try:
                    self.docdb_versions[id] = md['docdb_version']
                    self.docdb_content_modified[id]  = md['content_last_modified_date']
                    self.docdb_metadata_modified[id] = md['metadata_last_modified_date']
                except:
                    logger.info(f"error with {id}")
                    continue
            self.metadata[id] = md
            self.documents[id]=doc

            self.doc_ids.add(id)

            event_id = id.split("_")[0]
            self.events_ids.add(event_id)


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
            else:
                logger.error(f"Cannot add {i} because didn't extract text from it")

        for id_ in up_ids:
            md = {}
            doc_idx = ids_to_idx_map[id_]
            metadata_of_doc_to_update = documents[doc_idx]
            doc_texts.append(metadata_of_doc_to_update['cleaned_text'])

            for k,v in metadata_of_doc_to_update.items():
                if k not in ['cleaned_text', 'raw_text', 'document_id']:
                    md[k] = v
            metadatas.append(md)


        length = 0
        for d in doc_texts:
            length += len(d)
        logger.info(f'Storing document of length {length} in chunks')

        if length == 0: return 0
        logger.info(f"Length = {length}")
        logger.info(f"up_ids = {up_ids}, mode={mode}")
        print(f"ids :{type(up_ids)}\ndocuments: {type(doc_texts)}\nmetadatas: {type(metadatas)}")

        for i in range(0, len(up_ids), MAX_VARIABLE_NUMBER):
            if mode == 'update':
                self.chroma_collection.update(
                    ids = up_ids[i:i+MAX_VARIABLE_NUMBER],
                    documents = doc_texts[i:i+MAX_VARIABLE_NUMBER],
                    metadatas = metadatas[i:i+MAX_VARIABLE_NUMBER],
                )

            elif mode == 'add':
                self.chroma_collection.add(
                    ids = up_ids[i:i+MAX_VARIABLE_NUMBER],
                    documents = doc_texts[i:i+MAX_VARIABLE_NUMBER],
                    metadatas = metadatas[i:i+MAX_VARIABLE_NUMBER],
                )
                self.chroma_ntotal += len(up_ids[i:i+MAX_VARIABLE_NUMBER])
            else:
                logger.error(f"Invalid argument mode={mode}. Must be 'add' or 'update")
                raise ValueError
            logger.info(f"Added {len(up_ids[i: i+ MAX_VARIABLE_NUMBER])} to Chroma")
        return len(up_ids)


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

    def add_documents(self, documents: List[Dict[str, Any]], num_events: int) -> int:
        logger.info("in add documents")
        self.num_events+=num_events
        #which index each doc id is at in documents
        map_ids_to_idx = {d['document_id']:idx for idx,d in enumerate(documents)}


        self.update_indico_docdb_record(documents)
        logger.info("updated global var")


        #all ids to add/update
        ids = set(map_ids_to_idx.keys())

        #update existing ids in chroma
        ids_to_update = list(self.doc_ids.intersection(ids))
        print(self.doc_ids, ids)
        added = self.add_to_chroma(documents = documents, ids = ids_to_update, ids_to_idx_map =  map_ids_to_idx, mode = 'update')
        logger.info(" updated to chroma documents")


        #add new ids to chroma
        ids_to_add = ids - self.doc_ids

        if ids_to_add is not None:
            added += self.add_to_chroma(documents= documents, ids= ids_to_add, ids_to_idx_map=map_ids_to_idx, mode='add')
        self.doc_ids.update(ids_to_add)

        return added
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
    def get_content_modification_dates(self) -> Dict[int, str]:
        return self.docdb_content_modified

    #def metamodf
    def get_metadata_modification_dates(self) -> Dict[int, str]:
        return self.docdb_metadata_modified


    def without_reranker_search(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
            Finds the docID_number associated with the retrieved text, then goes back to the docID to find header info
        """
        results = self.chroma_collection.query(query_texts=[query],  n_results=top_k)
        snippets, refs = [], []
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
    def search_old(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """
            Finds the docID_number associated with the retrieved text, then goes back to the docID to find header info
        """
        results = self.chroma_collection.query(query_texts=[query],  n_results=top_k*6)
# rerank the results with original query and documents returned from Chroma
        #scores = self.reranker.predict([(query, doc) for doc in results["documents"][0]])

        scores = []
        scores = self.reranker.predict(
            [(query, doc) for doc in results["documents"][0]],
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
        #might be able to zip these for oops together.
        #check if results are returned as connected idices
        #ie if metadata idx1 is assoc w dcyments idx 1
        return snippets, refs

    def tokenize(self, doc):
        return re.findall(r"\w+", doc.lower())
    
    def build_bm25_index(self, documents):
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        self.bm25_cache = BM25Okapi(tokenized_docs)
        
    def keyword_search(self, query, top_k):
        all_entries = self.chroma_collection.get()
        if not self.bm25_cache:
            self.build_bm25_index(all_entries['documents'])
        
        bm25_scores = self.bm25_cache.get_scores(query.split())
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        doc_ids = list(self.metadata.keys())
        results = [(doc_ids[idx], bm25_scores[idx]) for idx in top_indices]
        return results

    def semantic_search(self, query, doc_type, top_k):

        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=top_k,
            where={'document_type': doc_type}
        )
        if not results['ids'][0]:
            return []
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        # Assuming cosine distance, similarity = 1 - distance
        similarities = [1 - dist for dist in results['distances'][0]]
        
        return list(zip(results['ids'][0], similarities))
    
    def merge(self, keyword_doc_ids,semantic_doc_ids):
        def normalize_scores(results):
            if not results:
                return {}
            scores = [score for _, score in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score if max_score > min_score else 1
            return {
                doc_id: (score - min_score) / score_range 
                for doc_id, score in results
            }
        keyword_scores = normalize_scores(keyword_doc_ids)
        semantic_scores = normalize_scores(semantic_doc_ids)
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

        

    def reranker_search(self, query, merged_docids, top_k):
        documents=[]
        try:
            documents = [self.documents[doc_id] for doc_id in merged_docids]
        except:
            print("No documents for")
            return []
                
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get cross-encoder scores
        ce_scores = self.reranker.predict(pairs)
        
        # Return sorted by score
        results = list(zip(merged_docids, ce_scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


    def get_links(self, reranked_docids):
        return [self.metadata[id_[0]]['event_url'] for id_ in reranked_docids]
    def get_content(self, reranked_docids):
        print(f"Looking at ids: {[id_[0] for id_ in reranked_docids]}")
        return [self.documents[id_[0]] for id_ in reranked_docids]

    def search(self, query: str, top_k: int = 3, keyword=True) -> Tuple[List[str], List[str]]:
        """
            Finds the docID_number associated with the retrieved text, then goes back to the docID to find header info
        """
        if keyword:
            keyword_doc_ids = self.keyword_search(query, top_k)
        else:
            keyword_doc_ids = []
        semantic_doc_ids = self.semantic_search(query, 'document', top_k)
        semantic_doc_ids.extend(self.semantic_search(query, 'slides', top_k))

        merged_docids = self.merge(keyword_doc_ids,semantic_doc_ids )

        reranked_docids = self.reranker_search(query, merged_docids, top_k)

        links = self.get_links(reranked_docids)
        content = self.get_content(reranked_docids)
        logger.info(f"{len(content)} sources eaxc of len {len(content[0])}")
        return content, links

    def save_all(self):
        """Persist metadata, doc_ids, and FAISS index."""
        pass

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_documents": self.num_events,
            "total_embeddings": self.chroma_collection.count(),
            "total_number_attachments_in_metadata": self.chroma_ntotal,
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


