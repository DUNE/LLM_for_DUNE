
from datetime import datetime
import time
import threading
from typing import List, Dict, Any, Optional, Set
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.extractors.docdb_extractor_multithreaded import DocDBExtractor
from src.extractors.indico_extractor_multithreaded import IndicoExtractor

from config import DOC_LIMIT_DOCDB, DOC_LIMIT_INDICO
from src.utils.logger import get_logger
import src.indexing.chroma_manager as chroma
logger = get_logger(__name__)

class DocumentProcessor:
    """Orchestrates document extraction, processing, and indexing"""

    def __init__(self, data, chunk_size):
        self.chunk_size=chunk_size
        logger.info("Initiated chroma")
        self.chroma_manager = chroma.ChromaManager(data) 
       
 
        self.docdb_extractor = DocDBExtractor(self.chroma_manager)
        self.indico_extractor = IndicoExtractor(self.chroma_manager)

    def process_all_documents(
        self,
        start_ddb=0,
        start_ind=0,
        docdb_limit: int = DOC_LIMIT_DOCDB,
        indico_limit: int = DOC_LIMIT_INDICO,
        force: bool = False,
    ) -> Dict[str, int]:
        logger.info("Starting document processing pipeline")
        results = {
            "docdb_parsed":0,
            "indico_parsed":0,
            "docdb_processed": 0,
            "indico_processed": 0,
            "total_embeddings_added": 0
        }

        #
        # --- DocDB portion ---
        #
        to_reindex: List[Dict[str, Any]] = []
        def docdb_extraction(name):
            try:
                logger.info("Processing DocDB documents")

                # 1) What versions (and thus IDs) do we already have?
                indexed_versions = self.chroma_manager.get_docdb_versions()
                #indexed_ids: Set[int] = set(indexed_versions.keys())
                indexed_ids: Set[int] = { int(did) for did in indexed_versions.keys() }

                # 2) Fetch up to `docdb_limit` pages, skipping any IDs in indexed_ids

                # 3) Optionally filter by version bump or force‐flag
                
                for docs_processed, raw_records, docs_parsed in self.docdb_extractor.extract_documents(start=start_ddb, limit=docdb_limit,
                                                                    indexed_doc_ids=indexed_ids,
                                                                    mode="incremental",
                                                                    stop_after_seen=100,
                                                                    max_missing=1000,
                                                                    chunk_size=self.chunk_size,
                                                                    existing_versions=indexed_versions
                                                                ):
                    
                    log_to_db_docdb(raw_records, num_processed=docs_processed, num_parsed=docs_parsed)
                    
                    print(docs_parsed , " parsed")           
            except Exception as e:
                logger.error(f"Error in extracting documents from dune docdb {e}"  )
            return to_reindex
        
        def log_to_db_docdb(to_reindex, num_processed, num_parsed):
            try:
                with log_lock:
                    
                    logger.info(f"to_reindex is {len(to_reindex)}")
                    # 5) Add the new/updated docs
                    added = self.chroma_manager.add_documents(to_reindex, num_processed)
                    results['docdb_parsed']+=num_parsed
                    results["docdb_processed"] += num_processed
                    results["total_embeddings_added"] += added


                    logger.info(
                        f"Added DocDB to Chroma: reindexed {len(to_reindex)}, added {added} vectors"
                    )

                            


            except Exception as e:
                logger.error(f"Error processing documents: {e}")

        #
        # --- Indico portion ---
        #
        indico_to_add=[None]
        def indico_extraction(name):
            try:
                
                logger.info("Processing Indico documents")

                for num_events, indico_records, docs_parsed in self.indico_extractor.extract_documents(start=start_ind, limit=indico_limit, chunk_size=self.chunk_size):
                    logger.info(f"Indico records returns {len(indico_records)} from {num_events} events")
                   

                    log_to_db_indico(indico_records,num_events, docs_parsed)
                    


                    

            except Exception as e:
                logger.error(f"Error processing Indico documents: {e}")

        def log_to_db_indico(docs,num_events, num_parsed):
            with log_lock:
                added = self.chroma_manager.add_documents(docs,num_events)
                results['indico_parsed'] += num_parsed
                results["indico_processed"] += num_events
                
                results["total_embeddings_added"] += added
                
                logger.info(f"Added Indico to Chroma: added {added} new vectors to index")
                return added
            
        docdb_thread = threading.Thread(target=docdb_extraction, args=('docdb',))
        indico_thread = threading.Thread(target=indico_extraction, args=('indico',))

        log_lock = threading.Lock()
        docdb_thread.start()
        indico_thread.start()

        # Wait for everything to finish
        docdb_thread.join()
        indico_thread.join()


        # Final summary & return
        logger.info(f"Document processing completed. Total new docs added: {results['total_embeddings_added']}")


        return results


    def get_index_stats(self) -> Dict[str, int]:
        """Get current index statistics"""
        return self.chroma_manager.get_stats()

    def cleanup(self):
        """Cleanup resources"""
        self.chroma_manager.cleanup()
