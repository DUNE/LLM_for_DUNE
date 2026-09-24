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

class EntryProcessor:
    """Orchestrates entry extraction, processing, and indexing"""

    def __init__(self, data, chunk_size):
        self.chunk_size=chunk_size
        logger.debug("Initiated chroma")
        self.chroma_manager = chroma.ChromaManager(data) 
       
 
        self.docdb_extractor = DocDBExtractor(self.chroma_manager)
        self.indico_extractor = IndicoExtractor(self.chroma_manager)

    def process_all_entries(
        self,
        start_ddb=0,
        start_ind=0,
        docdb_limit: int = DOC_LIMIT_DOCDB,
        indico_limit: int = DOC_LIMIT_INDICO,
        force: bool = False,
    ) -> Dict[str, int]:
        logger.info("Starting entry processing pipeline")
        results = {
            "docdb_chunks_parsed": 0,
            "indico_chunks_parsed": 0,
            "docdb_documents_processed": 0,
            "indico_events_processed": 0,
            "total_chunks_added": 0
        }

        #
        # --- DocDB portion ---
        #
        def docdb_extraction(name):
            try:
                logger.info("Processing DocDB documents")
                if docdb_limit == -1: return []
                
                indexed_versions = self.chroma_manager.get_docdb_versions()
                indexed_document_ids: Set[int] = { int(did) for did in indexed_versions.keys() }

                
                for documents_processed, chunks, chunks_parsed in self.docdb_extractor.extract_documents(
                                                                    start=start_ddb, 
                                                                    limit=docdb_limit,
                                                                    indexed_document_ids=indexed_document_ids,
                                                                    mode="incremental",
                                                                    stop_after_seen=100,
                                                                    max_missing=1000,
                                                                    chunk_size=self.chunk_size,
                                                                    existing_versions=indexed_versions
                                                                ):
                    
                    log_to_db_docdb(chunks, num_documents_processed=documents_processed, num_chunks_parsed=chunks_parsed)
                    
            except Exception as e:
                logger.error(f"Error in extracting documents from dune docdb {e}")
            return []
        
        def log_to_db_docdb(chunks_batch, num_documents_processed, num_chunks_parsed):
            try:
                with log_lock:
                    
                    logger.debug(f"chunks_batch size = {len(chunks_batch)}")
                    added = self.chroma_manager.add_entries(chunks_batch)
                    results['docdb_chunks_parsed'] += num_chunks_parsed
                    results["docdb_documents_processed"] += num_documents_processed
                    results["total_chunks_added"] += added


                    logger.debug(
                        f"Added DocDB batch: documents={num_documents_processed}, chunks_added={added}"
                    )

            except Exception as e:
                logger.error(f"Error processing DocDB chunks: {e}")

        #
        # --- Indico portion ---
        #
        def indico_extraction(name):
            try:
                
                logger.info("Processing Indico events")
                if indico_limit==-1: return []

                for events_processed, chunks, chunks_parsed in self.indico_extractor.extract_documents(start=start_ind, limit=indico_limit, chunk_size=self.chunk_size):
                    logger.debug(f"Indico records returns {len(chunks)} chunks from {events_processed} events")
                   

                    log_to_db_indico(chunks, events_processed, chunks_parsed)
                    
            except Exception as e:
                logger.error(f"Error processing Indico events: {e}")

        def log_to_db_indico(chunks_batch, num_events_processed, num_chunks_parsed):
            with log_lock:
                added = self.chroma_manager.add_entries(chunks_batch)
                results['indico_chunks_parsed'] += num_chunks_parsed
                results["indico_events_processed"] += num_events_processed
                
                results["total_chunks_added"] += added
                
                logger.debug(f"Added Indico to Chroma: added {added} new chunks to index")
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
        logger.info(f"Entry processing completed. Total new chunks added: {results['total_chunks_added']}")


        return results


    def get_index_stats(self) -> Dict[str, int]:
        """Get current index statistics"""
        return self.chroma_manager.get_stats()

    def cleanup(self):
        """Cleanup resources"""
        self.chroma_manager.cleanup()
