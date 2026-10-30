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
        self.chroma_manager = chroma.ChromaManager(data, indexing_only=True)
       
 
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
        run_start = time.perf_counter()
        initial_disk_size = self.chroma_manager.get_stats()["disk_size"]

        results = {
            "docdb_documents_retrieved": 0,
            "indico_events_retrieved": 0,
            "docdb_documents_processed": 0,
            "indico_events_processed": 0,
            "total_chunks_added": 0,
            "docdb_pages_new": 0,
            "docdb_pages_updated": 0,
            "docdb_attachments_indexed": 0,
            "docdb_chunks_written": 0,
        }

        #
        # --- DocDB portion ---
        #
        def docdb_extraction(name):
            try:
                logger.info("Processing DocDB documents")
                if docdb_limit == -1:
                    return

                indexed_versions = dict(
                    self.chroma_manager.get_docdb_versions()
                )

                for docid, version, chunks in (
                    self.docdb_extractor.iter_prepared_docdb_documents(
                        start=start_ddb,
                        limit=docdb_limit,
                        existing_versions=indexed_versions,
                        chunk_size=self.chunk_size,
                        max_missing=1000,
                    )
                ):
                    with log_lock:
                        was_indexed = str(docid) in indexed_versions

                        replaced = self.chroma_manager.replace_docdb_document(
                            docid, version, chunks
                        )

                        if replaced:
                            if was_indexed:
                                results["docdb_pages_updated"] += 1
                            else:
                                results["docdb_pages_new"] += 1

                            results["docdb_attachments_indexed"] += len({
                                str(chunk["filename"])
                                for chunk in chunks
                                if chunk.get("filename")
                            })
                            results["docdb_chunks_written"] += len(chunks)

                            indexed_versions[str(docid)] = int(version)
                            results["docdb_documents_retrieved"] += 1
                            results["docdb_documents_processed"] += 1
                            results["total_chunks_added"] += len(chunks)

                            logger.info(
                                "Indexed DocDB %s version %s (%s chunks)",
                                docid, version, len(chunks)
                            )

            except Exception as e:
                logger.exception("DocDB indexing failed")
                thread_errors.append(e)

        #
        # --- Indico portion ---
        #
        def indico_extraction(name):
            try:
                
                logger.info("Processing Indico events")
                if indico_limit==-1: return []

                for events_processed, chunks, events_retrieved in self.indico_extractor.extract_documents(start=start_ind, limit=indico_limit, chunk_size=self.chunk_size):
                    logger.debug(f"Indico records returns {len(chunks)} chunks from {events_processed} events")
                   

                    log_to_db_indico(chunks, events_processed, events_retrieved)
                    
            except Exception as e:
                logger.error(f"Error processing Indico events: {e}")

        def log_to_db_indico(chunks_batch, num_events_processed, num_events_retrieved):
            with log_lock:
                added = self.chroma_manager.add_entries(chunks_batch)
                results['indico_events_retrieved'] += num_events_retrieved
                results["indico_events_processed"] += num_events_processed
                
                results["total_chunks_added"] += added
                
                logger.debug(f"Added Indico to Chroma: added {added} new chunks to index")
                return added
            
        thread_errors = []

        docdb_thread = threading.Thread(target=docdb_extraction, args=('docdb',))
        indico_thread = threading.Thread(target=indico_extraction, args=('indico',))

        log_lock = threading.Lock()
        docdb_thread.start()
        indico_thread.start()

        # Wait for everything to finish
        docdb_thread.join()
        indico_thread.join()

        if thread_errors:
            raise RuntimeError(
                "DocDB indexing failed; see preceding error log"
            ) from thread_errors[0]


        # Final DocDB indexing report.
        extractor_stats = dict(self.docdb_extractor.run_stats)
        cumulative_stats = self.chroma_manager.get_docdb_index_stats()
        final_disk_size = self.chroma_manager.get_stats()["disk_size"]
        elapsed_seconds = time.perf_counter() - run_start
        disk_change = final_disk_size - initial_disk_size

        results.update({
            "docdb_pages_checked": extractor_stats["pages_checked"],
            "docdb_pages_missing": extractor_stats["pages_missing"],
            "docdb_pages_unchanged": extractor_stats["pages_unchanged"],
            "docdb_pages_failed": extractor_stats["pages_failed"],
            "docdb_pages_prepared": extractor_stats["pages_prepared"],
            "elapsed_seconds": elapsed_seconds,
            "database_size_bytes": final_disk_size,
            "database_growth_bytes": disk_change,
            **cumulative_stats,
        })

        return results


    def get_index_stats(self) -> Dict[str, int]:
        """Get current index statistics"""
        return self.chroma_manager.get_stats()

    def cleanup(self):
        """Cleanup resources"""
        self.chroma_manager.cleanup()
