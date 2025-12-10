from datetime import datetime
from typing import List, Dict, Any, Optional, Set

from src.extractors.docdb_extractor import DocDBExtractor
from src.extractors.indico_extractor import IndicoExtractor
from src.indexing.faiss_manager_reindexed import FAISSManager
from config import DOC_LIMIT_DOCDB, DOC_LIMIT_INDICO
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Orchestrates document extraction, processing, and indexing"""

    def __init__(self):
        
        self.faiss_manager = FAISSManager()
        self.docdb_extractor = DocDBExtractor(self.faiss_manager)
        self.indico_extractor = IndicoExtractor(self.faiss_manager)


    def process_all_documents(
        self,
        docdb_limit: int = DOC_LIMIT_DOCDB,
        indico_limit: int = DOC_LIMIT_INDICO,
        force: bool = False,
        docdb_latest_hint: Optional[int] = None
    ) -> Dict[str, int]:
        logger.info("Starting document processing pipeline")
        results = {
            "docdb_processed": 0,
            "indico_processed": 0,
            "total_added": 0
        }

        #
        # --- DocDB portion ---
        #
        try:
            logger.info("Processing DocDB documents")

            # 1) What versions (and thus IDs) do we already have?
            indexed_versions = self.faiss_manager.get_docdb_versions()
            content_modif_dates = self.faiss_manager.get_content_modification_dates()
            metadata_modif_dates = self.faiss_manager.get_metadata_modification_dates()
            #indexed_ids: Set[int] = set(indexed_versions.keys())
            indexed_ids: Set[int] = { int(did) for did in indexed_versions.keys() }

            # 2) Fetch up to `docdb_limit` pages, skipping any IDs in indexed_ids
            raw_records = self.docdb_extractor.extract_documents(
                limit=docdb_limit,
                indexed_doc_ids=indexed_ids,
                mode="incremental",
                stop_after_seen=100,
                max_missing=1000,
                latest_hint=docdb_latest_hint
            )
            logger.info(f"Fetched {len(raw_records)} records")
                

            # 3) Optionally filter by version bump or force‐flag
            to_reindex: List[Dict[str, Any]] = []
            for rec in raw_records:
                did = int(rec["document_id"])

                last_modified_md = rec['metadata_last_modified_date']
                if not metadata_modif_dates.get(did,None):
                    modified_date_md = datetime.min
                else:
                    metadata_modif_date = metadata_modif_dates[did]
                    modified_date_md = datetime.strptime(metadata_modif_date, '%Y-%m-%d') 
                last_scraped_date_md = datetime.strptime(last_modified_md, '%Y-%m-%d') 

                last_modified_ct= rec['content_last_modified_date']
                if not content_modif_dates.get(did,None):
                    modified_date_ct =datetime.min
                else:
                    content_modified_date = content_modif_dates[did]
                    modified_date_ct = datetime.strptime(content_modified_date, '%Y-%m-%d') 
                last_scraped_date_ct = datetime.strptime(last_modified_ct, '%Y-%m-%d') 

               
                incoming_v = int(rec.get("docdb_version", "0") or "0")
                if force or  incoming_v > indexed_versions.get(did, 0) or modified_date_md > last_scraped_date_md or modified_date_ct > last_scraped_date_ct:
                    to_reindex.append(rec)
            logger.info(f"Reindexed records")
            

            # 4) Prune old versions & rebuild only if there are updates
            if to_reindex:
                logger.info(f"Found {len(to_reindex)} new/updated DocDB docs; pruning & rebuilding…")
                for rec in to_reindex:
                    did_str = rec["document_id"]
                    self.faiss_manager.metadata_store.pop(did_str, None)
                    if did_str in self.faiss_manager.doc_ids:
                        self.faiss_manager.doc_ids.remove(did_str)
                self.faiss_manager._save_metadata()
                self.faiss_manager._save_doc_ids()
                self.faiss_manager.rebuild_index()
            
            logger.info(f"to_reindex is {len(to_reindex)}")
            # 5) Add the new/updated docs
            added = self.faiss_manager.add_documents(to_reindex)
            results["docdb_processed"] = len(to_reindex)
            results["total_added"] += added

            logger.info(
                f"DocDB: scanned {len(raw_records)} pages, "
                f"reindexed {len(to_reindex)}, added {added} vectors"
            )

        except Exception as e:
            logger.error(f"Error processing DocDB documents: {e}")

        #
        # --- Indico portion ---
        #
        try:
            logger.info("Processing Indico documents")

            indico_records = self.indico_extractor.extract_documents(limit=indico_limit)

            logger.info(f"Indico records returns {len(indico_records[0])}")
            # Skip any Indico docs already in FAISS
            existing_ids = set(self.faiss_manager.doc_ids)
            to_add = [
                doc for doc in indico_records
                if doc["document_id"] not in existing_ids
            ]

            added = self.faiss_manager.add_documents(to_add)
            results["indico_processed"] = len(to_add)
            results["total_added"] += added

            logger.info(
                f"Indico: extracted {len(to_add)} docs, "
                f"added {added} new vectors to index"
            )

        except Exception as e:
            logger.error(f"Error processing Indico documents: {e}")

        # Final summary & return
        logger.info(f"Document processing completed. Total new docs added: {results['total_added']}")
        return results

    def get_index_stats(self) -> Dict[str, int]:
        """Get current index statistics"""
        return self.faiss_manager.get_stats()

    def cleanup(self):
        """Cleanup resources"""
        self.faiss_manager.cleanup()
