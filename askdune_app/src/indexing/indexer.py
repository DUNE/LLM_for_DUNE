# src/indexing/indexer.py
import logging
from src.indexing.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

class IndexingJob:
    def __init__(self, chroma_path: str):
        self.cm = ChromaManager(chroma_path)

    def fetch_ddb_ind_data(self):
        logger.info("Fetching DocDB / Indico metadata from Chroma")

        results = self.cm.chroma_collection.get(
            include=["metadatas"]   # 🚨 NOT documents
        )

        indico_ids = set()
        docdb_versions = {}
        events_ids = set()

        for _id, md in zip(results["ids"], results["metadatas"]):
            source = md.get("source")

            if source == "indico":
                indico_ids.add(_id.split("_")[0])
            elif source == "docdb":
                docdb_versions[_id] = md.get("docdb_version")

            events_ids.add(_id.split("_")[0])

        logger.info(f"Found {len(events_ids)} events")
        return {
            "indico_ids": indico_ids,
            "docdb_versions": docdb_versions,
            "events_ids": events_ids,
        }

    def run(self):
        logger.info("Starting indexing job")
        meta = self.fetch_ddb_ind_data()

        # Later:
        # - crawl Indico
        # - crawl DocDB
        # - compare versions
        # - embed new docs
        # - upsert into Chroma

        logger.info("Indexing job completed")
