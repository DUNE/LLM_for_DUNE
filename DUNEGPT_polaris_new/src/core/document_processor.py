from typing import List, Dict, Any
from src.extractors.docdb_extractor import DocDBExtractor
from src.extractors.indico_extractor import IndicoExtractor
from src.indexing.faiss_manager import FAISSManager
from config import DOC_LIMIT_DOCDB, DOC_LIMIT_INDICO
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """Orchestrates document extraction, processing, and indexing"""
    
    def __init__(self):
        self.docdb_extractor = DocDBExtractor()
        self.indico_extractor = IndicoExtractor()
        self.faiss_manager = FAISSManager()
    
    def process_all_documents(
        self, 
        docdb_limit: int = DOC_LIMIT_DOCDB,
        indico_limit: int = DOC_LIMIT_INDICO
    ) -> Dict[str, int]:
        """Process documents from all sources"""
        
        logger.info("Starting document processing pipeline")
        
        results = {
            "docdb_processed": 0,
            "indico_processed": 0,
            "total_added": 0
        }
        
        # Process DocDB documents
        try:
            logger.info("Processing DocDB documents")
            docdb_documents = self.docdb_extractor.extract_documents(docdb_limit)
            docdb_added = self.faiss_manager.add_documents(docdb_documents)
            results["docdb_processed"] = len(docdb_documents)
            results["total_added"] += docdb_added
            logger.info(f"DocDB: {len(docdb_documents)} extracted, {docdb_added} added to index")
        except Exception as e:
            logger.error(f"Error processing DocDB documents: {e}")
        
        # Process Indico documents
        try:
            logger.info("Processing Indico documents")
            indico_documents = self.indico_extractor.extract_documents(indico_limit)
            indico_added = self.faiss_manager.add_documents(indico_documents)
            results["indico_processed"] = len(indico_documents)
            results["total_added"] += indico_added
            logger.info(f"Indico: {len(indico_documents)} extracted, {indico_added} added to index")
        except Exception as e:
            logger.error(f"Error processing Indico documents: {e}")
        
        logger.info(f"Document processing completed. Total new documents added: {results['total_added']}")
        
        return results
    
    def get_index_stats(self) -> Dict[str, int]:
        """Get current index statistics"""
        return self.faiss_manager.get_stats()
    
    def cleanup(self):
        """Cleanup resources"""
        self.faiss_manager.cleanup() 