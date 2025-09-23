#!/usr/bin/env python3
"""
Aurora Indexer - Heavy compute indexing for DUNE-GPT
Run this on Aurora/ANL machines to generate embeddings and FAISS index
"""

import os
import sys
import argparse
import logging
import tarfile
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from core.document_processor import DocumentProcessor
from utils.logger import setup_logger

class AuroraIndexer:
    """Aurora-specific indexing with packaging for transfer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__)
        
    def run_indexing(self, docdb_limit: int = 1000, indico_limit: int = 1000):
        """Run the full indexing pipeline optimized for Aurora"""
        
        self.logger.info("🚀 Starting Aurora indexing pipeline...")
        self.logger.info(f"Target: {docdb_limit} DocDB + {indico_limit} Indico documents")
        
        # Initialize document processor  
        processor = DocumentProcessor(self.config)
        
        # Process documents with high limits for Aurora
        stats = processor.process_documents(
            docdb_limit=docdb_limit,
            indico_limit=indico_limit
        )
        
        self.logger.info("✅ Indexing completed!")
        return stats
    
    def package_for_transfer(self, output_dir: str = "transfer_package"):
        """Package FAISS index and metadata for transfer to Fermilab"""
        
        self.logger.info("📦 Packaging index for transfer...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"dune_gpt_index_{timestamp}.tar.gz"
        package_path = output_path / package_name
        
        # Create tar archive
        with tarfile.open(package_path, "w:gz") as tar:
            # Add FAISS index files
            faiss_dir = Path(self.config.FAISS_INDEX_PATH).parent
            if faiss_dir.exists():
                tar.add(faiss_dir, arcname="faiss")
                self.logger.info(f"Added FAISS index: {faiss_dir}")
            
            # Create metadata file
            metadata = {
                "created_at": datetime.now().isoformat(),
                "created_on": "Aurora/ANL",
                "python_version": sys.version,
                "config": {
                    "sentence_transformer_model": self.config.SENTENCE_TRANSFORMER_MODEL,
                    "faiss_index_type": "IndexFlatIP",
                    "embedding_dimension": 384
                }
            }
            
            # Write metadata to temp file and add to archive
            metadata_file = output_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            tar.add(metadata_file, arcname="metadata.json")
            metadata_file.unlink()  # Clean up temp file
        
        self.logger.info(f"✅ Package created: {package_path}")
        self.logger.info(f"📊 Package size: {package_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Create transfer instructions
        instructions_file = output_path / "transfer_instructions.txt"
        with open(instructions_file, "w") as f:
            f.write(f"""
DUNE-GPT Index Transfer Instructions
==================================

Package: {package_name}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location: Aurora/ANL

Transfer Commands:
-----------------

1. Secure Copy to Fermilab:
   scp {package_name} username@fnal.gov:/path/to/dunegpt/

2. GridFTP Transfer (if available):
   globus-url-copy file://{package_path.absolute()} gsiftp://fnal.gov/path/to/dunegpt/

3. Extract on Fermilab:
   cd /path/to/dunegpt/
   tar -xzf {package_name}
   
4. Verify integrity:
   python scripts/verify_index.py

Next Steps:
----------
1. Transfer the package to Fermilab
2. Extract in DUNE-GPT directory  
3. Run: python scripts/fermilab_setup.py
4. Start application: python cli.py serve

Package Contents:
----------------
- faiss/: FAISS index files and embeddings
- metadata.json: Index creation metadata
- This instruction file

Security Notes:
--------------
- Package contains no credentials or sensitive data
- Only document embeddings and search index
- Safe for transfer via standard secure channels
""")
        
        self.logger.info(f"📋 Instructions saved: {instructions_file}")
        return package_path, instructions_file

def main():
    parser = argparse.ArgumentParser(description="Aurora Document Indexer for DUNE-GPT")
    parser.add_argument("--docdb-limit", type=int, default=1000, 
                       help="Maximum DocDB documents to process")
    parser.add_argument("--indico-limit", type=int, default=1000,
                       help="Maximum Indico documents to process") 
    parser.add_argument("--package-only", action="store_true",
                       help="Only package existing index (skip indexing)")
    parser.add_argument("--output-dir", default="transfer_package",
                       help="Output directory for transfer package")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = Config()
    indexer = AuroraIndexer(config)
    
    try:
        if not args.package_only:
            # Run indexing
            stats = indexer.run_indexing(
                docdb_limit=args.docdb_limit,
                indico_limit=args.indico_limit
            )
            
            print("\n" + "="*50)
            print("AURORA INDEXING RESULTS")
            print("="*50)
            for key, value in stats.items():
                print(f"{key}: {value}")
            print("="*50)
        
        # Package for transfer
        package_path, instructions_path = indexer.package_for_transfer(args.output_dir)
        
        print(f"\n🎉 Aurora indexing complete!")
        print(f"📦 Transfer package: {package_path}")
        print(f"📋 Instructions: {instructions_path}")
        print(f"\nNext: Transfer package to Fermilab and run setup script.")
        
    except Exception as e:
        logging.error(f"Aurora indexing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 