#!/usr/bin/env python3
"""
Fermilab Setup - Load pre-built FAISS index for DUNE-GPT serving
Run this on Fermilab machines after receiving Aurora-generated index
"""

import os
import sys
import json
import argparse
import logging
import tarfile
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from indexing.faiss_manager import FAISSManager
from utils.logger import setup_logger

class FermilabSetup:
    """Fermilab-specific setup for serving pre-built indices"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__)
    
    def extract_index_package(self, package_path: str):
        """Extract Aurora-generated index package"""
        package_path = Path(package_path)
        
        if not package_path.exists():
            raise FileNotFoundError(f"Package not found: {package_path}")
        
        self.logger.info(f"📦 Extracting index package: {package_path}")
        
        # Extract to current directory
        with tarfile.open(package_path, "r:gz") as tar:
            tar.extractall()
            members = tar.getnames()
            self.logger.info(f"Extracted {len(members)} files/directories")
        
        # Load and validate metadata
        metadata_path = Path("metadata.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            self.logger.info("📋 Index Metadata:")
            self.logger.info(f"  Created: {metadata.get('created_at')}")
            self.logger.info(f"  Source: {metadata.get('created_on')}")
            self.logger.info(f"  Model: {metadata.get('config', {}).get('sentence_transformer_model')}")
            
            return metadata
        else:
            self.logger.warning("⚠️ No metadata found in package")
            return {}
    
    def verify_index_integrity(self):
        """Verify that the FAISS index is properly loaded"""
        self.logger.info("🔍 Verifying index integrity...")
        
        try:
            # Initialize FAISS manager with existing index
            faiss_manager = FAISSManager(self.config)
            
            # Try to load the index
            if not faiss_manager.load_index():
                raise RuntimeError("Failed to load FAISS index")
            
            # Get index statistics
            stats = faiss_manager.get_index_stats()
            
            self.logger.info("✅ Index verification successful!")
            self.logger.info(f"  Documents: {stats.get('total_documents', 0)}")
            self.logger.info(f"  Vectors: {stats.get('total_vectors', 0)}")
            self.logger.info(f"  Dimension: {stats.get('dimension', 0)}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Index verification failed: {e}")
            raise
    
    def test_search_functionality(self):
        """Test basic search functionality"""
        self.logger.info("🔍 Testing search functionality...")
        
        try:
            faiss_manager = FAISSManager(self.config)
            faiss_manager.load_index()
            
            # Test search with sample query
            test_queries = [
                "DUNE detector design",
                "neutrino oscillation",
                "liquid argon"
            ]
            
            for query in test_queries:
                results = faiss_manager.search(query, top_k=3)
                if results:
                    self.logger.info(f"✅ Search test '{query}': {len(results)} results")
                else:
                    self.logger.warning(f"⚠️ Search test '{query}': No results")
            
            self.logger.info("✅ Search functionality verified!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Search test failed: {e}")
            return False
    
    def setup_fermilab_config(self):
        """Configure environment for Fermilab serving"""
        self.logger.info("⚙️ Setting up Fermilab configuration...")
        
        # Check if .env exists, create from template if not
        env_file = Path(".env")
        if not env_file.exists():
            env_example = Path("env.example")
            if env_example.exists():
                import shutil
                shutil.copy(env_example, env_file)
                self.logger.info("📝 Created .env from template")
                self.logger.warning("⚠️ Please edit .env with your Fermilab credentials!")
            else:
                self.logger.error("❌ No env.example template found")
                return False
        
        # Verify required Fermilab settings
        required_settings = [
            "ENABLE_AUTHENTICATION",
            "FERMILAB_CLIENT_ID", 
            "FERMILAB_CLIENT_SECRET",
            "FERMILAB_REDIRECT_URI",
            "ARGO_API_USERNAME",
            "ARGO_API_KEY"
        ]
        
        missing = []
        for setting in required_settings:
            if not getattr(self.config, setting, None):
                missing.append(setting)
        
        if missing:
            self.logger.warning(f"⚠️ Missing configuration: {', '.join(missing)}")
            self.logger.warning("Please update .env file before starting the server")
        else:
            self.logger.info("✅ Fermilab configuration looks good!")
        
        return len(missing) == 0
    
    def create_startup_script(self):
        """Create convenient startup script for Fermilab"""
        script_content = """#!/bin/bash
# DUNE-GPT Fermilab Startup Script
# Auto-generated by fermilab_setup.py

set -e

echo "🚀 Starting DUNE-GPT at Fermilab..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies
echo "📋 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Verify index
echo "🔍 Verifying FAISS index..."
python scripts/verify_index.py

# Start server
echo "✅ Starting DUNE-GPT server..."
echo "🌐 Access at: http://localhost:8000"
echo "🔐 Fermilab authentication enabled"
echo ""
python cli.py serve
"""
        
        script_path = Path("start_fermilab.sh")
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        self.logger.info(f"📜 Created startup script: {script_path}")
        return script_path

def main():
    parser = argparse.ArgumentParser(description="Fermilab Setup for DUNE-GPT")
    parser.add_argument("--package", type=str,
                       help="Path to Aurora-generated index package")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip package extraction (already extracted)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing index")
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
    setup = FermilabSetup(config)
    
    try:
        print("🏢 DUNE-GPT Fermilab Setup")
        print("="*40)
        
        if args.verify_only:
            # Only verify existing setup
            stats = setup.verify_index_integrity()
            setup.test_search_functionality()
            print("\n✅ Verification complete!")
            return
        
        # Extract package if provided and not skipped
        if args.package and not args.skip_extraction:
            metadata = setup.extract_index_package(args.package)
            print(f"📦 Extracted package from {metadata.get('created_on', 'unknown')}")
        
        # Verify index integrity
        stats = setup.verify_index_integrity()
        
        # Test search functionality
        search_ok = setup.test_search_functionality()
        
        # Setup Fermilab configuration
        config_ok = setup.setup_fermilab_config()
        
        # Create startup script
        startup_script = setup.create_startup_script()
        
        print("\n" + "="*50)
        print("FERMILAB SETUP COMPLETE")
        print("="*50)
        print(f"📊 Documents indexed: {stats.get('total_documents', 0)}")
        print(f"🔍 Search functionality: {'✅ Working' if search_ok else '❌ Failed'}")
        print(f"⚙️ Configuration: {'✅ Ready' if config_ok else '⚠️ Needs attention'}")
        print(f"📜 Startup script: {startup_script}")
        print("="*50)
        
        if config_ok and search_ok:
            print("\n🎉 Ready to serve DUNE users!")
            print(f"Run: ./{startup_script.name}")
            print("Or: python cli.py serve")
        else:
            print("\n⚠️ Please fix configuration issues before starting")
            
    except Exception as e:
        logging.error(f"Fermilab setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 