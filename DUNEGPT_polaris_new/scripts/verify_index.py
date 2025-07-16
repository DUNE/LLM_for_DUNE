#!/usr/bin/env python3
"""
Index Verification - Verify FAISS index integrity and functionality
Can be run on both Aurora (after indexing) and Fermilab (after transfer)
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from indexing.faiss_manager import FAISSManager
from utils.logger import setup_logger

def verify_faiss_index(config: Config):
    """Comprehensive FAISS index verification"""
    logger = setup_logger(__name__)
    
    print("🔍 FAISS Index Verification")
    print("="*40)
    
    try:
        # Initialize FAISS manager
        faiss_manager = FAISSManager(config)
        
        # Check if index files exist
        index_path = Path(config.FAISS_INDEX_PATH)
        metadata_path = Path(config.FAISS_METADATA_PATH)
        
        if not index_path.exists():
            print(f"❌ Index file not found: {index_path}")
            return False
            
        if not metadata_path.exists():
            print(f"❌ Metadata file not found: {metadata_path}")
            return False
        
        print(f"✅ Index files found")
        
        # Load index
        if not faiss_manager.load_index():
            print("❌ Failed to load FAISS index")
            return False
            
        print("✅ Index loaded successfully")
        
        # Get statistics
        stats = faiss_manager.get_index_stats()
        
        print(f"\n📊 Index Statistics:")
        print(f"  Documents: {stats.get('total_documents', 0)}")
        print(f"  Vectors: {stats.get('total_vectors', 0)}")
        print(f"  Dimension: {stats.get('dimension', 0)}")
        print(f"  Index type: {stats.get('index_type', 'Unknown')}")
        
        # Verify minimum requirements
        min_docs = 10
        min_vectors = 50
        expected_dim = 384
        
        issues = []
        
        if stats.get('total_documents', 0) < min_docs:
            issues.append(f"Too few documents ({stats.get('total_documents', 0)} < {min_docs})")
            
        if stats.get('total_vectors', 0) < min_vectors:
            issues.append(f"Too few vectors ({stats.get('total_vectors', 0)} < {min_vectors})")
            
        if stats.get('dimension', 0) != expected_dim:
            issues.append(f"Wrong dimension ({stats.get('dimension', 0)} != {expected_dim})")
        
        if issues:
            print(f"\n⚠️ Index issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print(f"\n✅ Index statistics look good!")
        
        # Test search functionality
        print(f"\n🔍 Testing Search Functionality:")
        
        test_queries = [
            "DUNE detector",
            "neutrino oscillation", 
            "liquid argon",
            "beam",
            "physics"
        ]
        
        search_results = {}
        for query in test_queries:
            try:
                results = faiss_manager.search(query, top_k=3)
                search_results[query] = len(results) if results else 0
                print(f"  '{query}': {search_results[query]} results")
            except Exception as e:
                print(f"  '{query}': ERROR - {e}")
                search_results[query] = -1
        
        # Check search quality
        total_results = sum(r for r in search_results.values() if r > 0)
        successful_searches = sum(1 for r in search_results.values() if r > 0)
        
        if successful_searches < len(test_queries) * 0.8:
            print(f"\n⚠️ Search functionality issues:")
            print(f"  Only {successful_searches}/{len(test_queries)} queries returned results")
            return False
        
        print(f"\n✅ Search functionality verified!")
        print(f"  {successful_searches}/{len(test_queries)} queries successful")
        print(f"  {total_results} total results returned")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        print(f"❌ Verification failed: {e}")
        return False

def verify_metadata_integrity():
    """Verify metadata files are consistent and complete"""
    print(f"\n📋 Metadata Verification:")
    
    config = Config()
    metadata_path = Path(config.FAISS_METADATA_PATH)
    
    if not metadata_path.exists():
        print(f"❌ Metadata file missing: {metadata_path}")
        return False
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        required_fields = ['documents', 'embeddings', 'index_info']
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            print(f"❌ Missing metadata fields: {', '.join(missing_fields)}")
            return False
        
        # Check document metadata
        doc_count = len(metadata.get('documents', {}))
        embedding_count = len(metadata.get('embeddings', {}))
        
        print(f"  Documents in metadata: {doc_count}")
        print(f"  Embeddings in metadata: {embedding_count}")
        
        if doc_count != embedding_count:
            print(f"⚠️ Document/embedding count mismatch: {doc_count} != {embedding_count}")
        
        print(f"✅ Metadata structure is valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Metadata JSON is corrupted: {e}")
        return False
    except Exception as e:
        print(f"❌ Metadata verification failed: {e}")
        return False

def verify_configuration(config: Config):
    """Verify configuration is suitable for operation"""
    print(f"\n⚙️ Configuration Verification:")
    
    # Check essential paths
    paths_to_check = [
        ("FAISS Index", config.FAISS_INDEX_PATH),
        ("FAISS Metadata", config.FAISS_METADATA_PATH),
        ("Data Directory", Path(config.FAISS_INDEX_PATH).parent)
    ]
    
    for name, path in paths_to_check:
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (not found)")
    
    # Check model configuration
    print(f"  Model: {config.SENTENCE_TRANSFORMER_MODEL}")
    
    # Check API configuration (without revealing credentials)
    api_configs = [
        ("Argo API Username", bool(config.ARGO_API_USERNAME)),
        ("Argo API Key", bool(config.ARGO_API_KEY)),
        ("DocDB Username", bool(config.DUNE_DOCDB_USERNAME)),
        ("DocDB Password", bool(config.DUNE_DOCDB_PASSWORD))
    ]
    
    for name, configured in api_configs:
        status = "✅ Configured" if configured else "❌ Not configured"
        print(f"  {status}: {name}")
    
    # Check authentication setup
    if config.ENABLE_AUTHENTICATION:
        auth_configured = all([
            config.FERMILAB_CLIENT_ID,
            config.FERMILAB_CLIENT_SECRET,
            config.FERMILAB_REDIRECT_URI
        ])
        print(f"  {'✅' if auth_configured else '❌'} Fermilab Authentication: {'Configured' if auth_configured else 'Incomplete'}")
    else:
        print(f"  ⚠️ Authentication: Disabled")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify DUNE-GPT FAISS Index")
    parser.add_argument("--skip-search", action="store_true",
                       help="Skip search functionality tests")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick verification (index loading only)")
    parser.add_argument("--log-level", default="WARNING",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = Config()
    
    print("🔬 DUNE-GPT Index Verification")
    print("="*50)
    
    all_passed = True
    
    # Configuration check
    verify_configuration(config)
    
    # Quick mode - just check if index loads
    if args.quick:
        try:
            faiss_manager = FAISSManager(config)
            if faiss_manager.load_index():
                print("\n✅ Quick verification: Index loads successfully")
                return 0
            else:
                print("\n❌ Quick verification: Index failed to load")
                return 1
        except Exception as e:
            print(f"\n❌ Quick verification failed: {e}")
            return 1
    
    # Full verification
    if not verify_faiss_index(config):
        all_passed = False
    
    if not verify_metadata_integrity():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED")
        print("🎉 Index is ready for use!")
    else:
        print("❌ VERIFICATION ISSUES FOUND")
        print("🔧 Please fix issues before using")
    print("="*50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 