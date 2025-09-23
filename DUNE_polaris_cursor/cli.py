#!/usr/bin/env python3
"""
DUNE-GPT CLI: Command-line interface for document processing and management
"""
import time
from dotenv import load_dotenv
import os
import click
from pathlib import Path
import sys

# this loads variables from your .env file into the environment
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import validate_config, create_directories, DOC_LIMIT_DOCDB, DOC_LIMIT_INDICO
from src.core.document_processor_multithread import DocumentProcessor
#from src.core.document_processor import DocumentProcessor
#from src.core.document_processor_chroma import DocumentProcessor
from src.indexing.faiss_manager_reindexed import FAISSManager
from src.indexing.chroma_manager import ChromaManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

@click.group()
def cli():
    """DUNE-GPT: RAG-based LLM for DUNE scientific documentation"""
    pass

@cli.command()
@click.option(
    '--docdb-limit',
    default=DOC_LIMIT_DOCDB,
    show_default=True,
    help=f'Number of DocDB documents to process'
)
@click.option(
    '--indico-limit',
    default=DOC_LIMIT_INDICO,
    show_default=True,
    help=f'Number of Indico documents to process'
)
@click.option(
    '--docdb-latest-hint',
    type=int,
    default=None,
    metavar='DOCID',
    help='Optional DocDB docid to start probing from (speeds up latest‐ID detection)'
)

@click.option(
    '--start_idx_ddb',
    type=int,
    default=0,
    metavar='DOCID',
    help='Optional DocDB docid to start probing from (speeds up latest‐ID detection)'
)

@click.option(
    '--start_idx_ind',
    type=int,
    default=0,
    metavar='DOCID',
    help='Optional DocDB docid to start probing from (speeds up latest‐ID detection)'
)

@click.option(
    '--data-path',
    type=str,
    default='data',
)

@click.option(
    '--chunk-size',
    type=int,
    default=7000000000000,
)
@click.option('--force', is_flag=True, help='Force reprocessing of existing documents')
def index(docdb_limit, indico_limit, start_idx_ddb, start_idx_ind, docdb_latest_hint, chunk_size, data_path, force):
    """Extract, embed, and index documents from DocDB and Indico"""
    start=time.time()
    try:
        logger.info("Starting document indexing process")

        # Validate configuration
        validate_config()
        create_directories(data_path)

        # Initialize document processor
        processor = DocumentProcessor(data_path, chunk_size)

        # Process documents; pass the new latest_hint through
        results = processor.process_all_documents(
            start_ddb=start_idx_ddb,
            start_ind=start_idx_ind,
            docdb_limit=docdb_limit,
            indico_limit=indico_limit,
            force=force
        )

        # Display results
        click.echo(f"\n{'='*50}")
        click.echo("INDEXING RESULTS")
        click.echo(f"{'='*50}")
        click.echo(f"DocDB documents processed: {results['docdb_processed']}")
        click.echo(f"Indico documents processed: {results['indico_processed']}")
        click.echo(f"Total new documents added: {results['total_added']}")

        # Show index stats
        stats = processor.get_index_stats()
        click.echo(f"\nCurrent index statistics:")
        click.echo(f"Total documents: {stats['total_documents']}")
        click.echo(f"Total vectors: {stats['total_vectors']}")
        click.echo(f"Metadata entries: {stats['metadata_entries']}")

        # Cleanup
        processor.cleanup()
        end=time.time()



        logger.info(f"Document indexing completed successfully taking {end-start} seconds")

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
def stats():
    """Show current index statistics"""
    try:
        faiss_manager = FAISSManager()
        stats = faiss_manager.get_stats()

        click.echo(f"\n{'='*40}")
        click.echo("INDEX STATISTICS")
        click.echo(f"{'='*40}")
        click.echo(f"Total documents: {stats['total_documents']}")
        click.echo(f"Total vectors: {stats['total_vectors']}")
        click.echo(f"Metadata entries: {stats['metadata_entries']}")

        faiss_manager.cleanup()

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('question')
@click.option('--top-k', default=3, show_default=True, help='Number of similar documents to retrieve')
def query(question, top_k):
    """Query the indexed documents"""
    try:
        from src.api.argo_client import ArgoAPIClient
        from config import ARGO_API_USERNAME, ARGO_API_KEY

        validate_config()

        # Initialize components
        faiss_manager = FAISSManager()
        argo_client = ArgoAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)

        # Check if index has documents
        stats = faiss_manager.get_stats()
        if stats['total_documents'] == 0:
            click.echo("Error: No documents in index. Run 'cli.py index' first.", err=True)
            sys.exit(1)

        # Search and get answer
        click.echo(f"Searching for: {question}")
        context_snippets, references = faiss_manager.search(question, top_k=top_k)
        context = "\n\n".join(context_snippets)

        click.echo(f"\nFound {len(context_snippets)} relevant documents")

        # Get answer from Argo API
        click.echo("Getting answer from LLM...")
        answer = argo_client.chat_completion(question, context)

        # Display results
        click.echo(f"\n{'='*60}")
        click.echo("ANSWER")
        click.echo(f"{'='*60}")
        click.echo(answer)

        if references:
            click.echo(f"\n{'='*60}")
            click.echo("REFERENCES")
            click.echo(f"{'='*60}")
            for i, ref in enumerate(references, 1):
                click.echo(f"{i}. {ref}")

        # Cleanup
        faiss_manager.cleanup()

    except Exception as e:
        logger.error(f"Query failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
def serve():
    """Start the web server"""
    try:
        from main import main
        main()
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@cli.command()
def health():
    """Check system health"""
    try:
        from src.api.argo_client import ArgoAPIClient
        from config import ARGO_API_USERNAME, ARGO_API_KEY

        click.echo("Checking system health...")

        # Check configuration
        try:
            validate_config()
            click.echo("✓ Configuration: OK")
        except Exception as e:
            click.echo(f"✗ Configuration: {e}")
            return

        # Check authentication configuration
        from config import ENABLE_AUTHENTICATION, FERMILAB_CLIENT_ID, FERMILAB_CLIENT_SECRET
        if ENABLE_AUTHENTICATION:
            if FERMILAB_CLIENT_ID and FERMILAB_CLIENT_SECRET:
                click.echo("✓ Fermilab Authentication: Configured")
            else:
                click.echo("✗ Fermilab Authentication: Missing credentials")
        else:
            click.echo("ℹ Fermilab Authentication: Disabled")

        # Check FAISS index
        try:
            faiss_manager = FAISSManager()
            stats = faiss_manager.get_stats()
            click.echo(f"✓ FAISS Index: {stats['total_documents']} documents")
            faiss_manager.cleanup()
        except Exception as e:
            click.echo(f"✗ FAISS Index: {e}")

        # Check Argo API
        try:
            argo_client = ArgoAPIClient(ARGO_API_USERNAME, ARGO_API_KEY)
            if argo_client.health_check():
                click.echo("✓ Argo API: Available")
            else:
                click.echo("✗ Argo API: Unavailable")
        except Exception as e:
            click.echo(f"✗ Argo API: {e}")

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()
