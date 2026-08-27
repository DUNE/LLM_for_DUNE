#!/usr/bin/env python3
"""
DUNE-GPT CLI: Command-line interface for document processing and management
"""
import time
import json
from dotenv import load_dotenv
import os
import click
from pathlib import Path
import sys
from src.indexing.indexer import IndexingJob
from config import DB_PATH

# this loads variables from your .env file into the environment
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import STORE, validate_config, create_directories, DOC_LIMIT_DOCDB, DOC_LIMIT_INDICO, CHROMA_PATH, CHUNK_SIZE, EMBEDDING_MODEL
if STORE == 'faiss':
    from src.core.document_processor_faiss import DocumentProcessor
elif STORE == 'chroma':
    from src.core.document_processor_chroma import DocumentProcessor
else:
    raise Exception(f"Invalid Store. Must be FAISS or CHROMA. Got {store}")

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

@click.option('--force', is_flag=True, help='Force reprocessing of existing documents')
def index(docdb_limit, indico_limit, start_idx_ddb, start_idx_ind, docdb_latest_hint,  data_path, force):
    """Extract, embed, and index documents from DocDB and Indico"""
    start=time.time()
    try:
        logger.info("Starting document indexing process")
        logger.info(f"chunk size is {CHUNK_SIZE}")

        # Validate configuration
        validate_config()
        create_directories(data_path)
        
        start_idx_ddb = int(os.getenv('DDB_START_IDX', start_idx_ddb))
        start_idx_ind = int(os.getenv('IND_START_IDX', start_idx_ind))
        docdb_limit = int(os.getenv('DOCUMENT_LIMIT', docdb_limit))
        indico_limit = int(os.getenv('DOCUMENT_LIMIT', indico_limit))
        data_path = os.getenv("DB_PATH", data_path)

        # Initialize document processor
        logger.info("Init processor")
        processor = DocumentProcessor(data_path, int(CHUNK_SIZE))
        logger.info("Processing all docs")
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
        click.echo(f"DocDB Events parsed: {results['docdb_parsed']}")
        click.echo(f"Indico Events parsed: {results['indico_parsed']}")
        click.echo(f"DocDB Events processed: {results['docdb_processed']}")
        click.echo(f"Indico Events processed: {results['indico_processed']}")
        click.echo(f"Total new events added: {results['indico_processed'] + results['docdb_processed']}")
        click.echo(f"Total new embeddings added: {results['total_embeddings_added']}")

        # Show index stats
        stats = processor.get_index_stats()
        click.echo(f"\nCurrent index statistics:")
        click.echo(f"Total Events: {stats['total_documents']}")
        click.echo(f"Total Embeddings: {stats['total_embeddings']}")
        click.echo(f"Total Number of Attachments in Metadata: {stats['total_number_attachments_in_metadata']}")
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

@cli.command("inspect-index")
def inspect_index():
    job = IndexingJob(DB_PATH)
    job.run()

@cli.command("index-local")
@click.option(
    "--benchmark-config",
    type=str,
    default=None,
    help="Optional JSON config for local benchmarking/indexing.",
)
@click.option(
    "--cache-path",
    type=str,
    default=None,
    help="Path containing cached raw attachment manifests and files.",
)
@click.option(
    "--source",
    type=click.Choice(["both", "docdb", "indico"]),
    default=None,
    help="Which cached source to index.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional maximum number of cached attachments to inspect.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Chunk size for local benchmark indexing.",
)
@click.option(
    "--chunk-strategy",
    type=click.Choice(["word", "char"]),
    default=None,
    help="Chunking strategy for local benchmark indexing.",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=None,
    help="Chunk overlap for local benchmark indexing, in words or chars depending on strategy.",
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="SentenceTransformers embedding model for this benchmark run.",
)
@click.option(
    "--data-path",
    type=str,
    default=None,
    help="Chroma persistence path. Defaults to DB_PATH from .env/config.",
)
def index_local(benchmark_config, cache_path, source, limit, chunk_size, chunk_strategy, chunk_overlap, embedding_model, data_path):
    """Embed and index cached local attachments without crawling DocDB/Indico."""
    start = time.time()
    try:
        validate_config()
        from src.core.local_attachment_processor import LocalAttachmentProcessor

        config_data = {}
        if benchmark_config:
            config_path = Path(benchmark_config)
            if not config_path.is_absolute():
                config_path = Path(__file__).parent / config_path
            with config_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)

        cache_path = cache_path or config_data.get("cache_path", "benchmarking/raw_attachments")
        source = source or config_data.get("source", "both")
        limit = limit if limit is not None else config_data.get("limit")
        chunk_size = chunk_size or int(config_data.get("chunk_size", CHUNK_SIZE))
        chunk_strategy = chunk_strategy or config_data.get("chunk_strategy", "word")
        chunk_overlap = chunk_overlap if chunk_overlap is not None else int(config_data.get("chunk_overlap", 0))
        embedding_model = embedding_model or config_data.get("embedding_model")
        chroma_path = data_path or config_data.get("data_path") or os.getenv("DB_PATH", "data")

        logger.info(f"Indexing local cached attachments from {cache_path}")
        logger.info(f"Chroma path: {chroma_path}")
        logger.info(f"chunk size is {chunk_size}")
        logger.info(f"chunk strategy is {chunk_strategy}")
        logger.info(f"chunk overlap is {chunk_overlap}")
        logger.info(f"embedding model is {embedding_model or 'config default'}")

        processor = LocalAttachmentProcessor(
            chroma_path,
            cache_path,
            int(chunk_size),
            chunk_strategy=chunk_strategy,
            chunk_overlap=int(chunk_overlap),
            embedding_model=embedding_model or EMBEDDING_MODEL,
        )
        results = processor.process(source=source, limit=limit)

        click.echo(f"\n{'='*50}")
        click.echo("LOCAL INDEXING RESULTS")
        click.echo(f"{'='*50}")
        click.echo(f"Attachments inspected: {results['attachments_seen']}")
        click.echo(f"Attachments with extracted text: {results['attachments_with_text']}")
        click.echo(f"Chunks created: {results['chunks_created']}")
        click.echo(f"Embeddings added: {results['embeddings_added']}")

        stats = processor.get_index_stats()
        click.echo(f"\nCurrent index statistics:")
        click.echo(f"Total Events/Attachments: {stats['total_documents']}")
        click.echo(f"Total Embeddings: {stats['total_embeddings']}")
        click.echo(f"Total Number of Attachments in Metadata: {stats['total_number_attachments_in_metadata']}")

        processor.cleanup()
        logger.info(f"Local indexing completed successfully taking {time.time() - start} seconds")

    except Exception as e:
        logger.error(f"Local indexing failed: {e}")
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
