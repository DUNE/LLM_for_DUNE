# Core document processing

This package contains application-level document processing integrations used by AskDUNE.

The repository retains legacy ChromaDB document-processing components for compatibility with historical workflows and tools. The FastAPI web application now uses the unified parent-child retrieval backend.

## Relationship to other modules

- `src/extractors/`: Retrieves source documents, attachments, and metadata from DocDB and Indico.
- `src/indexing/`: Contains ChromaDB indexing components, including unified parent-child indexing.
- `src/retrieval/`: Implements unified hybrid retrieval, reranking, metadata filtering, and parent-context reconstruction.
- `src/api/`: Contains the Argonne Argo and Fermilab LLM clients.
- `main.py`: Initializes the unified retrieval backend and the configured LLM client for the FastAPI application.

## Legacy compatibility

Legacy ChromaDB document-processing utilities remain in the repository but are not used by the unified FastAPI retrieval path.

Some historical scripts and the separate Slack bot may still depend on these components. Retain them until their dependencies have been reviewed and any necessary migration has been completed.

Changes to shared extraction or document-processing interfaces should be checked against both unified and legacy call sites.

Do not run legacy indexing commands against unified parent-child collections.

## Storage boundaries

The unified FastAPI application reads from the index configured by `PROD_DB_PATH`.

Unified indexing scripts write to the separate directory configured by `INDEXING_DB_PATH`, or to an explicitly supplied `--index-path`.

The existing live production database must remain unchanged until a deliberate migration is approved and validated.

## Historical FAISS components

The obsolete FAISS processors have been removed from the active application workflow. Do not restore old FAISS imports or assume that FAISS-backed storage is available.

Any remaining legacy references should be reviewed before those tools are reused.
