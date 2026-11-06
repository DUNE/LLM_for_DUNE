# Core document processing

This package contains application-level document processing integrations. The legacy ChromaDB document-processing path remains available for compatibility while the unified retrieval backend is developed and validated.

## Relationship to other modules

- `src/extractors/`: obtains source documents, attachments, and metadata.
- `src/indexing/`: persists records in ChromaDB.
- `src/retrieval/`: searches stored records and reconstructs parent contexts.
- `src/api/`: contains the Argo and Fermilab LLM clients.
- `main.py`: selects the retrieval backend at startup and initializes the configured LLM client.

The obsolete FAISS processors have been removed. Do not restore old FAISS imports or assume FAISS-backed storage exists.

## Compatibility considerations

The legacy ChromaDB manager and processing integration should be retained until the unified backend has completed validation and a migration plan is approved. Changes to extraction or document-processing interfaces should be checked against both active and legacy call sites. Never run legacy indexing commands against the experimental parent-child collections.
