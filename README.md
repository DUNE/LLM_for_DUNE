# AskDUNE

AskDUNE is a retrieval-augmented generation (RAG) application for documentation associated with the Deep Underground Neutrino Experiment (DUNE).

## Repository Structure

### AskDUNE Application (`askdune_app/`)

The main application includes:

- DocDB and Indico document ingestion and incremental indexing
- Unified parent-child retrieval across DocDB and Indico
- Embedding-based semantic search and cross-encoder reranking
- Parent-context reconstruction for RAG-based question answering
- Benchmarking and evaluation tools
- Application deployment and operational documentation

### BALSAM (`BALSAM/`)

HPC job submission and workflow orchestration components.

Both directories are maintained within this Git repository.

## Release

Development release branch: `release/askdune-v1.0.0`

The v1.0.0 development release introduces unified parent-child retrieval and includes a frozen retrieval benchmark baseline.

Initial functional tests have validated DocDB and Indico indexing, incremental indexing of unchanged records, and retrieval from an isolated test database.

Production indexing and deployment require separate validation. The existing production service remains on the legacy retrieval system until a validated unified index is ready for migration.

## Documentation

- [Application README](askdune_app/README.md)
- [Operations Guide](askdune_app/OPERATIONS.md)
- [Indexing Scripts](askdune_app/scripts/README.md)
