# AskDUNE

AskDUNE is a retrieval-augmented generation (RAG) application for
documentation associated with the Deep Underground Neutrino Experiment (DUNE).

## Repository Structure

### AskDUNE Application (`askdune_app/`)

The main application includes:

- DocDB and Indico document ingestion and indexing
- Hybrid retrieval, reranking, and context reconstruction
- RAG-based question answering
- Benchmarking and evaluation
- Application deployment and operational documentation

### BALSAM (`BALSAM/`)

HPC job submission and workflow orchestration components.

Both directories are maintained within this Git repository.

## Release

Development release branch: `release/askdune-v1.0.0`

The v1.0.0 development release includes a frozen retrieval benchmark
baseline. Production indexing and deployment require separate validation.

## Documentation

- [Application README](askdune_app/README.md)
- [Operations Guide](askdune_app/OPERATIONS.md)
- [Indexing Scripts](askdune_app/scripts/README.md)
