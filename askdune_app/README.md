# AskDUNE (DUNE-GPT)

AskDUNE is a retrieval-augmented generation (RAG) application for the Deep Underground Neutrino Experiment (DUNE). It retrieves scientific documentation from DUNE DocDB and Indico and uses a large language model (LLM) to answer questions with source references.

The web application uses FastAPI, supports Fermilab PingFederate single sign-on (SSO), and can use Argonne Argo or Fermilab LLM services.

## Features

- **Document extraction:** DocDB documents and Indico events and attachments.
- **Parent-child indexing:** Small searchable child chunks associated with larger document contexts.
- **Hybrid retrieval:** Semantic vector search and BM25 keyword search across both sources.
- **Metadata filtering:** Explicit person, title, and inclusive date-range constraints, with source-aware field matching.
- **Direct metadata lookup:** Additional matching records can be found beyond the initial semantic and BM25 candidate limits.
- **Reranking:** Cross-encoder relevance scoring plus metadata relevance bonuses.
- **Ordering:** Relevance, newest first, or oldest first.
- **References:** Parent-context reconstruction, deduplication, and DocDB/Indico source links.
- **Authentication:** Fermilab PingFederate SSO for authenticated web access.

## Architecture

The AskDUNE FastAPI web application uses the **unified parent-child retrieval backend** for both DocDB and Indico.

The unified backend uses ChromaDB with the following models:

| Component | Model |
| --- | --- |
| Embeddings | `intfloat/e5-small-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

### Unified retrieval pipeline

1. Interpret the query, including requested relevance or chronological ordering.
2. Retrieve DocDB and Indico child candidates using semantic search and BM25.
3. When filters are explicitly supplied, constrain candidates by person, title, and/or date range.
4. Perform direct metadata candidate lookup so matching documents are not limited to the initial semantic/BM25 result sets.
5. Merge candidates and rerank with a cross-encoder.
6. Apply metadata relevance bonuses and order by relevance, newest, or oldest.
7. Deduplicate selected parents and reconstruct the larger parent contexts.
8. Supply context and normalized source references to the answer-generation pipeline.

DocDB retrieval uses an active-version manifest. Indico retrieval reconstructs parent contexts from indexed event attachments.

### Index separation

AskDUNE uses separate database paths for serving and indexing:

| Variable | Purpose |
| --- | --- |
| `PROD_DB_PATH` | ChromaDB directory containing the validated unified index used by the FastAPI application |
| `INDEXING_DB_PATH` | Separate ChromaDB directory used to build or update an index |
| `EMBEDDING_MODEL` | Embedding model used for indexing and retrieval |
| `RERANKER_MODEL` | Cross-encoder model used for reranking |
| `K_DOCS` | Number of embedding candidates considered during retrieval |

The indexing scripts refuse to write to the configured `PROD_DB_PATH`.

**Important:** The unified index is still being populated and validated. The existing live production service and its legacy database must remain unchanged until a separate, deliberate migration is completed. Do not point development indexing jobs at production storage.

### Legacy components

The repository retains historical ChromaDB processing utilities, benchmarking scripts, and a separate Slack bot (`app.py`) that still uses legacy retrieval.

These components are not part of the validated unified FastAPI web-serving path. The Slack integration has not yet been migrated to unified retrieval.

## Quick start

Requires Python 3.11, appropriate access to DUNE documentation, and credentials for the selected LLM provider. Fermilab SSO configuration is required when authentication is enabled.

```bash
git clone https://github.com/DUNE/LLM_for_DUNE.git
cd LLM_for_DUNE/askdune_app

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cp env.example .env
```

Edit `.env` to configure your environment. The unified retrieval settings include:

```dotenv
PROD_DB_PATH=/path/to/validated/unified/index
INDEXING_DB_PATH=/path/to/separate/indexing

EMBEDDING_MODEL=intfloat/e5-small-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

K_DOCS=30
DEFAULT_TOP_K=3
```

`PROD_DB_PATH` must point to an existing, populated unified index compatible with the configured embedding model. The indexing path must be different from the serving path.

### Start a local development server

For localhost-only development with authentication disabled:

```bash
ENABLE_AUTHENTICATION=false PYTHONPATH=. \
  python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Visit `http://127.0.0.1:8001/` locally.

For a remote development host, use an SSH tunnel. Do not expose an unauthenticated development server on a public interface.

Do not replace or restart the running production service while testing this release.

### Unified indexing

The DocDB and Indico indexing scripts support separate development databases.

To inspect their options without starting indexing:

```bash
python -m scripts.run_docdb_indexing --help
python -m scripts.run_indico_indexing --help
```

By default, the scripts use `INDEXING_DB_PATH`. An explicit `--index-path` can be supplied to select another indexing destination. Both scripts reject a destination matching `PROD_DB_PATH`.

Read [INSTALL.md](INSTALL.md) and [scripts/README.md](scripts/README.md) before running indexing.

## Container deployment

The Dockerfile launches the FastAPI application directly using Uvicorn.

Docker Compose is configured to mount a validated unified index at `/data/chroma`, with `PROD_DB_PATH=/data/chroma` inside the container.

The serving index must be populated and validated before deployment. The current Compose configuration and read-only database mount have not yet been runtime-tested.

Container deployment should be validated in an isolated development environment before use in production.

## Project layout

| Location | Responsibility |
| --- | --- |
| `main.py` | FastAPI application and unified retrieval integration |
| `app.py` | Separate legacy Slack bot integration |
| `cli.py` | Command-line utilities, including legacy workflows |
| `config.py`, `env.example` | Application configuration |
| `src/api/` | Argo and Fermilab LLM clients |
| `src/auth/` | Fermilab SSO |
| `src/core/` | Document processing and core integrations |
| `src/extractors/` | Source extraction and text preparation |
| `src/indexing/` | ChromaDB storage and parent-child indexing |
| `src/retrieval/` | Unified hybrid search, metadata filtering, reranking, reconstruction |
| `src/analytics/` | Index, token, and timing utilities |
| `scripts/` | Indexing entry points and maintenance scripts |
| `templates/`, `static/` | Web interface |
| `benchmarking/` | Evaluation and benchmarking utilities |

## Documentation

- [Installation and configuration](INSTALL.md)
- [Serving and development operations](OPERATIONS.md)
- [Indexing scripts](scripts/README.md)
- [Extraction](src/extractors/README.md)
- [Indexing internals](src/indexing/README.md)
- [Retrieval internals](src/retrieval/README.md)
- [Core processing](src/core/README.md)
- [Analytics](src/analytics/README.md)
- [Parent-child benchmarking](benchmarking/parent_child_comparison/README.md)

## Security and contributions

Never commit `.env`, passwords, API keys, SSO secrets, or private document contents.

Keep serving and indexing databases separate. Test new indexing and retrieval changes against isolated development data before considering a production migration.

The unified web backend has undergone development smoke testing, but container startup, database mount compatibility, and production deployment still require validation.
