# AskDUNE (DUNE-GPT)

AskDUNE is a retrieval-augmented generation (RAG) application for the Deep Underground Neutrino Experiment (DUNE). It retrieves scientific documentation from DUNE DocDB and Indico and uses a large language model (LLM) to answer questions with source references.

The application uses FastAPI, supports Fermilab PingFederate single sign-on (SSO), and can use Argonne Argo or Fermilab LLM services.

## Features

- **Document extraction:** DocDB documents and Indico events and attachments.
- **Parent-child indexing:** Small searchable child chunks associated with larger document contexts.
- **Hybrid retrieval:** Semantic vector search and BM25 keyword search across both sources.
- **Metadata filtering:** Explicit person, title, and inclusive date-range constraints, with source-aware field matching.
- **Direct metadata lookup:** Additional matching records can be found beyond the initial semantic and BM25 candidate limits.
- **Reranking:** Cross-encoder relevance scoring plus metadata relevance bonuses.
- **Ordering:** Relevance, newest first, or oldest first.
- **References:** Parent-context reconstruction, deduplication, and DocDB/Indico source links.
- **Compatibility:** Existing ChromaDB retrieval remains available while the unified backend is validated.

## Architecture

| Backend | Setting | Description |
| --- | --- | --- |
| Legacy | `RETRIEVAL_BACKEND=legacy` | Existing ChromaDB retrieval; default for backward compatibility |
| Unified | `RETRIEVAL_BACKEND=unified` | Experimental DocDB + Indico parent-child retrieval |

### Unified retrieval pipeline

1. Interpret the query, including requested relevance or chronological ordering.
2. Retrieve DocDB and Indico child candidates using semantic search and BM25.
3. When filters are explicitly supplied, constrain candidates by person, title, and/or date range.
4. Perform direct metadata candidate lookup so matching documents are not limited to the initial semantic/BM25 result sets.
5. Merge candidates and rerank with a cross-encoder.
6. Apply metadata relevance bonuses and order by relevance, newest, or oldest.
7. Deduplicate selected parents and reconstruct the larger parent contexts.
8. Supply context and normalized source references to the answer-generation pipeline.

The unified backend uses ChromaDB, the `intfloat/e5-small-v2` embedding model, and the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker. DocDB retrieval uses an active-version manifest; Indico retrieval reconstructs parent contexts from indexed event attachments.

**Important:** The unified index is currently experimental and separate from the legacy and production ChromaDB indexes. Never point experimental indexing jobs at production storage.

## Quick start

Requires Python 3.11, the relevant DUNE document access, and credentials for the selected LLM provider. Fermilab SSO configuration is required when authentication is enabled.

```bash
git clone https://github.com/DUNE/LLM_for_DUNE.git
cd LLM_for_DUNE/askdune_app
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp env.example .env
```

Edit `.env` for your environment. For unified retrieval, set:

```dotenv
RETRIEVAL_BACKEND=unified
EMBEDDING_MODEL=intfloat/e5-small-v2
UNIFIED_INDEX_PATH=/path/to/parent_child_experiment
PROD_DB_PATH=/path/to/existing/chromadb
INDEXING_DB_PATH=/path/to/separate/indexing
```

`PROD_DB_PATH` is currently required by `config.py` even when the unified backend is selected. The `UNIFIED_INDEX_PATH` must point to an existing directory. Do not use the same directory for production and indexing.

To start a **localhost-only development** server (not production):

```bash
ENABLE_AUTHENTICATION=false RETRIEVAL_BACKEND=unified PYTHONPATH=. \
  python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Visit `http://127.0.0.1:8001/` locally. For a remote development host, use an SSH tunnel; do not expose an unauthenticated server on a public interface. Do not replace or restart the production service.

The experimental indexing scripts offer command-line help without writing data:

```bash
python scripts/run_docdb_indexing.py --help
python scripts/run_indico_indexing.py --help
```

Read [INSTALL.md](INSTALL.md) and [scripts/README.md](scripts/README.md) before running indexing.

## Project layout

| Location | Responsibility |
| --- | --- |
| `main.py` | FastAPI application and backend selection |
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
- [Indexing scripts](scripts/README.md)
- [Extraction](src/extractors/README.md)
- [Indexing internals](src/indexing/README.md)
- [Retrieval internals](src/retrieval/README.md)
- [Core processing](src/core/README.md)
- [Analytics](src/analytics/README.md)

## Security and contributions

Never commit `.env`, passwords, API keys, SSO secrets, or private document contents. Preserve existing indexes when developing new retrieval and indexing features. Test changes against isolated development data and document any new configuration variables or retrieval behavior. Production deployment, Docker support, and HTTP endpoint details should be verified against the target environment before being advertised as supported.

## Operational and benchmark command guides

- [Indexing commands, options, and examples](scripts/README.md)
- [Serving and development operations](OPERATIONS.md)
- [Parent-child benchmark commands, options, and examples](benchmarking/parent_child_comparison/README.md)
