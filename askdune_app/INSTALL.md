# AskDUNE installation and configuration

This guide describes installing and configuring the AskDUNE FastAPI web application with unified parent-child retrieval across DUNE DocDB and Indico.

These instructions are intended for development and validation. The existing production AskDUNE service and its legacy ChromaDB database must remain unchanged until a separately planned migration.

## Prerequisites

- Python 3.11 and access to the AskDUNE repository.
- Network access and credentials for DUNE DocDB and Indico when indexing those sources.
- Credentials for either Argonne Argo or the Fermilab LLM API.
- Fermilab SSO application settings when authentication is enabled.
- Sufficient disk space for model caches and separate ChromaDB indexing and serving databases.
- A populated unified ChromaDB index for running retrieval.

## Installation

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/DUNE/LLM_for_DUNE.git
cd LLM_for_DUNE/askdune_app

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
cp env.example .env
```

If `uv` is available, the following is an alternative dependency installation command:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

On quota-limited systems, configure `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, and `XDG_CACHE_HOME` to use a filesystem with sufficient space.

## Environment configuration

Edit `.env` for your environment. Never commit this file to Git.

The provided `env.example` contains placeholders, not valid credentials or ready-to-use database paths.

### LLM and authentication

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `argo` or `fermilab` |
| `LLM_MODEL` | Model selected for the configured provider |
| `ARGO_API_USERNAME`, `ARGO_API_KEY`, `ARGO_API_URL` | Argonne Argo configuration |
| `DUNE_DOCDB_USERNAME`, `DUNE_DOCDB_PASSWORD`, `DUNE_DOCDB_API_KEY` | DocDB access as required by extraction |
| `DUNE_INDICO_ACCESS_KEY`, `INDICO_API_TOKEN` | Indico access as required by extraction |
| `ENABLE_AUTHENTICATION` | Enables Fermilab SSO when `true` |
| `FERMILAB_CLIENT_ID`, `FERMILAB_CLIENT_SECRET` | Fermilab SSO client configuration |
| `FERMILAB_SESSION_SECRET`, `FERMILAB_REDIRECT_URI` | Session and callback configuration |

The separate Indico authentication utilities may use `FERMI_SSO_USERNAME` and `FERMI_SSO_PASSWORD` for automated cookie acquisition. These are not substitutes for the web application's OIDC client settings.

### Unified retrieval configuration

The FastAPI application uses the unified parent-child retrieval backend exclusively.

| Variable | Purpose |
| --- | --- |
| `PROD_DB_PATH` | Existing unified ChromaDB index used for serving queries |
| `INDEXING_DB_PATH` | Separate ChromaDB directory used for indexing |
| `EMBEDDING_MODEL` | Embedding model used for indexing and retrieval |
| `RERANKER_MODEL` | Cross-encoder reranking model |
| `K_DOCS` | Number of embedding candidates considered before reranking |
| `DEFAULT_TOP_K` | Default number of retrieved contexts |
| `HOST`, `PORT`, `DEBUG`, `LOG_LEVEL` | Web server and logging configuration |

Example:

```dotenv
PROD_DB_PATH=/path/to/validated/unified/index
INDEXING_DB_PATH=/path/to/separate/indexing

EMBEDDING_MODEL=intfloat/e5-small-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

K_DOCS=30
DEFAULT_TOP_K=3
```

The serving index must contain embeddings compatible with the configured embedding model.

`PROD_DB_PATH` must reference an existing unified ChromaDB directory. The retrieval backend does not create or populate missing collections at application startup.

`INDEXING_DB_PATH` must be a different directory. The indexing scripts reject a destination matching the configured serving path.

**Do not point `PROD_DB_PATH` to the existing legacy production index.** The new unified index must be populated and validated separately.

## Local development

Run the FastAPI application on a private development port with loopback binding:

```bash
ENABLE_AUTHENTICATION=false PYTHONPATH=. \
  python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/` from the development host.

For remote development, establish an SSH tunnel from your workstation:

```bash
ssh -N -L 8001:127.0.0.1:8001 YOUR_SSH_HOST
```

Then open `http://127.0.0.1:8001/` on your workstation.

Do not expose an authentication-disabled development instance on a public interface. Do not replace or restart the existing production service on port 8000.

## Unified indexing

DocDB and Indico indexing use separate scripts.

From the `askdune_app` directory, inspect their options:

```bash
python -m scripts.run_docdb_indexing --help
python -m scripts.run_indico_indexing --help
```

By default, the scripts use `INDEXING_DB_PATH`. An explicit `--index-path` overrides that destination.

Both scripts can initialize the required ChromaDB collections in a separate indexing directory.

Read [scripts/README.md](scripts/README.md) for supported options, examples, and indexing safeguards.

Indexing modifies the target database. Do not run indexing merely to verify the installation or documentation.

## Container deployment

The Dockerfile starts the FastAPI application directly using Uvicorn:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Docker Compose configures `PROD_DB_PATH=/data/chroma` inside the container and mounts a host-provided unified index at that location.

The host index directory is selected through `HOST_DB_PATH`. It must point to a validated unified index, not the existing legacy production database.

The container configuration uses a read-only serving index mount. ChromaDB compatibility with this mount has not yet been runtime-tested.

Docker and Podman are not available on the current development VM, so container builds and startup validation must be performed in a suitable isolated environment before deployment.

## Validation and troubleshooting

- **Missing `PROD_DB_PATH`:** Configure a valid unified serving index directory.
- **Missing collections:** Verify that `DUNE_DOCDB_CHILDREN`, `DUNE_DOCDB_VERSION_MANIFEST`, and `DUNE_INDICO_CHILDREN` exist in the selected index.
- **Embedding mismatch:** Use the embedding model that generated the stored vectors. The current unified indexing workflow uses `intfloat/e5-small-v2`.
- **Incorrect indexing destination:** Verify `INDEXING_DB_PATH` or the explicit `--index-path`. The target must differ from `PROD_DB_PATH`.
- **Authentication errors:** Check the SSO client ID, client secret, session secret, callback URI, and approved configuration.
- **Model cache or disk quota errors:** Relocate model caches to a filesystem with sufficient available space.
- **Empty search results:** Confirm that the selected unified index contains indexed documents and that metadata filters are not excluding expected results.
- **LLM provider errors:** Verify the selected provider, model, API credentials, and network access.

For syntax and formatting validation:

```bash
python -m py_compile config.py main.py
git diff --check
```

These commands do not start the application or modify the database.

Application startup, end-to-end retrieval, and container runtime behavior require separate development testing.

## Production migration

The unified retrieval code and the live production database are currently separate.

Before migrating production, complete full-scale indexing, verify retrieval quality and source references, confirm application authentication, and test the deployment configuration.

Production migration must be a deliberate operation with an appropriate rollback plan. Do not overwrite the legacy production index or restart the live service as part of release development.
