# AskDUNE installation and configuration

This guide describes the current Python development setup. Production deployment is separately managed; do not use these instructions to restart or replace the running AskDUNE service.

## Prerequisites

- Python 3.11 and access to the repository.
- Network access and credentials for DUNE DocDB/Indico when indexing those sources.
- Credentials for either Argonne Argo or the Fermilab LLM API.
- Fermilab SSO application settings when authentication is enabled.
- Sufficient writable disk space for model caches and **separate** ChromaDB indexes.

## Install

```bash
git clone https://github.com/DUNE/LLM_for_DUNE.git
cd LLM_for_DUNE/askdune_app
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp env.example .env
```

If `uv` is available, `uv pip install --python .venv/bin/python -r requirements.txt` is an alternative dependency installer. On quota-limited systems, configure `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, and `XDG_CACHE_HOME` on a filesystem with adequate space.

## Environment variables

Edit `.env`; never commit it. The example file provides placeholders, **not** valid credentials or ready-to-use storage paths.

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `argo` or `fermilab` |
| `LLM_MODEL` | Model selection for the configured LLM provider |
| `ARGO_API_USERNAME`, `ARGO_API_KEY`, `ARGO_API_URL` | Argo configuration when selected |
| `DUNE_DOCDB_USERNAME`, `DUNE_DOCDB_PASSWORD`, `DUNE_DOCDB_API_KEY` | DocDB access as required by extraction |
| `DUNE_INDICO_ACCESS_KEY`, `INDICO_API_TOKEN` | Indico access as required by extraction |
| `ENABLE_AUTHENTICATION` | Fermilab SSO enabled when `true` |
| `FERMILAB_CLIENT_ID`, `FERMILAB_CLIENT_SECRET`, `FERMILAB_SESSION_SECRET`, `FERMILAB_REDIRECT_URI` | SSO application/session settings |
| `RETRIEVAL_BACKEND` | `legacy` (default) or `unified` |
| `VECTOR_STORE` | `chroma` for the legacy backend |
| `PROD_DB_PATH` | Existing application ChromaDB directory; required by current `config.py` |
| `INDEXING_DB_PATH` | Separate indexing directory; cannot resolve to the production directory |
| `UNIFIED_INDEX_PATH` | Existing unified parent-child ChromaDB directory |
| `EMBEDDING_MODEL` | Indexing model; must match the existing index |
| `HOST`, `PORT`, `DEBUG`, `LOG_LEVEL` | Web server and logging settings |
| `DEFAULT_TOP_K`, `K_DOCS` | Retrieval/application result limits |

The current unified retrieval implementation loads `intfloat/e5-small-v2` explicitly. The parent-child indexing scripts require `EMBEDDING_MODEL=intfloat/e5-small-v2`. **Do not change the legacy model or rebuild a legacy index merely to match this setting**; use the embedding model compatible with each index.

## Retrieval backend selection

**Legacy:** `RETRIEVAL_BACKEND=legacy` uses the existing ChromaDB manager and `PROD_DB_PATH`.

**Unified:** `RETRIEVAL_BACKEND=unified` loads the experimental ChromaDB collections from `UNIFIED_INDEX_PATH`. This directory must exist. The current `config.py` still requires `PROD_DB_PATH` on startup, even in unified mode.

The unified backend expects DocDB children, DocDB version manifest, and Indico children collections. Do not point it at a legacy-only database.

## Local development

Use a private development port and loopback binding:

```bash
ENABLE_AUTHENTICATION=false RETRIEVAL_BACKEND=unified PYTHONPATH=. \
  python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

From a local workstation, a remote host can be accessed through an SSH tunnel:

```bash
ssh -N -L 8001:127.0.0.1:8001 YOUR_SSH_HOST
```

Then open `http://127.0.0.1:8001/`. Do not expose an authentication-disabled instance beyond localhost. Do not use development port 8001 to alter the production service on port 8000.

## Experimental indexing

Read [scripts/README.md](scripts/README.md) before indexing. The parent-child scripts derive their experimental storage location from `INDEXING_DB_PATH`; they are not general production indexers. Check the actual target path and credentials before running them. Running `--help` is safe and does not index documents.

## Validation and troubleshooting

- **Missing `PROD_DB_PATH`:** Configure it even for unified retrieval, as current `config.py` requires it.
- **Missing unified index:** Check `UNIFIED_INDEX_PATH` and whether its expected collections exist.
- **Embedding mismatch:** Use the model that created the index. Unified parent-child indexing uses `intfloat/e5-small-v2`.
- **Authentication errors:** Check the client ID, client secret, session secret, callback URI, and approved SSO configuration. Do not print secrets in logs.
- **Model cache or disk quota errors:** Relocate caches to a high-quota filesystem.
- **Empty search results:** Verify that the correct backend and index were selected and that data was indexed. Metadata constraints can legitimately narrow results.

Use `git diff --check` for documentation changes and test imports/application startup in the **development** environment before deploying code changes. Existing production indexes and services must remain untouched.
