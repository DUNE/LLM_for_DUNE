# ChromaDB indexing

AskDUNE uses unified parent-child indexing for DocDB documents and Indico events. The FastAPI web application retrieves information from the unified index.

The repository also retains legacy ChromaDB processing components for historical workflows and benchmarking. These must remain separate from the unified index.

## Unified collections

| Collection | Purpose |
| --- | --- |
| `DUNE_DOCDB_CHILDREN` | Searchable DocDB child records |
| `DUNE_DOCDB_VERSION_MANIFEST` | Active DocDB document-version tracking |
| `DUNE_INDICO_CHILDREN` | Searchable Indico child records |

Parent-child indexing stores child text and metadata that allow the retrieval layer to reconstruct larger parent contexts.

The unified embedding model is `intfloat/e5-small-v2`. The current indexing workflow uses a 480-token child target.

## DocDB versioning

The DocDB indexing workflow consults the active-version manifest and indexes documents through `index_docdb_document(...)`.

Retrieval uses `reconstruct_active_parent(...)` to select the active DocDB document version and avoid returning obsolete versions.

The DocDB indexing script can skip unchanged documents based on existing version information.

## Indico indexing

The Indico indexing workflow builds parent contexts from event attachments and calls `index_indico_event(...)`.

It uses `index_status=complete` to identify previously completed event records. This differs from the DocDB version-manifest mechanism.

## Index configuration

The indexing scripts use the following configuration:

| Variable | Purpose |
| --- | --- |
| `INDEXING_DB_PATH` | Default directory where unified indexing writes data |
| `PROD_DB_PATH` | Unified serving index; indexing into this directory is prohibited |
| `EMBEDDING_MODEL` | Model used to generate child embeddings |

An explicit `--index-path` argument can override `INDEXING_DB_PATH`.

The scripts can create the indexing directory and initialize the required ChromaDB collections.

Both indexing scripts reject a destination whose resolved path matches `PROD_DB_PATH`.

This protection does not replace manually verifying the destination before running an indexing operation.

## Indexing entry points

Run the scripts as Python modules from the `askdune_app` directory:

```bash
python -m scripts.run_docdb_indexing --help
python -m scripts.run_indico_indexing --help
```

These commands display options without modifying the index.

Actual indexing commands write to the configured destination and should only be run after verifying the target directory.

See [Indexing scripts](../../scripts/README.md) for CLI options and examples, and [Unified retrieval](../retrieval/README.md) for how indexed records are searched.

## Production deployment and operational boundaries

AskDUNE v1.0.0 uses unified parent-child retrieval for DocDB and Indico documentation.

The production FastAPI service reads its unified ChromaDB index from `PROD_DB_PATH`.
As of November 12, 2026, the validated production index contains 22,642 DocDB children and 5,016 Indico children (27,658 total embeddings).

These counts describe the current production snapshot, not all available documents.

### Production and staging directories

- `PROD_DB_PATH`: `/exp/dune/data/users/dune/askdune/data_prod`
- `INDEXING_DB_PATH`: `/exp/dune/data/users/dune/askdune/data_indexing`

The production index serves live queries and must not be modified by indexing jobs.
The staging index is used to build and validate replacements independently.

### Full indexing rebuild

A fresh staging rebuild is planned to incorporate updated XLSX extraction and expand DocDB and Indico coverage.

Use `python -m scripts.run_docdb_indexing --help` and `python -m scripts.run_indico_indexing --help` to inspect available options.

Both scripts support `--start`, `--limit`, `--batch-size`, and `--index-path`.
The Indico script additionally supports `--event-id`.

DocDB `--start` is the first document ID; `--limit` is the number of IDs to examine.
Indico `--start` is an event-selection offset; `--limit` selects eligible events subject to the discovery behavior.

Do not run independent indexing processes concurrently against the same ChromaDB directory.

### Deployment safeguards

1. Keep the production service running against `data_prod` during staging indexing.
2. Never index directly into `PROD_DB_PATH`.
3. Validate staging collections, extraction quality, retrieval, and integrity.
4. Back up the production database before promoting the new index.
5. Deploy the validated replacement through a controlled service restart.
6. Verify `/api/health` and representative queries after deployment.

Legacy indexing components remain available for historical workflows and benchmarking.

