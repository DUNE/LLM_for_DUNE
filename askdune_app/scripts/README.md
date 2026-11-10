# AskDUNE unified indexing scripts

These scripts build and update the unified parent-child ChromaDB index used by the AskDUNE FastAPI web application.

DocDB and Indico indexing are performed independently, but their child collections can coexist in the same unified ChromaDB directory.

**Important:** The scripts write to an indexing database, not the running production database. Do not run indexing against the existing live production index or its legacy ChromaDB storage.

## Configuration

The indexing destination is controlled by `INDEXING_DB_PATH` in `.env` or by the explicit `--index-path` command-line argument.

```dotenv
PROD_DB_PATH=/path/to/validated/unified/index
INDEXING_DB_PATH=/path/to/separate/indexing

EMBEDDING_MODEL=intfloat/e5-small-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

| Variable | Purpose |
| --- | --- |
| `PROD_DB_PATH` | Serving index; indexing into this path is prohibited |
| `INDEXING_DB_PATH` | Default destination for indexing |
| `EMBEDDING_MODEL` | Embedding model used to generate child vectors |
| `RERANKER_MODEL` | Retrieval reranker; not required for generating embeddings |

The scripts refuse to index into a destination whose resolved path matches `PROD_DB_PATH`. They can create the indexing directory and initialize the required ChromaDB collections.

The path comparison is a safeguard, not a substitute for checking the target directory. Always verify the destination before starting a write operation.

## DocDB indexing

Inspect the available options:

```bash
python -m scripts.run_docdb_indexing --help
```

The DocDB workflow retrieves documents and attachments, prepares parent contexts and child chunks, generates embeddings, and maintains active-version information.

The relevant ChromaDB collections are:

- `DUNE_DOCDB_CHILDREN`
- `DUNE_DOCDB_VERSION_MANIFEST`

The version manifest supports detection of unchanged documents and management of updated document versions.

### Example

The following command writes to the configured indexing database:

```bash
python -m scripts.run_docdb_indexing \
  --start 1 \
  --limit 5 \
  --batch-size 32
```

To select an explicit destination:

```bash
python -m scripts.run_docdb_indexing \
  --index-path /path/to/separate/test_index \
  --start 1 \
  --limit 5 \
  --batch-size 32
```

## Indico indexing

Inspect the available options:

```bash
python -m scripts.run_indico_indexing --help
```

The Indico workflow retrieves selected events and their attachments, prepares parent contexts, generates child embeddings, and writes to:

- `DUNE_INDICO_CHILDREN`

The script can skip events whose existing child records are marked `index_status=complete`. This differs from the DocDB active-version manifest.

### Example: Process a batch of events

```bash
python -m scripts.run_indico_indexing \
  --start 0 \
  --limit 5 \
  --batch-size 32
```

### Example: Process a specific event

Replace `EVENT_ID` with a real Indico event identifier:

```bash
python -m scripts.run_indico_indexing \
  --event-id EVENT_ID
```

## Command-line options

| Script | Option | Default | Description |
| --- | --- | --- | --- |
| DocDB | `--start` | `1` | Starting DocDB document ID |
| DocDB | `--limit` | `5` | Maximum documents to attempt |
| DocDB | `--batch-size` | `32` | Embedding batch size |
| DocDB | `--index-path` | `INDEXING_DB_PATH` | Explicit indexing destination |
| Indico | `--start` | `0` | Starting event offset |
| Indico | `--limit` | `5` | Maximum events to attempt |
| Indico | `--batch-size` | `32` | Embedding batch size |
| Indico | `--event-id` | Not set | Process a specific event |
| Indico | `--index-path` | `INDEXING_DB_PATH` | Explicit indexing destination |

These options reflect the current development workflow. The installed script's `--help` output is authoritative.

## Validation and safety

Before indexing, verify that the intended destination is not the live production database and that the appropriate DocDB or Indico credentials are configured.

After indexing, inspect the logs and collection counts. A successful process exit does not necessarily mean every requested document or attachment was extracted successfully.

Development smoke tests have confirmed that both scripts can populate a separate unified index and skip previously indexed, unchanged records without duplicating their child embeddings. Those tests do not constitute full production validation.

The unified index should be populated and independently validated before it is used as the serving database.

## Development workflow

Use the following sequence when developing or updating the unified index:

1. Configure an isolated `INDEXING_DB_PATH`.
2. Inspect the indexing options with `--help`.
3. Run a small DocDB or Indico indexing batch.
4. Check the logs, collection counts, and representative retrieval results.
5. Repeat indexing to check for unintended duplicates.
6. Expand the indexing scope only after the initial results are validated.

The current production AskDUNE service and its database must remain unchanged until an explicitly planned migration.
