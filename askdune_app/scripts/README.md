# AskDUNE indexing scripts

The current parent-child scripts populate an **isolated experimental** ChromaDB index. They are not production indexing commands. Do not run them against existing production or legacy indexes.

## DocDB

```bash
python scripts/run_docdb_indexing.py --help
```

Supported arguments: `--start` (default 1), `--limit` (default 5), and `--batch-size` (default 32). The script requires `EMBEDDING_MODEL=intfloat/e5-small-v2`. It reads existing active versions from the DocDB manifest, prepares documents and attachments, and indexes parent-child records with version tracking. The child target is 480 tokens; the extractor preparation uses a 2,000-character chunk setting.

## Indico

```bash
python scripts/run_indico_indexing.py --help
```

Supported arguments: `--event-id` (optional), `--limit` (default 5), `--start` (default 0), and `--batch-size` (default 32). The script checks Indico access, loads selected events and attachments, builds parent contexts, and indexes children. It skips events whose existing children are all marked `index_status=complete`. This is not the same version-manifest mechanism used for DocDB.

## Storage safeguards

Both scripts derive the experimental directory as the sibling `parent_child_experiment` of `INDEXING_DB_PATH` and require that experimental directory to exist and differ from the stable indexing directory. The expected ChromaDB collections include `DUNE_DOCDB_CHILDREN`, `DUNE_DOCDB_VERSION_MANIFEST`, and `DUNE_INDICO_CHILDREN`.

**Before indexing:** inspect the current `.env`, confirm the target directory, ensure source access is authorized, and preserve backups of important indexes. Do not run indexing as a documentation test. CLI `--help` is a read-only way to inspect options.

## Copy-and-run indexing examples

Run these from the repository root after activating `.venv` and verifying the **experimental write destination**. These commands **modify an index**; do not run them merely to test the documentation.

```bash
# Inspect the actual options supported by the installed scripts (read-only).
python scripts/run_docdb_indexing.py --help
python scripts/run_indico_indexing.py --help

# DocDB: process up to five documents starting at document ID 1.
PYTHONPATH=. python -u scripts/run_docdb_indexing.py \
  --start 1 --limit 5 --batch-size 32

# Indico: process up to five events starting at offset 0.
PYTHONPATH=. python -u scripts/run_indico_indexing.py \
  --start 0 --limit 5 --batch-size 32

# Indico: target one known event (replace the placeholder with a real ID).
PYTHONPATH=. python -u scripts/run_indico_indexing.py --event-id EVENT_ID
```

| Script | Option | Default | Meaning |
|---|---|---|---|
| DocDB | `--start` | `1` | Starting DocDB identifier |
| DocDB | `--limit` | `5` | Maximum documents to attempt in this batch |
| DocDB | `--batch-size` | `32` | Embedding/indexing batch size |
| Indico | `--start` | `0` | Starting event offset |
| Indico | `--limit` | `5` | Maximum events to attempt |
| Indico | `--batch-size` | `32` | Embedding/indexing batch size |
| Indico | `--event-id` | not set | Target a specific event |

These are the options previously reviewed for the experimental scripts; `--help` on the deployed version is authoritative. A successful command does not guarantee that every source was extracted or indexed; inspect its logs and collection counts.
