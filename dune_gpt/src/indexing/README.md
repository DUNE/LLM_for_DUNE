# ChromaDB indexing

AskDUNE maintains a legacy ChromaDB backend and a separate experimental unified parent-child index. Do not mix their collections or overwrite existing production data.

## Unified collections

| Collection | Purpose |
| --- | --- |
| `DUNE_DOCDB_CHILDREN` | Searchable DocDB child records |
| `DUNE_DOCDB_VERSION_MANIFEST` | Active DocDB document-version tracking |
| `DUNE_INDICO_CHILDREN` | Searchable Indico child records |

Parent-child indexing stores child text and metadata that allow the retrieval layer to reconstruct a larger parent context. The unified embedding model is `intfloat/e5-small-v2`; scripts enforce this setting and use a 480-token child target.

## DocDB versioning

The DocDB indexing path consults the active-version manifest and indexes documents through `index_docdb_document(...)`. Retrieval uses `reconstruct_active_parent(...)` to avoid selecting obsolete DocDB versions.

## Indico indexing

The Indico path builds parent contexts from event attachments and calls `index_indico_event(...)`. It uses `index_status=complete` to identify already-completed event records; this differs from the DocDB version manifest.

## Operational boundaries

The experimental indexing scripts target a sibling directory named `parent_child_experiment` next to `INDEXING_DB_PATH`. They must not write into `PROD_DB_PATH` or the stable indexing directory. See [../../scripts/README.md](../../scripts/README.md) for CLI options and [../retrieval/README.md](../retrieval/README.md) for how records are searched.
