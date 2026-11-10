# Unified retrieval

This package implements the unified parent-child retrieval backend for AskDUNE's FastAPI web application. It searches indexed DocDB documents and Indico events using semantic retrieval, BM25 keyword matching, metadata constraints, cross-encoder reranking, and parent-context reconstruction.

The FastAPI application uses unified retrieval exclusively. Backend selection through `RETRIEVAL_BACKEND` and `UNIFIED_INDEX_PATH` is no longer supported.

## Entry points

- `backend_config.py`: Resolves and validates the unified serving index directory using `PROD_DB_PATH`.
- `unified_backend.py`: Opens the DocDB and Indico ChromaDB collections, initializes the configured embedding and reranking models, and formats retrieval results.
- `unified_retriever.py`: Orchestrates candidate retrieval, metadata constraints, merging, reranking, sorting, and parent reconstruction.

## Models and configuration

The default models are:

| Component | Configuration | Default |
| --- | --- | --- |
| Embeddings | `EMBEDDING_MODEL` | `intfloat/e5-small-v2` |
| Reranking | `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Serving index | `PROD_DB_PATH` | Required |
| Embedding candidates | `K_DOCS` | `30` |

The embedding model used for retrieval must be compatible with the model used to generate the stored embeddings.

`INDEXING_DB_PATH` is configured separately for the indexing scripts and must not point to the same directory as `PROD_DB_PATH`.

## Retrieval stages

1. **Query interpretation:** `interpret_query(query)` determines the default `sort_by` mode when none is explicitly provided.
2. **Parallel retrieval methods:** DocDB and Indico each contribute semantic and BM25 keyword child candidates.
3. **Optional metadata constraints:** When `person`, `title`, `date_from`, or `date_to` is explicitly supplied, semantic and keyword candidates are filtered with `matches_metadata(...)`.
4. **Direct metadata lookup:** `search_metadata_children(...)` searches both sources for additional matching records that might not appear in the initial semantic/BM25 candidate limits.
5. **Merge and rerank:** Candidate lists are merged, cross-encoder reranked, and assigned `metadata_ranking_score(...)` bonuses; the combined score is the reranker score plus metadata bonus.
6. **Ordering:** `sort_contexts(...)` sorts by combined relevance or chronologically (`newest`/`oldest`).
7. **Reconstruction:** Duplicate parents are suppressed; DocDB active parents and Indico parents are reconstructed. Source URLs are normalized and deduplicated.

## Metadata filtering

| Filter | DocDB fields | Indico fields |
| --- | --- | --- |
| `person` | `authors`, `author` | `speaker_name`, `conveners` |
| `title` | `title`, `document_title` | `presentation_title`, `meeting_name` |
| `date_from`, `date_to` | `created_date`, `content_last_modified_date`, `date`, `creation_date`, `created_at`, `document_date` | `start_date` |

Person matching normalizes case, accents, punctuation, and name order. Title matching uses normalized substring matching. Date bounds are **inclusive** and use ISO dates or the date portion of ISO timestamps.

Indico metadata matching excludes records not marked `index_status=complete`.

**Important distinction:** Metadata constraints are applied only when explicitly supplied to the retrieval function. Query interpretation currently controls default sorting; it should not be described as automatically extracting all metadata filters from arbitrary natural-language questions.

## Retrieval API

The primary interface is:

```python
UnifiedRetrievalBackend.search(
    question,
    top_k=5,
    k_embeddings=30,
    **filters
)
```

This returns formatted parent-child retrieval results.

Supported filtering and ordering arguments include:

- `person`
- `title`
- `date_from`
- `date_to`
- `sort_by`

The underlying retrieval orchestration also supports `max_references`.

The FastAPI application uses this backend to retrieve source contexts and references before generating answers through the configured LLM provider.

## ChromaDB collections

The unified backend expects the following collections in the configured serving index:

| Collection | Purpose |
| --- | --- |
| `DUNE_DOCDB_CHILDREN` | DocDB child embeddings and metadata |
| `DUNE_DOCDB_VERSION_MANIFEST` | Active DocDB document versions |
| `DUNE_INDICO_CHILDREN` | Indico child embeddings and metadata |

The serving backend does not create or populate these collections on startup. They must already exist in the selected database.

## Storage and deployment safety

`PROD_DB_PATH` points to the unified index used for serving queries.

`INDEXING_DB_PATH` points to a separate development or indexing database. The unified indexing scripts refuse to write to the configured serving index.

The existing live AskDUNE production service still uses its legacy database. The new unified serving index must remain separate until it has been sufficiently populated, tested, and deliberately approved for migration.

Do not replace the live production index or restart the production service as part of development testing.

The unified backend has passed limited development retrieval smoke tests. Full-scale indexing, container deployment, and production migration remain to be validated.
