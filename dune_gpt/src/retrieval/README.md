# Unified retrieval

This package implements experimental parent-child retrieval across DocDB and Indico. The application selects it using `RETRIEVAL_BACKEND=unified` and `UNIFIED_INDEX_PATH`.

## Entry points

- `backend_config.py`: selects `legacy` or `unified` and validates the unified index directory.
- `unified_backend.py`: opens the three ChromaDB collections, loads the `intfloat/e5-small-v2` embedder and `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, and formats results.
- `unified_retriever.py`: orchestrates candidate retrieval, metadata constraints, merging, reranking, sorting, and parent reconstruction.

## Retrieval stages

1. **Query interpretation:** `interpret_query(query)` determines the default `sort_by` mode when none is explicitly provided.
2. **Parallel retrieval methods:** DocDB and Indico each contribute semantic and BM25 keyword child candidates.
3. **Optional metadata constraints:** When `person`, `title`, `date_from`, or `date_to` is explicitly supplied, semantic and keyword candidates are filtered with `matches_metadata(...)`.
4. **Direct metadata lookup:** `search_metadata_children(...)` searches both sources for additional matching records that might not appear in the initial semantic/BM25 candidate limits.
5. **Merge and rerank:** Candidate lists are merged, cross-encoder reranked, and assigned `metadata_ranking_score(...)` bonuses; combined score is reranker score plus metadata bonus.
6. **Ordering:** `sort_contexts(...)` sorts by combined relevance or chronologically (`newest`/`oldest`).
7. **Reconstruction:** Duplicate parents are suppressed; DocDB active parents and Indico parents are reconstructed. Source URLs are normalized and deduplicated.

## Metadata filtering

| Filter | DocDB fields | Indico fields |
| --- | --- | --- |
| `person` | `authors`, `author` | `speaker_name`, `conveners` |
| `title` | `title`, `document_title` | `presentation_title`, `meeting_name` |
| `date_from`, `date_to` | `created_date`, `content_last_modified_date`, `date`, `creation_date`, `created_at`, `document_date` | `start_date` |

Person matching normalizes case, accents, punctuation, and name order; title matching uses normalized substring matching. Date bounds are **inclusive** and use ISO dates or ISO timestamp date portions. Indico metadata matching excludes records not marked `index_status=complete`.

**Important distinction:** Metadata constraints are applied only when explicitly supplied to the retrieval function. Query interpretation currently controls default sorting; do not describe it as automatically extracting every metadata filter from arbitrary natural-language questions without verifying that behavior.

## Retrieval API

`UnifiedRetrievalBackend.search(question, top_k=5, k_embeddings=30, **filters)` returns formatted parent-child search results. The orchestration function additionally accepts `max_references`, `person`, `title`, `date_from`, `date_to`, and `sort_by`.

## Storage and safety

The unified backend expects collections `DUNE_DOCDB_CHILDREN`, `DUNE_DOCDB_VERSION_MANIFEST`, and `DUNE_INDICO_CHILDREN` at the configured index path. It does not create or populate them on startup. Keep this index separate from legacy and production ChromaDB storage.
