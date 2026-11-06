# Document extraction

This package prepares source text and metadata for indexing. AskDUNE currently handles DUNE DocDB documents and Indico events/attachments.

## DocDB

`docdb_extractor_multithreaded.py` provides the `DocDBExtractor` used by the experimental DocDB parent-child indexing script. Its `iter_prepared_docdb_documents(...)` interface iterates prepared records with starting position, limit, active-version information, and chunk-size settings. The extractor supports attachment preparation and document metadata needed for later filtering and references.

## Indico

The experimental Indico indexing script uses `IndicoParentChildExtractor` to verify source access, retrieve event details, and prepare attachments for parent-child indexing. Event and attachment metadata are retained for source-aware retrieval.

## Chunking and processing

The parent-child design separates smaller searchable child chunks from larger parent contexts used to answer questions. Extraction and token preparation are distinct from the ChromaDB indexing and retrieval stages. See [../indexing/README.md](../indexing/README.md) and [../retrieval/README.md](../retrieval/README.md).

## Safety

Extractors may access restricted collaboration content. Keep credentials out of source control and logs; use authorized source access. Avoid changing extraction output fields without checking indexing and metadata-filtering consumers. Legacy extraction/processing integrations remain in the repository for compatibility.
