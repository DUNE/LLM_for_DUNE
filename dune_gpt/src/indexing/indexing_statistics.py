"""Formatting utilities for AskDUNE indexing statistics."""


def format_indexing_report(results):
    """Format current-run and cumulative ChromaDB indexing statistics."""

    def format_size(num_bytes):
        sign = "-" if num_bytes < 0 else ""
        value = abs(float(num_bytes))
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{sign}{value:.2f} {unit}"
            value /= 1024

    return (
        "\n"
        "========== AskDUNE DocDB Indexing Summary ==========\n"
        "CURRENT RUN\n"
        f"  Pages checked:          {results['docdb_pages_checked']:,}\n"
        f"  New pages indexed:      {results['docdb_pages_new']:,}\n"
        f"  Existing pages updated: {results['docdb_pages_updated']:,}\n"
        f"  Unchanged pages:        {results['docdb_pages_unchanged']:,}\n"
        f"  Missing pages:          {results['docdb_pages_missing']:,}\n"
        f"  Failed/skipped pages:   {results['docdb_pages_failed']:,}\n"
        f"  Attachments indexed:    {results['docdb_attachments_indexed']:,}\n"
        f"  Chunks written:         {results['docdb_chunks_written']:,}\n"
        f"  Net database growth:    {format_size(results['database_growth_bytes'])}\n"
        f"  Elapsed time:           {results['elapsed_seconds']:.1f} seconds\n"
        "\n"
        "CHROMADB TOTALS AFTER RUN\n"
        f"  Total DocDB pages:      {results['total_docdb_pages']:,}\n"
        f"  Total DocDB attachments:{results['total_docdb_attachments']:>8,}\n"
        f"  Total DocDB chunks:     {results['total_docdb_chunks']:,}\n"
        f"  All ChromaDB chunks:    {results['total_chromadb_chunks']:,}\n"
        f"  Total database size:    {format_size(results['database_size_bytes'])}\n"
        "=====================================================\n"
    )
