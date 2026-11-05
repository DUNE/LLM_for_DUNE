"""Configuration for selecting the AskDUNE retrieval backend."""

import os
from pathlib import Path


DEFAULT_UNIFIED_INDEX_PATH = (
    "/exp/dune/app/users/dune/askdune/dev/"
    "docdb-integration/test_data/parent_child_experiment"
)


def get_retrieval_config() -> tuple[str, str]:
    """Return the selected retrieval backend and unified index path."""

    backend = os.getenv("RETRIEVAL_BACKEND", "legacy").strip().lower()

    if backend not in {"legacy", "unified"}:
        raise ValueError(
            f"Invalid RETRIEVAL_BACKEND={backend!r}. "
            "Expected 'legacy' or 'unified'."
        )

    index_path = os.getenv(
        "UNIFIED_INDEX_PATH",
        DEFAULT_UNIFIED_INDEX_PATH,
    )

    if backend == "unified" and not Path(index_path).is_dir():
        raise FileNotFoundError(
            f"Unified retrieval index does not exist: {index_path}"
        )

    return backend, index_path
