"""Configuration for AskDUNE unified parent-child retrieval."""

from pathlib import Path
from config import PROD_DB_PATH


def get_retrieval_config() -> str:
    """Return the production ChromaDB index path."""

    path = Path(PROD_DB_PATH).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"Production retrieval index does not exist: {path}"
        )

    return str(path)
