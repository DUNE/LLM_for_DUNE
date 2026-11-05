"""Token accounting for parent-child indexing."""

from src.extractors.docdb_token_chunker import get_tokenizer


def count_parent_tokens(parents):
    """Count tokens in parent text, without embedding prefixes."""
    tokenizer = get_tokenizer()
    return sum(
        len(tokenizer.encode(
            parent["cleaned_text"],
            add_special_tokens=False,
            truncation=False,
        ))
        for parent in parents
    )


def count_child_embedding_tokens(children):
    """Count child tokens including the E5 passage prefix and special tokens."""
    tokenizer = get_tokenizer()
    return sum(
        len(tokenizer.encode(
            "passage: " + child["cleaned_text"],
            add_special_tokens=True,
            truncation=False,
        ))
        for child in children
    )
