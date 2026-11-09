"""Token-aware DocDB chunking for the configured embedding model."""

from functools import lru_cache

from transformers import AutoTokenizer

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def chunk_docdb_text(text: str, target_tokens: int = 480) -> list[str]:
    """Split text into token-limited chunks without losing words or adding overlap."""
    if not text or not text.strip():
        return []

    tokenizer = get_tokenizer()
    prefix = "passage: "
    max_length = min(target_tokens, tokenizer.model_max_length)

    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=True))
    if max_length <= prefix_tokens:
        raise ValueError("Token budget is too small for the passage prefix")

    words = text.split()
    original_words = words.copy()

    # Split long whitespace-free strings into token-safe pieces.
    # Handles extracted tables, HTML fragments, and long URLs.
    safe_words = []

    for word in words:
        if len(tokenizer.encode(
            prefix + word,
            add_special_tokens=True,
            truncation=False,
        )) <= max_length:
            safe_words.append(word)
            continue

        # Split oversized words without discarding any characters.
        start = 0

        while start < len(word):
            low = start + 1
            high = len(word)
            best = start

            while low <= high:
                mid = (low + high) // 2
                candidate = word[start:mid]

                token_count = len(tokenizer.encode(
                    prefix + candidate,
                    add_special_tokens=True,
                    truncation=False,
                ))

                if token_count <= max_length:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1

            if best == start:
                raise ValueError(
                    "Cannot split long word within token budget"
                )

            safe_words.append(word[start:best])
            start = best

    words = safe_words

    chunks = []
    current = []
    estimated_tokens = prefix_tokens

    def token_length(value):
        return len(tokenizer.encode(
            prefix + value,
            add_special_tokens=True,
            truncation=False,
        ))

    for word in words:
        word_tokens = len(tokenizer.encode(
            word, add_special_tokens=False, truncation=False
        ))

        if token_length(word) > max_length:
            raise ValueError(f"Single word exceeds token budget: {word[:80]!r}")

        # Fast estimate; verify using the complete candidate near the limit.
        if current and estimated_tokens + word_tokens >= max_length - 20:
            candidate = " ".join(current + [word])
            if token_length(candidate) > max_length:
                chunks.append(" ".join(current))
                current = []
                estimated_tokens = prefix_tokens

        current.append(word)
        estimated_tokens += word_tokens

        # Verify near the limit; recover safely from estimation errors.
        if estimated_tokens >= max_length - 20:
            candidate = " ".join(current)
            if token_length(candidate) > max_length:
                overflow_word = current.pop()
                if not current:
                    raise ValueError("Single word exceeds token budget")
                chunks.append(" ".join(current))
                current = [overflow_word]
                estimated_tokens = prefix_tokens + word_tokens

    if current:
        chunks.append(" ".join(current))

    assert "".join("".join(chunk.split()) for chunk in chunks) == "".join(original_words)
    assert all(token_length(chunk) <= max_length for chunk in chunks)

    return chunks
