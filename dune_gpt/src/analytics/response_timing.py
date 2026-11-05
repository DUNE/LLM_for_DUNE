"""Lightweight timing utilities for AskDUNE streaming responses."""

import time
import logging

logger = logging.getLogger(__name__)


def timed_stream(stream, request_start, retrieval_seconds=0.0):
    """Yield the original stream while logging response timing."""
    first_chunk_time = None
    chunk_count = 0

    try:
        for chunk in stream:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
                logger.info(
                    "AskDUNE time to first chunk: %.2f seconds",
                    first_chunk_time - request_start,
                )

            chunk_count += 1
            yield chunk

    finally:
        total_seconds = time.perf_counter() - request_start

        logger.info(
            "AskDUNE timing | retrieval=%.2fs | "
            "total=%.2fs | chunks=%d",
            retrieval_seconds,
            total_seconds,
            chunk_count,
        )
