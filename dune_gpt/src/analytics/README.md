# AskDUNE analytics

This package contains utilities for observing indexing and retrieval behavior. Its scope includes index statistics, token-related metrics, and response timing.

## Current usage

The development application records unified retrieval elapsed time in `main.py`. A `response_timing.py` utility also exists, but its full streaming integration should not be assumed without verifying its call sites.

The experimental indexing scripts log counts and elapsed time; DocDB indexing additionally reports page/token and storage inventory information.

## Interpretation

Retrieval timing measures the retrieval portion of a request, **not** necessarily the full time to first token or end-to-end answer latency. Distinguish model generation time, network latency, and indexing time in performance comparisons.

## Privacy and safety

Avoid logging API credentials, session secrets, or restricted source text. Benchmark against development indexes; never alter production data for analytics experiments.
