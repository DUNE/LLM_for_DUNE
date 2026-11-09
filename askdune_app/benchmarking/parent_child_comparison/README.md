# AskDUNE parent-child retrieval benchmark

This benchmark compares the experimental unified parent-child retriever with a legacy large-chunk baseline on **the same frozen source corpus**. Run all commands from the `dune_gpt/` repository root, with `.venv` active. The benchmark ChromaDB path below is read-only for manifest and question-generation operations. **Do not rebuild, overwrite, or point an indexing job at `benchmark_200`.**

## Frozen corpus

Current manifest: **108 DocDB documents + 106 Indico events = 214 sources**, with 14,421 DocDB and 1,937 Indico child records. These numbers describe the audited index, not guaranteed completeness of remote attachments. The manifest checksum should be retained in question and evaluation outputs.

```bash
# Inspect the manifest script's supported options.
PYTHONPATH=. python benchmarking/parent_child_comparison/build_corpus_manifest.py --help

# Rebuild the metadata-only manifest from the frozen benchmark index.
PYTHONPATH=. python -u benchmarking/parent_child_comparison/build_corpus_manifest.py \
  --index-path /exp/dune/app/users/dune/askdune/dev/docdb-integration/test_data/benchmark_200
```

| Manifest option | Default | Meaning |
|---|---|---|
| `--index-path` | required | Existing benchmark ChromaDB directory; script checks that its directory name is `benchmark_200` |
| `--output` | `benchmarking/parent_child_comparison/data/corpus_manifest.json` | Destination for the JSON source inventory and integrity report |
| `--help` | n/a | Show CLI options without running the audit |

Example with an explicit output path:

```bash
PYTHONPATH=. python benchmarking/parent_child_comparison/build_corpus_manifest.py \
  --index-path /exp/dune/app/users/dune/askdune/dev/docdb-integration/test_data/benchmark_200 \
  --output benchmarking/parent_child_comparison/data/corpus_manifest.json
```

## Generate source-grounded benchmark questions

`generate_questions.py` uses the configured Fermilab model (`gpt-oss:120b` in the current pilot) to create questions, reference answers, and evidence quotations from selected indexed passages. The deterministic seed is `2026`. The target is **50 DocDB and 50 Indico questions**, selected from distinct sources. The generator writes successes incrementally to `data/questions_generated.jsonl` and logs unsuccessful attempts to `data/question_failures.jsonl`. It resumes by skipping previously saved source IDs.

```bash
# Inspect all flags supported by your deployed version.
PYTHONPATH=. python benchmarking/parent_child_comparison/generate_questions.py --help

# Small pilot: attempt one not-yet-completed source.
PYTHONPATH=. python -u benchmarking/parent_child_comparison/generate_questions.py \
  --max-new 1

# Larger pilot: attempt up to 10 unfinished sources.
PYTHONPATH=. python -u benchmarking/parent_child_comparison/generate_questions.py \
  --max-new 10
```

| Question-generation option | Meaning |
|---|---|
| `--max-new N` | With the **updated pilot implementation**, attempt at most `N` unfinished sources in this invocation. Count attempts, not successful generations. Failures do not consume a source permanently; they can be retried on the next run. |
| `--help` | List the deployed script's full argparse options and defaults. |

**Important:** `--max-new` originally limited *successful* questions. The pilot code was changed to count *attempted* sources; verify the change is present before relying on this behavior. Other CLI flags/defaults of `generate_questions.py` are **not verified in this documentation**; inspect `--help` rather than assuming they exist. Do not remove the saved JSONL to rerun a pilot unless intentionally resetting its progress.

### Evidence verification

The generator strips surrounding quotation marks and normalizes formatting such as HTML `<br>` tags and Markdown emphasis before checking that the model's evidence quotation occurs within the indexed passage. This does **not** authorize semantic paraphrases or unsupported quotations. Store source IDs, versions, evidence child IDs, and manifest checksum for traceability.

### Current pilot example

```text
Already completed: 5
Generating docdb 17 ... Saved (6/100)
...
New questions this run: 10
Failures this run: 0
Total saved: 15
DocDB questions: 15
Indico questions: 0
```

A first-attempt evidence mismatch followed by a successful retry is **not** a permanent failure. Review generated questions manually: avoid overrepresenting page-number lookups, arbitrary table-cell positions, or obsolete-document trivia. Do not call the 100-question dataset complete until it contains 50 valid questions from each source type.

## Planned retrieval and answer evaluation

The following are evaluation requirements, **not yet verified runnable commands**:

1. Build a legacy large-chunk index from the **same 214 frozen sources**, separately from production and the parent-child benchmark index.
2. Compare Recall@5, Recall@10, MRR, evidence/source citation correctness, and latency.
3. Evaluate generated answers using the **same model and prompt** for both retrieval systems; separately assess answer correctness and groundedness.
4. Manually review a sample, especially when the generator and judge use the same model family.
5. Record script version, model, seed, corpus checksum, retrieval parameters, and output paths with every run.

Do not assume scripts for retrieval benchmarking, answer benchmarking, or analysis exist until they have been implemented and tested.
