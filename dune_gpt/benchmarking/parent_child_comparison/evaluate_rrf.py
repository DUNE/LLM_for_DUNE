import json
from pathlib import Path

from src.retrieval.unified_backend import UnifiedRetrievalBackend
from src.retrieval.unified_retriever import (
    search_docdb_children,
    search_docdb_children_bm25,
    search_indico_children,
    search_indico_children_bm25,
)

BASE = Path("benchmarking/parent_child_comparison")
INDEX = (
    "/exp/dune/app/users/dune/askdune/dev/"
    "docdb-integration/test_data/benchmark_200"
)

questions = [
    json.loads(line)
    for line in (BASE / "data/questions_reviewed.jsonl").open()
    if json.loads(line)["quality_status"] == "accepted"
]

backend = UnifiedRetrievalBackend(INDEX)

output = BASE / "results/parent_child_rrf.jsonl"

def source_id(source, item):
    if source == "docdb":
        return str(item["docdb_id"])
    return str(item["indico_id"])

def evaluate(q):
    query = q["question"]

    lists = [
        ("docdb", search_docdb_children(
            backend.docdb, backend.manifest, query,
            candidate_count=30, embedder=backend.embedder
        )),
        ("docdb", search_docdb_children_bm25(
            backend.docdb, backend.manifest, query,
            candidate_count=30
        )),
        ("indico", search_indico_children(
            backend.indico, backend.embedder, query,
            candidate_count=30
        )),
        ("indico", search_indico_children_bm25(
            backend.indico, query, candidate_count=30
        )),
    ]

    scores = {}
    candidates = {}

    for source, items in lists:
        for rank, item in enumerate(items, 1):
            key = (source, str(item["document_id"]))
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            candidates[key] = item

    ranked = sorted(
        scores, key=lambda key: scores[key], reverse=True
    )

    parents = []
    seen = set()

    for key in ranked:
        item = candidates[key]
        parent_key = (key[0], str(item["parent_id"]))

        if parent_key in seen:
            continue

        seen.add(parent_key)

        parents.append({
            "source": key[0],
            "source_id": source_id(key[0], item),
            "parent_id": str(item["parent_id"]),
            "rrf_score": scores[key],
        })

        if len(parents) >= 10:
            break

    expected_source = q["source"]
    expected_id = str(q["source_id"])
    expected_parent = str(q["parent_id"])

    source_rank = next(
        (
            i for i, p in enumerate(parents, 1)
            if p["source"] == expected_source
            and p["source_id"] == expected_id
        ),
        None
    )

    parent_rank = next(
        (
            i for i, p in enumerate(parents, 1)
            if p["source"] == expected_source
            and p["parent_id"] == expected_parent
        ),
        None
    )

    return {
        "question_id": q["question_id"],
        "source": expected_source,
        "source_rank": source_rank,
        "parent_rank": parent_rank,
        "source_recall_at_5": source_rank is not None and source_rank <= 5,
        "source_recall_at_10": source_rank is not None and source_rank <= 10,
        "parent_recall_at_5": parent_rank is not None and parent_rank <= 5,
        "parent_recall_at_10": parent_rank is not None and parent_rank <= 10,
        "retrieved": parents,
    }

results = []

with output.open("w") as f:
    for i, q in enumerate(questions, 1):
        result = evaluate(q)
        results.append(result)
        f.write(json.dumps(result) + "\n")
        f.flush()
        print(
            f"{i}/{len(questions)} {q['question_id']} "
            f"source={result['source_rank']} "
            f"parent={result['parent_rank']}",
            flush=True,
        )

print("\nRESULTS")
print("=" * 65)

for source in ("all", "docdb", "indico"):
    subset = [
        r for r in results
        if source == "all" or r["source"] == source
    ]
    n = len(subset)

    print(f"\n{source.upper()} (n={n})")

    for metric in (
        "source_recall_at_5",
        "source_recall_at_10",
        "parent_recall_at_5",
        "parent_recall_at_10",
    ):
        count = sum(r[metric] for r in subset)
        print(f"{metric}: {count}/{n} ({100 * count / n:.2f}%)")

print("\nSaved:", output)
