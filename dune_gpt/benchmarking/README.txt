DUNE-GPT Benchmarking Methodology
=================================

This document describes the benchmarking methodology used to evaluate the
retrieval component of DUNE-GPT. The benchmark separates upstream
embedding/index construction from downstream retrieval and context construction.
This separation allows us to first identify a strong vector index configuration,
and then evaluate how different retrieval strategies and final context policies
affect evidence retrieval quality.


1. Benchmarking Objective
-------------------------

The goal of the benchmark is to evaluate whether the retrieval system can return
the correct DUNE attachment and, more importantly, the correct evidence span
inside that attachment. This is stricter than checking whether the system
retrieves a generally relevant document. For a RAG system, the retrieved context
must contain the evidence needed to answer the question, not merely come from a
related file.

The benchmark therefore evaluates retrieval at two levels. The first is
file-level correctness: whether the retrieved context includes chunks from the
expected attachment. The second is evidence-level correctness: whether the
retrieved context includes chunks that overlap the specific evidence window used
to generate the ground-truth question.

The full benchmark is divided into two stages. The upstream stage selects the
best embedding and Chroma index configuration. The downstream stage fixes that
index and evaluates retrieval strategies and final context construction
policies.


2. Benchmark Corpus
-------------------

The corpus consists of locally cached DUNE attachments collected from DocDB and
Indico. The use of local cached attachments avoids repeated crawling and ensures
that all model and retrieval experiments are evaluated on the same underlying
corpus. Each attachment is associated with metadata such as source, filename,
local path, attachment URL, content type, and document type.

The benchmark includes both slide-like and document-like materials. Native
PowerPoint files are labeled as slides. PDF files are text-extracted with
pdfplumber, and their document_type is assigned using a Poppler/pdf2image-based
first-page aspect-ratio heuristic. This document_type label is used for
retrieval analysis and final context filtering, but it should be interpreted as
a coarse metadata label rather than a perfect semantic file classifier.


3. Ground-Truth QA Construction
-------------------------------

The final benchmark uses Paired QA v2 as the main ground-truth QA dataset:

benchmarking/qa_sets/retrieval_qa_paired_v2_candidates.csv

The purpose of Paired QA v2 is to provide questions that are both evidence
aligned and representative of realistic RAG usage. Each QA item is tied to a
specific source attachment and a specific target evidence window inside that
attachment.

For each selected attachment, the text is first extracted and normalized. Two
text windows are then prepared. The first is an overview window consisting of
the first 5000 characters of the attachment. This gives the question generation
model broad context about the attachment. The second is a 1500-character target
window sampled from the full attachment text. The sampling is deterministic
under a fixed random seed, but the target window is not restricted to the
beginning of the attachment. This avoids biasing the benchmark toward early
chunks.

Both windows are sent to LiteLLM. The model is instructed to use the overview
window only for contextual orientation and to focus on the target window as the
evidence source. For each target window, two question variants are generated:
a lexical_anchor question and a paraphrased question. The lexical_anchor
variant tends to preserve important terms from the evidence span, while the
paraphrased variant expresses the same evidence need with less direct lexical
overlap.

This paired construction makes the benchmark more balanced. The lexical version
tests whether retrievers can exploit explicit terminology, while the paraphrased
version tests semantic retrieval behavior. This is important because the
downstream benchmark compares BM25, dense retrieval, hybrid retrieval, and
hybrid retrieval with reranking.

Each QA row records the expected attachment path and the anchor span location
inside the extracted text. These anchor fields allow the evaluation to measure
whether retrieved chunks overlap the actual evidence window.


4. Upstream Embedding and Indexing Benchmark
--------------------------------------------

The first experimental stage evaluates embedding and chunking configurations.
The retrieval method is kept simple in this stage: each Chroma index is queried
with dense retrieval, and the resulting chunks are evaluated against the
ground-truth QA set.

The upstream benchmark varies embedding model, document prefix, query prefix,
chunk size, chunk overlap, and chunking strategy. The embedding models tested
are:

- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/multi-qa-mpnet-base-dot-v1
- BAAI/bge-base-en-v1.5
- intfloat/e5-small-v2

The prefix protocol follows the expected usage pattern of each model:

- all-MiniLM-L6-v2:
  document prefix = "passage: "
  query prefix    = "passage: "

- multi-qa-mpnet-base-dot-v1:
  document prefix = "passage: "
  query prefix    = "passage: "

- BAAI/bge-base-en-v1.5:
  document prefix = "passage: "
  query prefix    = "Represent this sentence for searching relevant passages: "

- intfloat/e5-small-v2:
  document prefix = "passage: "
  query prefix    = "query: "

The chunking strategy is word-based. The tested chunk sizes are 1000 and 2000
words, and the tested overlaps are 0, 100, and 200 words. This creates 24
embedding-layer combinations:

4 embedding models x 2 chunk sizes x 3 overlaps = 24 combinations.

The purpose of this stage is not to evaluate complex retriever behavior, but to
choose a strong and stable upstream index. The selected configuration for the
downstream experiments is:

- embedding model: intfloat/e5-small-v2
- chunk size: 2000 words
- chunk overlap: 0
- source: both DocDB and Indico

The corresponding Chroma index is:

benchmarking/chroma_experiments/intfloat_e5-small-v2_word_chunk2000_overlap0_both/


5. Retrieval Evaluation Metrics
-------------------------------

The benchmark uses five retrieval metrics. These metrics are computed per query
and then averaged over the QA set.

path@k measures whether the top-k retrieved chunks contain at least one chunk
from the expected attachment file. This is a file-level hit metric. It answers
the question: did retrieval find the correct source file?

anchor@k measures whether the top-k retrieved chunks contain a chunk that
overlaps the ground-truth anchor span by at least the configured threshold. This
is stricter than path@k because it requires the retrieved context to contain the
evidence region, not just the correct file.

mrr_path is the reciprocal rank of the first retrieved chunk from the expected
attachment. If the expected file first appears at rank 1, the score is 1. If it
first appears at rank 2, the score is 1/2. If it never appears, the score is 0.
This metric rewards correct files appearing earlier in the retrieval list.

mrr_anchor is the reciprocal rank of the first retrieved chunk that overlaps
the ground-truth anchor span. This metric rewards evidence-bearing chunks
appearing earlier in the retrieval list.

best_ov is the best overlap ratio between any retrieved chunk and the
ground-truth anchor span, averaged over all questions. This captures partial
evidence alignment even when the strict anchor hit threshold is not met.


6. Aggregate Score: ERS
-----------------------

The main aggregate score is Evidence Retrieval Score, or ERS:

ERS =
    0.24 * path@k
  + 0.36 * anchor@k
  + 0.12 * mrr_path
  + 0.18 * mrr_anchor
  + 0.10 * best_ov

The weighting emphasizes evidence-level correctness. anchor@k and mrr_anchor
receive the largest combined weight because the benchmark is intended to
measure whether the system retrieves answer-supporting evidence, not merely a
related file.

Latency is recorded separately as mean milliseconds per query. It is used as an
efficiency diagnostic, but it is not included in ERS.

Some intermediate downstream experiments also used RERS:

RERS = ERS - 0.01 * final_top_k

RERS penalizes larger returned contexts. In the final filter-ratio experiment,
all configurations return the same number of chunks, so the final comparison
uses ERS directly.


7. Downstream Retriever Benchmark
---------------------------------

After the upstream index is fixed, the downstream benchmark evaluates retrieval
strategies without rebuilding the Chroma index. This isolates the impact of the
retriever and context construction logic from the embedding/indexing layer.

The downstream methods include dense retrieval, BM25 retrieval, hybrid
retrieval, and hybrid retrieval with reranking. Dense retrieval queries the
Chroma vector index directly. BM25 ranks chunks using sparse lexical matching.
Hybrid retrieval combines normalized dense and BM25 scores:

hybrid_score =
    dense_weight * normalized_dense_score
  + (1 - dense_weight) * normalized_bm25_score

The reranker pipeline first retrieves a candidate pool using hybrid retrieval,
then reranks the candidate chunks with a cross-encoder reranker.

The strongest configurations identified before the final filter-ratio study
were:

1. without reranker:
   hybrid retrieval with dense weight 0.50.

2. with reranker:
   hybrid retrieval with dense weight 0.50, followed by
   BAAI/bge-reranker-v2-m3.


8. Final Context Composition Filter
-----------------------------------

The final stage evaluates a practical RAG context construction policy: the
ratio of slide chunks to document chunks in the final retrieved context.

This stage fixes the retrieval pipelines and varies only the final
slides/document filter. The goal is to measure whether retrieval evidence
quality changes when the final context is biased toward slides, toward
documents, or balanced between the two.

Two fixed pipelines are evaluated.

Pipeline A, without reranker:

- base retriever: hybrid
- dense weight: 0.50
- filter pool: top 10 retrieved chunks

Pipeline B, with reranker:

- base retriever: hybrid
- dense weight: 0.50
- candidate_k before reranking: 15
- reranker: BAAI/bge-reranker-v2-m3
- filter pool: top 8 reranked chunks

For both pipelines, the final context size is fixed at 6 chunks. The following
seven ratios are evaluated:

- slides:6, document:0
- slides:5, document:1
- slides:4, document:2
- slides:3, document:3
- slides:2, document:4
- slides:1, document:5
- slides:0, document:6

For each ratio, the filter selects the highest-ranked chunks of each requested
document type from the filter pool. If one type is underrepresented in the
filter pool, the remaining slots are filled by the highest-ranked chunks from
the other type. This preserves ranking as much as possible while enforcing the
desired context composition.


9. Final Filter-Ratio Results
-----------------------------

The final filter-ratio sweep uses Paired QA v2 and the fixed E5 Chroma index.
The primary score is ERS. Latency is recorded as mean milliseconds per query.

Final result directory:

benchmarking/retriever_runs/filter_ratio_sweep_paired_v2_top6_context/

Final CSV table:

benchmarking/retriever_runs/filter_ratio_sweep_paired_v2_top6_context/final_filter_ratio_ers_latency_table.csv

Final image table:

benchmarking/retriever_runs/filter_ratio_sweep_paired_v2_top6_context/final_filter_ratio_ers_latency_table.png

Final ERS results:

filter ratio             without reranker ERS    with reranker ERS
slides:6, document:0     0.765971                0.771657
slides:5, document:1     0.857237                0.822109
slides:4, document:2     0.857237                0.822019
slides:3, document:3     0.857176                0.817350
slides:2, document:4     0.860279                0.821868
slides:1, document:5     0.845339                0.811989
slides:0, document:6     0.751062                0.769820

In this final sweep, the best without-reranker configuration is
slides:2, document:4. The best with-reranker configuration is
slides:5, document:1.


10. Interpretation
------------------

The benchmark shows that retrieval performance depends not only on the
embedding model and retriever, but also on the final context construction
policy. In the final filter-ratio experiment, mixed slide/document contexts
perform better than all-slide or all-document contexts for the no-reranker
pipeline. This suggests that the DUNE corpus contains complementary evidence
across slides and documents.

The reranker pipeline is more computationally expensive, with latency dominated
by cross-encoder scoring. In the final filter-ratio sweep, reranking does not
outperform the best no-reranker hybrid configuration under the selected Paired
QA v2 setting and final context size. This result is important because it shows
that a simpler hybrid retriever with a well-chosen final context composition
policy can be competitive or stronger for this benchmark.


11. Reproducibility
-------------------

The final benchmark assumes the following local assets:

- Paired QA v2:
  benchmarking/qa_sets/retrieval_qa_paired_v2_candidates.csv

- E5 Chroma index:
  benchmarking/chroma_experiments/intfloat_e5-small-v2_word_chunk2000_overlap0_both/

- BGE reranker cache:
  .sentence_transformers_cache/models--BAAI--bge-reranker-v2-m3/

The final filter-ratio sweep can be rerun with:

python benchmarking/scripts/run_filter_ratio_sweep.py --run-name filter_ratio_sweep_paired_v2_top6_context --overwrite

