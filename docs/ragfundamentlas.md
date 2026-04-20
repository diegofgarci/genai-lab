# RAG Fundamentals: Embeddings, Vector Stores, and Retrieval Design

A technical reference for designing the retrieval half of a Retrieval-Augmented
Generation system. Covers what embeddings are, how similarity works, how vector
stores operate, ChromaDB in practice, and — most importantly — how to diagnose
and mitigate the failures that embeddings have in production.

---

## 1. What Problem RAG Solves

LLMs have three structural limitations that make them unsuitable as standalone
question-answering systems over your data:

1. **Knowledge cutoff.** Models don't know anything after their training date.
2. **No access to private domain.** Your contracts, invoices, internal policies,
   customer data — none of that is in the model weights.
3. **Confident hallucination.** When a model doesn't know, it invents with
   syntactic elegance. Unacceptable in production.

RAG solves all three by injecting relevant information **into the prompt** at
query time. The model shifts from "remembering" to "reading and synthesizing".

### Architectural Impact

| Approach | Viable? | Why |
|---|---|---|
| Stuff all documents into the prompt | ❌ | Context windows are finite; expensive; attention degrades mid-context |
| Keyword search (BM25, grep) | Partial | No synonyms, no paraphrases |
| Fine-tune the model on your data | Rarely | Expensive, stale fast, doesn't cite sources, still hallucinates |
| **RAG** | ✅ | Scalable, cheap per query, cites sources, data is always fresh |

**Rule:** when in doubt between fine-tuning and RAG, choose RAG. Almost always.

---

## 2. Embeddings — The Core Primitive

### Definition

An embedding is a fixed-size vector of floats that represents the semantic
content of a piece of text. Produced by a neural network pre-trained on the
task: *make similar text produce similar vectors, dissimilar text produce
distant vectors*.

```
"the dog barks"  →  [0.021, -0.134, 0.087, ..., 0.045]   (384 numbers)
```

### Key Properties

- **Fixed dimensionality.** Each model produces vectors of a specific size
  (384, 768, 1536, 3072…). You don't choose — the model does.
- **No interpretable dimensions.** There is no "animal-ness" or "formality"
  axis. Meaning is distributed across all dimensions.
- **Pre-trained, not trained by you.** You consume embeddings as a service:
  `text → vector`. Training your own is rarely necessary.
- **Deterministic.** Same text + same model → same vector, every time.

### Why Geometry Captures Meaning

Based on the distributional hypothesis (Firth, 1957): *words with similar
meanings appear in similar contexts*. A model trained on billions of sentences
naturally learns to place words used interchangeably near each other in the
embedding space. Geometry becomes a proxy for semantics.

```
"perro ladra" ─── near ───── "can ladra"
     │
     └─ far ── "receta de tortilla"
     └─ far ── "firmé el contrato"
```

---

## 3. Measuring Similarity: Cosine

### The Formula

```
cos(A, B) = (A · B) / (|A| × |B|)
```

Returns a value in [-1, 1]:
- **1.0** — identical direction (same meaning)
- **0.0** — orthogonal (unrelated)
- **-1.0** — opposite direction (rare in practice)

For normalized embeddings (common), the practical range is [0, 1].

### Why Cosine, Not Euclidean

Cosine measures the *angle* between vectors, ignoring magnitude. This matters
because:
- Magnitude often reflects text length, not meaning.
- Two documents with equivalent semantics but different lengths should score
  as similar.

For L2-normalized vectors, cosine and Euclidean are mathematically equivalent
up to a constant. Most modern embedding models output normalized vectors, so
the choice is largely convention. **Default to cosine** unless you have a
reason not to.

### Distance vs Similarity

Many vector databases return **distance** (lower = closer), not similarity.
In a cosine-configured store:

```
similarity = 1 - distance
```

Pay attention to which one you're reading. A distance of 0.3 means
similarity of 0.7.

---

## 4. Choosing an Embedding Model

### Model Comparison (April 2026)

| Model | Dims | Deployment | Quality | Use Case |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Local, CPU | Baseline | Prototyping, learning |
| `all-mpnet-base-v2` | 768 | Local, CPU/GPU | Good | Production with modest resources |
| `BAAI/bge-m3` | 1024 | Local, GPU pref. | SOTA open | Multilingual, production |
| `text-embedding-3-small` (OpenAI) | 1536 | API | Very good | Production default |
| `text-embedding-3-large` (OpenAI) | 3072 | API | Top-tier | When quality is critical |
| `voyage-3-large` | 1024 | API | SOTA for RAG | Specialized retrieval |
| `cohere-embed-v4` | 1024-1536 | API | SOTA multilingual | Enterprise, multilingual |

### Selection Heuristics

1. **Start small for prototyping.** MiniLM on CPU is fine for the first 100
   documents. Iterate on system design, not model choice.
2. **Match the language.** MiniLM was trained mostly on English. For Spanish,
   French, German, or multilingual work, pick a multilingual model.
3. **Specialize by domain when quality matters.** Voyage, Cohere, and
   specialized BGE variants outperform general-purpose models by 5-15% in
   domain-specific retrieval benchmarks.
4. **Dimensions are a secondary axis.** More dimensions ≠ automatically better.
   A well-trained 768-dim model beats a poorly-trained 3072-dim model.
5. **API vs local.** API embeddings are more accurate but create a dependency
   and a per-call cost. Local models are private and cheap at scale.

---

## 5. Vector Stores — Why Not Just an Array?

Computing similarity against 8 documents in a Python loop is trivial. At
8,000 it's slow. At 8 million it's impossible. Vector stores solve four
problems a plain array cannot:

### Capabilities

1. **Approximate Nearest Neighbor (ANN) search.** Algorithms like HNSW, IVF,
   and LSH trade 1-2% of precision for logarithmic-time search. Essential
   at scale.
2. **Persistence.** The index lives on disk. No need to re-embed on every
   restart.
3. **Metadata + filters.** Each vector has structured attributes (date,
   author, category, amount) that can be filtered independently of similarity.
4. **CRUD.** Add, update, delete individual vectors without rebuilding the
   index.

### Store Comparison

| Store | Type | When to Use |
|---|---|---|
| **ChromaDB** | Embedded (SQLite + HNSW) | Development, prototypes, datasets up to ~1M vectors |
| **Qdrant** | Client-server (Rust) | Production, high throughput |
| **Pinecone** | SaaS managed | Production, don't want to manage infra |
| **pgvector** | Postgres extension | Already have Postgres, want one DB |
| **Weaviate** | Client-server | Production with advanced features (hybrid, multi-tenancy) |
| **FAISS** | Library (Meta) | Max control, no persistence, research |

The API surface is similar across stores: `collection.add()`,
`collection.query()`, metadata filters. Learning one teaches you 90% of
the others.

---

## 6. ChromaDB in Practice

### Architecture

ChromaDB is SQLite + binary HNSW index files. Metadata lives in SQLite tables;
vectors live in the index. This means:

- Backups are trivial (`cp -r chroma_db/ backup/`).
- Inspection is possible (`sqlite3 chroma.sqlite3`).
- Portability is automatic (move the folder).
- Scale limit is where SQLite's concurrency stops being enough.

### Core Concepts

```
PersistentClient (disk-backed)
  └── Collection (like a table)
        ├── Embedding function (fixed at creation time)
        ├── Distance metric (cosine, L2, IP)
        └── Documents
              ├── id (string, unique)
              ├── text (the document itself)
              ├── embedding (computed automatically)
              └── metadata (dict of str/int/float/bool)
```

### Minimal Example

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    ),
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    ids=["doc_001"],
    documents=["Invoice processing automation with AI"],
    metadatas=[{"domain": "invoicing", "year": 2026}],
)

results = collection.query(
    query_texts=["How do I automate billing?"],
    n_results=3,
    where={"year": {"$gte": 2026}},
)
```

### Metadata Filter Syntax

ChromaDB uses a MongoDB-style filter dialect:

| Operator | Meaning |
|---|---|
| `{"field": value}` | Equality |
| `{"field": {"$eq": value}}` | Equality (explicit) |
| `{"field": {"$ne": value}}` | Not equal |
| `{"field": {"$gt": value}}` | Greater than |
| `{"field": {"$gte": value}}` | Greater or equal |
| `{"field": {"$lt": value}}` | Less than |
| `{"field": {"$lte": value}}` | Less or equal |
| `{"field": {"$in": [v1, v2]}}` | Value in list |
| `{"field": {"$nin": [v1, v2]}}` | Value not in list |
| `{"$and": [filter1, filter2]}` | Logical AND |
| `{"$or": [filter1, filter2]}` | Logical OR |

**Filters run before the vector search.** Only documents matching the filter
are considered for similarity ranking. This is how you combine exact conditions
with semantic search.

### Embedding Function Is Sticky

The embedding function is fixed when the collection is created. It's used for
both ingestion and query. You can't change it later without re-indexing
everything — plan the choice accordingly.

---

## 7. Where Embeddings Fail

Critical section. These are not theoretical problems — they show up immediately
in real workloads. Know them, design for them.

### Failure Modes

| Failure | Example | Why |
|---|---|---|
| **Negation** | "includes clause X" vs "does NOT include clause X" | Embeddings weight content words heavily; "not" is a small signal |
| **Exact numbers** | "€1,247" vs "€12,470" | The model encodes "a number", not its value |
| **Very short strings** | "yes" vs "no" | Too little signal to produce distinct vectors |
| **Rare named entities** | "Aceitunas Pérez S.L." | Infrequent in training data, weak representation |
| **Similarity floor** | Random sentences in the same language show similarity 0.3-0.5 | Shared substrate: language, grammar, register |
| **Domain transfer weakness** | Technical vocabulary with small/generalist models | MiniLM can't distinguish "developer tools" from "Michelin restaurants" when both are abstract noun phrases |

### The Similarity Floor — Detailed

In a collection of same-language sentences, unrelated pairs rarely score below
0.3-0.4 with small generalist models. This happens because embeddings encode
many non-semantic features: language, formality, syntactic structure, sentence
length. Two Spanish sentences share this "substrate" regardless of topic.

**Implication: absolute similarity values are meaningless in isolation.**
Only *relative* rankings matter. A result at 0.55 is excellent if the next
one is at 0.30. It's noise if the next one is at 0.52.

Stop using fixed similarity thresholds. Use top-K retrieval and trust the
ordering (or add re-ranking, see §9).

---

## 8. Diagnostic Signals for Bad Retrieval

When inspecting query results, these patterns signal problems:

| Signal | What It Tells You | Mitigation |
|---|---|---|
| Top-K distances all within 0.15 of each other | Model can't distinguish; ranking is effectively random | Upgrade embedding model; add re-ranker |
| Top-1 distance > 0.7 (cosine space) | Nothing in the collection is truly relevant | Widen corpus; improve chunking; reformulate query |
| Obviously irrelevant doc in top-3 | Domain transfer failure | Upgrade model; add metadata filters; add BM25 hybrid |
| Relevant docs scattered across ranks | Weak chunking or weak model | Revisit chunking strategy; upgrade model |
| High-score outliers that don't match query intent | Spurious lexical overlap | Add re-ranker; consider hybrid search |

**Practical habit:** after building the retriever, run 10-20 real user
queries and inspect top-5 manually *before* wiring up the LLM. If retrieval
is broken, the LLM will amplify the failure, not fix it.

---

## 9. Mitigation Hierarchy

When retrieval quality is insufficient, try fixes in this order. Higher cost
= later resort.

### Level 1 — Upgrade the embedding model

Swap MiniLM → `bge-m3` or `text-embedding-3-small`. Often eliminates 60-80%
of retrieval issues with zero code changes beyond the `embedding_function`.

### Level 2 — Hybrid search (dense + sparse)

Combine semantic search (embeddings) with keyword search (BM25):
- Embeddings capture synonyms and paraphrases.
- BM25 eliminates any document that shares no terms with the query.
- Fuse results via Reciprocal Rank Fusion or weighted scoring.

Hybrid outperforms pure semantic search in virtually every public benchmark
since 2024. This is industry consensus, not a niche optimization.

### Level 3 — Metadata filters

Move anything exact out of the embedding and into metadata: numbers, dates,
categories, IDs, boolean flags. Filters run before vector search and
eliminate whole classes of retrieval failures (wrong year, wrong vendor,
wrong amount range).

**Critical for any domain involving negation, numbers, or structured
attributes.** Invoices. Contracts. Medical records. Product catalogs.

### Level 4 — Re-ranking with a cross-encoder

Retrieve top-50 with a fast bi-encoder (the embedding model), then re-rank
them with a slower but more accurate cross-encoder that reads each
(query, document) pair jointly. Cross-encoders capture negation, logical
relationships, and fine-grained relevance that bi-encoders miss.

Typical cost: 50-200ms per query. Typical quality gain: substantial,
especially on top-5 precision.

### Level 5 — Fine-tune the embedding model

Only when levels 1-4 are exhausted. Requires:
- Hundreds or thousands of labeled query-document pairs
- Training infrastructure
- Rigorous evaluation
- Commitment to retrain as the domain evolves

Most teams that think they need fine-tuning actually need better chunking,
hybrid search, or a re-ranker. Fine-tuning is a last resort.

---

## 10. Design Principles for Production RAG

Distilled from everything above.

1. **Embeddings retrieve by topic, not by truth or logic.** Use them for
   domain/topic matching. Do not use them for comparing numbers, evaluating
   conditions, or reasoning about negation.
2. **Metadata is not optional.** Any production system has structured
   attributes. Put them in metadata, not in the embedding.
3. **Top-K, not thresholds.** Always retrieve a fixed K and rely on relative
   ordering. Absolute similarity values drift across models and corpora.
4. **Evaluate retrieval before wiring the LLM.** Precision@K on 20 real
   queries tells you more than any LLM output.
5. **Hybrid search is the default for production.** Semantic alone leaves
   quality on the table.
6. **The model is a lever, not a magic wand.** Upgrading the embedding model
   is the cheapest and biggest lever. Pull it before anything else.

---

## 11. Common Pitfalls Checklist

Things that go wrong early, in order of frequency:

- [ ] Storing exact numbers in embeddings instead of metadata
- [ ] Using fixed similarity thresholds instead of top-K
- [ ] Not inspecting retrieval quality before adding the LLM
- [ ] Using a generalist model for a specialized domain
- [ ] Ignoring the similarity floor and assuming 0.5 means "relevant"
- [ ] Embedding very short strings (tokens, single words) and expecting
      meaningful rankings
- [ ] Changing the embedding model without re-indexing
- [ ] Committing the vector DB folder to git
- [ ] Not handling the case where queries return zero results after filtering

---

## 12. Glossary

- **Embedding** — fixed-size vector representation of text produced by a
  neural network.
- **Dense retrieval** — retrieval based on embedding similarity.
- **Sparse retrieval** — retrieval based on keyword matching (BM25, TF-IDF).
- **Hybrid retrieval** — combination of dense and sparse.
- **Bi-encoder** — model that encodes query and document independently; fast
  but less accurate. Used for the initial retrieval step.
- **Cross-encoder** — model that encodes query and document jointly; slow
  but more accurate. Used for re-ranking.
- **Top-K** — the number of results retrieved per query. Typical values 3-10.
- **HNSW** — Hierarchical Navigable Small World. Graph-based ANN algorithm,
  the default in most modern vector stores.
- **ANN** — Approximate Nearest Neighbor. Sacrifices a small amount of
  precision for large speedups in search.
- **Collection** — a ChromaDB unit grouping documents, embeddings, and
  metadata. Roughly equivalent to a table.
- **Similarity floor** — the minimum cosine similarity observed between
  arbitrary same-language sentences; depends on the model.
- **Re-ranker** — a second-stage model applied to top-K results to improve
  ordering.
- **Precision@K** — fraction of the top-K retrieved documents that are
  actually relevant.
- **Recall@K** — fraction of all relevant documents that appear in the top-K.
- **MRR** — Mean Reciprocal Rank. Average of 1/rank_of_first_relevant_result
  across a query set.