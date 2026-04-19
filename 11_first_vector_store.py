"""
11_first_vector_store.py — First hands-on with ChromaDB
========================================================
Week 3 Day 1: Go from embeddings-in-an-array to a real persistent
vector store with CRUD, metadata filtering, and scalable search.

What changes from script 10:
  - Embeddings are computed and stored automatically by the collection.
  - The index (HNSW) lives on disk and loads in milliseconds.
  - Each document has metadata you can filter on.
  - Queries scale to millions of vectors, not dozens.

What you'll build by running this:
  1. A persistent ChromaDB on disk (./chroma_db/)
  2. A collection with 15 documents spanning 3 domains
  3. Semantic queries with and without metadata filters
  4. CRUD operations (add, update, delete)
  5. Introspection (count, peek, get by ID)

Usage:
    python 11_first_vector_store.py

To reset the DB between runs, delete the ./chroma_db/ directory.
"""

import os
import shutil
import chromadb
from chromadb.utils import embedding_functions


# =============================================================================
# CONFIG
# =============================================================================
# Where Chroma persists data. Each collection is a subdirectory here.
# Delete this folder to reset everything.
CHROMA_PATH = "./chroma_db"

# Collection name — think of it as a "table" in a relational DB.
# You can have many collections per client (e.g. one per tenant, per project).
COLLECTION_NAME = "week3_demo"

# Whether to wipe the DB on each run. Useful while learning; turn off for real use.
RESET_ON_START = True


# =============================================================================
# BLOCK 1: Initialize the client
# =============================================================================
# PersistentClient saves everything to disk. There's also an in-memory client
# (chromadb.Client()) for ephemeral tests, and an HttpClient for talking to a
# ChromaDB server. PersistentClient is the right default for local development.

def init_client() -> chromadb.ClientAPI:
    print("=" * 70)
    print("BLOCK 1: Initialize persistent ChromaDB client")
    print("=" * 70)

    if RESET_ON_START and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Wiped existing DB at {CHROMA_PATH}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    print(f"Client ready. Data persists at: {os.path.abspath(CHROMA_PATH)}")
    print(f"Existing collections: {[c.name for c in client.list_collections()]}")
    return client


# =============================================================================
# BLOCK 2: Create a collection with an embedding function
# =============================================================================
# A collection ties together documents, their embeddings, and metadata.
#
# The embedding_function tells Chroma HOW to turn text into vectors.
# Chroma ships with several built-in options:
#   - SentenceTransformerEmbeddingFunction (local, what we used in script 10)
#   - OpenAIEmbeddingFunction (API)
#   - CohereEmbeddingFunction (API)
#   - You can also provide your own callable
#
# KEY POINT: once you set the embedding function at creation time, Chroma uses
# it for BOTH ingestion and query. You never call model.encode() manually again
# — Chroma handles it. This is why the rest of the code looks magical: you pass
# strings, you get results.

def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    print("\n" + "=" * 70)
    print("BLOCK 2: Create collection with embedding function")
    print("=" * 70)

    # Same model we used in script 10 so results are comparable
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "Week 3 Day 1 demo", "hnsw:space": "cosine"},
        # ↑ hnsw:space chooses the distance metric. Default is L2 (euclidean).
        # For normalized text embeddings, cosine is the industry standard.
    )

    print(f"Collection '{collection.name}' ready.")
    print(f"Embedding function: SentenceTransformer (all-MiniLM-L6-v2, 384 dims)")
    print(f"Distance metric: cosine")
    print(f"Current document count: {collection.count()}")
    return collection


# =============================================================================
# BLOCK 3: Ingest documents with rich metadata
# =============================================================================
# Three things per document:
#   1. id (string, unique within the collection) — your primary key
#   2. document (the text itself) — will be embedded automatically
#   3. metadata (dict of str/int/float/bool values) — filterable attributes
#
# Metadata is where a lot of real-world retrieval quality comes from.
# Numbers, dates, categories — put them here, not in the embedding.
# Remember what you saw in script 10: embeddings don't distinguish 1.247 EUR
# from 12.470 EUR. Metadata does.

def ingest_documents(collection: chromadb.Collection):
    print("\n" + "=" * 70)
    print("BLOCK 3: Ingest 15 documents across 3 domains")
    print("=" * 70)

    documents = [
        # --- Domain 1: invoicing / business automation (5 docs) ---
        {
            "id": "doc_001",
            "text": "Proceso de emisión de facturas electrónicas en España según la normativa de Hacienda.",
            "metadata": {"domain": "invoicing", "year": 2025, "author": "Ana", "lang": "es"},
        },
        {
            "id": "doc_002",
            "text": "Extracción automática de datos de facturas en PDF usando OCR y modelos de lenguaje.",
            "metadata": {"domain": "invoicing", "year": 2026, "author": "Diego", "lang": "es"},
        },
        {
            "id": "doc_003",
            "text": "Validación de NIF, CIF y NIE en sistemas contables: reglas y casos límite.",
            "metadata": {"domain": "invoicing", "year": 2024, "author": "Ana", "lang": "es"},
        },
        {
            "id": "doc_004",
            "text": "Integración con la API de Hacienda para envío de facturas electrónicas B2B.",
            "metadata": {"domain": "invoicing", "year": 2026, "author": "Diego", "lang": "es"},
        },
        {
            "id": "doc_005",
            "text": "Plazos legales para conservar documentación fiscal en empresas españolas.",
            "metadata": {"domain": "invoicing", "year": 2023, "author": "Luis", "lang": "es"},
        },

        # --- Domain 2: cooking (5 docs) ---
        {
            "id": "doc_006",
            "text": "Receta tradicional de tortilla de patatas con cebolla pochada a fuego lento.",
            "metadata": {"domain": "cooking", "year": 2025, "author": "Marta", "lang": "es"},
        },
        {
            "id": "doc_007",
            "text": "Cómo preparar un buen pulpo a feira gallego: cocción, aliño y presentación.",
            "metadata": {"domain": "cooking", "year": 2024, "author": "Marta", "lang": "es"},
        },
        {
            "id": "doc_008",
            "text": "Fermentación de masa madre para pan artesanal: hidratación y tiempos.",
            "metadata": {"domain": "cooking", "year": 2026, "author": "Pablo", "lang": "es"},
        },
        {
            "id": "doc_009",
            "text": "Recomendaciones de restaurantes con estrella Michelin en A Coruña.",
            "metadata": {"domain": "cooking", "year": 2026, "author": "Marta", "lang": "es"},
        },
        {
            "id": "doc_010",
            "text": "Maridaje de vinos de Galicia con mariscos y pescados del Atlántico.",
            "metadata": {"domain": "cooking", "year": 2025, "author": "Pablo", "lang": "es"},
        },

        # --- Domain 3: software / programming (5 docs) ---
        {
            "id": "doc_011",
            "text": "Arquitectura de agentes multi-tool con Claude API y function calling.",
            "metadata": {"domain": "programming", "year": 2026, "author": "Diego", "lang": "es"},
        },
        {
            "id": "doc_012",
            "text": "Implementación de retrieval augmented generation con ChromaDB y Python.",
            "metadata": {"domain": "programming", "year": 2026, "author": "Diego", "lang": "es"},
        },
        {
            "id": "doc_013",
            "text": "Patrones de manejo de errores en APIs asíncronas con FastAPI y Pydantic.",
            "metadata": {"domain": "programming", "year": 2025, "author": "Luis", "lang": "es"},
        },
        {
            "id": "doc_014",
            "text": "Containerización de servicios Python con Docker y despliegue en Railway.",
            "metadata": {"domain": "programming", "year": 2026, "author": "Luis", "lang": "es"},
        },
        {
            "id": "doc_015",
            "text": "Comparativa de frameworks agénticos: LangGraph, CrewAI y Claude Agent SDK.",
            "metadata": {"domain": "programming", "year": 2026, "author": "Diego", "lang": "es"},
        },
    ]

    # Chroma expects parallel lists, not a list of dicts
    collection.add(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[d["metadata"] for d in documents],
    )

    print(f"Ingested {len(documents)} documents.")
    print(f"Collection now has: {collection.count()} documents.")
    print("Behind the scenes: Chroma embedded each text with MiniLM and stored")
    print("the vectors + documents + metadata in the HNSW index on disk.")


# =============================================================================
# BLOCK 4: Basic semantic query
# =============================================================================
# This is the core retrieval operation. You pass text, Chroma embeds it with
# the same function used at ingestion, and returns the N closest documents.
#
# n_results is the "K" in "top-K retrieval". Typical production values: 3-10.
# Too few → miss relevant context. Too many → noise + cost in the LLM step.

def basic_query(collection: chromadb.Collection):
    print("\n" + "=" * 70)
    print("BLOCK 4: Basic semantic query (no filters)")
    print("=" * 70)

    query = "¿Cómo automatizo el proceso de facturación con IA?"
    print(f"\nQuery: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=5,
    )

    print(f"\nTop 5 results (sorted by similarity):\n")
    print(f"{'Rank':<5} {'Distance':>10} {'Domain':<14} {'ID':<8}  Document")
    print("-" * 100)

    for rank, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), start=1):
        print(f"{rank:<5} {dist:>10.4f} {meta['domain']:<14} {doc_id:<8}  {doc}")

    print("\n→ Note: Chroma returns DISTANCE, not similarity.")
    print("  With cosine space, similarity = 1 - distance.")
    print("  Lower distance = more similar. The top results should all be 'invoicing'.")


# =============================================================================
# BLOCK 5: Queries with metadata filters
# =============================================================================
# This is where real RAG systems get their quality. Metadata filters apply
# BEFORE the vector search — only documents matching the filter are considered
# for similarity ranking. This combines the precision of relational queries
# with the semantic power of embeddings.
#
# Chroma's filter syntax (simplified MongoDB-style):
#   {"field": value}                        → equality
#   {"field": {"$eq": value}}               → equality (explicit)
#   {"field": {"$ne": value}}               → not equal
#   {"field": {"$gt": value}}               → greater than
#   {"field": {"$gte": value}}              → greater or equal
#   {"field": {"$lt": value}}               → less than
#   {"field": {"$lte": value}}              → less or equal
#   {"field": {"$in": [v1, v2]}}            → value in list
#   {"field": {"$nin": [v1, v2]}}           → value not in list
#   {"$and": [filter1, filter2]}            → logical AND
#   {"$or": [filter1, filter2]}             → logical OR

def filtered_queries(collection: chromadb.Collection):
    print("\n" + "=" * 70)
    print("BLOCK 5: Queries with metadata filters")
    print("=" * 70)

    # --- Filter 1: single equality ---
    print("\n--- Filter: domain = 'cooking' ---")
    query = "platos con productos del mar"
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"domain": "cooking"},
    )
    print(f"Query: '{query}'")
    for doc_id, doc, dist in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        print(f"  [{dist:.4f}] {doc_id}: {doc}")

    # --- Filter 2: numeric range ---
    print("\n--- Filter: year >= 2026 ---")
    query = "herramientas modernas para desarrolladores"
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"year": {"$gte": 2026}},
    )
    print(f"Query: '{query}'")
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  [{dist:.4f}] {doc_id} ({meta['year']}): {doc}")

    # --- Filter 3: combined conditions (AND) ---
    print("\n--- Filter: author = 'Diego' AND year = 2026 ---")
    query = "sistemas de IA que aprendan"
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={
            "$and": [
                {"author": "Diego"},
                {"year": 2026},
            ]
        },
    )
    print(f"Query: '{query}'")
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  [{dist:.4f}] {doc_id} ({meta['author']}, {meta['year']}): {doc}")

    # --- Filter 4: $in for multiple values ---
    print("\n--- Filter: domain in ['invoicing', 'programming'] ---")
    query = "automatización"
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={"domain": {"$in": ["invoicing", "programming"]}},
    )
    print(f"Query: '{query}'")
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  [{dist:.4f}] {doc_id} ({meta['domain']}): {doc}")

    print("\n→ Filters run BEFORE the vector search. This is exactly what you'll")
    print("  need for the invoice capstone: 'find invoices semantically similar")
    print("  to X, but only for vendor Y, year 2026, amount > 1000€'.")


# =============================================================================
# BLOCK 6: CRUD operations (update, delete)
# =============================================================================
# You'll need these in any real system where data changes.

def crud_operations(collection: chromadb.Collection):
    print("\n" + "=" * 70)
    print("BLOCK 6: CRUD — update and delete")
    print("=" * 70)

    # --- Update: change text and metadata for one doc ---
    print("\n--- Update doc_001 ---")
    print(f"Before: {collection.get(ids=['doc_001'])['documents'][0]}")

    collection.update(
        ids=["doc_001"],
        documents=["UPDATED: Nueva normativa de facturación electrónica para 2027."],
        metadatas=[{"domain": "invoicing", "year": 2027, "author": "Ana", "lang": "es"}],
    )

    print(f"After:  {collection.get(ids=['doc_001'])['documents'][0]}")
    print("Chroma re-embedded the document automatically during update.")

    # --- Delete: remove by ID ---
    print("\n--- Delete doc_015 ---")
    print(f"Before delete: count = {collection.count()}")
    collection.delete(ids=["doc_015"])
    print(f"After delete:  count = {collection.count()}")

    # --- Delete with filter: remove all cooking docs ---
    print("\n--- Delete all docs where domain = 'cooking' ---")
    collection.delete(where={"domain": "cooking"})
    print(f"After delete:  count = {collection.count()}")


# =============================================================================
# BLOCK 7: Introspect the collection
# =============================================================================
# Useful for debugging and building dashboards over your RAG system.

def inspect_collection(collection: chromadb.Collection):
    print("\n" + "=" * 70)
    print("BLOCK 7: Introspect the collection")
    print("=" * 70)

    # Count total
    print(f"\nTotal documents: {collection.count()}")

    # Peek at first few (without running a query)
    peek = collection.peek(limit=3)
    print(f"\nFirst 3 documents (arbitrary order):")
    for doc_id, doc in zip(peek["ids"], peek["documents"]):
        print(f"  {doc_id}: {doc[:70]}...")

    # Get all docs matching a filter (no similarity ranking)
    invoicing_docs = collection.get(where={"domain": "invoicing"})
    print(f"\nAll remaining 'invoicing' documents: {len(invoicing_docs['ids'])}")
    for doc_id, doc, meta in zip(
        invoicing_docs["ids"],
        invoicing_docs["documents"],
        invoicing_docs["metadatas"],
    ):
        print(f"  {doc_id} ({meta['year']}): {doc[:60]}...")


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = init_client()
    collection = get_or_create_collection(client)
    ingest_documents(collection)
    basic_query(collection)
    filtered_queries(collection)
    crud_operations(collection)
    inspect_collection(collection)

    print("\n" + "=" * 70)
    print("DONE. Key takeaways:")
    print("=" * 70)
    print("  1. ChromaDB handles embedding automatically once you set an")
    print("     embedding_function on the collection.")
    print("  2. Data persists on disk — next run loads in milliseconds.")
    print("  3. Metadata filters run BEFORE vector search: that's how you")
    print("     combine exact conditions (year, author, amount) with semantic")
    print("     similarity. This is the pattern for production RAG.")
    print("  4. CRUD is straightforward — update re-embeds automatically.")
    print("  5. You've just built the retriever half of a RAG system.")
    print("     Tomorrow (Day 2) we add real document ingestion + chunking")
    print("     strategies. Day 3 connects this to Claude for generation.")
    print()
    print(f"DB files are at: {os.path.abspath(CHROMA_PATH)}")
    print("Explore with: `ls -la chroma_db/` and `sqlite3 chroma_db/chroma.sqlite3`")


if __name__ == "__main__":
    main()