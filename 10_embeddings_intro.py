"""
10_embeddings_intro.py — First hands-on contact with embeddings
================================================================
Week 3 Day 1: See embeddings for what they are — just vectors of floats
that capture meaning. No vector database, no LLM, no magic.

We use sentence-transformers with the MiniLM model:
  - 384 dimensions
  - Runs locally on CPU
  - ~22MB
  - Good enough to build intuition (and for many real use cases)

What you'll see by running this:
  1. What an embedding actually looks like (just numbers)
  2. Semantic similarity between related sentences
  3. Clear separation between unrelated concepts
  4. Cross-lingual behavior (Spanish vs English)
  5. Where embeddings break (negation, exact numbers)

Usage:
    python 10_embeddings_intro.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# =============================================================================
# BLOCK 0: Load the embedding model
# =============================================================================
# First run downloads ~22MB of model weights to ~/.cache/huggingface/
# Subsequent runs load from cache in ~1 second.
#
# WHY this model: all-MiniLM-L6-v2 is the sweet spot for learning:
#   - Small enough to run anywhere (no GPU needed)
#   - Decent multilingual behavior (trained mostly on English but handles
#     Spanish reasonably for demonstration purposes)
#   - Fast inference (~1000 sentences/sec on CPU)
# In production you'd probably use OpenAI text-embedding-3-small,
# Voyage, or Cohere Embed v3 for better retrieval quality.

print("Loading embedding model (first run downloads ~22MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")


# =============================================================================
# BLOCK 1: What does an embedding look like?
# =============================================================================
# Take a single sentence, embed it, and actually LOOK at the numbers.
# The goal: demystify. There's nothing magical — it's just a fixed-size
# array of floats.

def block_1_show_raw_embedding():
    print("\n" + "=" * 70)
    print("BLOCK 1: What does an embedding actually look like?")
    print("=" * 70)

    sentence = "El perro ladra en el parque"
    embedding = model.encode(sentence)

    print(f"\nInput sentence: '{sentence}'")
    print(f"Output type:    {type(embedding).__name__}")
    print(f"Output shape:   {embedding.shape}")
    print(f"Data type:      {embedding.dtype}")
    print(f"\nFirst 10 dimensions:  {embedding[:10]}")
    print(f"Last 10 dimensions:   {embedding[-10:]}")
    print(f"\nMin value:  {embedding.min():.4f}")
    print(f"Max value:  {embedding.max():.4f}")
    print(f"Mean:       {embedding.mean():.4f}")
    print(f"L2 norm:    {np.linalg.norm(embedding):.4f}")

    print("\n→ Key takeaway: an embedding is just 384 floats. Nothing more.")
    print("  The 'magic' is that these numbers were chosen by a neural network")
    print("  so that similar sentences produce similar vectors.")


# =============================================================================
# BLOCK 2: Cosine similarity between related sentences
# =============================================================================
# Now we verify the main claim: sentences with similar meaning should
# have similar vectors. We compute cosine similarity between pairs and
# expect to see high values (close to 1.0) for related sentences.

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: cos(A, B) = (A · B) / (|A| * |B|)

    Returns a value between -1 and 1:
      - 1.0  = identical direction (same meaning)
      - 0.0  = orthogonal (unrelated)
      - -1.0 = opposite direction (rare in practice)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def block_2_related_sentences():
    print("\n" + "=" * 70)
    print("BLOCK 2: Similarity between RELATED sentences (expect HIGH values)")
    print("=" * 70)

    # Pairs we expect to be very similar
    pairs = [
        ("El perro ladra",              "El can ladra"),                    # synonym
        ("Voy a emitir una factura",    "Necesito crear un comprobante"),   # paraphrase
        ("El coche es rojo",            "El automóvil es de color rojo"),   # synonym + expansion
        ("Me duele la cabeza",          "Tengo un fuerte dolor de cabeza"), # intensification
        ("El contrato incluye NDA",     "El acuerdo contempla confidencialidad"),  # domain paraphrase
    ]

    print(f"\n{'Sentence A':<40} {'Sentence B':<40} {'Similarity':>10}")
    print("-" * 92)

    for s1, s2 in pairs:
        emb1 = model.encode(s1)
        emb2 = model.encode(s2)
        sim = cosine_similarity(emb1, emb2)
        print(f"{s1:<40} {s2:<40} {sim:>10.4f}")

    print("\n→ Notice: all values are high (typically 0.5+), confirming that")
    print("  the model captures semantic relatedness, not just word overlap.")


# =============================================================================
# BLOCK 3: Cosine similarity between UNRELATED sentences
# =============================================================================
# The opposite test: sentences from totally different domains should
# produce vectors that are roughly orthogonal (similarity near 0).
# This is what makes retrieval work — irrelevant content gets filtered.

def block_3_unrelated_sentences():
    print("\n" + "=" * 70)
    print("BLOCK 3: Similarity between UNRELATED sentences (expect LOW values)")
    print("=" * 70)

    pairs = [
        ("El perro ladra en el parque",      "La receta lleva tres huevos"),
        ("Firmé el contrato con el abogado", "El volcán entró en erupción"),
        ("Necesito emitir una factura",      "La pelota cayó al agua"),
        ("El coche es rojo",                 "Python es un lenguaje de programación"),
        ("Hace mucho calor hoy",             "La ecuación diferencial tiene solución única"),
    ]

    print(f"\n{'Sentence A':<40} {'Sentence B':<40} {'Similarity':>10}")
    print("-" * 92)

    for s1, s2 in pairs:
        emb1 = model.encode(s1)
        emb2 = model.encode(s2)
        sim = cosine_similarity(emb1, emb2)
        print(f"{s1:<40} {s2:<40} {sim:>10.4f}")

    print("\n→ Notice: values are much lower (typically 0.0–0.3).")
    print("  Unrelated content produces nearly orthogonal vectors.")
    print("  THIS is why retrieval works: the query vector only lands close")
    print("  to vectors of genuinely related documents.")


# =============================================================================
# BLOCK 4: Cross-lingual behavior
# =============================================================================
# MiniLM is primarily English but was trained on some multilingual data.
# It's not the best multilingual model, but good enough to demonstrate
# the concept. For production multilingual RAG you'd use:
#   - paraphrase-multilingual-mpnet-base-v2
#   - BAAI/bge-m3 (state of the art multilingual)
#   - Cohere embed-multilingual-v3

def block_4_cross_lingual():
    print("\n" + "=" * 70)
    print("BLOCK 4: Cross-lingual similarity")
    print("=" * 70)
    print("NOTE: MiniLM is English-first. Multilingual models do this better.")

    pairs = [
        ("El perro ladra",              "The dog barks"),
        ("Necesito emitir una factura", "I need to issue an invoice"),
        ("Buenos días",                 "Good morning"),
        ("Me gusta el café",            "I like coffee"),
        ("El contrato es válido",       "The contract is valid"),
    ]

    print(f"\n{'Spanish':<35} {'English':<35} {'Similarity':>10}")
    print("-" * 82)

    for es, en in pairs:
        emb_es = model.encode(es)
        emb_en = model.encode(en)
        sim = cosine_similarity(emb_es, emb_en)
        print(f"{es:<35} {en:<35} {sim:>10.4f}")

    print("\n→ Results are decent but not amazing. For serious multilingual RAG,")
    print("  switch to a proper multilingual model (see docstring).")


# =============================================================================
# BLOCK 5: Where embeddings FAIL
# =============================================================================
# Critical section. Theory was clear: embeddings struggle with negation,
# exact numbers, and very short text. Now you'll see it with your own eyes.
# This shapes how you design real RAG systems (metadata filters, re-ranking,
# hybrid search).

def block_5_failure_modes():
    print("\n" + "=" * 70)
    print("BLOCK 5: Failure modes — where embeddings break")
    print("=" * 70)

    # --- Failure 1: Negation ---
    print("\n--- Negation (embeddings barely distinguish yes/no) ---")
    pairs_negation = [
        ("El contrato incluye cláusula de no competencia",
         "El contrato NO incluye cláusula de no competencia"),
        ("Me gusta el café",
         "No me gusta el café"),
        ("Este producto es seguro",
         "Este producto no es seguro"),
    ]
    for s1, s2 in pairs_negation:
        sim = cosine_similarity(model.encode(s1), model.encode(s2))
        print(f"  sim = {sim:.4f}  |  '{s1}' vs '{s2}'")

    print("\n  → Opposite meanings return very high similarity.")
    print("    Production fix: use metadata flags, hybrid search with keywords,")
    print("    or a re-ranker that handles negation better.")

    # --- Failure 2: Exact numbers ---
    print("\n--- Exact numbers (embeddings see 'a number', not its value) ---")
    pairs_numbers = [
        ("La factura es de 1.247 euros",
         "La factura es de 12.470 euros"),
        ("El contrato tiene 10 páginas",
         "El contrato tiene 100 páginas"),
        ("Plazo de pago: 30 días",
         "Plazo de pago: 90 días"),
    ]
    for s1, s2 in pairs_numbers:
        sim = cosine_similarity(model.encode(s1), model.encode(s2))
        print(f"  sim = {sim:.4f}  |  '{s1}' vs '{s2}'")

    print("\n  → Different values, nearly identical vectors.")
    print("    Production fix: extract numeric values as metadata and filter")
    print("    at query time (e.g. where amount > 10000).")

    # --- Failure 3: Very short strings ---
    print("\n--- Very short strings (low signal, noisy similarities) ---")
    pairs_short = [
        ("Sí", "No"),
        ("Ok", "Vale"),
        ("Hola", "Adiós"),
    ]
    for s1, s2 in pairs_short:
        sim = cosine_similarity(model.encode(s1), model.encode(s2))
        print(f"  sim = {sim:.4f}  |  '{s1}' vs '{s2}'")

    print("\n  → Short strings have little semantic content to encode.")
    print("    Production fix: embed larger chunks (sentences, paragraphs),")
    print("    never isolated tokens.")


# =============================================================================
# BLOCK 6: Bonus — batch encoding and the "semantic neighborhood"
# =============================================================================
# Real-world pattern: encode many texts at once (much faster than one by one),
# then rank a query against all of them. This is exactly what a vector store
# does internally, just more efficiently.

def block_6_semantic_neighborhood():
    print("\n" + "=" * 70)
    print("BLOCK 6: Query against a small corpus — the retrieval preview")
    print("=" * 70)

    corpus = [
        "Proceso de emisión de facturas electrónicas en España",
        "Cómo extraer datos de facturas con OCR",
        "Validación de NIF y CIF en el sistema contable",
        "Receta de tortilla de patatas tradicional",
        "Recomendaciones de restaurantes en A Coruña",
        "Integración de la API de Hacienda para e-invoicing",
        "Plazos legales para conservar documentación fiscal",
        "Historia del arte románico en Galicia",
    ]

    query = "¿Cómo automatizo el proceso de facturación?"

    # Encode everything at once — this is the fast path
    corpus_embeddings = model.encode(corpus)
    query_embedding = model.encode(query)

    # Compute similarity of the query against every document
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in corpus_embeddings
    ]

    # Rank documents by similarity
    ranked = sorted(
        zip(corpus, similarities),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\nQuery: '{query}'\n")
    print(f"{'Rank':<6} {'Similarity':>10}  Document")
    print("-" * 90)
    for rank, (doc, sim) in enumerate(ranked, start=1):
        marker = "  ← TOP" if rank <= 3 else ""
        print(f"{rank:<6} {sim:>10.4f}  {doc}{marker}")

    print("\n→ The top-3 documents are all about invoicing, even though the query")
    print("  uses the word 'facturación' and documents use 'facturas' or 'e-invoicing'.")
    print("  This is retrieval in its purest form — no DB, no LLM, just math.")
    print("  In the next script we do exactly this with ChromaDB at scale.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    block_1_show_raw_embedding()
    block_2_related_sentences()
    block_3_unrelated_sentences()
    block_4_cross_lingual()
    block_5_failure_modes()
    block_6_semantic_neighborhood()

    print("\n" + "=" * 70)
    print("DONE. Key takeaways:")
    print("=" * 70)
    print("  1. An embedding is just an array of floats (384 here).")
    print("  2. Semantically related text → high cosine similarity.")
    print("  3. Unrelated text → similarity near zero.")
    print("  4. Negation, exact numbers, and tiny strings are weak spots.")
    print("  5. Block 6 is already RAG retrieval — just without the DB.")
    print("\nNext: 11_first_vector_store.py — same idea, but with ChromaDB")
    print("      and a proper persistent index you can scale to thousands of docs.")


if __name__ == "__main__":
    main()