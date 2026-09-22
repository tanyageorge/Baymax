"""
Day 1 goal: prove the RAG retrieval pipeline works on our seed corpus,
BEFORE wiring up an LLM.

What this script does, step by step:
1. Load the .txt files from data/raw/ as LlamaIndex "documents"
2. Split each document into smaller "chunks" (nodes)
3. Turn each chunk into a vector (a list of numbers representing its content)
4. Store those vectors in Chroma, a local vector database
5. Given a plain-English query, vectorize the query the same way and ask
   Chroma for the chunks whose vectors are closest to it
6. Print the retrieved chunks + their source file, so we can eyeball whether
   retrieval is actually pulling the right content

This is the core of RAG's "R" (Retrieval) — the "AG" (Augmented Generation)
part comes later, once this is solid and we wire in an LLM to write an
answer using these retrieved chunks as context.

--------------------------------------------------------------------------
A NOTE ON THE EMBEDDING MODEL (important to understand, not just skip over)
--------------------------------------------------------------------------
The "right" way to do this is with a neural embedding model (e.g. a small
HuggingFace sentence-transformer) that understands MEANING — so "my kid has
a fever" and "child's temperature is high" would land near each other in
vector space even though they share almost no words. That's what makes RAG
powerful over simple keyword search.

For today, this script uses TF-IDF (TfidfVectorizer, from scikit-learn) as
a simple first pass: it represents each chunk as a vector of *weighted word
frequencies* — common words score low, rare/distinctive words score high.
It's a classic "sparse" retrieval technique (the same family as the BM25
algorithm search engines used for decades before neural embeddings existed)
and it WILL correctly retrieve on queries that share vocabulary with the
source text, but it won't understand synonyms or paraphrasing the way a
real embedding model does.

>>> Once we've proven this works, flip EMBEDDING_MODE below to
>>> "huggingface" to use real semantic embeddings — the rest of the
>>> pipeline (chunking, indexing, retrieval) doesn't change at all. That's
>>> the benefit of building on LlamaIndex's abstractions: the embedding
>>> model is swappable without touching the rest of the code.

Worth noting in your study notes: "sparse retrieval" (TF-IDF/BM25, keyword-
based) vs. "dense retrieval" (neural embeddings, meaning-based) is a real
distinction interviewers ask about — production RAG systems sometimes use
BOTH (hybrid search) because each catches things the other misses.
"""

from pathlib import Path
from typing import List

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

EMBEDDING_MODE = "tfidf"  # "tfidf" (works today) or "huggingface" (swap in later)


class TfidfEmbedding(BaseEmbedding):
    """
    A minimal LlamaIndex-compatible embedding class backed by scikit-learn's
    TF-IDF vectorizer, so we can prove out the retrieval pipeline before
    setting up a real neural embedding model.

    Must be `fit()` once on the full corpus before use (see build_index()) —
    unlike a pretrained neural embedding model, TF-IDF's vocabulary has to be
    learned from the documents it will be used on.
    """

    _vectorizer: TfidfVectorizer = None
    _fitted: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_vectorizer", TfidfVectorizer(max_features=512))
        object.__setattr__(self, "_fitted", False)

    def fit(self, corpus_texts: List[str]) -> None:
        self._vectorizer.fit(corpus_texts)
        object.__setattr__(self, "_fitted", True)

    def _vectorize(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedding must be fit() on the corpus before use.")
        return self._vectorizer.transform([text]).toarray()[0].tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._vectorize(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._vectorize(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


def get_embed_model(documents) -> BaseEmbedding:
    if EMBEDDING_MODE == "huggingface":
        # Real semantic embeddings — needs internet access to huggingface.co.
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Default: TF-IDF stand-in.
    embed_model = TfidfEmbedding()
    embed_model.fit([doc.text for doc in documents])
    return embed_model


def build_index() -> VectorStoreIndex:
    print(f"Loading documents from {DATA_DIR} ...")
    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    print(f"Loaded {len(documents)} documents.")

    Settings.embed_model = get_embed_model(documents)
    Settings.llm = None  # retrieval only for now — no generation yet

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Fresh collection each run, so re-running this script doesn't duplicate chunks
    try:
        chroma_client.delete_collection("baymax_seed_corpus")
    except Exception:
        pass
    chroma_collection = chroma_client.get_or_create_collection("baymax_seed_corpus")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Embedding + indexing chunks into Chroma (mode={EMBEDDING_MODE})...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    print("Index built.\n")
    return index


def run_query(index: VectorStoreIndex, query: str, top_k: int = 3) -> None:
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    print(f"\n=== Query: {query!r} ===")
    if not results:
        print("    (no results)")
        return
    for i, node in enumerate(results, start=1):
        source = node.node.metadata.get("file_name", "unknown")
        score = node.score
        snippet = node.node.get_text()[:220].replace("\n", " ")
        print(f"\n[{i}] source={source}  score={score:.3f}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    index = build_index()

    test_queries = [
        "I have a really bad headache and my neck feels stiff",
        "my kid has a fever and won't stop crying",
        "I cut my finger pretty deep and it won't stop bleeding",
        "I've been really stressed and can't sleep",
        "I got stung by a bee and my throat feels tight",
    ]

    for q in test_queries:
        run_query(index, q)