"""
The "real" retrieval module used by the FastAPI backend — the production
version of the exploratory pipeline built in scripts/test_retrieval.py on
Day 1. Same TF-IDF-based approach for now (see that script's docstring for
the full explanation of why, and what to swap in later).

Unlike the script, this module is imported once when the API server starts
(see main.py's startup event), and the index it builds is reused for every
/chat request — rebuilding it per-request would be slow and pointless
since the corpus doesn't change while the server is running.
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

# backend/app/retrieval.py -> parents: app/, backend/, <project root>/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


class TfidfEmbedding(BaseEmbedding):
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


def _get_embed_model(documents) -> BaseEmbedding:
    embed_model = TfidfEmbedding()
    embed_model.fit([doc.text for doc in documents])
    return embed_model


def build_index() -> VectorStoreIndex:
    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()

    Settings.embed_model = _get_embed_model(documents)
    Settings.llm = None  # generation happens separately, via app/llm_client.py

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection("baymax_seed_corpus")
    except Exception:
        pass
    chroma_collection = chroma_client.get_or_create_collection("baymax_seed_corpus")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=False
    )


def retrieve_chunks(index: VectorStoreIndex, query: str, top_k: int = 3) -> List[dict]:
    """
    Returns a list of {"source": "01_fever.txt", "text": "..."} dicts —
    the exact format app/llm_client.py's generate_answer() expects.
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    return [
        {
            "source": node.node.metadata.get("file_name", "unknown"),
            "text": node.node.get_text(),
        }
        for node in results
    ]