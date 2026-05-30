from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEndpointEmbeddings:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing in .env")
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=token,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_embedder().embed_documents(texts)
    for v in vectors:
        if len(v) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding dim mismatch: got {len(v)}, expected {EMBEDDING_DIM}. "
                f"Check EMBEDDING_MODEL ({EMBEDDING_MODEL}) — must output {EMBEDDING_DIM}-dim vectors."
            )
    return vectors


def embed_query(text: str) -> list[float]:
    return get_embedder().embed_query(text)
