from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

RAG_DATABASE_URL = os.getenv("RAG_DATABASE_URL") or os.getenv("DATABASE_URL")
HOTEL_DATABASE_URL = os.getenv("HOTEL_DATABASE_URL")


def _normalize(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def _ensure_vector_extension(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()


@contextmanager
def rag_conn() -> Iterator[psycopg.Connection]:
    if not RAG_DATABASE_URL:
        raise RuntimeError("RAG_DATABASE_URL (or DATABASE_URL) is not set in .env")
    conn = psycopg.connect(_normalize(RAG_DATABASE_URL), autocommit=False)
    try:
        _ensure_vector_extension(conn)
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def hotel_conn(read_only: bool = True) -> Iterator[psycopg.Connection]:
    if not HOTEL_DATABASE_URL:
        raise RuntimeError("HOTEL_DATABASE_URL is not set in .env")
    conn = psycopg.connect(_normalize(HOTEL_DATABASE_URL), autocommit=False)
    try:
        if read_only:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
