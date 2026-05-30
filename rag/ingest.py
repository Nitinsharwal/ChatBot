from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.db import rag_conn
from rag.embeddings import embed_texts

DOCS_DIR = Path(__file__).parent / "docs"
SUPPORTED_EXT = {".md", ".markdown", ".txt"}

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class ParsedDoc:
    source: str
    title: str
    category: str
    body: str
    sha: str


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip().strip("'\"")
    return meta, text[m.end():]


def parse_doc(path: Path) -> ParsedDoc:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    title = meta.get("title") or path.stem.replace("-", " ").title()
    category = meta.get("category") or "general"
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return ParsedDoc(source=str(path.name), title=title, category=category, body=body.strip(), sha=sha)


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def _upsert_document(cur, doc: ParsedDoc) -> tuple[int | None, str]:
    cur.execute(
        "SELECT id, content_sha FROM documents WHERE source = %s",
        (doc.source,),
    )
    row = cur.fetchone()
    if row and row[1] == doc.sha:
        return row[0], "unchanged"
    if row:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (row[0],))
        cur.execute(
            "UPDATE documents SET title=%s, category=%s, content_sha=%s, updated_at=now() "
            "WHERE id=%s",
            (doc.title, doc.category, doc.sha, row[0]),
        )
        return row[0], "updated"
    cur.execute(
        "INSERT INTO documents (source, title, category, content_sha) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (doc.source, doc.title, doc.category, doc.sha),
    )
    return cur.fetchone()[0], "inserted"


def _insert_chunks(cur, document_id: int, chunks: list[str], vectors: list[list[float]]) -> None:
    cur.executemany(
        "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
        "VALUES (%s, %s, %s, %s)",
        [(document_id, i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors))],
    )


def ingest_path(path: Path) -> dict:
    doc = parse_doc(path)
    with rag_conn() as conn, conn.cursor() as cur:
        doc_id, action = _upsert_document(cur, doc)
        if action == "unchanged":
            return {"source": doc.source, "action": "unchanged", "chunks": 0}
        chunks = chunk_text(doc.body)
        if not chunks:
            return {"source": doc.source, "action": action, "chunks": 0}
        vectors = embed_texts(chunks)
        _insert_chunks(cur, doc_id, chunks, vectors)
        return {"source": doc.source, "action": action, "chunks": len(chunks)}


def ingest_folder(folder: Path) -> list[dict]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXT)
    if not files:
        print(f"⚠️  No supported files found in {folder}")
        return []
    results = []
    for path in files:
        try:
            r = ingest_path(path)
            results.append(r)
            print(f"  {r['action']:>9}  {r['chunks']:>3} chunks  {r['source']}")
        except Exception as e:
            print(f"  ❌ FAILED  {path.name}: {e}")
            results.append({"source": path.name, "action": "error", "error": str(e)})
    return results


def stats() -> None:
    with rag_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        d = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        c = cur.fetchone()[0]
        cur.execute(
            "SELECT category, COUNT(*) FROM documents GROUP BY category ORDER BY category"
        )
        by_cat = cur.fetchall()
    print(f"\n📊 KB total: {d} documents, {c} chunks")
    for cat, n in by_cat:
        print(f"     {cat}: {n} docs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    ap.add_argument("path", nargs="?", default=str(DOCS_DIR),
                    help=f"File or folder to ingest (default: {DOCS_DIR})")
    args = ap.parse_args()
    target = Path(args.path)
    if not target.exists():
        print(f"❌ Path not found: {target}")
        sys.exit(1)
    print(f"🔍 Ingesting from {target} (chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    if target.is_file():
        r = ingest_path(target)
        print(f"  {r['action']:>9}  {r.get('chunks', 0):>3} chunks  {r['source']}")
    else:
        ingest_folder(target)
    stats()


if __name__ == "__main__":
    main()
