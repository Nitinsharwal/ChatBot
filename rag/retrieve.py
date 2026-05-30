from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from pgvector.psycopg import Vector

from rag.db import hotel_conn, rag_conn
from rag.embeddings import embed_query


@dataclass
class Chunk:
    content: str
    source: str
    title: str
    category: str
    score: float


@dataclass
class BookingInfo:
    reference: str
    guest_name: str
    guest_email: str
    hotel_name: str
    hotel_location: str
    room_name: str
    room_type: str
    start_date: str
    end_date: str
    num_guests: int
    total_amount: str
    status: str
    payment_status: Optional[str]
    cancelled_at: Optional[str]
    cancellation_reason: Optional[str]


def retrieve(
    question: str,
    k: int = 4,
    category: Optional[str] = None,
    max_distance: float = 0.65,
) -> list[Chunk]:
    if not question.strip():
        return []
    vec = Vector(embed_query(question))

    sql = """
        SELECT c.content, d.source, d.title, d.category,
               c.embedding <=> %s AS distance
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    params: list = [vec]
    where = ""
    if category:
        where = "WHERE d.category = %s"
        params.append(category)
    params.extend([vec, k])
    sql = sql.format(where=where)

    with rag_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        Chunk(
            content=content,
            source=source,
            title=title,
            category=cat,
            score=round(1.0 - float(distance), 4),
        )
        for content, source, title, cat, distance in rows
        if float(distance) <= max_distance
    ]


def lookup_booking(reference: str, email: str) -> Optional[BookingInfo]:
    if not reference or not email:
        return None
    sql = """
        SELECT b.reference,
               b.guest_first_name || ' ' || b.guest_last_name AS guest_name,
               b.guest_email,
               h.hotel_name, h.hotel_location,
               r.name AS room_name, r.room_type,
               b.start_date, b.end_date, b.num_guests,
               b.total_amount, b.status,
               (SELECT status FROM accounts_payment
                 WHERE booking_id = b.id
                 ORDER BY id DESC LIMIT 1) AS payment_status,
               b.cancelled_at, b.cancellation_reason
        FROM accounts_booking b
        JOIN accounts_room r ON r.id = b.room_id
        JOIN accounts_hotels h ON h.id = r.hotel_id
        WHERE b.reference = %s AND lower(b.guest_email) = lower(%s)
        LIMIT 1
    """
    with hotel_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (reference.strip(), email.strip()))
        row = cur.fetchone()
    if not row:
        return None
    return BookingInfo(
        reference=row[0],
        guest_name=row[1],
        guest_email=row[2],
        hotel_name=row[3],
        hotel_location=row[4],
        room_name=row[5],
        room_type=row[6],
        start_date=str(row[7]),
        end_date=str(row[8]),
        num_guests=int(row[9]),
        total_amount=str(row[10]),
        status=row[11],
        payment_status=row[12],
        cancelled_at=str(row[13]) if row[13] else None,
        cancellation_reason=row[14] or None,
    )


def log_support_gap(question: str, session_id: Optional[str] = None) -> None:
    with rag_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO support_gaps (question, session_id) VALUES (%s, %s)",
            (question.strip()[:2000], session_id),
        )


def format_booking(b: BookingInfo) -> str:
    lines = [
        f"Booking {b.reference} — {b.status.upper()}",
        f"Guest: {b.guest_name} ({b.guest_email})",
        f"Hotel: {b.hotel_name}, {b.hotel_location}",
        f"Room: {b.room_name} ({b.room_type}) for {b.num_guests} guest(s)",
        f"Dates: {b.start_date} → {b.end_date}",
        f"Total: {b.total_amount}",
    ]
    if b.payment_status:
        lines.append(f"Payment: {b.payment_status}")
    if b.cancelled_at:
        lines.append(f"Cancelled at: {b.cancelled_at}")
        if b.cancellation_reason:
            lines.append(f"Reason: {b.cancellation_reason}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the RAG retriever.")
    ap.add_argument("question", help="The question to search for")
    ap.add_argument("-k", type=int, default=4, help="Number of chunks (default 4)")
    ap.add_argument("--category", default=None, help="Filter by category")
    ap.add_argument("--booking", nargs=2, metavar=("REF", "EMAIL"),
                    help="Also look up a booking by reference + email")
    args = ap.parse_args()

    print(f"\n🔎 Question: {args.question}\n")
    chunks = retrieve(args.question, k=args.k, category=args.category)
    if not chunks:
        print("  (no relevant chunks found above threshold)")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] score={c.score}  category={c.category}  source={c.source}")
        snippet = c.content[:240].replace("\n", " ")
        print(f"    {snippet}{'…' if len(c.content) > 240 else ''}\n")

    if args.booking:
        ref, email = args.booking
        b = lookup_booking(ref, email)
        print("─" * 60)
        print(format_booking(b) if b else f"❌ No booking found for {ref} / {email}")


if __name__ == "__main__":
    main()
