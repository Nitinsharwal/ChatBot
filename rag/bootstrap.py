from __future__ import annotations

from pathlib import Path

from rag.db import HOTEL_DATABASE_URL, hotel_conn, rag_conn

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def apply_schema() -> None:
    sql = SCHEMA_PATH.read_text()
    with rag_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
    print("RAG schema applied (extension, tables, indexes).")


def verify_rag() -> None:
    with rag_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ext = cur.fetchone()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        tables = [r[0] for r in cur.fetchall()]
    print(f"   pgvector installed: {bool(ext)}")
    print(f"   RAG tables: {tables}")


HOTEL_TABLES = {
    "booking": "accounts_booking",
    "payment": "accounts_payment",
    "room": "accounts_room",
    "hotels": "accounts_hotels",
}


def verify_hotel() -> None:
    if not HOTEL_DATABASE_URL:
        print("⚠️  HOTEL_DATABASE_URL not set — skipping hotel-DB verification.")
        return
    with hotel_conn() as conn, conn.cursor() as cur:
        found = {}
        for label, table in HOTEL_TABLES.items():
            cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
            found[label] = bool(cur.fetchone()[0])
        print("   Hotel tables found: " + " ".join(f"{k}={v}" for k, v in found.items()))

        if found["booking"]:
            cur.execute(f"SELECT COUNT(*) FROM {HOTEL_TABLES['booking']}")
            print(f"   Live bookings: {cur.fetchone()[0]}")
            cur.execute(
                "SELECT column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=%s "
                "ORDER BY ordinal_position",
                (HOTEL_TABLES["booking"],),
            )
            cols = cur.fetchall()
            print(f"   {HOTEL_TABLES['booking']} columns:")
            for name, dtype in cols:
                print(f"      - {name}: {dtype}")


def main() -> None:
    apply_schema()
    verify_rag()
    verify_hotel()


if __name__ == "__main__":
    main()
