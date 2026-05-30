from __future__ import annotations

import re
from pathlib import Path

from rag.db import hotel_conn, rag_conn
from rag.ingest import ingest_path

DOCS_DIR = Path(__file__).parent / "docs"
HOTELS_DIR = DOCS_DIR / "hotels"
LOCATIONS_DOC = DOCS_DIR / "hotels-locations.md"

QUERY = """
SELECT
  h.id, h.hotel_name, h.hotel_description, h.hotel_slug,
  h.hotel_location, h.hotel_price, h.hotel_offer_price,
  h.rating_avg, h.rating_count,
  COALESCE(
    (SELECT string_agg(a.amenities_name, ', ' ORDER BY a.amenities_name)
       FROM accounts_hotels_hotel_amenities ha
       JOIN accounts_amenities a ON a.id = ha.amenities_id
      WHERE ha.hotels_id = h.id),
    ''
  ) AS amenities,
  COALESCE(
    (SELECT string_agg(
       r.name || ' (' || r.room_type || ', sleeps ' || r.capacity || ', ₹' || r.base_price || '/night)',
       '; ' ORDER BY r.room_type
     )
       FROM accounts_room r
      WHERE r.hotel_id = h.id AND r.is_active = true),
    ''
  ) AS rooms
FROM accounts_hotels h
WHERE h.is_active = true
ORDER BY h.rating_avg DESC NULLS LAST, h.hotel_name
"""


def fetch_hotels() -> list[dict]:
    with hotel_conn() as conn, conn.cursor() as cur:
        cur.execute(QUERY)
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "untitled"


def _price_line(h: dict) -> str:
    line = f"₹{float(h['hotel_price']):.0f} per night"
    if h.get("hotel_offer_price") and float(h["hotel_offer_price"]) < float(h["hotel_price"]):
        line += f" (offer: ₹{float(h['hotel_offer_price']):.0f})"
    return line


def _rating(h: dict) -> str:
    if h.get("rating_avg") is not None and h.get("rating_count"):
        return f"{float(h['rating_avg']):.1f}/5 from {h['rating_count']} reviews"
    return "no reviews yet"


def build_hotel_doc(h: dict) -> str:
    name = h["hotel_name"]
    loc = h["hotel_location"]
    lines = [
        "---",
        f"title: {name} — {loc}",
        "category: hotels",
        "---",
        "",
        f"# {name}",
        f"Located in **{loc}**.",
        "",
        "## Quick facts",
        f"- **Location / city:** {loc}",
        f"- **Price:** {_price_line(h)}",
        f"- **Guest rating:** {_rating(h)}",
    ]
    if h.get("amenities"):
        lines.append(f"- **Amenities:** {h['amenities']}")
    if h.get("rooms"):
        lines.append(f"- **Rooms available:** {h['rooms']}")
    if h.get("hotel_description"):
        lines += ["", "## About this hotel", h["hotel_description"].strip()]
    lines += [
        "",
        "## Booking",
        f"To book {name}, visit our website, select dates and a room type, "
        "and complete checkout. See the cancellation policy for refund details.",
    ]
    return "\n".join(lines)


def build_locations_doc(hotels: list[dict]) -> str:
    lines = [
        "---",
        "title: Where Our Hotels Are Located",
        "category: hotels",
        "---",
        "",
        "# Where we operate",
        "",
        "Use this list to answer questions like *which cities do you operate in*, "
        "*do you have a hotel in <city>*, or *recommend a hotel*.",
        "",
    ]
    if not hotels:
        lines.append("_We have no active hotels listed at the moment._")
        return "\n".join(lines)

    by_city: dict[str, list[dict]] = {}
    for h in hotels:
        by_city.setdefault(h["hotel_location"], []).append(h)

    lines.append(f"We currently operate in **{len(by_city)}** location(s): "
                 + ", ".join(sorted(by_city.keys())) + ".")
    lines.append("")
    for city in sorted(by_city.keys()):
        lines.append(f"## {city}")
        for h in by_city[city]:
            lines.append(f"- **{h['hotel_name']}** — {_price_line(h)} · {_rating(h)}")
        lines.append("")
    return "\n".join(lines)


def cleanup_stale_docs(current_sources: set[str]) -> int:
    """Remove DB documents that came from a previous hotel sync but no longer exist."""
    with rag_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, source FROM documents "
            "WHERE source LIKE 'hotel-%' OR source = 'hotels-locations.md' OR source = 'hotels-directory.md'"
        )
        rows = cur.fetchall()
        stale = [r for r in rows if r[1] not in current_sources]
        for doc_id, _ in stale:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    return len(stale)


def main() -> None:
    print("📥 Fetching live hotels from Django DB…")
    hotels = fetch_hotels()
    print(f"   Found {len(hotels)} active hotel(s).")

    HOTELS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-hotel files
    written_sources: set[str] = set()
    for h in hotels:
        fname = f"hotel-{slugify(h['hotel_slug'] or h['hotel_name'])}.md"
        path = HOTELS_DIR / fname
        path.write_text(build_hotel_doc(h), encoding="utf-8")
        written_sources.add(fname)

    # Remove local files for hotels that vanished
    for f in HOTELS_DIR.glob("hotel-*.md"):
        if f.name not in written_sources:
            f.unlink()

    # Locations summary
    LOCATIONS_DOC.write_text(build_locations_doc(hotels), encoding="utf-8")
    written_sources.add(LOCATIONS_DOC.name)

    # Drop old/legacy directory doc if it ever existed locally
    legacy = DOCS_DIR / "hotels-directory.md"
    if legacy.exists():
        legacy.unlink()

    print("\n🔁 Re-ingesting hotel docs…")
    for fname in sorted(written_sources):
        path = (HOTELS_DIR / fname) if fname.startswith("hotel-") else (DOCS_DIR / fname)
        r = ingest_path(path)
        print(f"   {r['action']:>9}  {r.get('chunks', 0):>3} chunks  {r['source']}")

    removed = cleanup_stale_docs(written_sources)
    if removed:
        print(f"\n🧹 Removed {removed} stale hotel doc(s) from DB.")

    print("\n✅ Done. Per-hotel chunks are now searchable in the chatbot KB.")


if __name__ == "__main__":
    main()
