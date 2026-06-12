from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import json
import re

from rag.db import rag_conn
from rag.retrieve import (
    Chunk,
    format_booking,
    log_support_gap,
    lookup_booking,
    retrieve,
)


def _log_conversation(
    session_id: str | None,
    question: str,
    answer_text: str,
    sources: list,
    handoff: bool,
    booking_ref: str | None,
    booking_email: str | None,
    booking_found: bool,
) -> None:
    try:
        payload = [
            {"index": s.index, "title": s.title, "source": s.source,
             "category": s.category, "score": s.score}
            for s in sources
        ]
        with rag_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations "
                "(session_id, question, answer, sources, handoff, booking_ref, booking_email, booking_found) "
                "VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s)",
                (
                    session_id or "anon",
                    question[:2000],
                    answer_text[:8000],
                    json.dumps(payload),
                    handoff,
                    (booking_ref or None),
                    (booking_email or None),
                    booking_found,
                ),
            )
    except Exception:
        pass

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if os.getenv("GOOGLE_API_KEY") else "huggingface").lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                          if LLM_PROVIDER == "gemini"
                          else "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "8221985564")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "nomapvtltd@gmail.com")
LOGIN_URL = os.getenv("LOGIN_URL", "https://sharwal-nitin-hotel.vercel.app/accounts/login/")
BOT_NAME = os.getenv("BOT_NAME", "Kelly")
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))
BOOKING_REF_RE = re.compile(r"\b(HTL[-\s]?\d{3,8})\b", re.IGNORECASE)

HANDOFF_MESSAGE = (
    f"I don't have that in our knowledge base yet — but you can reach our team directly:\n\n"
    f"📞 **Phone:** {SUPPORT_PHONE}\n"
    f"📧 **Email:** {SUPPORT_EMAIL}\n\n"
    f"For anything booking-specific, please sign in to your account:\n"
    f"👉 [Login here]({LOGIN_URL})\n\n"
    f"**How to log in:**\n"
    f"1. Click the login link above.\n"
    f"2. Enter your registered email and password, or use **Sign in with Google**.\n"
    f"3. Open **My Bookings** to view and manage your reservations."
)

RAG_PROMPT = f"""You are **{BOT_NAME}**, the dedicated AI concierge for our hotel booking platform.
You help guests with bookings, cancellations, payments, room/amenity questions, account help, and recommending hotels.

# Identity
- Always speak as {BOT_NAME} — warm, professional, and crisp.
- Never reveal that you're an LLM, what model you are, or how you were built. If asked, say "I'm {BOT_NAME}, your hotel concierge."
- Use the guest's name if it's in the booking context. Don't recite their profile back.

# Hard rules (grounding)
- Answer ONLY using the "Hotel knowledge base" and "Guest booking" sections below, plus the recent conversation.
- Do NOT invent hotels, cities, prices, dates, policies, or contact details. If a fact isn't in the context, say so honestly.
- Cite supporting snippets inline as [1], [2], etc., matching the numbered sources.
- If the question is about a *city* and no hotel in the knowledge base is in that city, say clearly which cities we DO operate in, and offer those.

# Booking-specific behavior
- If a "Guest booking" section is present, use it to personalize the answer (status, dates, hotel, amount).
- If the guest asks about *their* booking but no "Guest booking" section is present, ask once for their booking reference (format: HTL-XXXX). Don't re-ask if they ignore it.

# Conversation
- Use the recent chat history to resolve references ("the cheaper one", "that hotel", "what about the suite?").
- For follow-ups, modify your previous answer instead of repeating it.
- Match the guest's language. If they write in Hindi, reply in Hindi.

# Style
- Concise by default. One short paragraph or 3–5 bullets, not an essay.
- Use Markdown: **bold** for emphasis, bullet lists for options, no headings inside a chat reply.
- End with a clear next step ("Would you like me to confirm?", "Want me to find dates?").

# When you genuinely can't answer
If after using all context you still cannot answer factually, reply EXACTLY with this — no edits, no additions:
---
{HANDOFF_MESSAGE}
---
"""


@dataclass
class Source:
    index: int
    title: str
    source: str
    category: str
    score: float
    content: str


@dataclass
class Answer:
    answer: str
    sources: list[Source]
    handoff: bool
    booking_found: bool


@lru_cache(maxsize=4)
def _get_model(repo_id: str, temperature: float, max_tokens: int):
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set (LLM_PROVIDER=gemini)")
        return ChatGoogleGenerativeAI(
            model=repo_id,
            google_api_key=key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set")
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        task="text-generation",
        temperature=temperature,
        max_new_tokens=max_tokens,
    )
    return ChatHuggingFace(llm=llm)


def _format_context(chunks: list[Chunk], booking_text: Optional[str]) -> str:
    parts: list[str] = []
    if chunks:
        parts.append("# Hotel knowledge base (numbered sources)")
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] {c.title} — {c.source} (category: {c.category})\n{c.content}")
    if booking_text:
        parts.append("# Guest booking (live from our system)\n" + booking_text)
    return "\n\n".join(parts)


def _extract_booking_ref(text: str) -> Optional[str]:
    m = BOOKING_REF_RE.search(text or "")
    if not m:
        return None
    return m.group(1).upper().replace(" ", "-")


def _format_history(history: Optional[list[dict]]) -> str:
    if not history:
        return ""
    trimmed = history[-(HISTORY_TURNS * 2):]
    lines = []
    for m in trimmed:
        role = "Guest" if m.get("role") == "user" else BOT_NAME
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _retrieval_query(question: str, history: Optional[list[dict]]) -> str:
    """Join the last user turn with the new question so vague follow-ups
    ('what about the other one?') still retrieve relevant chunks."""
    if not history:
        return question
    prev_user = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"),
        "",
    )
    return f"{prev_user} {question}".strip() if prev_user else question


def answer_question(
    question: str,
    booking_ref: Optional[str] = None,
    booking_email: Optional[str] = None,
    session_id: Optional[str] = None,
    history: Optional[list[dict]] = None,
    k: int = 5,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer(answer="Please ask a question.", sources=[], handoff=False, booking_found=False)

    if not booking_ref:
        booking_ref = _extract_booking_ref(question)

    retrieval_q = _retrieval_query(question, history)
    chunks = retrieve(retrieval_q, k=k)

    booking_text: Optional[str] = None
    booking_found = False
    if booking_ref and booking_email:
        b = lookup_booking(booking_ref, booking_email)
        if b:
            booking_text = format_booking(b)
            booking_found = True

    if not chunks and not booking_text:
        try:
            log_support_gap(question, session_id)
        except Exception:
            pass
        result = Answer(answer=HANDOFF_MESSAGE, sources=[], handoff=True, booking_found=False)
        _log_conversation(session_id, question, result.answer, result.sources, True,
                          booking_ref, booking_email, False)
        return result

    context = _format_context(chunks, booking_text)
    history_block = _format_history(history)
    parts = [RAG_PROMPT, context]
    if history_block:
        parts.append("# Recent conversation\n" + history_block)
    parts.append(f"# Current guest question\n{question}")
    prompt = "\n\n".join(p for p in parts if p)

    chat = _get_model(model, temperature, max_tokens)
    chat_result = chat.invoke([{"role": "user", "content": prompt}])
    text = getattr(chat_result, "content", str(chat_result)).strip()

    sources = [
        Source(
            index=i,
            title=c.title,
            source=c.source,
            category=c.category,
            score=c.score,
            content=c.content,
        )
        for i, c in enumerate(chunks, 1)
    ]
    final = Answer(answer=text, sources=sources, handoff=False, booking_found=booking_found)
    _log_conversation(session_id, question, text, sources, False,
                      booking_ref, booking_email, booking_found)
    return final


def answer_to_dict(a: Answer) -> dict:
    return {
        "answer": a.answer,
        "sources": [asdict(s) for s in a.sources],
        "handoff": a.handoff,
        "booking_found": a.booking_found,
    }
