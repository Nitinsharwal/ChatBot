from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import json

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

DEFAULT_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "8221985564")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "nomapvtltd@gmail.com")
LOGIN_URL = os.getenv("LOGIN_URL", "https://sharwal-nitin-hotel.vercel.app/accounts/login/")

HANDOFF_MESSAGE = (
    f"I don't have that information in our knowledge base — but you can reach us directly:\n\n"
    f"📞 **Phone:** {SUPPORT_PHONE}\n"
    f"📧 **Email:** {SUPPORT_EMAIL}\n\n"
    f"For anything booking-specific, please sign in to your account:\n"
    f"👉 [Login here]({LOGIN_URL})\n\n"
    f"**How to log in:**\n"
    f"1. Click the login link above.\n"
    f"2. Enter your registered email and password, or use **Sign in with Google**.\n"
    f"3. Open **My Bookings** to view and manage your reservations."
)

RAG_PROMPT = f"""You are the official support assistant for our hotel booking platform.
Your job is to answer guest questions about bookings, policies, payments, rooms, and account issues.

# Hard rules
- Answer ONLY using the information in the "Hotel knowledge base" and "Guest booking" sections below.
- Do NOT use outside knowledge. Do NOT invent prices, dates, policies, or contact details.
- Cite the supporting snippets inline as [1], [2], etc., matching the numbered sources.
- If the answer is not in the context, reply EXACTLY with this message (no edits, no additions):
---
{HANDOFF_MESSAGE}
---
- If the guest asks about their own booking and a "Guest booking" section is present, use it to personalize the answer.

# Style
- Be warm, concise, and professional. Use short paragraphs and bullet lists.
- Confirm what the guest asked, then give the answer, then any next step they need to take.
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
def _get_model(repo_id: str, temperature: float, max_tokens: int) -> ChatHuggingFace:
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


def answer_question(
    question: str,
    booking_ref: Optional[str] = None,
    booking_email: Optional[str] = None,
    session_id: Optional[str] = None,
    k: int = 4,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer(answer="Please ask a question.", sources=[], handoff=False, booking_found=False)

    chunks = retrieve(question, k=k)

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
    prompt = f"{RAG_PROMPT}\n\n{context}\n\n# Guest question\n{question}"

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
