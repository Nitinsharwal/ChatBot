from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from rag.retrieve import (
    Chunk,
    format_booking,
    log_support_gap,
    lookup_booking,
    retrieve,
)

load_dotenv()

DEFAULT_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

RAG_PROMPT = """You are the official support assistant for our hotel booking platform.
Your job is to answer guest questions about bookings, policies, payments, rooms, and account issues.

# Hard rules
- Answer ONLY using the information in the "Hotel knowledge base" and "Guest booking" sections below.
- Do NOT use outside knowledge. Do NOT invent prices, dates, policies, or contact details.
- Cite the supporting snippets inline as [1], [2], etc., matching the numbered sources.
- If the answer is not in the context, reply exactly:
  "I don't have that information in our knowledge base. Let me connect you with a human agent — please email support@hotel.example or call +1-800-HOTEL-00."
- If the guest asks about their own booking and a "Guest booking" section is present, use it to personalize the answer.

# Style
- Be warm, concise, and professional. Use short paragraphs and bullet lists.
- Confirm what the guest asked, then give the answer, then any next step they need to take.
"""

HANDOFF_MESSAGE = (
    "I don't have that information in our knowledge base. "
    "Let me connect you with a human agent — please email "
    "support@hotel.example or call +1-800-HOTEL-00."
)


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
        return Answer(answer=HANDOFF_MESSAGE, sources=[], handoff=True, booking_found=False)

    context = _format_context(chunks, booking_text)
    prompt = f"{RAG_PROMPT}\n\n{context}\n\n# Guest question\n{question}"

    chat = _get_model(model, temperature, max_tokens)
    result = chat.invoke([{"role": "user", "content": prompt}])
    text = getattr(result, "content", str(result)).strip()

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
    return Answer(answer=text, sources=sources, handoff=False, booking_found=booking_found)


def answer_to_dict(a: Answer) -> dict:
    return {
        "answer": a.answer,
        "sources": [asdict(s) for s in a.sources],
        "handoff": a.handoff,
        "booking_found": a.booking_found,
    }
