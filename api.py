from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.answer import answer_question, answer_to_dict

load_dotenv()

API_KEY = os.getenv("CHATBOT_API_KEY")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Hotel ChatBot API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


_RATE_BUCKET: dict[str, list[float]] = {}
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = [t for t in _RATE_BUCKET.get(ip, []) if now - t < 60]
    if len(bucket) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests")
    bucket.append(now)
    _RATE_BUCKET[ip] = bucket


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    booking_ref: Optional[str] = Field(default=None, max_length=64)
    booking_email: Optional[str] = Field(default=None, max_length=255)
    session_id: Optional[str] = Field(default=None, max_length=64)
    k: int = Field(default=4, ge=1, le=8)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def chat(req: ChatRequest) -> dict:
    try:
        result = answer_question(
            question=req.question,
            booking_ref=req.booking_ref,
            booking_email=req.booking_email,
            session_id=req.session_id,
            k=req.k,
        )
        return answer_to_dict(result)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e.__class__.__name__}")
