from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
try:
    HF_TOKEN = HF_TOKEN or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
except (FileNotFoundError, KeyError):
    pass

DATA_DIR = Path(__file__).parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
PROFILE_FILE = DATA_DIR / "profile.json"
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

MODELS = {
    "Gemma 2 2B Instruct": "google/gemma-2-2b-it",
    "Qwen2.5 7B Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral 7B Instruct v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama 3.1 8B Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Phi-3 Mini 4K Instruct": "microsoft/Phi-3-mini-4k-instruct",
}

BASE_PROMPT = """You are Nitin's AI Assistant, an expert AI built by Nitin Sharwal.
Your job is to give the user the most accurate, useful, and well-reasoned answer possible.

# Core principles
- Accuracy first: never invent facts, APIs, numbers, names, or citations. If unsure, say so.
- Think before you answer: reason silently; share reasoning only when it genuinely helps.
- Be concise by default, thorough when needed. Match length to complexity.
- Direct and decisive: when asked for a recommendation, give one and justify it briefly.

# Style
- Plain, professional English. No filler ("Certainly!", "Great question!").
- Markdown: short paragraphs, bullet lists, fenced code blocks with language tags.
- Code: runnable, idiomatic, production-quality. Comment only when intent is non-obvious.
- Math: LaTeX ($...$ inline, $$...$$ block) when it improves clarity.

# Conversation
- Use prior turns and the user's profile as context. Personalize naturally — don't recite their profile back.
- On follow-ups, modify the previous answer rather than restarting.
- If ambiguous, ask one focused clarifying question; otherwise pick the most reasonable interpretation and proceed.

# Boundaries
- Decline illegal, harmful, or deceptive requests; offer a safer alternative when possible.
- For medical, legal, or financial topics: give general info and recommend a professional.
- Never claim internet/file access unless a tool is actually available.

Goal: every response should feel like it came from a sharp, senior expert who respects the user's time."""


def load_profile() -> dict:
    if PROFILE_FILE.exists():
        try:
            return json.loads(PROFILE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "name": "",
        "role": "",
        "expertise": "",
        "interests": "",
        "tone": "Professional, concise",
        "language": "English",
        "extra": "",
    }


def save_profile(p: dict) -> None:
    PROFILE_FILE.write_text(json.dumps(p, indent=2))


def profile_block(p: dict) -> str:
    fields = [
        ("Name", p.get("name")),
        ("Role", p.get("role")),
        ("Expertise", p.get("expertise")),
        ("Interests", p.get("interests")),
        ("Preferred tone", p.get("tone")),
        ("Preferred language", p.get("language")),
        ("Other notes", p.get("extra")),
    ]
    lines = [f"- {k}: {v.strip()}" for k, v in fields if v and v.strip()]
    if not lines:
        return ""
    return "\n\n# User profile (use to personalize answers)\n" + "\n".join(lines)


def list_sessions() -> list[dict]:
    items = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            items.append({
                "id": f.stem,
                "title": data.get("title", "Untitled"),
                "updated": data.get("updated", ""),
                "path": f,
            })
        except json.JSONDecodeError:
            continue
    items.sort(key=lambda x: x["updated"], reverse=True)
    return items


def load_session(sid: str) -> dict:
    f = SESSIONS_DIR / f"{sid}.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"title": "New chat", "messages": [], "updated": ""}


def save_session(sid: str, title: str, messages: list[dict]) -> None:
    payload = {
        "title": title,
        "messages": messages,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    (SESSIONS_DIR / f"{sid}.json").write_text(json.dumps(payload, indent=2))


def delete_session(sid: str) -> None:
    f = SESSIONS_DIR / f"{sid}.json"
    if f.exists():
        f.unlink()


def derive_title(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            t = m["content"].strip().splitlines()[0]
            return (t[:40] + "…") if len(t) > 40 else t
    return "New chat"


@st.cache_resource(show_spinner=False)
def load_model(repo_id: str, temperature: float, max_new_tokens: int):
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing in .env")
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return ChatHuggingFace(llm=llm)


def build_messages(messages: list[dict], system_text: str) -> list[dict]:
    msgs = [dict(m) for m in messages]
    sp = (system_text or "").strip()
    if sp and msgs and msgs[0]["role"] == "user":
        msgs[0] = {"role": "user", "content": f"{sp}\n\n{msgs[0]['content']}"}
    return msgs


st.set_page_config(page_title="ChatBot", page_icon="🤖", layout="wide")

if "profile" not in st.session_state:
    st.session_state.profile = load_profile()
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []
    st.session_state.title = "New chat"

if "streaming_buffer" in st.session_state:
    partial = st.session_state.pop("streaming_buffer")
    if partial.strip():
        st.session_state.messages.append({
            "role": "assistant",
            "content": partial.rstrip() + "\n\n_⏹ Stopped by user._",
        })
        if st.session_state.title in ("New chat", "Untitled"):
            st.session_state.title = derive_title(st.session_state.messages)
        save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)

with st.sidebar:
    st.header("💬 Chats")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.session_state.title = "New chat"
        st.rerun()

    for s in list_sessions():
        active = s["id"] == st.session_state.session_id
        label = ("🟢 " if active else "• ") + s["title"]
        c1, c2 = st.columns([5, 1])
        if c1.button(label, key=f"open_{s['id']}", use_container_width=True):
            data = load_session(s["id"])
            st.session_state.session_id = s["id"]
            st.session_state.messages = data["messages"]
            st.session_state.title = data["title"]
            st.rerun()
        if c2.button("🗑", key=f"del_{s['id']}"):
            delete_session(s["id"])
            if active:
                st.session_state.session_id = uuid.uuid4().hex[:12]
                st.session_state.messages = []
                st.session_state.title = "New chat"
            st.rerun()

    st.divider()
    with st.expander("👤 Profile", expanded=False):
        p = st.session_state.profile
        p["name"] = st.text_input("Name", p.get("name", ""))
        p["role"] = st.text_input("Role / occupation", p.get("role", ""))
        p["expertise"] = st.text_area("Expertise / skills", p.get("expertise", ""), height=70)
        p["interests"] = st.text_area("Interests / projects", p.get("interests", ""), height=70)
        p["tone"] = st.text_input("Preferred tone", p.get("tone", "Professional, concise"))
        p["language"] = st.text_input("Preferred language", p.get("language", "English"))
        p["extra"] = st.text_area("Other notes (anything I should remember)", p.get("extra", ""), height=90)
        if st.button("💾 Save profile", use_container_width=True):
            save_profile(p)
            st.success("Profile saved.")

    with st.expander("⚙️ Model & generation", expanded=False):
        model_label = st.selectbox("Model", list(MODELS.keys()), index=list(MODELS).index("Llama 3.1 8B Instruct"))
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max new tokens", 64, 2048, 512, 64)

    with st.expander("📜 System prompt", expanded=False):
        custom_prompt = st.text_area("Base instructions", BASE_PROMPT, height=300)

    if st.session_state.messages:
        rows = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download chat (CSV)",
            csv,
            file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.title(f"🤖 {st.session_state.title}")
st.caption(f"Model: `{MODELS[model_label]}`  •  temp {temperature}  •  max {max_tokens}")

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    system_text = custom_prompt + profile_block(st.session_state.profile)
    stop_slot = st.empty()
    stop_slot.button("⏹ Stop generating", key=f"stop_{uuid.uuid4().hex[:6]}", use_container_width=True)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        st.session_state.streaming_buffer = ""
        full = ""
        try:
            model = load_model(MODELS[model_label], temperature, max_tokens)
            payload = build_messages(st.session_state.messages, system_text)
            try:
                for chunk in model.stream(payload):
                    piece = getattr(chunk, "content", "") or ""
                    full += piece
                    st.session_state.streaming_buffer = full
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full or "_(empty response)_")
            except (AttributeError, NotImplementedError):
                result = model.invoke(payload)
                full = getattr(result, "content", str(result))
                placeholder.markdown(full)
        except Exception as e:
            full = f"⚠️ Error: {e}"
            placeholder.error(full)

    st.session_state.pop("streaming_buffer", None)
    stop_slot.empty()
    st.session_state.messages.append({"role": "assistant", "content": full})
    if st.session_state.title in ("New chat", "Untitled"):
        st.session_state.title = derive_title(st.session_state.messages)
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)
    st.rerun()
