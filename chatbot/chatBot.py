from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
import os

def main_model():
    load_dotenv()
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    model = ChatHuggingFace(llm=llm)
    return model


SYSTEM_PROMPT = """
You are an intelligent and concise AI assistant developed by Nitin Sharwal.
Your goal is to provide clear, accurate, and helpful answers in as few lines as possible use less words and lines to show answers.
Always maintain context from previous messages to make responses relevant and coherent.
If the user asks follow-up questions, refer to the chat history naturally without repeating information.
"""

st.set_page_config(page_title="ChatBot By Nitin Sharwal", page_icon="🤖")
st.title("ChatBot By Nitin Sharwal !!")

if "model" not in st.session_state:
    st.session_state.model = main_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

user_input = st.text_input("How can I help you today?", key="user_input")

if st.button("Run"):
    if user_input.lower() == "exit":
        st.write("Goodbye..!")
    elif user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        model = st.session_state.model
        result = model.invoke(st.session_state.chat_history)

        reply = result.content if hasattr(result, "content") else str(result)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

for message in st.session_state.chat_history[1:]:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**AI:** {message['content']}")
    