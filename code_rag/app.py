import os
import streamlit as st
import pickle
import asyncio
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Groq / Pydantic AI
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

# =========================
# Load environment variables
# =========================
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

# =========================
# Initialize Groq model and Agent
# =========================
try:
    model = GroqModel(
        "llama-3.3-70b-versatile",
        provider=GroqProvider(api_key=API_KEY)
    )
    agent = Agent(model)
except Exception as e:
    st.error(f"Error initializing Groq model/agent: {e}")
    st.stop()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="RAG AI Chat", layout="wide")
st.title("RAG AI Chat Assistant")
st.write("Upload TXT/Markdown files to chat with your documents:")

# =========================
# Upload documents
# =========================
uploaded_files = st.file_uploader(
    "Choose files", type=["txt", "md"], accept_multiple_files=True
)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8")
            docs.append(content)
        except Exception as e:
            st.warning(f"Failed to read {file.name}: {e}")

    if not docs:
        st.warning("No valid documents uploaded.")
    else:
        st.success(f"{len(docs)} document(s) loaded.")

        # Serialize documents
        with open("docs.pkl", "wb") as f:
            pickle.dump(docs, f)
        st.write("Documents are saved and ready for RAG processing.")

        # =========================
        # Initialize session state for chat history
        # =========================
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Input query
        query = st.text_input("Ask a question about your documents:", key="query_input")

        if query:
            try:
                # RAG retrieval: TF-IDF
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(docs)
                query_vec = vectorizer.transform([query])
                sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
                best_doc_idx = sim_scores.argmax()
                best_doc = docs[best_doc_idx]

                # Prepare prompt
                prompt = f"Answer based on this document:\n{best_doc}\nQuestion: {query}"

                # Run agent (async)
                result = asyncio.run(agent.run(prompt))
                response_text = result.output

                # Save to chat history
                st.session_state.chat_history.append(("user", query))
                st.session_state.chat_history.append(("ai", response_text))

            except Exception as e:
                st.error(f"Error processing query: {e}")

        # =========================
        # Display chat history as WhatsApp-like bubbles with black text
        # =========================
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"""
                <div style="background-color:#DCF8C6; color:#000000; padding:10px; border-radius:10px; max-width:70%; margin-bottom:10px; margin-left:auto;">
                <b>You:</b> {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:#FFFFFF; color:#000000; padding:10px; border-radius:10px; max-width:70%; margin-bottom:10px;">
                <b>AI:</b> {message}
                </div>
                """, unsafe_allow_html=True)
