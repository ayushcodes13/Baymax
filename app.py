"""Baymax — a wellbeing conversation prototype, not a medical service.

Flask + LangChain RAG web app. Requires PINECONE_API_KEY and GROQ_API_KEY
in the environment (or in a .env file). See README.md for setup instructions.
"""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

from src.helper import download_embeddings
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)

# --- Configuration -----------------------------------------------------------

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "baymax-index")

# --- Error handling for missing keys ----------------------------------------

_missing_keys = []
if not PINECONE_API_KEY:
    _missing_keys.append("PINECONE_API_KEY")
if not GROQ_API_KEY:
    _missing_keys.append("GROQ_API_KEY")

if _missing_keys:
    print(f"WARNING: Missing environment variables: {', '.join(_missing_keys)}")
    print("The app will return 503 errors until these are set.")


# --- Lazy initialization (errors handled gracefully) ------------------------

_embeddings = None
_rag_chain = None


def _init_chain():
    """Build the RAG chain. Called lazily on first request."""
    global _embeddings, _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    if _missing_keys:
        raise RuntimeError(
            f"Missing required keys: {', '.join(_missing_keys)}. "
            "Set them in your environment or .env file."
        )

    _embeddings = download_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=_embeddings,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    _rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return _rag_chain


# --- Routes ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        chain = _init_chain()
    except (RuntimeError, Exception) as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        response = chain.invoke({"input": msg})
        return str(response["answer"])
    except Exception as exc:
        return jsonify({"error": f"Could not get a reply: {exc}"}), 502


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8080)), debug=os.environ.get("FLASK_DEBUG", "1") == "1")
