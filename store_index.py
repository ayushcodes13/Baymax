"""Build and populate the Baymax Pinecone vector index from PDF documents.

Usage:
    python3 store_index.py [--data-dir DATA] [--index-name NAME]

Requires:
    - PINECONE_API_KEY and GROQ_API_KEY set in the environment (or in a .env file).
    - PDF documents in the data directory (default: ./data/).

This script is idempotent: if the index already exists, it will add documents
to it rather than recreating it.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def load_pdf_files(data_dir: str) -> list[Document]:
    """Load all PDFs from a directory into LangChain Documents."""
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print(f"WARNING: No PDF files found in {data_dir}")
    return documents


def filter_to_minimal_docs(docs: list[Document]) -> list[Document]:
    """Strip all metadata except the source filename to reduce noise."""
    minimal_docs: list[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src},
            )
        )
    return minimal_docs


def text_split(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks suitable for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def download_embeddings():
    """Download and return HuggingFace sentence-transformer embeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_index(
    data_dir: str = "data",
    index_name: str = "baymax-index",
) -> None:
    """Load PDFs, embed, and upsert into a Pinecone index."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("ERROR: PINECONE_API_KEY is not set. Add it to your .env or environment.")
        sys.exit(1)

    print(f"Loading PDFs from {data_dir}/ ...")
    extracted_data = load_pdf_files(data_dir)
    if not extracted_data:
        print("ERROR: No documents to index. Place PDF files in the data directory.")
        sys.exit(1)
    print(f"Loaded {len(extracted_data)} document(s).")

    minimal_docs = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(minimal_docs)
    print(f"Split into {len(text_chunks)} chunk(s).")

    print(f"Loading embeddings ({EMBEDDING_MODEL}) ...")
    embedding = download_embeddings()

    print(f"Connecting to Pinecone (index: {index_name}) ...")
    pc = Pinecone(api_key=pinecone_api_key)

    if not pc.has_index(index_name):
        print(f"Creating index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Index '{index_name}' already exists; adding documents to it.")

    print("Upserting documents into the vector store ...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embedding,
        index_name=index_name,
    )
    print(f"Done. {len(text_chunks)} chunk(s) indexed into '{index_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Baymax Pinecone vector index.")
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing source PDFs (default: data/)"
    )
    parser.add_argument(
        "--index-name", default="baymax-index", help="Pinecone index name (default: baymax-index)"
    )
    args = parser.parse_args()
    build_index(data_dir=args.data_dir, index_name=args.index_name)
