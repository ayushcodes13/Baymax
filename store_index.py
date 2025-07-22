### Loading the Book
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()
import os
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    
    documents = loader.load()
    return documents    
extracted_data = load_pdf_files("data")
extracted_data
len(extracted_data)

def filter_to_minimal_docs(docs:List[Document]) -> List[Document]:
    
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadeta={"source": src,}
            )
        )
    return minimal_docs
minimal_docs = filter_to_minimal_docs(extracted_data)
minimal_docs 

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk
text_chunk = text_split(minimal_docs)
print(f"Number of text chunks: {len(text_chunk)}")
text_chunk
### Embedding
from langchain.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings

embedding = download_embeddings()
embedding
vector = embedding.embed_query("Hello world")
vector
print(f"Vector length: {len(vector)}")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
from pinecone import Pinecone
pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)
pc
from pinecone import ServerlessSpec
index_name = "baymax-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",  
        spec=ServerlessSpec(cloud ="aws", region="us-east-1")
    )
    
index = pc.Index(index_name)
from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name,
)

from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_existing_index(
    index_name =index_name,
    embedding=embedding
)
