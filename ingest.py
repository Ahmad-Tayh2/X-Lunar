"""
Ingest documents into a vector store to be later used for question-answering.
"""

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Define the paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

# Define Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "gemma3:4b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # Model that actually supports embeddings

def ingest_docs():
    """
    Load documents from the data directory, split them into chunks,
    and store them in a ChromaDB vector store.
    """
    print("Loading documents from", DATA_DIR)
    
    # Load documents from data directory
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Create embeddings using Ollama with an embedding-compatible model
    print(f"Creating embeddings with Ollama using {OLLAMA_EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # Create and persist the vector store
    print(f"Creating vector store at {VECTORSTORE_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    vectorstore.persist()
    print("Vector store created successfully")

if __name__ == "__main__":
    ingest_docs()