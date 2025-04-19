"""
Ingest documents into multiple language-specific vector stores to be later used for question-answering.
"""

import os
import time
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader, 
    PyPDFLoader,
    Docx2txtLoader)
from langchain_ollama import OllamaEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document

# Define the paths
DATA_DIR_AR = os.path.join(os.path.dirname(__file__), "data/ar")
DATA_DIR_FR = os.path.join(os.path.dirname(__file__), "data/fr")
DATA_DIR_ENG = os.path.join(os.path.dirname(__file__), "data/eng")

VECTORSTORE_DIR_AR = os.path.join(os.path.dirname(__file__), "vectorstore_ar")
VECTORSTORE_DIR_FR = os.path.join(os.path.dirname(__file__), "vectorstore_fr")
VECTORSTORE_DIR_ENG = os.path.join(os.path.dirname(__file__), "vectorstore_eng")

# Define Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "gemma3:4b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # Model that actually supports embeddings

# Custom loader function that handles individual files
def load_single_document(file_path):
    """Load a single document from a file path."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.docx' or file_ext == '.doc':
            loader = Docx2txtLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path)
        else:
            print(f"Unsupported file extension: {file_ext}")
            return None
        
        return loader.load()
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None

def process_language_directory(data_dir, vectorstore_dir, lang_name):
    """Process all documents in a language-specific directory and create a vector store."""
    print(f"\n=== Processing {lang_name} documents from {data_dir} ===")
    
    # Create embeddings using Ollama
    print(f"Initializing Ollama embeddings with model {OLLAMA_EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # Initialize or load the vector store
    print(f"Initializing vector store at {vectorstore_dir}")
    vectorstore = None
    if os.path.exists(vectorstore_dir):
        print("Loading existing vector store...")
        vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Process all files in the data directory
    total_docs = 0
    total_chunks = 0
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: {lang_name} directory {data_dir} does not exist. Skipping.")
        return 0, 0
    
    # Get all files in the data directory
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in ['.pdf', '.txt', '.doc', '.docx']:
                all_files.append(file_path)
    
    print(f"Found {len(all_files)} {lang_name} files to process")
    
    if len(all_files) == 0:
        print(f"No {lang_name} files found. Skipping.")
        return 0, 0
    
    # Process each file individually
    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        print(f"\nProcessing file {i+1}/{len(all_files)}: {file_name}")
        
        # Load the document with individual error handling
        docs = load_single_document(file_path)
        
        if not docs:
            print(f"  Skipping {file_name} due to loading errors")
            continue
            
        print(f"  Successfully loaded {file_name}")
        
        # Process each document from this file
        for j, doc in enumerate(docs):
            print(f"  Processing document {j+1}/{len(docs)} from {file_name}")
            
            # Split the document into chunks
            doc_chunks = text_splitter.split_documents([doc])
            print(f"    Split into {len(doc_chunks)} chunks")
            
            if not doc_chunks:
                print(f"    Warning: No chunks created for this document. It might be empty.")
                continue
            
            # Create or update the vector store with this document's chunks
            print(f"    Adding chunks to vector store...")
            start_time = time.time()
            
            try:
                if vectorstore is None:
                    # First document, create the vector store
                    vectorstore = Chroma.from_documents(
                        documents=doc_chunks,
                        embedding=embeddings,
                        persist_directory=vectorstore_dir
                    )
                else:
                    # Add to existing vector store
                    vectorstore.add_documents(doc_chunks)
                
                # No need to explicitly call persist() as Chroma with persist_directory 
                # automatically handles persistence
                print("    Vector store updated - persistence handled by Chroma")
                
                elapsed = time.time() - start_time
                print(f"    Added to vector store in {elapsed:.2f} seconds")
                
                total_chunks += len(doc_chunks)
                total_docs += 1
            except Exception as e:
                print(f"    Error adding chunks to vector store: {str(e)}")
    
    print(f"\n{lang_name} processing complete: Ingested {total_docs} documents with a total of {total_chunks} chunks")
    return total_docs, total_chunks

def ingest_docs():
    """
    Load documents from language-specific directories, split them into chunks,
    and store them in separate language-specific ChromaDB vector stores.
    """
    print("Starting multi-language document ingestion...")
    
    # Process each language directory
    ar_docs, ar_chunks = process_language_directory(DATA_DIR_AR, VECTORSTORE_DIR_AR, "Arabic")
    fr_docs, fr_chunks = process_language_directory(DATA_DIR_FR, VECTORSTORE_DIR_FR, "French")
    eng_docs, eng_chunks = process_language_directory(DATA_DIR_ENG, VECTORSTORE_DIR_ENG, "English")
    
    # Summary
    total_docs = ar_docs + fr_docs + eng_docs
    total_chunks = ar_chunks + fr_chunks + eng_chunks
    
    print("\n=== INGESTION SUMMARY ===")
    print(f"Arabic: {ar_docs} documents, {ar_chunks} chunks")
    print(f"French: {fr_docs} documents, {fr_chunks} chunks")
    print(f"English: {eng_docs} documents, {eng_chunks} chunks")
    print(f"Total: {total_docs} documents, {total_chunks} chunks")
    
    if total_docs == 0:
        print("No documents were processed. Please check your data directories.")
    else:
        print("Vector stores created and persisted successfully")

if __name__ == "__main__":
    ingest_docs()