from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import time
import pdfplumber

load_dotenv()

VECTOR_STORE_PATH = "vectr_store/faiss_index"  # Path to save/load the FAISS index

def initialize_vector_store():
    """Initializes the vector store, creates embeddings, and saves the index."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    # Extract text from the PDF using pdfplumber
    full_text = ""
    with pdfplumber.open("dataset/OS_NOTES.pdf") as pdf:
        for page in pdf.pages:
            full_text += page.extract_text()

    # Check if text extraction is successful
    if not full_text.strip():
        print("No text could be extracted from the PDF.")
        return

    # Create a list of Document objects
    docs = [Document(page_content=full_text)]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    # Create a vector store with the embeddings
    vector_store = FAISS.from_documents(final_documents, embeddings)

    # Save the vector store for reuse
    vector_store.save_local(VECTOR_STORE_PATH)
    print("Vector Store Created and Saved!")

# Call the function to initialize the vector store
initialize_vector_store()
