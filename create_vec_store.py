from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

VECTOR_STORE_PATH = "netflix_store/faiss_index"  # Path to save/load the FAISS index

def initialize_vector_store():
    """Initializes the vector store, creates embeddings, and saves the index."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    # Load the CSV file
    csv_file_path = "dataset/netflix_titles_short.csv"
    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} does not exist.")
        return

    # Read CSV and concatenate relevant text columns
    try:
        df = pd.read_csv(csv_file_path)
        full_text = " ".join(df.fillna("").apply(lambda row: " ".join(row.values.astype(str)), axis=1))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if text extraction is successful
    if not full_text.strip():
        print("No text could be extracted from the CSV.")
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
