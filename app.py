from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import time

load_dotenv()

# Load the GROQ API key
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Netflix Movies Query Platform")

# Define the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

VECTOR_STORE_PATH = "netflix_store/faiss_index"  # Path to save/load the FAISS index


def load_vector_store():
    """Loads the vector store from disk."""
    if "vectors" not in st.session_state:
        if os.path.exists(VECTOR_STORE_PATH):
            st.session_state.vectors = FAISS.load_local(
                VECTOR_STORE_PATH,
                HuggingFaceEmbeddings(model_name="BAAI/bge-small-en"),
                allow_dangerous_deserialization=True  # Enable this flag
            )
            st.write("Vector Store Loaded!")
        else:
            st.error("Vector Store not found. Please initialize it first.")


# Load the vector store directly when the app runs
load_vector_store()

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1:
    # Check if vectors are initialized
    if "vectors" not in st.session_state:
        st.error("Vectors are not initialized. Please initialize it first.")
    else:
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        elapsed_time = time.process_time() - start
        
        st.write(response['answer'])  # Only print the answer
        st.write(f"Response Time: {elapsed_time:.2f} seconds")