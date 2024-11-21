from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import time
import pdfplumber

load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Chatgroq With Llama3 Demo")

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

def vector_embedding():
    if "vectors" not in st.session_state:
        # Use HuggingFace Embeddings with BAAI model
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        
        # Extract text from the PDF using pdfplumber
        full_text = ""
        with pdfplumber.open("./OS_NOTES.pdf") as pdf:
            for page in pdf.pages:
                full_text += page.extract_text()

        # Check if text extraction is successful
        if not full_text.strip():
            st.error("No text could be extracted from the PDF.")
            return

        # Create a list of Document objects
        st.session_state.docs = [Document(page_content=full_text)]

        # Split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Create a vector store with the embeddings
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

        st.write("Vector Store Created!")  # Confirm the vector store is created

# Input field for question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to initialize document embeddings
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if prompt1:
    # Check if vectors are initialized
   if "vectors" not in st.session_state:
        st.error("Vectors are not initialized. Please click 'Documents Embedding' to initialize the vector store.")
   else:
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])  # Only print the answer