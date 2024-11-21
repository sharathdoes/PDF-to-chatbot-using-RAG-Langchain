# Operating System Chatbot

This is a chatbot application built to provide theoretical answers related to operating systems. It uses advanced natural language processing techniques to respond to user queries about various topics within operating systems, such as process management, memory management, file systems, synchronization, and more.

## Features

- **Theoretical Q&A**: Answers any theoretical questions related to operating systems.
- **PDF Document Ingestion**: The chatbot can load and process PDF documents (e.g., textbooks or notes) to enhance its ability to answer questions based on the content of those documents.
- **Vector Store for Fast Retrieval**: Uses FAISS vector store for fast document retrieval and precise answers.
- **Customizable with New Documents**: You can upload new documents to update the knowledge base.
  
## Technologies Used

- **Streamlit**: For creating the interactive web interface.
- **Langchain**: To manage documents and embeddings for information retrieval.
- **FAISS**: For efficient similarity search and document indexing.
- **Hugging Face Embeddings**: For generating vector embeddings from the documents.
- **Groq**: Integrated to enable powerful LLM interactions with `Llama3-8b-8192` model.
- **pdfplumber**: For extracting text from PDF documents.

## Installation

To get started, clone this repository and install the necessary dependencies:

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Set up your `.env` file with necessary API keys:
    - `GROQ_API_KEY=<your-groq-api-key>`
    - You may also need to add other API keys depending on your integration.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. The app will start on your local server (usually `http://localhost:8501`).
3. Use the text input to ask questions about operating systems.
4. Click the "Documents Embedding" button to initialize the knowledge base with documents (e.g., your OS notes in PDF format).
5. The chatbot will answer any theoretical questions based on the documents and its internal model.

## Contributing

Feel free to fork the repository, submit issues, and make pull requests. Contributions are always welcome!

