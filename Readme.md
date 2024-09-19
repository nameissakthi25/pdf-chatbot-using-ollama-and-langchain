## PDF Q&A Chatbot
This project is a PDF-based chatbot that allows users to upload a PDF, process its content, and ask questions related to the document using the Langchain library and OllamaLLM (gemma2:2b). The chatbot utilizes Streamlit for the user interface and Chroma for vector storage, enabling efficient document processing and retrieval-based question-answering.

## Features
* PDF Upload: Users can upload a PDF file which will be processed and split into smaller chunks for efficient retrieval.
* Persisted Vector Store: The PDF is processed once and stored in a vector database, allowing for fast responses to user queries without reprocessing the document.
* Q&A: Users can ask questions related to the uploaded PDF, and the chatbot will provide concise answers based on the document content.
* Fast PDF Processing: The use of vector storage ensures that once the PDF is processed, users can query it efficiently.
Tech Stack
* Langchain: For building the Q&A chatbot pipeline and vector-based document retrieval.
* OllamaLLM: The LLM used for generating answers to user queries.
* Chroma: Vector store for document embeddings, enabling fast document retrieval.

### Create and activate a virtual environment:
```bash
python3 -m venv ven
```
### Install the required libraries:
```bash
pip install -r requirements.txt
```

### Run the application:
```bash
streamlit run app.py
```