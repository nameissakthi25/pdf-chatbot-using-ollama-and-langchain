import streamlit as st
import tempfile
import os
from llm import load_and_process_pdf, create_vectorstore, create_rag_chain, get_response

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚")

st.title("PDF Q&A Chatbot")

# Initialize session state for vector store and chain
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and st.session_state.vectorstore is None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Process the PDF only once
    with st.spinner("Processing PDF..."):
        splits = load_and_process_pdf(tmp_file_path)
        st.session_state.vectorstore = create_vectorstore(splits)
        st.session_state.rag_chain = create_rag_chain()

    st.success("PDF processed successfully! Now you can ask questions.")

    # Clean up the temporary file
    os.unlink(tmp_file_path)

# Question input
if st.session_state.vectorstore is not None:
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Generating answer..."):
            answer = get_response(st.session_state.rag_chain, st.session_state.vectorstore, question)
        st.write("Answer:", answer)

else:
    st.info("Please upload a PDF file to get started.")
