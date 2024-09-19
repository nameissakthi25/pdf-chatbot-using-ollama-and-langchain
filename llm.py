from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    return splits

def create_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings()  # Initialize the embedding function
    vectorstore = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"  # Optional: Directory for local persistence
    )
    vectorstore.add_documents(splits)  # Add processed document splits to the vector store
    return vectorstore

def create_rag_chain():
    llm = OllamaLLM(model="gemma2:2b")
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Provide a clear and concise answer, using "
        "up to three sentences."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return question_answer_chain

def get_response(rag_chain, vectorstore, question):
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, rag_chain)
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]