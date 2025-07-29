from flask import Flask, request, render_template_string
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # New recommended import
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)

# Configuration
PDF_PATH = "union_contract.pdf"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO = "google/flan-t5-small"

# Initialize components
def initialize_components():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    docs = splitter.split_documents(pages)
    
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    print("Loading LLM...")
    llm = HuggingFaceHub(
        repo_id=LLM_REPO,
        model_kwargs={"temperature": 0.2, "max_length": 256},
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    
    print("Creating QA chain...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    return qa

qa_chain = initialize_components()

# [Rest of your Flask app code remains the same]