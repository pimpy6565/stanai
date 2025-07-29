from flask import Flask, request, render_template_string
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # New recommended import
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
app = Flask(__name__)

# Configuration
PDF_PATH = "union_contract.pdf"
MODEL_NAME = "intfloat/e5-small-v2"
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
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>Union Contract Assistant</title></head>
<body>
  <h1>Ask Your Union Contract</h1>
  <form method="POST">
    <input type="text" name="question" size="60" placeholder="Ask something..." required>
    <input type="submit" value="Ask">
  </form>
  {% if answer %}
    <h2>Answer:</h2>
    <p>{{ answer }}</p>
  {% endif %}
</body>
</html>"""

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        query = request.form["question"]
        try:
            answer = qa_chain.run(query)
        except Exception as e:
            answer = f"Error processing query: {str(e)}"
    return render_template_string(HTML_TEMPLATE, answer=answer)
# [Rest of your Flask app code remains the same]