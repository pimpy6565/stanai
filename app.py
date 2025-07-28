import os
from flask import Flask, request, render_template_string
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # More memory-efficient than Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Configuration
PDF_PATH = "union_contract.pdf"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO = "google/flan-t5-small"  # Smaller model for production

# Initialize components once at startup
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
    vectorstore = FAISS.from_documents(docs, embeddings)  # FAISS is more memory-efficient
    
    print("Loading LLM...")
    llm = HuggingFaceHub(
        repo_id=LLM_REPO,
        model_kwargs={"temperature": 0.2, "max_length": 256}
    )
    
    print("Creating QA chain...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    return qa

# Initialize during app startup
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))