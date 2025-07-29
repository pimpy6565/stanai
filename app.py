from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

PDF_PATH = "union_contract.pdf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Fixed
LLM_MODEL = "google/flan-t5-base"  # Fixed
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
qa_chain = None

def initialize_components():
    if not API_TOKEN:
        raise RuntimeError("Missing Hugging Face API token.")

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = HuggingFaceHub(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=API_TOKEN,
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": 256
        }
    )

    try:
        llm.invoke("Hello")  # quick call to validate
    except Exception as e:
        raise RuntimeError(f"LLM error — check token or model: {e}")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global qa_chain
    q = request.json.get("question", "").strip()
    if not q:
        return jsonify(error="No question provided"), 400

    try:
        if qa_chain is None:
            qa_chain = initialize_components()
        ans = qa_chain.run(q).strip() or "No answer from model."
        return jsonify(answer=ans)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)