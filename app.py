from flask import Flask, request, render_template_string
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# === Config ===
PDF_PATH = "union_contract.pdf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
LLM_MODEL = "google/flan-t5-base"
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

qa_chain = None  # Lazy-loaded

def initialize_components():
    logging.info("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    logging.info("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    docs = splitter.split_documents(pages)

    logging.info("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logging.info("Creating vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    logging.info("Initializing hosted LLM...")
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=API_TOKEN
    )

    # test model works
    try:
        response = llm.invoke("Say hello")
        logging.info("Model test response: %s", response)
    except Exception as e:
        logging.error("LLM error: %s", e)
        raise RuntimeError("LLM failed. Check your token or model ID.")

    logging.info("Creating QA chain...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return qa

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
    global qa_chain
    answer = ""

    if request.method == "POST":
        query = request.form["question"]
        try:
            if qa_chain is None:
                qa_chain = initialize_components()

            logging.info("Question: %s", query)
            answer = qa_chain.run(query)

            if not answer or answer.strip() == "":
                logging.warning("Empty model response.")
                answer = "Sorry, the model returned no answer."

        except Exception as e:
            logging.error("Error: %s", e)
            answer = f"Error: {e}"

    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)