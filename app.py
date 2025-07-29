from flask import Flask, request, render_template_string
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
PDF_PATH = "union_contract.pdf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
LLM_MODEL = "google/flan-t5-base"
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

qa_chain = None  # Lazy load on first request

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
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=API_TOKEN,
        model=EMBEDDING_MODEL
    )

    logging.info("Creating vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    logging.info("Loading hosted LLM...")
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=API_TOKEN,
        temperature=0.2,
        max_new_tokens=256
    )

    logging.info("Testing LLM output...")
    try:
        test_response = llm.invoke("Say hello")
        logging.info(f"Model test response: {test_response}")
    except Exception as e:
        logging.error(f"LLM error: {e}")
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

            logging.info(f"Question submitted: {query}")
            answer = qa_chain.run(query)

            if not answer or answer.strip() == "":
                logging.info("Empty model response.")
                answer = "Sorry, the model returned no answer."

            logging.info(f"Answer received: {answer}")

        except Exception as e:
            logging.error(f"Error: {e}")
            answer = f"Error: {e}"

    return render_template_string(HTML_TEMPLATE, answer=answer)

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # For Render or local fallback
    app.run(host="0.0.0.0", port=port)