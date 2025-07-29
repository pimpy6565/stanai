from flask import Flask, request, render_template_string
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Configuration
PDF_PATH = "union_contract.pdf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
LLM_MODEL = "google/flan-t5-small"
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

qa_chain = None  # Lazy loaded later

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

    print("Loading hosted embeddings...")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=API_TOKEN,
        model_name=EMBEDDING_MODEL
    )

    print("Creating vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("Loading hosted LLM...")
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=API_TOKEN,
        temperature=0.2,
        max_length=256
    )

    print("Creating QA chain...")
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
            answer = qa_chain.run(query)
        except Exception as e:
            answer = f"Error processing query: {str(e)}"

    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # For Render
    app.run(host="0.0.0.0", port=port)