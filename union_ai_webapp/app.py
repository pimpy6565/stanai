from flask import Flask, request, render_template_string
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)

# Load PDF once
pdf_path = "union_contract.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Use HuggingFace model for FREE
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding_model)

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature":0.2, "max_length":256}
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Union Contract Assistant</title></head>
<body>
  <h1>Ask Your Union Contract</h1>
  <form method="POST">
    <input type="text" name="question" size="60" placeholder="Ask something...">
    <input type="submit" value="Ask">
  </form>
  {% if answer %}
    <h2>Answer:</h2>
    <p>{{ answer }}</p>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        query = request.form["question"]
        answer = qa.run(query)
    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))