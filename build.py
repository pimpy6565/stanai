from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

PDF_PATH = "union_contract.pdf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
FAISS_PATH = "faiss_index"

if __name__ == "__main__":
    # Load PDF
    pdf_loader = PyPDFLoader(PDF_PATH)
    pdf_pages = pdf_loader.load()
    
    # Load TXT
    txt_loader = TextLoader("union.txt")
    txt_pages = txt_loader.load()
    
    # Combine documents
    all_pages = pdf_pages + txt_pages
    
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(all_pages)

    # Embed and build FAISS
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save to disk
    vectorstore.save_local(FAISS_PATH)
    print("FAISS index built and saved.")
