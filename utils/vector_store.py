import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


DATA_DIR = "Data"
VECTOR_DB_PATH = "faiss_store"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents():
    documents = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_DIR, file))
            documents.extend(loader.load())
    return documents

def create_vector_store():
    print("Creating new vector store...")
    documents = load_documents()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_DB_PATH)
    print("Vector Store Saved.")

def load_vector_store():
    if not os.path.exists(VECTOR_DB_PATH):
        create_vector_store()
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
