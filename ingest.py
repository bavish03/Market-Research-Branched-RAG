import os
import pandas as pd
import chromadb

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# -------------------------
# Load Embedding Model
# -------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded")

# -------------------------
# Chroma Database
# -------------------------

client = chromadb.PersistentClient(path="./chroma_db")
company_collection = client.get_or_create_collection(name="companies")
industry_collection = client.get_or_create_collection(name="industry")
news_collection = client.get_or_create_collection(name="news")

# -------------------------
# Read PDFs
# -------------------------

def process_pdf_folder(folder_path, collection):
    all_chunks = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            print(f"Reading {file}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
    print(f"Chunks found: {len(all_chunks)}")

    for i, chunk in enumerate(all_chunks):
        embedding = embedding_model.encode(chunk.page_content).tolist()
        collection.add(ids=[str(i)], documents=[chunk.page_content], embeddings=[embedding])
        
process_pdf_folder("data/companies", company_collection)
process_pdf_folder("data/industry", industry_collection)
print("Processing News Dataset...")

df = pd.read_csv(r"C:\Users\WELCOME\OneDrive\Desktop\MarketResearchRAG\data\news\train.csv")
df = df.head(500)
print(df.columns)

for i, row in df.iterrows():
    text = (
        str(row["Title"])
        + " "
        + str(row["Description"])
    )
    embedding = embedding_model.encode(text).tolist()
    news_collection.add(ids=[str(i)], documents=[text], embeddings=[embedding])
print("News data stored successfully")