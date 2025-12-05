import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_chroma import Chroma

# Path where vector database will be saved
PERSIST_DIRECTORY = "./vector_db"

def embed_and_store_pdf(pdf_path):
    """
    1. Loads PDF
    2. Splits into chunks
    3. Embeds using Gemini and stores in ChromaDB
    """
    
    # 1. Clean up existing DB to start fresh
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    # 2. Load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 3. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    
    # 4. Embed and Store (Gemini)
    # Ensure GOOGLE_API_KEY is in your .env file
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"Successfully saved {len(splits)} chunks to {PERSIST_DIRECTORY} using Gemini Embeddings.")
    return vectorstore