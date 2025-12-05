from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

PERSIST_DIRECTORY = "./vector_db"

def get_retriever():
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function
    )
    
    # --- FIX: USE MMR SEARCH ---
    # "mmr" (Maximal Marginal Relevance) is better for finding details.
    # It fetches diverse chunks instead of just the most similar ones.
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    return retriever