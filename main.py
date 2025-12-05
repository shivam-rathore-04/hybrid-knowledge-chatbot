import streamlit as st
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# LCEL Imports (The building blocks of Chains)
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Internal Imports
from rag.embed_and_store import embed_and_store_pdf
from rag.retrieve import get_retriever 
# NOTE: Ensure search.py has 'get_web_search_tool' as defined in previous steps
from internet_agent.search import get_web_search_tool

# Load Env
load_dotenv()

st.set_page_config(page_title="RAG Chains PoC", layout="wide")
st.title("PDF Query Chatbot")

# --- HELPER FUNCTION ---
# This converts the list of Documents (from ChromaDB) into a single string for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# ---------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Flow Control")
    mode = st.radio("Generate Response with:", ["PDF Only", "PDF + Web"])
    
    st.divider()
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file and "vector_db_ready" not in st.session_state:
        with st.spinner("Ingesting PDF..."):
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            embed_and_store_pdf(temp_path)
            st.session_state["vector_db_ready"] = True
            st.success("PDF Vectorized!")
            if os.path.exists(temp_path):
                os.remove(temp_path)

# --- CHAT UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask away..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Running Chain..."):
            try:
                # 1. Setup Common Components
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                output_parser = StrOutputParser()
                
                # --- CHAIN A: SEQUENTIAL (PDF ONLY) ---
                if mode == "PDF Only":
                    if "vector_db_ready" not in st.session_state:
                        st.error("Upload a PDF to use this chain.")
                        st.stop()

                    retriever = get_retriever()
                    
                    # The Prompt
                    template = """"You are a helpful assistant. Use the provided context to answer the question. "
                        "If the answer is not in the context, say you don't know."
                        "\n\nContext
                    {context}
                    
                    Question: {question}
                    """
                    prompt = ChatPromptTemplate.from_template(template)

                    # THE CHAIN DEFINITION
                    # 1. RunnablePassthrough() passes the user input to 'question'
                    # 2. retriever | format_docs passes the retrieved text to 'context'
                    chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | output_parser
                    )
                    
                    response_text = chain.invoke(prompt_input)

                # --- CHAIN B: PARALLEL (PDF + WEB) ---
                else:
                    # Initialize Tools
                    retriever = get_retriever()
                    web_search_tool = get_web_search_tool()

                    # The Prompt (Expects two inputs: pdf_context and web_context)
                    template = """AYou are a helpful assistant. Use the provided context to answer the question.
                    
                    --- PDF CONTEXT ---
                    {pdf_context}
                    
                    --- WEB CONTEXT ---
                    {web_context}
                    
                    Question: {question}
                    """
                    prompt = ChatPromptTemplate.from_template(template)

                    # PARALLEL STEP
                    # This runs the Vector Search AND the Tavily Web Search at the same time.
                    setup_and_retrieval = RunnableParallel(
                        {
                            "pdf_context": retriever | format_docs, 
                            "web_context": lambda x: web_search_tool.invoke(x),
                            "question": RunnablePassthrough()
                        }
                    )
                    
                    # THE FULL CHAIN
                    chain = (
                        setup_and_retrieval
                        | prompt
                        | llm
                        | output_parser
                    )
                    
                    response_text = chain.invoke(prompt_input)

                # Output
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"Chain Error: {str(e)}")