# Hybrid Knowledge Chatbot

A Streamlit-based AI assistant that combines **RAG (Retrieval-Augmented Generation)** with **Web Search**. It allows users to chat with their own PDF documents and, optionally, search the internet for real-time information simultaneously.

Built using **LangChain (LCEL)**, **Google Gemini**, **ChromaDB**, and **Tavily Search**.


## Features

*  PDF Only Mode (Sequential Chain):
    * Fast retrieval from uploaded documents.
    * Uses **MMR (Maximal Marginal Relevance)** to find diverse and detailed context from the PDF.
    * Strictly answers based *only* on the document to reduce hallucinations.

*  PDF + Web Mode (Parallel Chain):
    * Runs **Vector Search** (PDF) and **Tavily Web Search** (Internet) in parallel.
    * Synthesizes information from both sources to answer complex questions (e.g., "Compare the course grade policy with current industry standards").

*  Debug Mode:
    * Built-in "What the Bot is Reading" expander to view exactly which PDF chunks and web results are being fed to the LLM.

##  Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM & Embeddings:** [Google Gemini](https://ai.google.dev/) (gemini-2.5-flash)
* **Orchestration:** [LangChain](https://www.langchain.com/) (using LCEL - LangChain Expression Language)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Web Search:** [Tavily AI](https://tavily.com/)

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/shivam-rathore-04/hybrid-knowledge-chatbot.git]
    ```

2.  **Create a virtual environment:**
    ```bash
      First install uv (pip install uv)
       
      # Create the virtual environment
      uv venv

      # Activate it
      # On Windows:
      .venv\Scripts\activate
      
      # On Mac/Linux:
      source .venv/bin/activate

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_google_gemini_key
    TAVILY_API_KEY=your_tavily_api_key
    ```

##  Usage

Run the Streamlit app:

```bash
streamlit run main.py




