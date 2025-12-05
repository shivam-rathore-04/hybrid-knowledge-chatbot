import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

def get_web_search_tool():
    """
    Returns a configured Tavily Search tool optimized for LCEL chains.
    It takes a query string and returns search results.
    """
    # Ensure API Key is loaded
    if not os.getenv("TAVILY_API_KEY"):
        # You can hardcode it here for testing if .env fails:
        # os.environ["TAVILY_API_KEY"] = "your-key-here"
        raise ValueError("CRITICAL: TAVILY_API_KEY is missing from environment variables.")

    # Initialize the tool
    # k=3 means it fetches 3 search results
    # .invoke("query") will return a list of dictionaries
    tool = TavilySearchResults(k=3)
    
    return tool