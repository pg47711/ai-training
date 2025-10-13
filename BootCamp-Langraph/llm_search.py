"""
LLM-based search tool to replace Tavily when VPN blocks connectivity.
"""
from langchain_core.tools import tool
from custom_llm import ChatNetAppLLM
from langchain_core.messages import HumanMessage


@tool
def llm_web_search(query: str) -> str:
    """
    Search the web using LLM knowledge base.
    Use this when you need to find current information or facts.
    
    Args:
        query: The search query
        
    Returns:
        Search results as a string
    """
    llm = ChatNetAppLLM()
    
    prompt = f"""You are a helpful search engine. Answer this search query concisely based on your knowledge: {query}

Provide accurate, factual information. If you're not sure about current/real-time data, mention that."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def get_search_tool():
    """Get the LLM-based search tool."""
    return llm_web_search
