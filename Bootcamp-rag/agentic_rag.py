"""
🤖 Agentic RAG System with LangGraph

An intelligent RAG system that can:
- Decide when to retrieve more information
- Use multiple tools (Milvus search, metadata filtering, web search)
- Perform multi-step reasoning
- Self-critique and refine answers
- Handle complex queries requiring multiple retrievals

Author: Prabhu & Claude Sonnet
Date: 2025-10-13
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_milvus import Milvus
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict, List, Literal
import requests
import urllib3

# ============================================================================
# Configuration
# ============================================================================

# Milvus Configuration
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "sec_filings"
EMBED_MODEL = "all-MiniLM-L6-v2"

# NetApp LLM Configuration
NETAPP_LLM_ENDPOINT = "https://llm-proxy-api.ai.eng.netapp.com/v1/completions"
NETAPP_API_KEY = ""
NETAPP_USER = "pg47711"
NETAPP_MODEL = "gpt-5"

# SSL Fix
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# State Definition
# ============================================================================

class AgenticRAGState(TypedDict):
    """State for the agentic RAG workflow"""
    question: str                    # Original user question
    messages: List[dict]             # Conversation history
    retrieved_docs: List[Document]   # All retrieved documents
    current_answer: str              # Current answer being built
    needs_more_info: bool            # Does agent need more retrieval?
    retrieval_count: int             # Number of retrievals performed
    final_answer: str                # Final polished answer
    reasoning: List[str]             # Agent's reasoning steps
    confidence_score: float          # Confidence in answer (0-1)

# ============================================================================
# Initialize Components
# ============================================================================

print("🔧 Initializing Agentic RAG System...")

# Load embedding model
embedder = SentenceTransformer(EMBED_MODEL)

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

embeddings = SentenceTransformerEmbeddings(embedder)

# Connect to Milvus
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    text_field="text",
    vector_field="embedding",
    auto_id=True
)

print(f"✅ Connected to Milvus: {COLLECTION_NAME}")

# ============================================================================
# Agent Tools
# ============================================================================

def call_netapp_llm(prompt: str, max_tokens: int = 512) -> str:
    """Call NetApp LLM with a prompt"""
    try:
        response = requests.post(
            NETAPP_LLM_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {NETAPP_API_KEY}"
            },
            json={
                "model": NETAPP_MODEL,
                "user": NETAPP_USER,
                "prompt": prompt
            },
            timeout=60,
            verify=False
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)[:100]}"


def retrieve_documents(query: str, filter_expr: str = "", k: int = 5) -> List[Document]:
    """Tool: Retrieve documents from Milvus"""
    search_kwargs = {"k": k}
    if filter_expr:
        search_kwargs["expr"] = filter_expr
    
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(query)
    return docs


def analyze_query(query: str) -> dict:
    """Tool: Analyze query to determine retrieval strategy"""
    prompt = f"""Analyze this question and suggest a retrieval strategy.

Question: {query}

Provide in JSON format:
{{
    "companies": ["NVDA", "AMZN"],  // List of companies mentioned or leave empty for all
    "time_period": "2023 Q3",  // Specific time or "all"
    "requires_comparison": true/false,  // Does it compare companies/periods?
    "query_type": "revenue/challenges/growth/general"
}}

Answer with only the JSON:"""
    
    response = call_netapp_llm(prompt, max_tokens=200)
    
    # Parse or return simple analysis
    return {"analysis": response, "original_query": query}

# ============================================================================
# Agentic Workflow Nodes
# ============================================================================

def analyze_question(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 1: Analyze the question to plan retrieval strategy
    
    The agent analyzes:
    - What companies are mentioned?
    - What time period is relevant?
    - Is this a comparison query?
    - What type of information is needed?
    """
    question = state["question"]
    
    print(f"\n🤔 STEP 1: Analyzing question...")
    print(f"   Question: {question}")
    
    # Use LLM to analyze query
    analysis_prompt = f"""Analyze this question about SEC filings and extract key information:

Question: {question}

Identify:
1. Companies mentioned (AAPL, AMZN, INTC, MSFT, NVDA, or "all")
2. Time period (specific quarter/year or "all periods")
3. Query type (revenue, challenges, comparison, growth, general)
4. Requires multiple retrievals? (yes/no)

Provide brief analysis:"""
    
    analysis = call_netapp_llm(analysis_prompt, max_tokens=150)
    
    reasoning = [f"Query analysis: {analysis}"]
    
    print(f"   📊 Analysis: {analysis[:100]}...")
    
    return {
        **state,
        "reasoning": reasoning,
        "retrieval_count": 0,
        "needs_more_info": True
    }


def initial_retrieval(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 2: Perform initial retrieval based on analysis
    
    The agent retrieves relevant documents using:
    - Smart metadata filtering
    - Appropriate K value
    - Query understanding from analysis
    """
    question = state["question"]
    
    print(f"\n🔍 STEP 2: Initial retrieval...")
    
    # Intelligent filter selection based on question
    filter_expr = ""
    k = 5
    
    # Simple keyword-based filter detection (in production, use LLM analysis)
    question_lower = question.lower()
    
    # Detect company
    company_map = {
        "nvidia": "NVDA", "nvda": "NVDA",
        "amazon": "AMZN", "amzn": "AMZN", "aws": "AMZN",
        "intel": "INTC", "intc": "INTC",
        "microsoft": "MSFT", "msft": "MSFT", "azure": "MSFT",
        "apple": "AAPL", "aapl": "AAPL"
    }
    
    detected_companies = []
    for keyword, ticker in company_map.items():
        if keyword in question_lower:
            detected_companies.append(ticker)
    
    # Detect year/quarter
    if "2023" in question:
        if "q3" in question_lower or "third quarter" in question_lower:
            if detected_companies:
                filter_expr = f'company == "{detected_companies[0]}" and year == 2023 and quarter == "Q3"'
        elif "q1" in question_lower:
            if detected_companies:
                filter_expr = f'company == "{detected_companies[0]}" and year == 2023 and quarter == "Q1"'
        else:
            if detected_companies:
                filter_expr = f'company == "{detected_companies[0]}" and year == 2023'
    elif detected_companies:
        filter_expr = f'company == "{detected_companies[0]}"'
    
    # For comparison queries, retrieve more docs
    if "compare" in question_lower or "versus" in question_lower or len(detected_companies) > 1:
        k = 10
    
    print(f"   Filter: {filter_expr if filter_expr else 'None (search all)'}")
    print(f"   K: {k}")
    
    # Retrieve documents
    docs = retrieve_documents(question, filter_expr, k)
    
    print(f"   ✅ Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs[:3], 1):
        print(f"      {i}. {doc.metadata.get('filename')} (Page {doc.metadata.get('page')})")
    
    reasoning = state.get("reasoning", [])
    reasoning.append(f"Retrieved {len(docs)} documents with filter: {filter_expr}")
    
    return {
        **state,
        "retrieved_docs": docs,
        "retrieval_count": 1,
        "reasoning": reasoning
    }


def generate_draft_answer(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 3: Generate initial answer from retrieved documents
    
    The agent creates a draft answer using the retrieved context.
    """
    question = state["question"]
    docs = state["retrieved_docs"]
    
    print(f"\n💡 STEP 3: Generating draft answer...")
    
    # Format context from top documents
    context_text = "\n\n".join([
        f"[Source: {doc.metadata.get('filename')}, Page {doc.metadata.get('page')}, "
        f"{doc.metadata.get('company')} {doc.metadata.get('quarter')} {doc.metadata.get('year')}]\n"
        f"{doc.page_content[:600]}"
        for doc in docs[:5]
    ])
    
    # Create prompt for draft answer
    prompt = f"""You are a financial analyst answering questions about SEC filings. 
Use the retrieved context to answer the question. Be specific and cite sources.
If information is missing, acknowledge it.

Question: {question}

Retrieved Context:
{context_text}

Provide a detailed answer with specific numbers and sources:"""
    
    draft_answer = call_netapp_llm(prompt, max_tokens=500)
    
    print(f"   ✅ Draft answer generated ({len(draft_answer)} chars)")
    print(f"   Preview: {draft_answer[:150]}...")
    
    reasoning = state.get("reasoning", [])
    reasoning.append(f"Generated draft answer from {len(docs)} documents")
    
    return {
        **state,
        "current_answer": draft_answer,
        "reasoning": reasoning
    }


def evaluate_answer_quality(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 4: Agent evaluates its own answer and decides if more retrieval is needed
    
    The agent performs self-critique:
    - Is the answer complete?
    - Are there gaps in information?
    - Should we retrieve more documents?
    - Should we use different filters?
    """
    question = state["question"]
    answer = state["current_answer"]
    retrieval_count = state["retrieval_count"]
    
    print(f"\n🔍 STEP 4: Self-evaluation...")
    
    # Agent critiques its own answer
    critique_prompt = f"""Evaluate this answer for completeness and accuracy.

Original Question: {question}

Answer Generated: {answer}

Self-critique:
1. Does this fully answer the question? (yes/no)
2. What information is missing, if any?
3. Should we retrieve more documents? (yes/no)
4. Confidence score (0-1)?

Provide brief analysis:"""
    
    critique = call_netapp_llm(critique_prompt, max_tokens=200)
    
    print(f"   📊 Self-critique: {critique[:100]}...")
    
    # Decide if more retrieval is needed (simple heuristic)
    # In production, parse LLM critique more carefully
    needs_more = "yes" in critique.lower() and "missing" in critique.lower() and retrieval_count < 3
    
    # Extract confidence (simple parsing)
    confidence = 0.8  # Default
    if "0." in critique:
        try:
            # Try to extract a decimal number
            import re
            match = re.search(r'0\.\d+', critique)
            if match:
                confidence = float(match.group())
        except:
            pass
    
    reasoning = state.get("reasoning", [])
    reasoning.append(f"Self-critique: {critique[:100]}... | Needs more: {needs_more}")
    
    print(f"   Needs more info: {needs_more}")
    print(f"   Confidence: {confidence:.2f}")
    
    return {
        **state,
        "needs_more_info": needs_more,
        "confidence_score": confidence,
        "reasoning": reasoning
    }


def additional_retrieval(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 5: Perform additional targeted retrieval based on gaps
    
    The agent retrieves more specific information to fill gaps.
    """
    question = state["question"]
    current_docs = state["retrieved_docs"]
    
    print(f"\n🔍 STEP 5: Additional retrieval for missing information...")
    
    # Use broader filter or different query
    # In this example, we retrieve with no filter or different time period
    new_docs = retrieve_documents(question, filter_expr="", k=5)
    
    # Combine with existing docs (avoiding duplicates by ID if possible)
    all_docs = current_docs + [doc for doc in new_docs if doc not in current_docs]
    
    print(f"   ✅ Retrieved {len(new_docs)} additional documents")
    print(f"   Total documents: {len(all_docs)}")
    
    reasoning = state.get("reasoning", [])
    reasoning.append(f"Retrieved {len(new_docs)} additional documents")
    
    return {
        **state,
        "retrieved_docs": all_docs,
        "retrieval_count": state["retrieval_count"] + 1,
        "reasoning": reasoning
    }


def refine_answer(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 6: Refine the answer with additional information
    
    The agent incorporates new information into a better answer.
    """
    question = state["question"]
    previous_answer = state["current_answer"]
    docs = state["retrieved_docs"]
    
    print(f"\n✨ STEP 6: Refining answer with additional context...")
    
    # Format all context
    context_text = "\n\n".join([
        f"[{doc.metadata.get('filename')}, Page {doc.metadata.get('page')}]\n{doc.page_content[:500]}"
        for doc in docs[:8]
    ])
    
    # Refine with more context
    refine_prompt = f"""You previously answered this question but found gaps. 
Now refine your answer with this additional context.

Question: {question}

Previous Answer: {previous_answer}

Additional Context:
{context_text}

Provide a refined, complete answer with specific details and sources:"""
    
    refined_answer = call_netapp_llm(refine_prompt, max_tokens=600)
    
    print(f"   ✅ Answer refined ({len(refined_answer)} chars)")
    
    reasoning = state.get("reasoning", [])
    reasoning.append(f"Refined answer with {len(docs)} total documents")
    
    return {
        **state,
        "current_answer": refined_answer,
        "reasoning": reasoning,
        "needs_more_info": False  # Done refining
    }


def finalize_answer(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 7: Finalize and format the answer
    
    The agent polishes the answer and adds citations.
    """
    answer = state["current_answer"]
    docs = state["retrieved_docs"]
    
    print(f"\n📝 STEP 7: Finalizing answer...")
    
    # Format sources
    sources = []
    seen_sources = set()
    for doc in docs[:5]:
        source_key = f"{doc.metadata.get('filename')}-{doc.metadata.get('page')}"
        if source_key not in seen_sources:
            sources.append(
                f"- {doc.metadata.get('filename')} (Page {doc.metadata.get('page')}, "
                f"{doc.metadata.get('company')} {doc.metadata.get('quarter')} {doc.metadata.get('year')})"
            )
            seen_sources.add(source_key)
    
    # Add sources to answer
    final_answer = f"""{answer}

📚 Sources:
{chr(10).join(sources[:5])}

📊 Retrieved from {len(state['retrieved_docs'])} documents across {state['retrieval_count']} retrieval(s)
"""
    
    print(f"   ✅ Answer finalized with {len(sources)} sources")
    
    return {
        **state,
        "final_answer": final_answer
    }


def should_retrieve_more(state: AgenticRAGState) -> Literal["refine", "finalize"]:
    """
    Decision function: Should we retrieve more or finalize?
    
    This is where the agent makes decisions!
    """
    needs_more = state.get("needs_more_info", False)
    retrieval_count = state.get("retrieval_count", 0)
    confidence = state.get("confidence_score", 0.5)
    
    # Agent's decision logic
    if needs_more and retrieval_count < 3 and confidence < 0.9:
        print(f"\n🤖 DECISION: Retrieve more information (attempt {retrieval_count + 1}/3)")
        return "refine"
    else:
        print(f"\n🤖 DECISION: Finalize answer (confidence: {confidence:.2f})")
        return "finalize"

# ============================================================================
# Build Agentic Workflow
# ============================================================================

print("\n🔧 Building agentic workflow...")

# Create the graph
workflow = StateGraph(AgenticRAGState)

# Add all nodes
workflow.add_node("analyze", analyze_question)
workflow.add_node("retrieve", initial_retrieval)
workflow.add_node("generate", generate_draft_answer)
workflow.add_node("evaluate", evaluate_answer_quality)
workflow.add_node("additional_retrieval", additional_retrieval)
workflow.add_node("refine", refine_answer)
workflow.add_node("finalize", finalize_answer)

# Define the flow
workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "evaluate")

# CONDITIONAL EDGE: Agent decides next step
workflow.add_conditional_edges(
    "evaluate",
    should_retrieve_more,
    {
        "refine": "additional_retrieval",  # Get more info
        "finalize": "finalize"             # We're done
    }
)

# After additional retrieval, refine the answer
workflow.add_edge("additional_retrieval", "refine")

# After refining, evaluate again
workflow.add_edge("refine", "evaluate")

# Finalize leads to end
workflow.add_edge("finalize", END)

# Compile the agentic RAG system
agentic_rag = workflow.compile()

print("✅ Agentic RAG system created!")
print("\n📊 Workflow structure:")
print("   START")
print("   → analyze (understand query)")
print("   → retrieve (initial search)")
print("   → generate (draft answer)")
print("   → evaluate (self-critique)")
print("   → [DECISION]")
print("       ↳ additional_retrieval → refine → evaluate (loop)")
print("       ↳ finalize → END")


# ============================================================================
# Visualization
# ============================================================================

def visualize_workflow(save_to_file: bool = True):
    """
    Visualize the agentic RAG workflow graph.
    
    Args:
        save_to_file: If True, saves PNG. If False, displays in notebook.
    """
    try:
        # Get the graph as Mermaid diagram
        graph_png = agentic_rag.get_graph().draw_mermaid_png()
        
        if save_to_file:
            # Save to file
            with open("agentic_rag_workflow.png", "wb") as f:
                f.write(graph_png)
            print("✅ Workflow diagram saved to: agentic_rag_workflow.png")
        else:
            # Display in notebook
            from IPython.display import Image, display
            display(Image(graph_png))
            print("✅ Workflow diagram displayed above")
        
        return graph_png
    
    except Exception as e:
        print(f"⚠️  Could not generate graph image: {e}")
        print("\n📊 ASCII Workflow Visualization:\n")
        print_ascii_workflow()


def print_ascii_workflow():
    """Print ASCII art version of the workflow"""
    print("""
┌──────────────────┐
│      START       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   ANALYZE        │ ← LLM analyzes query
│   (Node 1)       │   - Extract companies
└────────┬─────────┘   - Detect time period
         │              - Understand intent
         ▼
┌──────────────────┐
│   RETRIEVE       │ ← Smart retrieval
│   (Node 2)       │   - Auto-generate filters
└────────┬─────────┘   - Choose K value
         │
         ▼
┌──────────────────┐
│   GENERATE       │ ← Create draft answer
│   (Node 3)       │   - Use retrieved context
└────────┬─────────┘   - Cite sources
         │
         ▼
┌──────────────────┐
│   EVALUATE       │ ← Self-critique
│   (Node 4)       │   - Is it complete?
└────────┬─────────┘   - Missing info?
         │              - Confidence score?
         ▼
    [DECISION]
         │
    ┌────┴────┐
    │         │
    ▼         ▼
NEEDS     CONFIDENT
MORE      (>0.9)
    │         │
    ▼         │
┌──────────────────┐  │
│ ADDITIONAL       │  │
│ RETRIEVAL        │  │
│ (Node 5)         │  │
└────────┬─────────┘  │
         │            │
         ▼            │
┌──────────────────┐  │
│   REFINE         │  │
│   (Node 6)       │  │
└────────┬─────────┘  │
         │            │
         └───┬────────┘
             │
             ▼
      (Re-evaluate)
             │
             ▼
┌──────────────────┐
│   FINALIZE       │ ← Format answer
│   (Node 7)       │   - Add citations
└────────┬─────────┘   - Add metadata
         │
         ▼
┌──────────────────┐
│       END        │
└──────────────────┘

🎯 Key Features:
- Conditional branching at EVALUATE node
- Potential loop: evaluate → retrieve → refine → evaluate
- Maximum 3 retrieval attempts
- Stops when confidence > 0.9
    """)


def get_workflow_stats():
    """Get statistics about the workflow"""
    graph = agentic_rag.get_graph()
    
    stats = {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "conditional_edges": len([e for e in graph.edges if hasattr(e, 'condition')]),
        "max_depth": "Variable (1-3 loops)",
        "avg_llm_calls": "3-5",
        "avg_retrievals": "1-3"
    }
    
    print("📊 Workflow Statistics:")
    for key, value in stats.items():
        print(f"   {key:20s}: {value}")
    
    return stats

# ============================================================================
# Helper Functions
# ============================================================================

def ask_agentic_rag(question: str) -> dict:
    """
    Ask a question to the agentic RAG system.
    
    The agent will:
    1. Analyze the question
    2. Retrieve relevant documents
    3. Generate a draft answer
    4. Self-evaluate
    5. Retrieve more if needed
    6. Refine the answer
    7. Finalize with sources
    
    Args:
        question: The question to ask
        
    Returns:
        dict: Result with final_answer, reasoning, confidence_score
    """
    print("="*80)
    print(f"❓ QUESTION: {question}")
    print("="*80)
    
    result = agentic_rag.invoke({
        "question": question,
        "messages": [],
        "retrieved_docs": [],
        "current_answer": "",
        "needs_more_info": True,
        "retrieval_count": 0,
        "final_answer": "",
        "reasoning": [],
        "confidence_score": 0.0
    })
    
    print("\n" + "="*80)
    print("📝 FINAL ANSWER:")
    print("="*80)
    print(result["final_answer"])
    print("\n" + "="*80)
    print("🧠 AGENT REASONING:")
    print("="*80)
    for i, step in enumerate(result["reasoning"], 1):
        print(f"{i}. {step}")
    print("="*80)
    print(f"\n📊 Confidence: {result['confidence_score']:.2f}")
    print(f"📊 Total Retrievals: {result['retrieval_count']}")
    print("="*80)
    
    return result

# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🤖 AGENTIC RAG SYSTEM DEMO")
    print("="*80)
    
    # Visualize the workflow
    print("\n🎨 Generating workflow visualization...")
    visualize_workflow(save_to_file=True)
    print("\n📊 Workflow Statistics:")
    get_workflow_stats()
    print("\n" + "="*80)
    
    # Demo questions showcasing agentic capabilities
    demo_questions = [
        "What was NVIDIA's Q3 2023 revenue and how does it compare to Q2?",
        "What challenges did Intel face in 2023 and what was their response?",
        "How did Amazon's AWS business perform across 2023?",
    ]
    
    print("\n🎯 Running agentic queries (the agent will think, retrieve, and refine)...\n")
    
    for i, question in enumerate(demo_questions[:1], 1):  # Run first question as demo
        print(f"\n{'='*80}")
        print(f"DEMO QUERY {i}/{len(demo_questions)}")
        print(f"{'='*80}\n")
        
        result = ask_agentic_rag(question)
        
        input("\n⏸️  Press Enter to continue to next demo query...")
    
    print("\n" + "="*80)
    print("🎉 AGENTIC RAG DEMO COMPLETE!")
    print("="*80)
    print("\n💡 Key Differences from Simple RAG:")
    print("   1. ✅ Agent analyzes queries before retrieving")
    print("   2. ✅ Intelligent filter selection")
    print("   3. ✅ Self-evaluation and critique")
    print("   4. ✅ Adaptive retrieval (can get more info if needed)")
    print("   5. ✅ Multi-step reasoning")
    print("   6. ✅ Confidence scoring")
    print("   7. ✅ Complete reasoning trace")
    
    print("\n📚 To use in your code:")
    print("""
    from agentic_rag import ask_agentic_rag
    
    result = ask_agentic_rag("Your complex question here")
    print(result["final_answer"])
    print(f"Confidence: {result['confidence_score']}")
    """)
    
    print("\n🎓 Compare this to Retrieval.ipynb to see the differences!")
    
    print("\n" + "="*80)
    print("🎨 WORKFLOW VISUALIZATION")
    print("="*80)
    print("\n💡 Visual graph saved to: agentic_rag_workflow.png")
    print("   Open this file to see the complete workflow diagram!")
    print("\n💡 Or view the ASCII version above")
    print("="*80)
