# LangGraph Bootcamp

A comprehensive hands-on learning project for mastering LangGraph - a framework for building stateful, multi-actor applications with LLMs.

## 📚 What's Included

This bootcamp covers essential LangGraph concepts through interactive Jupyter notebooks and Python scripts:

1. **Simple Graph** (`1_simple_graph.ipynb`) - Introduction to basic graph structures
2. **Graph with Conditions** (`2_graph_with_condition.ipynb`) - Adding conditional logic to graphs
3. **Chatbot** (`3_chatbot.ipynb`) - Building a conversational agent
4. **Tool Calling** (`4_tool_call.ipynb`) - Integrating external tools
5. **Tool Call Agent** (`5_tool_call_agent.ipynb`) - Creating agents that use tools
6. **Memory** (`6_memory.ipynb`) - Implementing conversation memory
7. **LangSmith Tracing** (`7_langsmith_tracing.ipynb`) - Debugging and tracing with LangSmith
8. **Human-in-the-Loop** (`8_HITL.py`) - Interactive workflows requiring human input
9. **Multi-Agent Workflow** (`9.Multi_agent_workflow.ipynb`) - Coordinating multiple AI agents

## 🔧 Prerequisites

- **Python 3.9 or higher**
- **uv** (Python package manager) - [Install from here](https://docs.astral.sh/uv/getting-started/installation/)
- **Jupyter Notebook** support
- API Keys (see Environment Setup below)

## 🚀 Installation

### Step 1: Clone or Download the Repository

```bash
cd /path/to/langgraph-bootcamp
```

### Step 2: Install Dependencies with uv

If you don't have `uv` installed, install it first:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

Then install project dependencies:

```bash
# Create virtual environment and install dependencies
uv sync
```

This will:
- Create a virtual environment in `.venv`
- Install all dependencies from `pyproject.toml`
- Lock dependencies in `uv.lock`

### Alternative: Install with pip

If you prefer using pip:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -e .
```

## 🔐 Environment Setup

### Step 1: Create Environment File

Copy the sample environment file:

```bash
cp sample.env .env
```

### Step 2: Configure API Keys

Edit `.env` and add your API keys:

```bash
# Required: NetApp LLM API Token
NETAPP_LLM_TOKEN=your_netapp_token_here

# Required: LangSmith API Key (for tracing/debugging)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT="langgraph-learning"

# Optional: Tavily API Key (for web search)
TAVILY_API_KEY=your_tavily_key_here
```

### Where to Get API Keys

- **NetApp LLM Token**: Get from your NetApp AI Platform portal
- **LangSmith API Key**: Sign up at [https://smith.langchain.com](https://smith.langchain.com)
- **Tavily API Key** (Optional): Sign up at [https://tavily.com](https://tavily.com)

> **⚠️ Important**: Never commit your `.env` file to version control! It's already included in `.gitignore`.

## 📓 Running the Notebooks

### Option 1: Using Jupyter Notebook

```bash
# Activate virtual environment (if using uv)
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Start Jupyter Notebook
jupyter notebook
```

Then navigate to any notebook in your browser and run the cells.

### Option 2: Using VS Code

1. Open the project in VS Code
2. Install the **Jupyter** extension
3. Open any `.ipynb` file
4. Select the Python interpreter from `.venv` (bottom right corner)
5. Run cells using the play button or `Shift+Enter`

### Option 3: Using JupyterLab

```bash
# Install JupyterLab (if not already installed)
uv pip install jupyterlab

# Activate virtual environment
source .venv/bin/activate

# Start JupyterLab
jupyter lab
```

## 🐍 Running Python Scripts

For the Python scripts (like `8_HITL.py`):

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the script
python 8_HITL.py
```

## 📖 Learning Path

Follow the notebooks in order for the best learning experience:

1. Start with **Simple Graph** to understand the basics
2. Progress through **Conditions** and **Chatbot** to build interactive flows
3. Learn **Tool Calling** to integrate external functionality
4. Explore **Memory** for stateful conversations
5. Use **LangSmith Tracing** for debugging
6. Master **Human-in-the-Loop** for interactive workflows
7. Complete with **Multi-Agent Workflows** for complex systems

## 🛠️ Utility Scripts

- `custom_llm.py` - Custom LLM implementation helpers
- `llm_search.py` - LLM-powered search functionality

## 📦 Project Structure

```
langgraph-bootcamp/
├── 1_simple_graph.ipynb           # Basic graph structures
├── 2_graph_with_condition.ipynb   # Conditional logic
├── 3_chatbot.ipynb                # Conversational agents
├── 4_tool_call.ipynb              # Tool integration
├── 5_tool_call_agent.ipynb        # Tool-using agents
├── 6_memory.ipynb                 # Conversation memory
├── 7_langsmith_tracing.ipynb      # Debugging with LangSmith
├── 8_HITL.py                      # Human-in-the-loop
├── 9.Multi_agent_workflow.ipynb   # Multi-agent systems
├── custom_llm.py                  # Custom LLM helpers
├── llm_search.py                  # Search utilities
├── pyproject.toml                 # Project dependencies
├── uv.lock                        # Locked dependencies
├── sample.env                     # Environment template
└── README.md                      # This file
```

## 🔍 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` when running notebooks

**Solution**: Make sure your Jupyter kernel is using the `.venv` Python interpreter

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=langgraph-bootcamp
```

Then select the `langgraph-bootcamp` kernel in Jupyter.

---

**Issue**: API key errors

**Solution**: 
1. Verify your `.env` file exists and contains valid keys
2. Make sure you're loading environment variables (notebooks should do this automatically)
3. Check that your API keys are active and have the correct permissions

---

**Issue**: `uv: command not found`

**Solution**: Install uv following the official instructions:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)

## 💡 Tips

- Use LangSmith tracing to debug your LangGraph applications
- Start with simple examples and gradually increase complexity
- Experiment with different LLM models and parameters
- Keep your API keys secure and never share them

## 🤝 Contributing

This is a learning project. Feel free to experiment, modify, and extend the examples!

## 📄 License

This project is for educational purposes.

---

**Happy Learning! 🚀**
