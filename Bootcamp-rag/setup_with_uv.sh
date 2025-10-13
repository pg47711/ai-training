#!/bin/bash
# 🚀 Quick Setup Script for RAG Demo using UV
# 
# This script sets up the entire RAG Demo environment in < 1 minute
# Uses UV (https://github.com/astral-sh/uv) - a blazing fast Python package manager

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "🚀 RAG Demo - Quick Setup with UV"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}📥 UV not found. Installing UV...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo -e "${GREEN}✅ UV installed!${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Please restart your terminal and run this script again${NC}"
    exit 0
fi

echo -e "${GREEN}✅ UV is installed${NC}"
echo ""

# Create virtual environment
echo -e "${BLUE}📦 Creating virtual environment...${NC}"
uv venv .venv
echo -e "${GREEN}✅ Virtual environment created: .venv${NC}"
echo ""

# Activate virtual environment
echo -e "${BLUE}🔌 Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Install dependencies with uv (FAST!)
echo -e "${BLUE}📥 Installing dependencies with UV (this is FAST!)...${NC}"
uv pip install -e .
echo -e "${GREEN}✅ All dependencies installed!${NC}"
echo ""

# Install Jupyter for notebooks
echo -e "${BLUE}📓 Installing Jupyter for notebooks...${NC}"
uv pip install jupyter jupyterlab ipywidgets
echo -e "${GREEN}✅ Jupyter installed!${NC}"
echo ""

# Check Milvus
echo "════════════════════════════════════════════════════════════════"
echo "🔍 Checking Milvus..."
echo "════════════════════════════════════════════════════════════════"
if docker ps | grep -q milvus; then
    echo -e "${GREEN}✅ Milvus is running${NC}"
else
    echo -e "${YELLOW}⚠️  Milvus is not running${NC}"
    echo ""
    echo "To start Milvus:"
    echo "  docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest"
    echo ""
fi

# Check Ollama
echo "════════════════════════════════════════════════════════════════"
echo "🦙 Checking Ollama (for RAGAS evaluation)..."
echo "════════════════════════════════════════════════════════════════"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama is installed${NC}"
    
    # Check if gemma2:2b is pulled
    if ollama list | grep -q "gemma2:2b"; then
        echo -e "${GREEN}✅ gemma2:2b model is available${NC}"
    else
        echo -e "${YELLOW}⚠️  gemma2:2b model not pulled${NC}"
        echo ""
        echo "To pull the model:"
        echo "  ollama pull gemma2:2b"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠️  Ollama is not installed${NC}"
    echo ""
    echo "To install Ollama:"
    echo "  macOS: brew install ollama"
    echo "  Linux: curl https://ollama.ai/install.sh | sh"
    echo "  Windows: Download from https://ollama.ai/download"
    echo ""
    echo "Then pull the model:"
    echo "  ollama pull gemma2:2b"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
echo "✅ Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}🎉 Your RAG Demo environment is ready!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Start Jupyter:"
echo "   jupyter lab"
echo ""
echo "2. Or run agentic RAG:"
echo "   python agentic_rag.py"
echo ""
echo "3. Or visualize workflows:"
echo "   python visualize_workflows.py"
echo ""
echo "📚 Documentation:"
echo "   - QUICK_REFERENCE.md - Quick start"
echo "   - FINAL_SUMMARY.md - Complete overview"
echo "   - TEACHING_GUIDE.md - How to teach this"
echo ""
echo "════════════════════════════════════════════════════════════════"
